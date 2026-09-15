# SGLang disaggregated TP1-attention + EP2, 4P4D

This recipe deploys a disaggregated SGLang serving topology (4 Prefill + 4 Decode workers across 16 GPUs) for `Qwen/Qwen3.6-35B-A3B-FP8` using NIXL/UCX state transfer and Dynamo KV-aware routing.

### Topology & Configuration Summary

| Component / Feature | Configuration | Details |
| :--- | :--- | :--- |
| **Worker Allocation** | 4 Prefill + 4 Decode (4P4D) | 8 workers × 2 GPUs/worker = 16 GPUs total |
| **Parallelism Specs** | Attention TP=1, EP=2 | `tp-size=2`, `dp-size=2`, `ep-size=2`, `moe-dense-tp-size=1` (DP attention/LM head) |
| **Inter-Pod Transfer** | NIXL over UCX / RDMA | Transfers KV & recurrent state between Prefill and Decode roles |
| **Routing Mode** | Dynamo KV-Aware Routing | Frontend routes requests based on SGLang ZMQ KV cache events (`--router-mode kv`) |
| **Telemetry Ports** | SGLang: `9090`, NIXL: `19090` | Prometheus metrics exported on worker system ports |
| **Deployment Variants** | [`deploy.yaml`](deploy.yaml) / [`deploy-kv-offloading.yaml`](deploy-kv-offloading.yaml) | Baseline vs. optional prefill-only GPU-to-CPU HiCache offload |

> [!NOTE]
> Startup logs and real request validation are mandatory acceptance gates before benchmarking. Decode-side KV offload is disabled due to runtime constraints with hybrid Mamba/GDN state.



## Variables

```bash
export NAMESPACE=dynamo-bench
export EXP_DIR=/ephemeral/shared/qwen3.6-35b-a3b/sglang/disagg/tp1-ep2-4p4d
export DEPLOYMENT=q36-sgl-pd-tp1ep2-4p4d
export PERF_JOB_NAME=qwen36-sglang-tp1ep2-4p4d-perf
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
export MODEL=Qwen/Qwen3.6-35B-A3B-FP8
```

Complete the shared namespace, PVC, model-cache, and RoCE recovery in the
[parent runbook](../README.md) first. This recipe consumes all 16 GPUs; run no
other GPU recipe concurrently.

## Optional KV-offloading configuration

The optional [`deploy-kv-offloading.yaml`](deploy-kv-offloading.yaml) enables GPU-to-CPU HiCache offload on **prefill workers only**, while retaining live NIXL P-to-D transfer over UCX/RDMA.

### Prefill HiCache Settings
- **Host Tier Allocation**: `--hicache-ratio 1.2` (allocates CPU host-cache tier per rank).
- **Layout & Backend**: `--hicache-mem-layout page_first_direct` with `--hicache-io-backend direct` (required for Qwen3.6 hybrid attention/Mamba state).

### Prohibited Flags & Known Runtime Pitfalls

> [!WARNING]
> Do **NOT** add the following unsupported flags for this model/runtime pair:
> 
> 1. **Do not add `--disaggregation-decode-enable-offload-kvcache` or `--disaggregation-decode-enable-radix-cache` on decode workers**:
>    - SGLang 0.5.14 rejects decode offloading/radix cache for Qwen3.6's hybrid `HybridLinearKVPool` (`Unsupported KV cache type for decode offload`).
> 2. **Do not add `--hicache-storage-backend nixl` on prefill workers**:
>    - NIXL HiCache implements v1 storage interface, whereas Qwen3.6 requires v2 (`NotImplementedError` in `batch_exists_v2()`).
> 3. **Do not use `page_first`/`kernel` memory layout**:
>    - Causes `MambaPoolHost` initialization failure. Must use `page_first_direct` with `direct` I/O.

---

## Preflight

Complete the 4-GPU canary in [`preflight.md`](preflight.md) before applying `deploy-kv-offloading.yaml`. It verifies resource checks, single-replica deployment, bounded request tests, CPU HiCache proof, and NIXL transfer counters.



## Production manifest

The baseline production manifest is [deploy.yaml](deploy.yaml). It has four
prefill and four decode replicas, cache reporting, NIXL telemetry, UCX, and
KV-aware routing, but no `--enable-hierarchical-cache` or `--hicache-*` flags.
Use [deploy-kv-offloading.yaml](deploy-kv-offloading.yaml) only when prefill CPU
KV offload is desired and the offload-specific preflight passes.

Create the production manifest on the cluster host:

```bash
mkdir -p "$EXP_DIR"
tee "$EXP_DIR/deploy.yaml" >/dev/null <<'EOF'
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: q36-sgl-pd-tp1ep2-4p4d
spec:
  backendFramework: sglang
  pvcs:
    - name: model-cache
      create: false
  services:
    Frontend:
      componentType: frontend
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0
          workingDir: /workspace/examples/backends/sglang
          command:
            - python3
            - -m
            - dynamo.frontend
          args:
            - --router-mode
            - kv
            - --router-host-cache-hit-weight
            - "0.75"
      envs:
        - name: HF_HOME
          value: /opt/models
      replicas: 6
      resources:
        requests:
          cpu: "16"
          memory: 64Gi
        limits:
          cpu: "32"
          memory: 128Gi
    SglangPrefillWorker:
      componentType: worker
      subComponentType: prefill
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 80Gi
      extraPodMetadata:
        annotations:
          k8s.v1.cni.cncf.io/networks: roce
      extraPodSpec:
        hostNetwork: false
        dnsPolicy: ClusterFirst
        nodeSelector:
          qwen.nvidia.com/role: prefill
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0
          workingDir: /workspace/examples/backends/sglang
          command:
            - /bin/sh
            - -c
          args:
            - |
              set -e
              ulimit -l unlimited
              # Model flags select the local checkpoint and public API model name.
              # Parallelism flags keep attention TP=1 and expert parallelism EP=2 across two GPUs.
              # Context, page, and memory flags size the GPU KV cache and maximum request window.
              # Parser flags enable Qwen reasoning and tool-call response handling through Dynamo.
              # Disaggregation flags make this the prefill role and transfer live state through NIXL.
              # Cache reporting exposes reused prompt-token details without CPU KV offload.
              # KV events feed Dynamo's KV-aware router with prefix-cache residency updates.
              # Metrics exposes SGLang/Dynamo Prometheus series on the worker system port.
              exec python3 -m dynamo.sglang \
                --model-path "$MODEL_PATH" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --tp-size 2 \
                --dp-size 2 \
                --ep-size 2 \
                --enable-dp-attention \
                --enable-dp-lm-head \
                --moe-dense-tp-size 1 \
                --context-length 131072 \
                --page-size 64 \
                --mem-fraction-static 0.85 \
                --reasoning-parser qwen3 \
                --dyn-reasoning-parser qwen3 \
                --dyn-tool-call-parser qwen3_coder \
                --disaggregation-mode prefill \
                --disaggregation-transfer-backend nixl \
                --disaggregation-bootstrap-port 30001 \
                --host 0.0.0.0 \
                --enable-cache-report \
                --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}' \
                --enable-metrics
          env: &worker-env
            - name: SERVED_MODEL_NAME
              value: Qwen/Qwen3.6-35B-A3B-FP8
            - name: MODEL_PATH
              value: /opt/models/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989
            - name: HF_HOME
              value: /opt/models
            - name: PYTHONHASHSEED
              value: "0"
            - name: SGLANG_DISAGGREGATION_NIXL_BACKEND
              value: UCX
            - name: UCX_TLS
              value: rc_x,rc,cuda_copy,cuda_ipc
            - name: UCX_IB_ADDR_TYPE
              value: eth
            - name: UCX_RNDV_SCHEME
              value: get_zcopy
            - name: UCX_RNDV_THRESH
              value: "0"
            - name: UCX_IB_REG_METHODS
              value: odp,rcache
            - name: UCX_RCACHE_MAX_UNRELEASED
              value: "1024"
            - name: UCX_RC_TIMEOUT
              value: 600s
            - name: UCX_KEEPALIVE_INTERVAL
              value: 300s
            - name: UCX_LOG_LEVEL
              value: info
            - name: NIXL_LOG_LEVEL
              value: INFO
            - name: DYN_SYSTEM_PORT
              value: "9090"
            - name: NIXL_TELEMETRY_ENABLE
              value: "y"
            - name: NIXL_TELEMETRY_EXPORTER
              value: prometheus
            - name: NIXL_TELEMETRY_PROMETHEUS_PORT
              value: "19090"
          securityContext: &worker-security-context
            runAsUser: 0
            capabilities:
              add:
                - IPC_LOCK
                - SYS_RESOURCE
      replicas: 4
      resources: &worker-resources
        requests:
          gpu: "2"
          cpu: "16"
          memory: 128Gi
          custom:
            rdma/ib: "2"
        limits:
          gpu: "2"
          cpu: "32"
          memory: 192Gi
          custom:
            rdma/ib: "2"
    SglangDecodeWorker:
      componentType: worker
      subComponentType: decode
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 80Gi
      extraPodMetadata:
        annotations:
          k8s.v1.cni.cncf.io/networks: roce
      extraPodSpec:
        hostNetwork: false
        dnsPolicy: ClusterFirst
        nodeSelector:
          qwen.nvidia.com/role: decode
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0
          workingDir: /workspace/examples/backends/sglang
          command:
            - /bin/sh
            - -c
          args:
            - |
              set -e
              ulimit -l unlimited
              # Model flags select the same checkpoint and API name used by the prefill worker.
              # Parallelism flags must match prefill so transferred KV and recurrent state align.
              # Context, page, and memory flags keep decode allocation compatible with prefill.
              # Parser flags enable the same Qwen reasoning and tool-call response handling.
              # Disaggregation flags make this the decode role and receive live state through NIXL.
              # Cache reporting adds cached-token details; hybrid decode KV offload stays disabled.
              # Metrics exposes SGLang/Dynamo Prometheus series on the worker system port.
              exec python3 -m dynamo.sglang \
                --model-path "$MODEL_PATH" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --tp-size 2 \
                --dp-size 2 \
                --ep-size 2 \
                --enable-dp-attention \
                --enable-dp-lm-head \
                --moe-dense-tp-size 1 \
                --context-length 131072 \
                --page-size 64 \
                --mem-fraction-static 0.85 \
                --reasoning-parser qwen3 \
                --dyn-reasoning-parser qwen3 \
                --dyn-tool-call-parser qwen3_coder \
                --disaggregation-mode decode \
                --disaggregation-transfer-backend nixl \
                --disaggregation-bootstrap-port 30002 \
                --host 0.0.0.0 \
                --enable-cache-report \
                --enable-metrics
          env: *worker-env
          securityContext: *worker-security-context
      replicas: 4
      resources: *worker-resources
EOF
```

## KV-offloading production manifest

Use this optional variant when prefill CPU KV offload is required. It is
identical to the baseline topology except for the prefill hierarchical-cache
flags. It intentionally does not enable decode-side KV offload or the NIXL
HiCache storage backend.

Create the offload manifest on the cluster host:

```bash
tee "$EXP_DIR/deploy-kv-offloading.yaml" >/dev/null <<'EOF'
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: q36-sgl-pd-tp1ep2-4p4d
spec:
  backendFramework: sglang
  pvcs:
    - name: model-cache
      create: false
  services:
    Frontend:
      componentType: frontend
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0
          workingDir: /workspace/examples/backends/sglang
          command:
            - python3
            - -m
            - dynamo.frontend
          args:
            - --router-mode
            - kv
            - --router-host-cache-hit-weight
            - "0.75"
      envs:
        - name: HF_HOME
          value: /opt/models
      replicas: 6
      resources:
        requests:
          cpu: "16"
          memory: 64Gi
        limits:
          cpu: "32"
          memory: 128Gi
    SglangPrefillWorker:
      componentType: worker
      subComponentType: prefill
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 80Gi
      extraPodMetadata:
        annotations:
          k8s.v1.cni.cncf.io/networks: roce
      extraPodSpec:
        hostNetwork: false
        dnsPolicy: ClusterFirst
        nodeSelector:
          qwen.nvidia.com/role: prefill
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0
          workingDir: /workspace/examples/backends/sglang
          command:
            - /bin/sh
            - -c
          args:
            - |
              set -e
              ulimit -l unlimited
              # Model flags select the local checkpoint and public API model name.
              # Parallelism flags keep attention TP=1 and expert parallelism EP=2 across two GPUs.
              # Context, page, and memory flags size the GPU KV cache and maximum request window.
              # Parser flags enable Qwen reasoning and tool-call response handling through Dynamo.
              # Disaggregation flags make this the prefill role and transfer live state through NIXL.
              # Cache flags enable usage reporting and the hybrid-compatible CPU host tier.
              # KV events feed Dynamo's KV-aware router with prefix-cache residency updates.
              # Metrics exposes SGLang/Dynamo Prometheus series on the worker system port.
              exec python3 -m dynamo.sglang \
                --model-path "$MODEL_PATH" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --tp-size 2 \
                --dp-size 2 \
                --ep-size 2 \
                --enable-dp-attention \
                --enable-dp-lm-head \
                --moe-dense-tp-size 1 \
                --context-length 131072 \
                --page-size 64 \
                --mem-fraction-static 0.85 \
                --reasoning-parser qwen3 \
                --dyn-reasoning-parser qwen3 \
                --dyn-tool-call-parser qwen3_coder \
                --disaggregation-mode prefill \
                --disaggregation-transfer-backend nixl \
                --disaggregation-bootstrap-port 30001 \
                --host 0.0.0.0 \
                --enable-cache-report \
                --enable-hierarchical-cache \
                --hicache-ratio 1.2 \
                --hicache-write-policy write_back \
                --hicache-mem-layout page_first_direct \
                --hicache-io-backend direct \
                --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}' \
                --enable-metrics
          env: &worker-env
            - name: SERVED_MODEL_NAME
              value: Qwen/Qwen3.6-35B-A3B-FP8
            - name: MODEL_PATH
              value: /opt/models/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989
            - name: HF_HOME
              value: /opt/models
            - name: PYTHONHASHSEED
              value: "0"
            - name: SGLANG_DISAGGREGATION_NIXL_BACKEND
              value: UCX
            - name: UCX_TLS
              value: rc_x,rc,cuda_copy,cuda_ipc
            - name: UCX_IB_ADDR_TYPE
              value: eth
            - name: UCX_RNDV_SCHEME
              value: get_zcopy
            - name: UCX_RNDV_THRESH
              value: "0"
            - name: UCX_IB_REG_METHODS
              value: odp,rcache
            - name: UCX_RCACHE_MAX_UNRELEASED
              value: "1024"
            - name: UCX_RC_TIMEOUT
              value: 600s
            - name: UCX_KEEPALIVE_INTERVAL
              value: 300s
            - name: UCX_LOG_LEVEL
              value: info
            - name: NIXL_LOG_LEVEL
              value: INFO
            - name: DYN_SYSTEM_PORT
              value: "9090"
            - name: NIXL_TELEMETRY_ENABLE
              value: "y"
            - name: NIXL_TELEMETRY_EXPORTER
              value: prometheus
            - name: NIXL_TELEMETRY_PROMETHEUS_PORT
              value: "19090"
          securityContext: &worker-security-context
            runAsUser: 0
            capabilities:
              add:
                - IPC_LOCK
                - SYS_RESOURCE
      replicas: 4
      resources: &worker-resources
        requests:
          gpu: "2"
          cpu: "16"
          memory: 128Gi
          custom:
            rdma/ib: "2"
        limits:
          gpu: "2"
          cpu: "32"
          memory: 192Gi
          custom:
            rdma/ib: "2"
    SglangDecodeWorker:
      componentType: worker
      subComponentType: decode
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 80Gi
      extraPodMetadata:
        annotations:
          k8s.v1.cni.cncf.io/networks: roce
      extraPodSpec:
        hostNetwork: false
        dnsPolicy: ClusterFirst
        nodeSelector:
          qwen.nvidia.com/role: decode
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0
          workingDir: /workspace/examples/backends/sglang
          command:
            - /bin/sh
            - -c
          args:
            - |
              set -e
              ulimit -l unlimited
              # Model flags select the same checkpoint and API name used by the prefill worker.
              # Parallelism flags must match prefill so transferred KV and recurrent state align.
              # Context, page, and memory flags keep decode allocation compatible with prefill.
              # Parser flags enable the same Qwen reasoning and tool-call response handling.
              # Disaggregation flags make this the decode role and receive live state through NIXL.
              # Cache reporting adds cached-token details; hybrid decode KV offload stays disabled.
              # Metrics exposes SGLang/Dynamo Prometheus series on the worker system port.
              exec python3 -m dynamo.sglang \
                --model-path "$MODEL_PATH" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --tp-size 2 \
                --dp-size 2 \
                --ep-size 2 \
                --enable-dp-attention \
                --enable-dp-lm-head \
                --moe-dense-tp-size 1 \
                --context-length 131072 \
                --page-size 64 \
                --mem-fraction-static 0.85 \
                --reasoning-parser qwen3 \
                --dyn-reasoning-parser qwen3 \
                --dyn-tool-call-parser qwen3_coder \
                --disaggregation-mode decode \
                --disaggregation-transfer-backend nixl \
                --disaggregation-bootstrap-port 30002 \
                --host 0.0.0.0 \
                --enable-cache-report \
                --enable-metrics
          env: *worker-env
          securityContext: *worker-security-context
      replicas: 4
      resources: *worker-resources
EOF
```

The two manifests use the same DynamoGraphDeployment name and are alternatives;
do not apply them as independent deployments at the same time.

## Deployment & Readiness Verification

Choose one deployment manifest to apply (both update the same `DynamoGraphDeployment` resource):

```bash
# Option A: Baseline (no CPU KV offload)
export DEPLOY_FILE="$EXP_DIR/deploy.yaml"

# Option B: Optional Prefill CPU KV Offload
# export DEPLOY_FILE="$EXP_DIR/deploy-kv-offloading.yaml"

# Apply deployment & wait for all Pods to become Ready
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$DEPLOY_FILE"
kubectl apply -n "$NAMESPACE" -f "$DEPLOY_FILE"

kubectl wait -n "$NAMESPACE" --for=condition=Ready pod \
  -l "$GRAPH_LABEL" --timeout=1800s
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  -L nvidia.com/dynamo-component-type -o wide
```

---

## Startup & Transfer Acceptance Gates

### Step 1: Startup Log Gate
Inspect worker logs to ensure no initialization errors or unsupported state exceptions occurred:

```bash
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --all-containers --prefix --tail=1500 |
  tee "$EXP_DIR/qwen36-sglang-tp1ep2-4p4d-startup.log"

# Check for failure keywords
if grep -Ei 'traceback|assertionerror|notimplementederror|nixl_err|out of memory|unsupported' \
  "$EXP_DIR/qwen36-sglang-tp1ep2-4p4d-startup.log"; then
  echo "STOP: SGLang startup compatibility gate failed!" >&2
fi

grep -Ei 'sglang|nixl|ucx|rdma|prefill|decode|transfer|ready' \
  "$EXP_DIR/qwen36-sglang-tp1ep2-4p4d-startup.log" | tail -300
```

### Step 2: In-Cluster NIXL Smoke Test
Execute a test completion request from the frontend Pod to verify NIXL P-to-D state transfer:

```bash
frontend_pod="$(kubectl get pods -n "$NAMESPACE" \
  -l "$GRAPH_LABEL,nvidia.com/dynamo-component-type=frontend" \
  -o jsonpath='{.items[0].metadata.name}')"

timeout 330s kubectl exec -n "$NAMESPACE" "$frontend_pod" -- \
  env "MODEL=$MODEL" \
  python3 -c '
import json, os, urllib.request

body = json.dumps({
    "model": os.environ["MODEL"],
    "messages": [{"role": "user", "content": "Production NIXL smoke-test prefix. " * 256 + "\nReply with: smoke-test-ok"}],
    "temperature": 0,
    "max_tokens": 32,
    "stream": False
}).encode()

req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", body, {"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=300) as resp:
    print(json.dumps(json.load(resp), indent=2))
'

# Verify NIXL / UCX transfer log entries
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --all-containers --prefix --since=10m |
  grep -Ei 'nixl|ucx|rdma|transfer|error|traceback'
```

---

## Performance Benchmarking

After all acceptance gates pass, follow the cluster benchmarking runbook in [benchmark.md](../../../../benchmark.md) to trigger perf jobs using `$EXP_DIR/perf.yaml`.

---

## Cleanup

Delete the deployment graph, benchmark jobs, and temporary test pods:

```bash
kubectl delete job -n "$NAMESPACE" "$PERF_JOB_NAME" --ignore-not-found
kubectl delete dynamographdeployment.nvidia.com -n "$NAMESPACE" \
  "$DEPLOYMENT" --wait=false --ignore-not-found
kubectl delete dynamographdeployment.nvidia.com -n "$NAMESPACE" \
  "${DEPLOYMENT}-pf" --wait=false --ignore-not-found
kubectl delete pod -n "$NAMESPACE" \
  sglang-runtime-preflight qwen-smoke qwen-sglang-preflight-request \
  --ignore-not-found
```
