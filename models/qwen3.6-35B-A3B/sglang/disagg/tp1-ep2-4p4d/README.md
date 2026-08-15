# SGLang disaggregated TP1-attention + EP2, 4P4D

This recipe runs four prefill workers and four decode workers. Every worker
uses two GPUs with `tp-size=dp-size=ep-size=2`, DP attention, and
`moe-dense-tp-size=1`, giving effective attention/dense TP=1 and expert
parallelism EP=2. It requests eight GPUs on each role node and all 16 GPUs in
the reservation.

The Dynamo frontend runs with KV-aware routing. Prefill workers publish SGLang
KV events, NIXL is configured to transfer KV/recurrent state over UCX/RDMA,
and every worker exports SGLang/Dynamo metrics on port 9090 plus NIXL transfer
metrics on port 19090. The baseline `deploy.yaml` enables cache reporting but
does not enable hierarchical cache or CPU KV offload. The optional
`deploy-kv-offloading.yaml` adds prefill-only GPU-to-CPU HiCache offload.
Decode workers retain live NIXL P-to-D transfer and telemetry in both variants,
but decode-side cache offload remains disabled because the pinned runtime does
not support it for this hybrid model.

This is an experimental backend comparison. SGLang supports NIXL
prefill/decode disaggregation generally, but this runbook does not assume that
Qwen3.6 recurrent/GDN state transfer works. Startup logs and a real request are
mandatory acceptance gates.

## Variables

```bash
export NAMESPACE=qwen32-bench
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

The optional `deploy-kv-offloading.yaml` keeps the baseline P/D topology and
enables hierarchical cache only on prefill. A conservative
`--hicache-ratio 1.2` allocates a CPU host-cache tier per rank without changing
the existing Pod resource template. The
`page_first_direct`/`direct` combination copies hybrid attention KV and Mamba
state between GPU and host memory. Both roles keep `--enable-cache-report`,
SGLang/Dynamo metrics, and NIXL P-to-D transfer metrics.

Do not add `--disaggregation-decode-enable-offload-kvcache` to the decode
worker for this image/model pair. SGLang 0.5.14 accepts that path only for
pure `MHATokenToKVPool` or `MLATokenToKVPool`; Qwen3.6 creates a hybrid
attention/Mamba `HybridLinearKVPool`, which raises `Unsupported KV cache type
for decode offload`. The following scheduler `EOFError` is only a consequence
of that initialization failure. `--disaggregation-decode-enable-radix-cache`
is not a workaround because the same runtime rejects decode radix cache for
hybrid Mamba/SSM models.

Do not add `--hicache-storage-backend nixl` on prefill either. Qwen3.6 passes
multiple hybrid pools to SGLang's v2 storage interface, while the NIXL HiCache
backend in SGLang 0.5.14 implements only the v1 interface. Its prefetch thread
therefore calls the base `batch_exists_v2()` and raises `NotImplementedError`.
The later `First message should be b'NixlMsgGuard'. Foreign traffic?` assertion
is a separate SGLang bug: an aborted request sends an `ABORT` tag that the NIXL
prefill receiver does not handle before its guard assertion. It is downstream
of the stalled request, not evidence that unrelated network traffic reached
the worker.

Qwen3.6 is a hybrid attention/Mamba-GDN model. Its SGLang 0.5.14 host pool
requires `--hicache-mem-layout page_first_direct` with
`--hicache-io-backend direct`; using the `page_first`/`kernel` combination
causes `MambaPoolHost` to fail during scheduler initialization.

`--disaggregation-transfer-backend nixl` still moves live P-to-D state over
UCX/RDMA on both roles. It does not provide the CPU HiCache tier; the HiCache
`direct` I/O backend provides that local GPU-to-host copy. `NIXL_TELEMETRY_*`
instruments P-to-D transfers, and `--enable-cache-report` exposes reused
prompt tokens in `usage.prompt_tokens_details.cached_tokens`.

## Preflight

Complete the four-GPU canary in [preflight.md](preflight.md) before applying
the optional KV-offloading manifest. It contains the resource checks,
single-replica deployment, bounded request tests, CPU HiCache proof,
cache-report proof, NIXL transfer counters, log acceptance gates, and canary
cleanup. The baseline manifest does not expose CPU HiCache metrics because it
does not allocate that tier; use the common startup and NIXL transfer gates
below for the baseline.

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
          k8s.v1.cni.cncf.io/networks: qwen-roce
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
            - name: UCX_NET_DEVICES
              value: mlx5_8:1
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
          k8s.v1.cni.cncf.io/networks: qwen-roce
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
          k8s.v1.cni.cncf.io/networks: qwen-roce
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
            - name: UCX_NET_DEVICES
              value: mlx5_8:1
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
          k8s.v1.cni.cncf.io/networks: qwen-roce
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

## Validate and deploy

Choose exactly one manifest. Both variants use the same
`DynamoGraphDeployment` name, so applying one updates the other rather than
creating an independent deployment:

```bash
# Baseline: no CPU KV offload.
export DEPLOY_FILE="$EXP_DIR/deploy.yaml"

# Optional prefill CPU KV offload; use this instead of the line above.
# export DEPLOY_FILE="$EXP_DIR/deploy-kv-offloading.yaml"

kubectl apply --dry-run=server -n "$NAMESPACE" -f "$DEPLOY_FILE"
kubectl apply -n "$NAMESPACE" -f "$DEPLOY_FILE"
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide -w
```

Successful request, cache-report, and NIXL transfer gates are mandatory for
both variants. Positive prefill CPU-HiCache total and used tokens are mandatory
only for `deploy-kv-offloading.yaml`; those series should be absent from the
baseline deployment.

Wait for all 9 Pods:

```bash
kubectl wait -n "$NAMESPACE" --for=condition=Ready pod \
  -l "$GRAPH_LABEL" --timeout=1800s
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  -L nvidia.com/dynamo-component-type -o wide
```

## Startup and transfer acceptance

```bash
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --all-containers --prefix --tail=1500 |
  tee "$EXP_DIR/qwen36-sglang-tp1ep2-4p4d-startup.log"

if grep -Ei 'traceback|assertionerror|notimplementederror|nixl_err|out of memory|does not support|unsupported' \
  "$EXP_DIR/qwen36-sglang-tp1ep2-4p4d-startup.log"; then
  echo "SGLang startup compatibility gate failed" >&2
  echo "STOP: do not continue to the request or benchmark steps" >&2
fi

grep -Ei 'sglang|nixl|ucx|rdma|prefill|decode|transfer|ready' \
  "$EXP_DIR/qwen36-sglang-tp1ep2-4p4d-startup.log" | tail -300
```

Do not benchmark merely because Pods are Ready. Send a real request:

```bash
frontend_pod="$(kubectl get pods -n "$NAMESPACE" \
  -l "$GRAPH_LABEL,nvidia.com/dynamo-component-type=frontend" \
  -o jsonpath='{.items[0].metadata.name}')"

timeout 330s kubectl exec -n "$NAMESPACE" "$frontend_pod" -- \
  env "MODEL=$MODEL" \
  python3 -c '
import json
import os
import urllib.request

body = json.dumps({
    "model": os.environ["MODEL"],
    "messages": [{
        "role": "user",
        "content": "Production NIXL smoke-test prefix. " * 256
            + "\nReply with exactly: smoke-test-ok"
    }],
    "temperature": 0,
    "max_tokens": 32,
    "stream": False
}).encode()

request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    body,
    {"Content-Type": "application/json"},
    method="POST"
)

with urllib.request.urlopen(request, timeout=300) as response:
    print(json.dumps(json.load(response), indent=2))
'

kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --all-containers --prefix --since=10m |
  grep -Ei 'nixl|ucx|rdma|transfer|error|traceback'
```

Accept only a successful response plus successful NIXL/UCX transfer evidence.
If logs report unsupported recurrent/GDN state, stop; changing TP size will not
repair backend support.

## Benchmark

After all acceptance gates pass, follow [benchmark.md](benchmark.md) to create
`$EXP_DIR/perf.yaml` and compare KV-aware routing against KV-aware routing with
prefill CPU offload. The benchmark runbook owns the prefill-heavy,
decode-heavy, and mixed presets, metric snapshots, A/B plots, cold-burst
diagnostic, required environment variables, and artifact locations.

## Cleanup

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

Cleanup preserves the namespace, PVCs, retained PV data, and RoCE objects.
