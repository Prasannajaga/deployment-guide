# TP1 horizontal scaling: 4 prefill + 4 decode

## Purpose

This is configuration A of the controlled comparison: four one-GPU TP1
prefill workers and four one-GPU TP1 decode workers. It consumes exactly eight
GPUs and uses the Dynamo KV-aware frontend plus NIXL/UCX transfer.

Read the [comparison README](../README.md) before publishing results, especially
the hybrid-cache compatibility gate and fairness checklist.

## Architecture and GPU topology

```text
AIPerf -> Dynamo frontend (KV router)
                    |-> 4 x prefill (TP1, 1 GPU) --NIXL--> 4 x decode (TP1, 1 GPU)
```

The prefill node selector consumes four GPUs on the node labeled `prefill`;
the decode selector consumes four GPUs on the node labeled `decode`.

## Variables

Run commands from this directory and set:

```bash
export NAMESPACE=qwen32-bench
export ROCE_NETWORK=qwen-roce
export EXP_DIR=/ephemeral/shared/qwen3.6-35b-a3b/vllm/disagg/tp1-4p4d
export MODEL_CACHE_DIR=/ephemeral/shared/qwen3.6-35b-a3b/model-cache
export DEPLOYMENT=qwen36-35b-a3b-vllm-disagg-kv-tp1-4p4d
export PERF_JOB_NAME=qwen36-vllm-tp1-4p4d-perf
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
export FRONTEND_SERVICE="${DEPLOYMENT}-frontend"
export MODEL=Qwen/Qwen3.6-35B-A3B-FP8
export MODEL_REVISION=95a723d08a9490559dae23d0cff1d9466213d989
```

Change `EXP_DIR` and `MODEL_CACHE_DIR` only if the shared checkout is elsewhere.

## Recover the namespace and bootstrap

This is the complete recovery path after `qwen32-bench` was deleted. The
existing PVs use reclaim policy `Retain`, so reuse them—do not recreate or
delete them.

### 1. Recreate the namespace and recover storage

```bash
mkdir -p "$EXP_DIR" "$MODEL_CACHE_DIR"
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml |
  kubectl apply -f -

for binding in \
  qwen32-model-cache-pv:model-cache \
  qwen32-vllm-perf-cache-pv:perf-cache; do
  pv="${binding%%:*}"
  pvc="${binding#*:}"

  reclaim="$(kubectl get pv "$pv" -o jsonpath='{.spec.persistentVolumeReclaimPolicy}')"
  [ "$reclaim" = Retain ] || {
    echo "$pv must exist with reclaim policy Retain" >&2
    exit 1
  }

  phase="$(kubectl get pv "$pv" -o jsonpath='{.status.phase}')"
  case "$phase" in
    Released)
      kubectl patch pv "$pv" --type=merge -p '{"spec":{"claimRef":null}}'
      kubectl wait --for=jsonpath='{.status.phase}'=Available \
        "pv/$pv" --timeout=120s
      ;;
    Available)
      ;;
    Bound)
      claim_namespace="$(kubectl get pv "$pv" -o jsonpath='{.spec.claimRef.namespace}')"
      claim_name="$(kubectl get pv "$pv" -o jsonpath='{.spec.claimRef.name}')"
      [ "$claim_namespace/$claim_name" = "$NAMESPACE/$pvc" ] || {
        echo "$pv is already bound to $claim_namespace/$claim_name" >&2
        exit 1
      }
      ;;
    *)
      echo "Cannot recover $pv while phase=$phase" >&2
      exit 1
      ;;
  esac
done

tee "$MODEL_CACHE_DIR/model-and-perf-cache-pvcs.yaml" >/dev/null <<'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: qwen-shared-manual
  volumeName: qwen32-model-cache-pv
  resources:
    requests:
      storage: 100Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: perf-cache
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: qwen-shared-manual
  volumeName: qwen32-vllm-perf-cache-pv
  resources:
    requests:
      storage: 50Gi
EOF

kubectl apply -n "$NAMESPACE" \
  -f "$MODEL_CACHE_DIR/model-and-perf-cache-pvcs.yaml"
kubectl wait -n "$NAMESPACE" --for=jsonpath='{.status.phase}'=Bound \
  pvc/model-cache pvc/perf-cache --timeout=120s
kubectl get pvc model-cache perf-cache -n "$NAMESPACE" -o wide
```

The static `qwen-shared-manual` class does not require a StorageClass object
because each PVC explicitly names an existing PV.

### 2. Restore and verify the RoCE attachment

The namespace-local attachment may be recreated automatically. If it is
missing, retarget the existing cluster-scoped Macvlan network:

```bash
if ! kubectl get network-attachment-definition "$ROCE_NETWORK" \
  -n "$NAMESPACE" >/dev/null 2>&1; then
  kubectl patch macvlannetwork "$ROCE_NETWORK" --type=merge \
    -p "{\"spec\":{\"networkNamespace\":\"$NAMESPACE\"}}"
fi

for attempt in $(seq 1 60); do
  kubectl get network-attachment-definition "$ROCE_NETWORK" \
    -n "$NAMESPACE" >/dev/null 2>&1 && break
  [ "$attempt" -lt 60 ] || {
    echo "Timed out waiting for $NAMESPACE/$ROCE_NETWORK" >&2
    exit 1
  }
  sleep 2
done

kubectl get crd dynamographdeployments.nvidia.com
kubectl get network-attachment-definition "$ROCE_NETWORK" -n "$NAMESPACE"
kubectl get nodes -L qwen.nvidia.com/role \
  -o custom-columns='NODE:.metadata.name,ROLE:.metadata.labels.qwen\.nvidia\.com/role,GPU:.status.allocatable.nvidia\.com/gpu,RDMA:.status.allocatable.rdma/ib'
kubectl get dynamographdeployments.nvidia.com -A
```

Require the `prefill` and `decode` nodes, at least four free GPUs and four
`rdma/ib` resources on each, and no competing eight-GPU deployment.

### 3. Create and run the model-download Job

The public model normally needs no token. If authentication is required,
create `hf-token-secret` from a protected file outside this repository before
running the Job.

```bash
tee "$MODEL_CACHE_DIR/model-download.yaml" >/dev/null <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: qwen36-35b-a3b-fp8-download
spec:
  backoffLimit: 3
  template:
    metadata:
      labels:
        app: qwen36-35b-a3b-fp8-download
    spec:
      restartPolicy: Never
      containers:
        - name: model-download
          image: python:3.10-slim
          command:
            - /bin/sh
            - -c
          args:
            - |
              set -eu
              pip install --no-cache-dir huggingface_hub==1.16.4
              hf download "$MODEL_NAME" --revision "$MODEL_REVISION"
          env:
            - name: MODEL_NAME
              value: Qwen/Qwen3.6-35B-A3B-FP8
            - name: MODEL_REVISION
              value: 95a723d08a9490559dae23d0cff1d9466213d989
            - name: HF_HOME
              value: /model-store
            - name: HF_XET_HIGH_PERFORMANCE
              value: "1"
          envFrom:
            - secretRef:
                name: hf-token-secret
                optional: true
          resources:
            requests:
              cpu: "2"
              memory: 64Gi
            limits:
              cpu: "8"
              memory: 64Gi
          volumeMounts:
            - name: model-cache
              mountPath: /model-store
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache
EOF

kubectl delete job -n "$NAMESPACE" qwen36-35b-a3b-fp8-download \
  --ignore-not-found
kubectl apply --dry-run=server -n "$NAMESPACE" \
  -f "$MODEL_CACHE_DIR/model-download.yaml"
kubectl apply -n "$NAMESPACE" -f "$MODEL_CACHE_DIR/model-download.yaml"
kubectl wait -n "$NAMESPACE" --for=condition=Ready pod \
  -l app=qwen36-35b-a3b-fp8-download --timeout=300s
kubectl logs -n "$NAMESPACE" -f job/qwen36-35b-a3b-fp8-download
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  job/qwen36-35b-a3b-fp8-download --timeout=3600s
```

Do not apply `deploy.yaml` unless both PVCs are `Bound`, `qwen-roce`
exists in `qwen32-bench`, and the download Job completed successfully.

## Create `deploy.yaml` with `tee <<'EOF'`

The quoted delimiter keeps worker-side variables such as `$MODEL_PATH`
literal until the containers start.

```bash
mkdir -p "$EXP_DIR"
tee "$EXP_DIR/deploy.yaml" >/dev/null <<'EOF'
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: qwen36-35b-a3b-vllm-disagg-kv-tp1-4p4d
spec:
  backendFramework: vllm
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
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          workingDir: /workspace/examples/backends/vllm
          command:
            - python3
            - -m
            - dynamo.frontend
          args:
            - --router-mode
            - kv
      envs:
        - name: HF_HOME
          value: /opt/models
      replicas: 1
      resources:
        requests:
          cpu: "16"
          memory: 64Gi
        limits:
          cpu: "32"
          memory: 128Gi
    VllmPrefillWorker:
      componentType: worker
      subComponentType: prefill
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 40Gi
      extraPodMetadata:
        annotations:
          k8s.v1.cni.cncf.io/networks: qwen-roce
      extraPodSpec:
        hostNetwork: false
        dnsPolicy: ClusterFirst
        affinity:
          podAntiAffinity:
            preferredDuringSchedulingIgnoredDuringExecution:
              - weight: 100
                podAffinityTerm:
                  labelSelector:
                    matchLabels:
                      nvidia.com/dynamo-graph-deployment-name: qwen36-35b-a3b-vllm-disagg-kv-tp1-4p4d
                      nvidia.com/dynamo-component-type: worker
                  topologyKey: kubernetes.io/hostname
        nodeSelector:
          qwen.nvidia.com/role: prefill
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          workingDir: /workspace/examples/backends/vllm
          command:
            - /bin/sh
            - -c
          args:
            - |
              ulimit -l unlimited
              exec python3 -m dynamo.vllm \
                --model "$MODEL_PATH" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --dtype auto \
                --kv-cache-dtype auto \
                --tensor-parallel-size 1 \
                --data-parallel-size 1 \
                --disaggregation-mode prefill \
                --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' \
                --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:0","enable_kv_cache_events":true}' \
                --gpu-memory-utilization 0.85 \
                --max-model-len 131072 \
                --block-size 128 \
                --max-num-seqs 128 \
                --max-num-batched-tokens 32768 \
                --scheduling-policy fcfs \
                --no-async-scheduling \
                --enable-chunked-prefill \
                --enable-prefix-caching \
                --mamba-cache-mode align \
                --no-disable-hybrid-kv-cache-manager \
                --reasoning-parser qwen3 \
                --enable-auto-tool-choice \
                --tool-call-parser qwen3_coder
          env: &worker-env
            - name: SERVED_MODEL_NAME
              value: Qwen/Qwen3.6-35B-A3B-FP8
            - name: MODEL_PATH
              value: /opt/models/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989
            - name: HF_HOME
              value: /opt/models
            - name: PYTHONHASHSEED
              value: "0"
            - name: VLLM_SSM_CONV_STATE_LAYOUT
              value: DS
            - name: VLLM_NIXL_SIDE_CHANNEL_HOST
              valueFrom:
                fieldRef:
                  fieldPath: status.podIP
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
          securityContext: &worker-security-context
            runAsUser: 0
            capabilities:
              add:
                - IPC_LOCK
                - SYS_RESOURCE
      replicas: 4
      resources: &worker-resources
        requests:
          gpu: "1"
          cpu: "8"
          memory: 64Gi
          custom:
            rdma/ib: "1"
        limits:
          gpu: "1"
          cpu: "16"
          memory: 96Gi
          custom:
            rdma/ib: "1"
    VllmDecodeWorker:
      componentType: worker
      subComponentType: decode
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 40Gi
      extraPodMetadata:
        annotations:
          k8s.v1.cni.cncf.io/networks: qwen-roce
      extraPodSpec:
        hostNetwork: false
        dnsPolicy: ClusterFirst
        affinity:
          podAntiAffinity:
            preferredDuringSchedulingIgnoredDuringExecution:
              - weight: 100
                podAffinityTerm:
                  labelSelector:
                    matchLabels:
                      nvidia.com/dynamo-graph-deployment-name: qwen36-35b-a3b-vllm-disagg-kv-tp1-4p4d
                      nvidia.com/dynamo-component-type: worker
                  topologyKey: kubernetes.io/hostname
        nodeSelector:
          qwen.nvidia.com/role: decode
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          workingDir: /workspace/examples/backends/vllm
          command:
            - /bin/sh
            - -c
          args:
            - |
              ulimit -l unlimited
              exec python3 -m dynamo.vllm \
                --model "$MODEL_PATH" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --dtype auto \
                --kv-cache-dtype auto \
                --tensor-parallel-size 1 \
                --data-parallel-size 1 \
                --disaggregation-mode decode \
                --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}' \
                --gpu-memory-utilization 0.85 \
                --max-model-len 131072 \
                --block-size 128 \
                --max-num-seqs 128 \
                --max-num-batched-tokens 32768 \
                --scheduling-policy fcfs \
                --no-async-scheduling \
                --enable-chunked-prefill \
                --enable-prefix-caching \
                --mamba-cache-mode align \
                --no-disable-hybrid-kv-cache-manager \
                --reasoning-parser qwen3 \
                --enable-auto-tool-choice \
                --tool-call-parser qwen3_coder
          env: *worker-env
          securityContext: *worker-security-context
      replicas: 4
      resources: *worker-resources
EOF
```

## Run the NIXL/Mamba compatibility preflight

Run this CPU-only Job before the main deployment. It mounts the downloaded
model, uses the exact Dynamo runtime image, verifies vLLM 0.23.0 and the
required `DS` convolution-state layout, imports the NIXL connector, and fails
early when a GDN model meets connector code that explicitly supports only
Mamba2 state transfer.

```bash
tee "$EXP_DIR/preflight.yaml" >/dev/null <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: qwen36-vllm-nixl-preflight
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        app: qwen36-vllm-nixl-preflight
    spec:
      restartPolicy: Never
      containers:
        - name: preflight
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          imagePullPolicy: IfNotPresent
          command:
            - /bin/bash
            - -lc
          args:
            - |
              set -euo pipefail
              python3 - <<'PY'
              import importlib.metadata
              import json
              import os
              from pathlib import Path

              model_path = Path(
                  "/opt/models/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/"
                  "snapshots/95a723d08a9490559dae23d0cff1d9466213d989"
              )
              config_path = model_path / "config.json"
              if not config_path.is_file():
                  raise SystemExit(f"FAIL: missing model config: {config_path}")

              version = importlib.metadata.version("vllm")
              print(f"vLLM version: {version}")
              if version != "0.23.0":
                  raise SystemExit(
                      f"FAIL: expected vLLM 0.23.0, found {version}"
                  )

              if os.environ.get("VLLM_SSM_CONV_STATE_LAYOUT") != "DS":
                  raise SystemExit(
                      "FAIL: VLLM_SSM_CONV_STATE_LAYOUT must be DS"
                  )

              from vllm.model_executor.layers.mamba.mamba_utils import (
                  get_conv_state_layout,
              )
              if get_conv_state_layout() != "DS":
                  raise SystemExit(
                      "FAIL: vLLM did not activate DS conv-state layout"
                  )

              from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
                  NixlConnector,
              )
              print(f"NIXL connector import: {NixlConnector.__name__}")

              config = json.loads(config_path.read_text())
              config_text = json.dumps(config).lower()
              recurrent_markers = sorted(
                  marker
                  for marker in ("gdn", "mamba", "ssm")
                  if marker in config_text
              )
              print(f"Model architectures: {config.get('architectures')}")
              print(f"Recurrent-state markers: {recurrent_markers or ['none']}")

              import vllm

              nixl_dir = (
                  Path(vllm.__file__).resolve().parent
                  / "distributed/kv_transfer/kv_connector/v1/nixl"
              )
              nixl_source = "\n".join(
                  path.read_text(errors="replace")
                  for path in nixl_dir.glob("*.py")
              ).lower()
              mamba2_only = (
                  "only supports mamba2" in nixl_source
                  or "only support mamba2" in nixl_source
              )
              if "gdn" in recurrent_markers and mamba2_only:
                  raise SystemExit(
                      "FAIL: model config contains GDN state but this NIXL "
                      "connector explicitly supports only Mamba2 transfer"
                  )

              print("PASS: static model/runtime/NIXL compatibility checks")
              print(
                  "NOTE: the main deployment must still prove GPU engine "
                  "initialization and cross-node state transfer"
              )
              PY
          env:
            - name: VLLM_SSM_CONV_STATE_LAYOUT
              value: DS
          resources:
            requests:
              cpu: "1"
              memory: 2Gi
            limits:
              cpu: "4"
              memory: 8Gi
          volumeMounts:
            - name: model-cache
              mountPath: /opt/models
              readOnly: true
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache
EOF

kubectl get pvc model-cache -n "$NAMESPACE"
grep -A1 -F 'name: VLLM_SSM_CONV_STATE_LAYOUT' "$EXP_DIR/deploy.yaml" |
  grep -q 'value: DS' || {
    echo "FAIL: deploy.yaml does not configure DS for its workers" >&2
    exit 1
  }
kubectl delete job -n "$NAMESPACE" qwen36-vllm-nixl-preflight \
  --ignore-not-found
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/preflight.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/preflight.yaml"

if ! kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  job/qwen36-vllm-nixl-preflight --timeout=300s; then
  kubectl logs -n "$NAMESPACE" job/qwen36-vllm-nixl-preflight --tail=-1
  exit 1
fi
kubectl logs -n "$NAMESPACE" job/qwen36-vllm-nixl-preflight --tail=-1
```

Proceed only when the final log contains `PASS`. A pass proves static
model/runtime compatibility; it does not guarantee GPU engine startup or
cross-node NIXL transfer. If it reports GDN with Mamba2-only transfer, stop:
changing `--mamba-cache-mode` will not make this disaggregated topology work
with that runtime.

## Validate and apply the deployment manifest

The repository's `deploy.yaml` is canonical. Validate against the installed
CRD, then apply it:

```bash
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide -w
```

Wait for all nine Pods (one frontend plus eight workers):

```bash
kubectl wait -n "$NAMESPACE" --for=condition=Ready pod \
  -l "$GRAPH_LABEL" --timeout=1800s
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  -L nvidia.com/dynamo-component-type -o wide
```

## Logs and required compatibility acceptance

Inspect every container and fail the run if hybrid HMA was disabled or cache
specs could not be unified:

```bash
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --all-containers --prefix --tail=1000 | tee /tmp/qwen36-tp1-startup.log

if grep -E 'Hybrid KV cache manager is disabled|failed to convert the KV cache specs|does not support HMA' \
  /tmp/qwen36-tp1-startup.log; then
  echo 'Hybrid NIXL compatibility gate failed' >&2
  exit 1
fi

grep -Ei 'vllm|nixl|ucx|mamba|prefix|kv.event|block.size' \
  /tmp/qwen36-tp1-startup.log | tail -200
```

Require vLLM 0.23.0, successful NIXL/UCX initialization, prefix caching,
Mamba `align` mode, HMA enabled, and block size 128. A real request in the next
step must produce P-to-D transfer evidence.

## Internal smoke test

Verify the exact served model and then issue a deterministic completion:

```bash
kubectl run qwen-smoke --rm -i --restart=Never \
  --namespace "$NAMESPACE" --image=curlimages/curl -- \
  curl -fsS "http://${FRONTEND_SERVICE}:8000/v1/models"

kubectl run qwen-smoke --rm -i --restart=Never \
  --namespace "$NAMESPACE" --image=curlimages/curl -- \
  curl -fsS -H 'Content-Type: application/json' \
  --data-binary "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: smoke-test-ok\"}],\"chat_template_kwargs\":{\"enable_thinking\":false},\"temperature\":0,\"max_tokens\":32,\"stream\":false}" \
  "http://${FRONTEND_SERVICE}:8000/v1/chat/completions"
```

Reinspect prefill and decode logs and require a successful transfer. For an
optional workstation smoke test:

```bash
kubectl port-forward -n "$NAMESPACE" "service/${FRONTEND_SERVICE}" 8000:8000
curl -fsS http://127.0.0.1:8000/v1/models
```

Do not use port-forwarding for AIPerf measurements.

## Create `perf.yaml` with `tee <<'EOF'`

The quoted delimiter likewise keeps the AIPerf Job's shell variables literal
until the benchmark container starts.

```bash
mkdir -p "$EXP_DIR"
tee "$EXP_DIR/perf.yaml" >/dev/null <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: qwen36-vllm-tp1-4p4d-perf
spec:
  backoffLimit: 0
  completions: 1
  parallelism: 1
  activeDeadlineSeconds: 14400
  template:
    metadata:
      labels:
        app: qwen36-vllm-tp1-4p4d-perf
    spec:
      restartPolicy: Never
      tolerations:
        - key: node-role.kubernetes.io/control-plane
          operator: Exists
          effect: NoSchedule
        - key: nvidia.com/gpu
          operator: Equal
          value: "true"
          effect: NoSchedule
      containers:
        - name: perf
          image: python:3.12-slim
          imagePullPolicy: IfNotPresent
          workingDir: /workspace
          command:
            - /bin/bash
            - -lc
          args:
            - |
              set -euo pipefail

              apt-get update
              apt-get install -y --no-install-recommends build-essential curl jq procps
              rm -rf /var/lib/apt/lists/*
              python -m pip install --no-cache-dir "aiperf==0.10.0"

              aiperf --version || true
              python --version

              case "$PREFIX_MODE" in
                isolated)
                  prompt_tokens="$ISL"
                  prefix_args=()
                  ;;
                shared)
                  prefix_tokens=$((ISL * PREFIX_REUSE_PERCENT / 100))
                  prompt_tokens=$((ISL - prefix_tokens))
                  if [ "$prefix_tokens" -lt 1 ] || [ "$prompt_tokens" -lt 1 ]; then
                    echo "PREFIX_REUSE_PERCENT must leave positive shared and unique lengths" >&2
                    exit 2
                  fi
                  prefix_args=(
                    --prefix-prompt-pool-size "$PREFIX_GROUPS"
                    --prefix-prompt-length "$prefix_tokens"
                  )
                  ;;
                *)
                  echo "PREFIX_MODE must be isolated or shared" >&2
                  exit 2
                  ;;
              esac

              endpoint_url="${ENDPOINT%/}"
              case "$endpoint_url" in
                http://*|https://*) ;;
                *) endpoint_url="http://${endpoint_url}" ;;
              esac

              echo "Waiting for ${TARGET_MODEL} at ${endpoint_url}/v1/models"
              attempt=1
              until models_json="$(curl -fsS --max-time 10 "${endpoint_url}/v1/models")" &&
                printf '%s' "$models_json" | jq -e --arg model "$TARGET_MODEL" \
                  '.data[]? | select(.id == $model)' >/dev/null; do
                if [ "$attempt" -ge 540 ]; then
                  echo "Model readiness timed out after 45 minutes" >&2
                  exit 1
                fi
                attempt=$((attempt + 1))
                sleep 5
              done
              printf '%s' "$models_json" | jq .

              test -e "$TOKENIZER" || {
                echo "Tokenizer path does not exist: $TOKENIZER" >&2
                exit 1
              }

              mkdir -p "$ARTIFACT_ROOT" "$HF_HOME" /perf-cache/tmp
              export TMPDIR=/perf-cache/tmp
              run_root="${ARTIFACT_ROOT}/${TOPOLOGY_NAME}/${PREFIX_MODE}/${WORKLOAD_NAME}/isl-${ISL}_osl-${OSL}"
              mkdir -p "$run_root"
              status_file="${run_root}/matrix-status.tsv"

              printf '%s\n' "$models_json" > "${run_root}/models.json"
              curl -fsS "${endpoint_url}/metrics" > "${run_root}/frontend-metrics-before.prom" || true
              jq -n \
                --arg topology "$TOPOLOGY_NAME" \
                --arg workload "$WORKLOAD_NAME" \
                --arg prefix_mode "$PREFIX_MODE" \
                --arg model "$TARGET_MODEL" \
                --arg tokenizer "$TOKENIZER" \
                --arg endpoint "$endpoint_url" \
                --arg concurrencies "$CONCURRENCIES" \
                --argjson isl "$ISL" \
                --argjson osl "$OSL" \
                --argjson duration "$BENCHMARK_DURATION" \
                --argjson warmup "$WARMUP_REQUESTS" \
                --argjson seed "$RANDOM_SEED" \
                --argjson request_timeout "$REQUEST_TIMEOUT" \
                --argjson prefix_groups "$PREFIX_GROUPS" \
                --argjson prefix_reuse_percent "$PREFIX_REUSE_PERCENT" \
                '{topology:$topology, workload:$workload, prefix_mode:$prefix_mode,
                  model:$model, tokenizer:$tokenizer, endpoint:$endpoint,
                  isl:$isl, osl:$osl, concurrencies:$concurrencies,
                  benchmark_duration_seconds:$duration, warmup_requests:$warmup,
                  random_seed:$seed, request_timeout_seconds:$request_timeout,
                  prefix_groups:$prefix_groups,
                  target_prefix_token_percent:$prefix_reuse_percent,
                  aiperf:"0.10.0", dynamo:"1.3.0", vllm:"0.23.0"}' \
                > "${run_root}/input-config.json"
              printf 'concurrency\tstatus\tfinished_utc\n' > "$status_file"

              server_metric_args=()
              if [ -n "$SERVER_METRICS_URLS" ]; then
                read -r -a metric_urls <<< "$SERVER_METRICS_URLS"
                server_metric_args=(--server-metrics "${metric_urls[@]}")
              fi

              failures=0
              for concurrency in $CONCURRENCIES; do
                artifact_dir="${run_root}/c${concurrency}"
                if [ -e "${artifact_dir}/profile_export_raw.jsonl" ]; then
                  echo "Refusing to overwrite existing raw artifacts: $artifact_dir" >&2
                  exit 1
                fi
                mkdir -p "$artifact_dir"

                if ! curl -fsS --max-time 10 "${endpoint_url}/v1/models" |
                  jq -e --arg model "$TARGET_MODEL" \
                    '.data[]? | select(.id == $model)' >/dev/null; then
                  echo "Expected model disappeared before concurrency $concurrency" >&2
                  exit 1
                fi

                echo "Starting workload=$WORKLOAD_NAME prefix=$PREFIX_MODE ISL=$ISL OSL=$OSL concurrency=$concurrency"
                if aiperf profile \
                  --model "$TARGET_MODEL" \
                  --tokenizer "$TOKENIZER" \
                  --url "$endpoint_url" \
                  --endpoint-type chat \
                  --streaming \
                  --concurrency "$concurrency" \
                  --benchmark-duration "$BENCHMARK_DURATION" \
                  --benchmark-grace-period "$GRACE_PERIOD" \
                  --warmup-request-count "$WARMUP_REQUESTS" \
                  --request-timeout-seconds "$REQUEST_TIMEOUT" \
                  --isl "$prompt_tokens" \
                  --isl-stddev 0 \
                  --osl "$OSL" \
                  --osl-stddev 0 \
                  --extra-inputs "max_tokens:$OSL" \
                  --extra-inputs "min_tokens:$OSL" \
                  --extra-inputs ignore_eos:true \
                  --extra-inputs temperature:0.0 \
                  --extra-inputs '{"chat_template_kwargs":{"enable_thinking":false}}' \
                  --num-dataset-entries "$NUM_DATASET_ENTRIES" \
                  --random-seed "$RANDOM_SEED" \
                  --server-metrics-formats json csv jsonl \
                  "${prefix_args[@]}" \
                  "${server_metric_args[@]}" \
                  --ui simple \
                  --artifact-dir "$artifact_dir"; then
                  status=PASS
                else
                  status=FAIL
                  failures=$((failures + 1))
                fi
                printf '%s\t%s\t%s\n' "$concurrency" "$status" \
                  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$status_file"
              done

              curl -fsS "${endpoint_url}/metrics" > "${run_root}/frontend-metrics-after.prom" || true
              echo "Benchmark artifacts: $run_root"
              [ "$failures" -eq 0 ]
          env:
            - name: TARGET_MODEL
              value: Qwen/Qwen3.6-35B-A3B-FP8
            - name: TOKENIZER
              value: /opt/models/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989
            - name: ENDPOINT
              value: qwen36-35b-a3b-vllm-disagg-kv-tp1-4p4d-frontend:8000
            - name: ISL
              value: "8000"
            - name: OSL
              value: "1024"
            - name: CONCURRENCIES
              value: "1 2 4 8 16 32 64 128"
            - name: BENCHMARK_DURATION
              value: "180"
            - name: WARMUP_REQUESTS
              value: "32"
            - name: RANDOM_SEED
              value: "42"
            - name: ARTIFACT_ROOT
              value: /perf-cache/aiperf/qwen36
            - name: TOPOLOGY_NAME
              value: tp1-4p4d
            - name: WORKLOAD_NAME
              value: balanced
            - name: PREFIX_MODE
              value: isolated
            - name: REQUEST_TIMEOUT
              value: "3600"
            - name: PREFIX_GROUPS
              value: "8"
            - name: PREFIX_REUSE_PERCENT
              value: "75"
            - name: NUM_DATASET_ENTRIES
              value: "4096"
            - name: GRACE_PERIOD
              value: "600"
            - name: SERVER_METRICS_URLS
              value: ""
            - name: AIPERF_HTTP_CONNECTION_LIMIT
              value: "512"
            - name: HF_HOME
              value: /perf-cache/hf-aiperf-cache
            - name: PYTHONUNBUFFERED
              value: "1"
          resources:
            requests:
              cpu: "4"
              memory: 8Gi
              ephemeral-storage: 4Gi
            limits:
              cpu: "16"
              memory: 32Gi
              ephemeral-storage: 16Gi
          volumeMounts:
            - name: perf-cache
              mountPath: /perf-cache
            - name: model-cache
              mountPath: /opt/models
              readOnly: true
      volumes:
        - name: perf-cache
          persistentVolumeClaim:
            claimName: perf-cache
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache
EOF
```

## Canonical AIPerf Job

`perf.yaml` runs inside Kubernetes against the internal frontend service. Its
default is the balanced, isolated-prefix, eight-point sweep. It pins AIPerf
0.10.0, prints AIPerf/Python versions, checks `/v1/models`, preserves raw
records, and saves frontend/server metrics.

```bash
kubectl delete job -n "$NAMESPACE" "$PERF_JOB_NAME" --ignore-not-found
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/perf.yaml"
kubectl logs -n "$NAMESPACE" -f "job/${PERF_JOB_NAME}"
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  "job/${PERF_JOB_NAME}" --timeout=14400s
```

## Quick single-point validation

Use a separate artifact root so this does not collide with a full run:

```bash
kubectl delete job -n "$NAMESPACE" "$PERF_JOB_NAME" --ignore-not-found
kubectl set env --local -f "$EXP_DIR/perf.yaml" -o yaml \
  ISL=8000 OSL=1024 WORKLOAD_NAME=quick-balanced \
  PREFIX_MODE=isolated CONCURRENCIES=4 BENCHMARK_DURATION=30 \
  WARMUP_REQUESTS=4 ARTIFACT_ROOT=/perf-cache/aiperf/qwen36-quick | \
  kubectl apply -n "$NAMESPACE" -f -
kubectl logs -n "$NAMESPACE" -f "job/${PERF_JOB_NAME}"
```

Continue only if the actual input/output lengths are near 8000/1024, all
requests succeed, streaming TTFT/ITL exist, and raw artifacts are present.

## Full isolated-prefix workloads

This helper deletes only the previous client Job; it does not touch the model
deployment or PVC artifacts:

```bash
run_workload () {
  workload="$1" isl="$2" osl="$3"
  kubectl delete job -n "$NAMESPACE" "$PERF_JOB_NAME" --ignore-not-found
  kubectl set env --local -f "$EXP_DIR/perf.yaml" -o yaml \
    "WORKLOAD_NAME=${workload}" "ISL=${isl}" "OSL=${osl}" \
    PREFIX_MODE=isolated CONCURRENCIES='1 2 4 8 16 32 64 128' \
    BENCHMARK_DURATION=180 WARMUP_REQUESTS=32 | \
    kubectl apply -n "$NAMESPACE" -f -
  kubectl logs -n "$NAMESPACE" -f "job/${PERF_JOB_NAME}"
  kubectl wait -n "$NAMESPACE" --for=condition=Complete \
    "job/${PERF_JOB_NAME}" --timeout=14400s
}

run_workload prefill-heavy 32000 256
run_workload balanced 8000 1024
run_workload decode-heavy 2000 4096
```

For publication reruns change `BENCHMARK_DURATION=300`. To extend a discovery
sweep after observing continued scaling, set
`CONCURRENCIES='1 2 4 8 16 32 64 128 256'`.

## Controlled shared-prefix workload

This creates eight deterministic prefix groups with 75% of the total 8K input
assigned to a shared prefix:

```bash
kubectl delete job -n "$NAMESPACE" "$PERF_JOB_NAME" --ignore-not-found
kubectl set env --local -f "$EXP_DIR/perf.yaml" -o yaml \
  WORKLOAD_NAME=balanced-kv-reuse ISL=8000 OSL=1024 \
  PREFIX_MODE=shared PREFIX_GROUPS=8 PREFIX_REUSE_PERCENT=75 \
  CONCURRENCIES='1 2 4 8 16 32 64 128' BENCHMARK_DURATION=180 | \
  kubectl apply -n "$NAMESPACE" -f -
kubectl logs -n "$NAMESPACE" -f "job/${PERF_JOB_NAME}"
```

During the run, inspect the installed metric names and router/worker logs. The
router-visible full-attention hit ratio is not complete recurrent-state reuse.

## One-off manifest with `tee <<EOF`

The checked-in `perf.yaml` remains canonical. This Kustomize overlay creates a
temporary variant without editing it:

```bash
export RUN_SUFFIX=oneoff
export ISL=8000 OSL=1024
export CONCURRENCIES='1 2 4 8 16 32 64 128'
export BENCHMARK_DURATION=180
mkdir -p /tmp/qwen36-tp1-oneoff
cp "$EXP_DIR/perf.yaml" /tmp/qwen36-tp1-oneoff/perf.yaml
tee /tmp/qwen36-tp1-oneoff/kustomization.yaml >/dev/null <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - perf.yaml
nameSuffix: -${RUN_SUFFIX}
patches:
  - target:
      kind: Job
    patch: |-
      apiVersion: batch/v1
      kind: Job
      metadata:
        name: ${PERF_JOB_NAME}
      spec:
        template:
          spec:
            containers:
              - name: perf
                env:
                  - name: ISL
                    value: "${ISL}"
                  - name: OSL
                    value: "${OSL}"
                  - name: CONCURRENCIES
                    value: "${CONCURRENCIES}"
                  - name: BENCHMARK_DURATION
                    value: "${BENCHMARK_DURATION}"
                  - name: ARTIFACT_ROOT
                    value: "/perf-cache/aiperf/qwen36-${RUN_SUFFIX}"
EOF
kubectl apply -n "$NAMESPACE" -k /tmp/qwen36-tp1-oneoff
kubectl logs -n "$NAMESPACE" -f "job/${PERF_JOB_NAME}-${RUN_SUFFIX}"
```

## Artifact inspection

Artifacts use:

```text
/perf-cache/aiperf/qwen36/tp1-4p4d/<prefix-mode>/<workload>/isl-<isl>_osl-<osl>/c<concurrency>/
```

Every `cN` retains AIPerf summaries, JSONL records, raw JSONL records, and
server metrics. The workload directory also contains `input-config.json`,
`models.json`, before/after frontend metrics, and `matrix-status.tsv`.
Create a temporary PVC inspector after the Job completes:

```bash
kubectl apply -n "$NAMESPACE" -f - <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: qwen-perf-inspector
spec:
  restartPolicy: Never
  containers:
    - name: shell
      image: busybox:1.36
      command: [sh, -c, 'sleep 3600']
      volumeMounts:
        - name: perf-cache
          mountPath: /perf-cache
  volumes:
    - name: perf-cache
      persistentVolumeClaim:
        claimName: perf-cache
EOF
kubectl wait -n "$NAMESPACE" --for=condition=Ready pod/qwen-perf-inspector --timeout=120s
kubectl exec -n "$NAMESPACE" qwen-perf-inspector -- \
  find /perf-cache/aiperf/qwen36/tp1-4p4d -maxdepth 7 -type f
kubectl delete pod -n "$NAMESPACE" qwen-perf-inspector
```

The Job refuses to overwrite an existing raw `cN` directory. Choose a new
`ARTIFACT_ROOT` for reruns.

## Cleanup

```bash
kubectl delete job -n "$NAMESPACE" "$PERF_JOB_NAME" --ignore-not-found
kubectl delete job -n "$NAMESPACE" qwen36-35b-a3b-fp8-download \
  --ignore-not-found
kubectl delete job -n "$NAMESPACE" qwen36-vllm-nixl-preflight \
  --ignore-not-found
kubectl delete dynamographdeployment.nvidia.com -n "$NAMESPACE" \
  "$DEPLOYMENT" --wait=false --ignore-not-found
kubectl delete pod -n "$NAMESPACE" \
  qwen-smoke qwen-perf-inspector model-cache-check \
  --ignore-not-found
```

Cleanup intentionally preserves `model-cache` and `perf-cache` PVC contents.
Do not delete the namespace, PVCs, or retained PVs as routine cleanup.

## Troubleshooting

- `Pending` workers: check role labels, four free GPUs per role node,
  `rdma/ib`, Multus, and `qwen-roce`.
- Missing PVC after namespace deletion: stop the download Job, clear the stale
  claim references only on the two retained PVs, and repeat storage step 2.
- Missing `qwen32-bench/qwen-roce`: confirm the `MacvlanNetwork`
  `networkNamespace` is `qwen32-bench` and inspect Network Operator events.
- Hybrid/HMA error: stop; verify the exact 1.3.0 image and vLLM 0.23.0. Do not
  disable HMA to force the experiment through.
- No KV events: verify prefill `--kv-events-config`, `PYTHONHASHSEED=0`, ZMQ
  logs, and `/metrics`; do not invent metric names.
- Shared prompts but zero router hits: inspect attention-group events and
  router selections; Qwen3.6 recurrent groups are not equivalent to standard
  attention KV blocks.
- AIPerf install failure: the client Pod needs package-index egress or a
  prebuilt internal image containing exactly AIPerf 0.10.0.
- Artifact collision: select a new `ARTIFACT_ROOT`; do not delete old results
  unless they were archived.
- Timeouts on 32K/256 or 2K/4096: increase `REQUEST_TIMEOUT` and
  `GRACE_PERIOD`; do not accept timeout-defined saturation.
