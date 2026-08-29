# SGLang disaggregated tp1-4p4d

This recipe runs 4 TP=1 prefill workers and 4 TP=1 decode workers.
It requests four GPUs on each role node and eight GPUs total.

This is an experimental backend comparison. SGLang supports NIXL
prefill/decode disaggregation generally, but this runbook does not assume that
Qwen3.6 recurrent/GDN state transfer works. Startup logs and a real request are
mandatory acceptance gates.

## Variables

```bash
export NAMESPACE=dynamo-bench
export EXP_DIR=/ephemeral/shared/qwen3.6-35b-a3b/sglang/disagg/tp1-4p4d
export DEPLOYMENT=q36-sgl-pd-tp1-4p4d
export PERF_JOB_NAME=qwen36-sglang-tp1-4p4d-perf
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
export FRONTEND_SERVICE="${DEPLOYMENT}-frontend"
export MODEL=Qwen/Qwen3.6-35B-A3B-FP8
```

Complete the shared namespace, PVC, model-cache, and RoCE recovery in the
[parent runbook](../README.md) first. Run only one eight-GPU recipe at a time.

## Preflight

```bash
kubectl get pvc model-cache perf-cache -n "$NAMESPACE"
kubectl get network-attachment-definition roce -n "$NAMESPACE"
kubectl get nodes -L qwen.nvidia.com/role \
  -o custom-columns='NODE:.metadata.name,ROLE:.metadata.labels.qwen\.nvidia\.com/role,GPU:.status.allocatable.nvidia\.com/gpu,RDMA:.status.allocatable.rdma/ib'
kubectl get dynamographdeployments.nvidia.com -A
kubectl get pods -A \
  -o custom-columns='NAMESPACE:.metadata.namespace,POD:.metadata.name,NODE:.spec.nodeName,PHASE:.status.phase,GPU:.spec.containers[*].resources.requests.nvidia\.com/gpu'
```

Require both PVCs to be `Bound`, `roce` to exist, four free GPUs and
four free `rdma/ib` resources on each role node, and no competing GPU
deployment.

Run a CPU-only image/CLI preflight:

```bash
kubectl delete pod -n "$NAMESPACE" sglang-runtime-preflight --ignore-not-found
kubectl run sglang-runtime-preflight -n "$NAMESPACE" --restart=Never \
  --image=nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0 \
  --command -- /bin/bash -lc '
    set -e
    python3 -c "import sglang, nixl; print(\"SGLANG_AND_NIXL_IMPORT_OK\")"
    python3 -m dynamo.sglang --help |
      grep -E -- "--disaggregation-mode|--disaggregation-transfer-backend"
  '
kubectl wait -n "$NAMESPACE" --for=jsonpath='{.status.phase}'=Succeeded \
  pod/sglang-runtime-preflight --timeout=300s
kubectl logs -n "$NAMESPACE" sglang-runtime-preflight
kubectl delete pod -n "$NAMESPACE" sglang-runtime-preflight
```

This first check validates only the image and CLI. Next run an actual
one-prefill/one-decode GPU canary using the same TP size and runtime settings
as the full recipe.

### GPU model and NIXL canary

These high-level Dynamo Operator 1.3 manifests intentionally use
`nvidia.com/v1alpha1`: that schema accepts `spec.pvcs` and `spec.services`
and the operator converts it internally to beta `spec.components`. The
deprecation warning is expected; changing only `apiVersion` to `v1beta1`
causes a strict-decoding failure.

```bash
(
set -e
tee "$EXP_DIR/preflight.yaml" >/dev/null <<'EOF'
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: q36-sgl-pd-tp1-4p4d-pf
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
    SglangPrefillWorker:
      componentType: worker
      subComponentType: prefill
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 40Gi
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
              ulimit -l unlimited
              exec python3 -m dynamo.sglang \
                --model-path "$MODEL_PATH" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --tp-size 1 \
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
          securityContext: &worker-security-context
            runAsUser: 0
            capabilities:
              add:
                - IPC_LOCK
                - SYS_RESOURCE
      replicas: 1
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
    SglangDecodeWorker:
      componentType: worker
      subComponentType: decode
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 40Gi
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
              ulimit -l unlimited
              exec python3 -m dynamo.sglang \
                --model-path "$MODEL_PATH" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --tp-size 1 \
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
                --enable-metrics
          env: *worker-env
          securityContext: *worker-security-context
      replicas: 1
      resources: *worker-resources
EOF

export PREFLIGHT_DEPLOYMENT="${DEPLOYMENT}-pf"
export PREFLIGHT_GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${PREFLIGHT_DEPLOYMENT}"
export PREFLIGHT_SERVICE="${PREFLIGHT_DEPLOYMENT}-frontend"

kubectl delete dynamographdeployment.nvidia.com -n "$NAMESPACE" \
  "$PREFLIGHT_DEPLOYMENT" --ignore-not-found
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/preflight.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/preflight.yaml"

found=0
for attempt in $(seq 1 120); do
  count="$(kubectl get pods -n "$NAMESPACE" \
    -l "$PREFLIGHT_GRAPH_LABEL" --no-headers 2>/dev/null | wc -l)"
  if [ "$count" -ge 3 ]; then
    found=1
    break
  fi
  sleep 5
done

if [ "$found" -ne 1 ]; then
  echo "FAIL: operator did not create all three canary Pods" >&2
  kubectl describe dynamographdeployment "$PREFLIGHT_DEPLOYMENT" \
    -n "$NAMESPACE"
  kubectl get events -n "$NAMESPACE" --sort-by=.lastTimestamp | tail -50
  exit 1
fi

if ! kubectl wait -n "$NAMESPACE" --for=condition=Ready pod \
  -l "$PREFLIGHT_GRAPH_LABEL" --timeout=1800s; then
  kubectl get pods -n "$NAMESPACE" -l "$PREFLIGHT_GRAPH_LABEL" -o wide
  kubectl logs -n "$NAMESPACE" -l "$PREFLIGHT_GRAPH_LABEL" \
    --all-containers --prefix --tail=1500 || true
  exit 1
fi

kubectl logs -n "$NAMESPACE" -l "$PREFLIGHT_GRAPH_LABEL" \
  --all-containers --prefix --tail=1500 |
  tee "/tmp/qwen36-sglang-tp1-4p4d-preflight.log"

if grep -Ei 'traceback|assertionerror|notimplementederror|nixl_err|out of memory|does not support|unsupported' \
  "/tmp/qwen36-sglang-tp1-4p4d-preflight.log"; then
  echo "FAIL: canary startup logs contain a fatal signature" >&2
  exit 1
fi

kubectl run qwen-sglang-preflight-request --rm -i --restart=Never \
  --namespace "$NAMESPACE" --image=curlimages/curl -- \
  curl -fsS -H 'Content-Type: application/json' \
  --data-binary "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: sglang-preflight-ok\"}],\"temperature\":0,\"max_tokens\":32,\"stream\":false}" \
  "http://${PREFLIGHT_SERVICE}:8000/v1/chat/completions"

kubectl logs -n "$NAMESPACE" -l "$PREFLIGHT_GRAPH_LABEL" \
  --all-containers --prefix --since=10m |
  tee -a "/tmp/qwen36-sglang-tp1-4p4d-preflight.log"

if grep -Ei 'traceback|assertionerror|notimplementederror|nixl_err|transfer.*fail|does not support|unsupported' \
  "/tmp/qwen36-sglang-tp1-4p4d-preflight.log"; then
  echo "FAIL: canary request or transfer logs contain a fatal signature" >&2
  exit 1
fi

echo "PASS: SGLang model load and one P-to-D request completed"
kubectl delete dynamographdeployment.nvidia.com -n "$NAMESPACE" \
  "$PREFLIGHT_DEPLOYMENT" --wait=true
)
```

Do not apply the full deployment unless the canary prints `PASS`. The canary
uses two GPUs and must be deleted before the full eight-GPU recipe. A pass
substantially reduces risk, but it cannot guarantee that higher concurrency or
long-context requests will never expose another runtime failure.

## Create deploy.yaml

The same API-version note applies to the full deployment.

```bash
mkdir -p "$EXP_DIR"
tee "$EXP_DIR/deploy.yaml" >/dev/null <<'EOF'
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: q36-sgl-pd-tp1-4p4d
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
    SglangPrefillWorker:
      componentType: worker
      subComponentType: prefill
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 40Gi
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
              ulimit -l unlimited
              exec python3 -m dynamo.sglang \
                --model-path "$MODEL_PATH" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --tp-size 1 \
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
    SglangDecodeWorker:
      componentType: worker
      subComponentType: decode
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 40Gi
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
              ulimit -l unlimited
              exec python3 -m dynamo.sglang \
                --model-path "$MODEL_PATH" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --tp-size 1 \
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
                --enable-metrics
          env: *worker-env
          securityContext: *worker-security-context
      replicas: 4
      resources: *worker-resources
EOF
```

## Validate and deploy

```bash
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide -w
```

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
  tee "/tmp/qwen36-sglang-tp1-4p4d-startup.log"

if grep -Ei 'traceback|assertionerror|notimplementederror|nixl_err|out of memory|does not support|unsupported' \
  "/tmp/qwen36-sglang-tp1-4p4d-startup.log"; then
  echo "SGLang startup compatibility gate failed" >&2
  exit 1
fi

grep -Ei 'sglang|nixl|ucx|rdma|prefill|decode|transfer|ready' \
  "/tmp/qwen36-sglang-tp1-4p4d-startup.log" | tail -300
```

Do not benchmark merely because Pods are Ready. Send a real request:

```bash
kubectl run qwen-smoke --rm -i --restart=Never \
  --namespace "$NAMESPACE" --image=curlimages/curl -- \
  curl -fsS -H 'Content-Type: application/json' \
  --data-binary "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: smoke-test-ok\"}],\"temperature\":0,\"max_tokens\":32,\"stream\":false}" \
  "http://${FRONTEND_SERVICE}:8000/v1/chat/completions"

kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --all-containers --prefix --since=10m |
  grep -Ei 'nixl|ucx|rdma|transfer|error|traceback'
```

Accept only a successful response plus successful NIXL/UCX transfer evidence.
If logs report unsupported recurrent/GDN state, stop; changing TP size will not
repair backend support.

## Create perf.yaml

The benchmark supports two sequence-length modes:

- `WORKLOAD_MODE=fixed` uses only `ISL` and `OSL`, with zero variance.
- `WORKLOAD_MODE=mixed` ignores `ISL` and `OSL` and passes
  `SEQUENCE_DISTRIBUTION` to AIPerf's native
  `--sequence-distribution` option.

Mixed mode requires `PREFIX_MODE=isolated`. The Job validates the distribution
format, requires positive lengths and weights totaling 100, and records the
selected mode and distribution in `input-config.json`.

```bash
tee "$EXP_DIR/perf.yaml" >/dev/null <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: qwen36-sglang-tp1-4p4d-perf
spec:
  backoffLimit: 0
  completions: 1
  parallelism: 1
  activeDeadlineSeconds: 14400
  template:
    metadata:
      labels:
        app: qwen36-sglang-tp1-4p4d-perf
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

              prefix_args=()
              length_args=()
              exact_output_args=()

              case "$WORKLOAD_MODE" in
                fixed)
                  case "$PREFIX_MODE" in
                    isolated)
                      prompt_tokens="$ISL"
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
                  length_args=(
                    --isl "$prompt_tokens"
                    --isl-stddev 0
                    --osl "$OSL"
                    --osl-stddev 0
                  )
                  exact_output_args=(
                    --extra-inputs "max_tokens:$OSL"
                    --extra-inputs "min_tokens:$OSL"
                  )
                  run_shape="isl-${ISL}_osl-${OSL}"
                  ;;
                mixed)
                  if [ "$PREFIX_MODE" != isolated ]; then
                    echo "WORKLOAD_MODE=mixed requires PREFIX_MODE=isolated" >&2
                    exit 2
                  fi
                  python - "$SEQUENCE_DISTRIBUTION" <<'PY'
              import sys

              raw = sys.argv[1]
              if not raw:
                  raise SystemExit("SEQUENCE_DISTRIBUTION must not be empty")

              total = 0
              for entry in raw.split(";"):
                  try:
                      pair, weight_text = entry.split(":", 1)
                      isl_text, osl_text = pair.split(",", 1)
                      isl = int(isl_text)
                      osl = int(osl_text)
                      weight = int(weight_text)
                  except ValueError as exc:
                      raise SystemExit(
                          f"Invalid sequence-distribution entry: {entry!r}"
                      ) from exc
                  if isl < 1 or osl < 1 or weight < 1:
                      raise SystemExit(
                          f"ISL, OSL, and weight must be positive: {entry!r}"
                      )
                  total += weight

              if total != 100:
                  raise SystemExit(
                      f"Sequence-distribution weights must total 100, got {total}"
                  )
              print(f"Validated mixed sequence distribution: {raw}")
              PY
                  aiperf profile --help |
                    grep -F -- '--sequence-distribution' >/dev/null || {
                      echo "Installed AIPerf lacks --sequence-distribution" >&2
                      exit 2
                    }
                  length_args=(
                    --sequence-distribution "$SEQUENCE_DISTRIBUTION"
                  )
                  distribution_id="$(
                    printf '%s' "$SEQUENCE_DISTRIBUTION" |
                      sha256sum |
                      cut -c1-12
                  )"
                  run_shape="mixed-${distribution_id}"
                  ;;
                *)
                  echo "WORKLOAD_MODE must be fixed or mixed" >&2
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
              run_root="${ARTIFACT_ROOT}/${TOPOLOGY_NAME}/${PREFIX_MODE}/${WORKLOAD_NAME}/${run_shape}"
              mkdir -p "$run_root"
              status_file="${run_root}/matrix-status.tsv"

              printf '%s\n' "$models_json" > "${run_root}/models.json"
              curl -fsS "${endpoint_url}/metrics" > "${run_root}/frontend-metrics-before.prom" || true
              jq -n \
                --arg topology "$TOPOLOGY_NAME" \
                --arg workload "$WORKLOAD_NAME" \
                --arg workload_mode "$WORKLOAD_MODE" \
                --arg prefix_mode "$PREFIX_MODE" \
                --arg sequence_distribution "$SEQUENCE_DISTRIBUTION" \
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
                '{topology:$topology, workload:$workload,
                  workload_mode:$workload_mode, prefix_mode:$prefix_mode,
                  sequence_distribution:$sequence_distribution,
                  model:$model, tokenizer:$tokenizer, endpoint:$endpoint,
                  isl:(if $workload_mode == "fixed" then $isl else null end),
                  osl:(if $workload_mode == "fixed" then $osl else null end),
                  concurrencies:$concurrencies,
                  benchmark_duration_seconds:$duration, warmup_requests:$warmup,
                  random_seed:$seed, request_timeout_seconds:$request_timeout,
                  prefix_groups:$prefix_groups,
                  target_prefix_token_percent:$prefix_reuse_percent,
                  aiperf:"0.10.0", dynamo:"1.3.0", backend:"sglang"}' \
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

                echo "Starting workload=$WORKLOAD_NAME mode=$WORKLOAD_MODE prefix=$PREFIX_MODE shape=$run_shape concurrency=$concurrency"
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
                  "${length_args[@]}" \
                  "${exact_output_args[@]}" \
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
              value: glm52-fp8-vllm-a-984b-0-frontend:8000
            - name: ISL
              value: "8000"
            - name: OSL
              value: "1024"
            - name: WORKLOAD_MODE
              value: fixed
            - name: SEQUENCE_DISTRIBUTION
              value: "1024,256:35;4096,512:30;8192,1024:20;16384,512:10;32768,256:5"
            - name: CONCURRENCIES
              value: "1 2 4 8 16 32 64 128"
            - name: BENCHMARK_DURATION
              value: "180"
            - name: WARMUP_REQUESTS
              value: "32"
            - name: RANDOM_SEED
              value: "42"
            - name: ARTIFACT_ROOT
              value: /perf-cache/aiperf/qwen36-sglang
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

## Quick benchmark

The quick benchmark exercises mixed-mode support with the configured weighted
distribution. It uses a new artifact root on every rerun:

```bash
kubectl delete job -n "$NAMESPACE" "$PERF_JOB_NAME" --ignore-not-found
kubectl set env --local -f "$EXP_DIR/perf.yaml" -o yaml \
  WORKLOAD_MODE=mixed WORKLOAD_NAME=quick-mixed \
  SEQUENCE_DISTRIBUTION='1024,256:35;4096,512:30;8192,1024:20;16384,512:10;32768,256:5' \
  PREFIX_MODE=isolated CONCURRENCIES=4 BENCHMARK_DURATION=30 \
  WARMUP_REQUESTS=4 \
  ARTIFACT_ROOT="/perf-cache/aiperf/qwen36-sglang-quick-$(date -u +%Y%m%dT%H%M%SZ)" |
  kubectl apply -n "$NAMESPACE" -f -
kubectl logs -n "$NAMESPACE" -f "job/$PERF_JOB_NAME"
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  "job/$PERF_JOB_NAME" --timeout=14400s
```

This 30-second run verifies feature operation; it is too short to guarantee
that completed requests exactly match the target percentages, particularly
for 32K inputs. For measurements, use at least 180 seconds and keep the
distribution, concurrency, duration, warmup, and random seed identical across
the compared deployments.

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
