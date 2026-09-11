# Experiment 3: speculative decoding × KV routing

This recipe implements the mandatory 2×2 design with four independent TP1 H100
workers. Cell A is round-robin without speculation, B is round-robin with MTP,
C is KV routing without speculation, and D is KV routing with MTP. The same
manifest is rendered for each cell so model revision, cache block size, Mamba
settings, memory utilization, and worker count cannot drift between cells.

| Cell | Routing Strategy | Speculative Decoding | Description |
| :--- | :--- | :--- | :--- |
| **A** | Round-Robin | Disabled (`off`) | Baseline control without speculative decoding or KV routing |
| **B** | Round-Robin | Multi-Token Prediction (`mtp`, 7 tokens) | Speculative decoding isolated under round-robin routing |
| **C** | KV-Aware Routing | Disabled (`off`) | KV cache-aware routing isolated without speculative decoding |
| **D** | KV-Aware Routing | Multi-Token Prediction (`mtp`, 7 tokens) | Combined speculative decoding and KV cache-aware routing |

The first stage uses built-in MTP with seven speculative tokens. This runbook
downloads the pinned base, DFlash, and DSpark checkpoints before creating the
graph. DFlash and DSpark should be introduced only after the MTP 2×2 has
produced stable results.

## 1. Variables and preflight

Run these commands from a cluster-administration host.

```bash
export NAMESPACE=qwen32-bench
export RECIPE_ROOT=/ephemeral/shared/nemotron-3.5-lightning
export MODEL_CACHE_DIR="$RECIPE_ROOT/model-cache"
export EXP_DIR=/ephemeral/shared/nemotron-3.5-lightning/vllm/experiments/03-spec-kv-routing/aggregated
export DOWNLOAD_JOB=nemotron35-model-download
export TOPOLOGY=aggregated
export ARTIFACT_ROOT=/perf-cache/specrouting/aggregated
export DEPLOYMENT=nemotron35-vllm-e3
export PERF_JOB=nemotron35-vllm-e3-perf
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=$DEPLOYMENT"
export RUNTIME_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1
export CAPABILITY_POD=nemotron35-vllm-capability
mkdir -p "$MODEL_CACHE_DIR" "$EXP_DIR"

kubectl get crd dynamographdeployments.nvidia.com
kubectl get pvc model-cache perf-cache -n "$NAMESPACE"
kubectl get nodes \
  -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
```

The graph needs four allocatable H100s. Verify that the runtime contains vLLM
0.26.0 before scheduling it.

```bash
kubectl delete pod "$CAPABILITY_POD" -n "$NAMESPACE" --ignore-not-found
kubectl run "$CAPABILITY_POD" -n "$NAMESPACE" \
  --image="$RUNTIME_IMAGE" --restart=Never --command -- \
  python3 -c 'import importlib.metadata as m; version=m.version("vllm"); print("vLLM", version); assert version == "0.26.0", version'
kubectl wait -n "$NAMESPACE" --for=jsonpath='{.status.phase}'=Succeeded \
  "pod/$CAPABILITY_POD" --timeout=300s
kubectl logs -n "$NAMESPACE" "$CAPABILITY_POD"
kubectl delete pod "$CAPABILITY_POD" -n "$NAMESPACE" --ignore-not-found
```

## 2. Download and verify the pinned models

The deployment uses `HF_HUB_OFFLINE=1` and exact snapshot paths, so model
download is a mandatory gate. The Job below runs before any
`DynamoGraphDeployment` is created. Re-running it is safe: the Job verifies the
exact snapshot's configuration and weight files first and skips the Hugging
Face download call when they are complete. Exact commit SHAs prevent a moving
`main` branch from changing the experiment.

```bash
tee "$MODEL_CACHE_DIR/model-download.yaml" >/dev/null <<'MODEL_EOF'
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
apiVersion: batch/v1
kind: Job
metadata:
  name: nemotron35-model-download
spec:
  backoffLimit: 3
  activeDeadlineSeconds: 43200
  template:
    metadata:
      labels:
        app: nemotron35-model-download
    spec:
      restartPolicy: Never
      containers:
        - name: model-download
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1
          imagePullPolicy: IfNotPresent
          command: [python3, -c]
          args:
            - |
              import json
              import os
              from pathlib import Path
              from huggingface_hub import snapshot_download

              cache_root = Path("/model-cache/hub")

              def snapshot_path(repo_id, revision):
                  return (
                      cache_root
                      / f"models--{repo_id.replace('/', '--')}"
                      / "snapshots"
                      / revision
                  )

              def complete(snapshot):
                  config = snapshot / "config.json"
                  if not config.is_file() or config.stat().st_size == 0:
                      return False
                  for index_name in (
                      "model.safetensors.index.json",
                      "pytorch_model.bin.index.json",
                  ):
                      index = snapshot / index_name
                      if not index.is_file() or index.stat().st_size == 0:
                          continue
                      try:
                          weight_map = json.loads(
                              index.read_text(encoding="utf-8")
                          )["weight_map"]
                      except (KeyError, json.JSONDecodeError):
                          return False
                      weights = set(weight_map.values())
                      return bool(weights) and all(
                          (snapshot / weight).is_file()
                          and (snapshot / weight).stat().st_size > 0
                          for weight in weights
                      )
                  direct_weights = tuple(snapshot.glob("*.safetensors")) + tuple(
                      snapshot.glob("pytorch_model*.bin")
                  )
                  return any(
                      weight.is_file() and weight.stat().st_size > 0
                      for weight in direct_weights
                  )

              models = (
                  ("BASE_MODEL", "BASE_REVISION"),
                  ("DFLASH_MODEL", "DFLASH_REVISION"),
                  ("DSPARK_MODEL", "DSPARK_REVISION"),
              )
              for model_key, revision_key in models:
                  repo_id = os.environ[model_key]
                  revision = os.environ[revision_key]
                  expected = snapshot_path(repo_id, revision)
                  if complete(expected):
                      snapshot = expected
                      print(
                          f"Cache hit for {repo_id}@{revision}; skipping download",
                          flush=True,
                      )
                  else:
                      print(f"Downloading {repo_id}@{revision}", flush=True)
                      snapshot = Path(
                          snapshot_download(repo_id=repo_id, revision=revision)
                      )
                  if not complete(snapshot):
                      raise RuntimeError(
                          f"incomplete model snapshot after download: {snapshot}"
                      )
                  print(f"Verified {repo_id}@{revision}: {snapshot}", flush=True)

              for path in sorted(cache_root.glob("*/snapshots/*")):
                  print(path)
          env:
            - name: BASE_MODEL
              value: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
            - name: BASE_REVISION
              value: cc84af2fe71647d87f4486c064f320e1e7535243
            - name: DFLASH_MODEL
              value: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash
            - name: DFLASH_REVISION
              value: 7fc1f1ff4b82b917efbd0710df0872c2bb89caa5
            - name: DSPARK_MODEL
              value: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark
            - name: DSPARK_REVISION
              value: d10c6ff40d6e69d1f92e407e027de3eafdb77645
            - name: HF_HOME
              value: /model-cache
            - name: HF_XET_HIGH_PERFORMANCE
              value: "1"
          resources:
            requests:
              cpu: "4"
              memory: 16Gi
            limits:
              cpu: "8"
              memory: 32Gi
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: [ALL]
          volumeMounts:
            - name: model-cache
              mountPath: /model-cache
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache
MODEL_EOF

kubectl delete job "$DOWNLOAD_JOB" -n "$NAMESPACE" --ignore-not-found
kubectl apply --dry-run=server -n "$NAMESPACE" \
  -f "$MODEL_CACHE_DIR/model-download.yaml"
kubectl apply -n "$NAMESPACE" -f "$MODEL_CACHE_DIR/model-download.yaml"
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  "job/$DOWNLOAD_JOB" --timeout=43200s
kubectl logs -n "$NAMESPACE" "job/$DOWNLOAD_JOB" --tail=200
```

Do not continue to deployment unless the Job is `Complete` and its log contains
all three `Verified ...@<commit>` lines. A failed or timed-out Job leaves the
deployment blocked; inspect it before retrying:

```bash
kubectl get job "$DOWNLOAD_JOB" -n "$NAMESPACE" -o wide
kubectl get pods -n "$NAMESPACE" -l "job-name=$DOWNLOAD_JOB" -o wide
kubectl describe job "$DOWNLOAD_JOB" -n "$NAMESPACE"
```

## 3. Create the cell-aware deployment template

```bash
tee "$EXP_DIR/deploy.template.yaml" >/dev/null <<'DEPLOY_EOF'
# Set experiment-cell to A, B, C, or D before applying. See README.md.
apiVersion: v1
kind: ConfigMap
metadata:
  name: nemotron35-vllm-e3-settings
  labels:
    research.nvidia.com/experiment: spec-kv-routing
data:
  experiment-cell: A
  speculative-config: |-
    {"method":"mtp","num_speculative_tokens":7}
---
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: nemotron35-vllm-e3
  labels:
    app.kubernetes.io/name: nemotron35-vllm-e3
    app.kubernetes.io/part-of: nemotron35-research
    research.nvidia.com/experiment: spec-kv-routing
spec:
  backendFramework: vllm
  components:
    - name: Frontend
      type: frontend
      replicas: 2
      podTemplate:
        metadata:
          labels:
            app.kubernetes.io/name: nemotron35-vllm-e3
            app.kubernetes.io/component: frontend
            research.nvidia.com/experiment: spec-kv-routing
        spec:
          containers:
            - name: main
              image: &runtime_image nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1
              imagePullPolicy: IfNotPresent
              command: [/bin/sh, -c]
              args:
                - |
                  set -eu
                  set -- python3 -m dynamo.frontend --trust-remote-code
                  case "$EXPERIMENT_CELL" in
                    A|B)
                      set -- "$@" --router-mode round-robin
                      ;;
                    C|D)
                      set -- "$@" \
                        --router-mode kv \
                        --router-kv-events \
                        --kv-cache-block-size 64
                      ;;
                    *)
                      echo "EXPERIMENT_CELL must be A, B, C, or D" >&2
                      exit 2
                      ;;
                  esac
                  exec "$@"
              env:
                - name: EXPERIMENT_CELL
                  valueFrom:
                    configMapKeyRef:
                      name: nemotron35-vllm-e3-settings
                      key: experiment-cell
                - name: DYN_HTTP_BODY_LIMIT_MB
                  value: "200"
                - name: HF_HOME
                  value: /opt/models
                - name: HF_HUB_OFFLINE
                  value: "1"
                - name: HF_MODULES_CACHE
                  value: /tmp/hf_modules
              resources:
                requests:
                  cpu: "8"
                  memory: 8Gi
                limits:
                  cpu: "16"
                  memory: 16Gi
              startupProbe:
                httpGet:
                  path: /health
                  port: http
                periodSeconds: 10
                timeoutSeconds: 5
                failureThreshold: 90
              readinessProbe:
                httpGet:
                  path: /health
                  port: http
                initialDelaySeconds: 10
                periodSeconds: 10
                timeoutSeconds: 3
                failureThreshold: 3
              livenessProbe:
                httpGet:
                  path: /health
                  port: http
                initialDelaySeconds: 15
                periodSeconds: 10
                timeoutSeconds: 3
                failureThreshold: 3
              ports:
                - name: http
                  containerPort: 8000
              volumeMounts:
                - name: model-cache
                  mountPath: /opt/models
                  readOnly: true
          volumes:
            - name: model-cache
              persistentVolumeClaim:
                claimName: model-cache

    - name: VllmWorker
      type: worker
      replicas: 4
      sharedMemorySize: 40Gi
      podTemplate:
        metadata:
          labels:
            app.kubernetes.io/name: nemotron35-vllm-e3
            app.kubernetes.io/component: worker
            research.nvidia.com/experiment: spec-kv-routing
        spec:
          affinity:
            nodeAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
                nodeSelectorTerms:
                  - matchExpressions:
                      - key: nvidia.com/gpu.present
                        operator: In
                        values: ["true"]
                      - key: nvidia.com/gpu.product
                        operator: In
                        values: [NVIDIA-H100-80GB-HBM3]
          tolerations:
            - key: nvidia.com/gpu
              operator: Equal
              value: "true"
              effect: NoSchedule
          volumes:
            - name: model-cache
              persistentVolumeClaim:
                claimName: model-cache
            - name: dshm
              emptyDir:
                medium: Memory
                sizeLimit: 40Gi
          containers:
            - name: main
              image: *runtime_image
              imagePullPolicy: IfNotPresent
              workingDir: /workspace
              command: [/bin/sh, -c]
              args:
                - |
                  set -eu
                  ulimit -l unlimited
                  set -- python3 -m dynamo.vllm \
                    --model "$MODEL_PATH" \
                    --served-model-name "$SERVED_MODEL_NAME" \
                    --trust-remote-code \
                    --tensor-parallel-size 1 \
                    --max-model-len 65536 \
                    --max-num-seqs 512 \
                    --max-num-batched-tokens 32768 \
                    --gpu-memory-utilization 0.90 \
                    --async-scheduling \
                    --enable-prefix-caching \
                    --block-size 64 \
                    --mamba-cache-mode align \
                    --mamba-backend flashinfer \
                    --mamba-ssm-cache-dtype float16 \
                    --enable-mamba-cache-stochastic-rounding \
                    --mamba-cache-philox-rounds 5 \
                    --dyn-tool-call-parser nemotron_nano \
                    --dyn-reasoning-parser nemotron_nano \
                    --reasoning-parser nemotron_v3 \
                    --disaggregation-mode agg

                  case "$EXPERIMENT_CELL" in
                    A)
                      ;;
                    B)
                      set -- "$@" --speculative-config "$SPECULATIVE_CONFIG"
                      ;;
                    C)
                      set -- "$@" --kv-events-config "$KV_EVENTS_CONFIG"
                      ;;
                    D)
                      set -- "$@" \
                        --speculative-config "$SPECULATIVE_CONFIG" \
                        --kv-events-config "$KV_EVENTS_CONFIG"
                      ;;
                    *)
                      echo "EXPERIMENT_CELL must be A, B, C, or D" >&2
                      exit 2
                      ;;
                  esac
                  exec "$@"
              env:
                - name: EXPERIMENT_CELL
                  valueFrom:
                    configMapKeyRef:
                      name: nemotron35-vllm-e3-settings
                      key: experiment-cell
                - name: SPECULATIVE_CONFIG
                  valueFrom:
                    configMapKeyRef:
                      name: nemotron35-vllm-e3-settings
                      key: speculative-config
                - name: KV_EVENTS_CONFIG
                  value: '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
                - name: SERVED_MODEL_NAME
                  value: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
                - name: MODEL_PATH
                  value: /opt/models/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/snapshots/cc84af2fe71647d87f4486c064f320e1e7535243
                - name: HF_HOME
                  value: /opt/models
                - name: HF_HUB_OFFLINE
                  value: "1"
                - name: HF_MODULES_CACHE
                  value: /tmp/hf_modules
                - name: TRITON_CACHE_DIR
                  value: /tmp/.triton-cache
                - name: VLLM_CONFIG_ROOT
                  value: /tmp/vllm-config
                - name: VLLM_CACHE_ROOT
                  value: /tmp/vllm-cache
                - name: NCCL_IB_DISABLE
                  value: "1"
                - name: PYTHONHASHSEED
                  value: "0"
              resources:
                requests:
                  cpu: "16"
                  memory: 96Gi
                  nvidia.com/gpu: "1"
                limits:
                  cpu: "32"
                  memory: 128Gi
                  nvidia.com/gpu: "1"
              securityContext:
                runAsUser: 0
                capabilities:
                  add: [IPC_LOCK, SYS_RESOURCE]
              startupProbe:
                httpGet:
                  path: /live
                  port: 9090
                periodSeconds: 30
                timeoutSeconds: 20
                failureThreshold: 120
              ports:
                - name: system
                  containerPort: 9090
              volumeMounts:
                - name: model-cache
                  mountPath: /opt/models
                  readOnly: true
                - name: dshm
                  mountPath: /dev/shm
DEPLOY_EOF
```

The template defaults to A. Render exactly one cell at a time and delete the
old graph before switching, because a ConfigMap-only edit does not guarantee a
worker restart.

```bash
export CELL=A
case "$CELL" in A|B|C|D) ;; *) echo 'CELL must be A, B, C, or D' >&2; exit 2 ;; esac
sed "s/experiment-cell: A/experiment-cell: $CELL/" \
  "$EXP_DIR/deploy.template.yaml" > "$EXP_DIR/deploy-$CELL.yaml"

kubectl delete job "$PERF_JOB" -n "$NAMESPACE" --ignore-not-found
kubectl delete dynamographdeployment "$DEPLOYMENT" -n "$NAMESPACE" --ignore-not-found
kubectl delete configmap nemotron35-vllm-e3-settings -n "$NAMESPACE" --ignore-not-found
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/deploy-$CELL.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy-$CELL.yaml"
kubectl wait -n "$NAMESPACE" --for=jsonpath='{.status.state}'=successful \
  "dynamographdeployment/$DEPLOYMENT" --timeout=60m
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide -w
kubectl get endpoints "${DEPLOYMENT}-frontend" -n "$NAMESPACE"
```

Before accepting a run, inspect a worker log and frontend metrics. B and D must
show MTP initialization; C and D must expose KV-router and applied-event metric
families. A and B must not publish KV events.

### Port-forward and test the frontend

Keep this command running in a dedicated terminal. The Kubernetes frontend
listens on port 8000; `LOCAL_PORT` is the port exposed only on the administration
host. Once the deployment endpoints are active, port-forward using the frontend
Service:

```bash
export LOCAL_PORT=8000
kubectl port-forward -n "$NAMESPACE" \
  "svc/${DEPLOYMENT}-frontend" "${LOCAL_PORT}:8000"
```

If you need to test a specific frontend pod directly without routing through the
Service object:

```bash
export LOCAL_PORT=8000
kubectl port-forward -n "$NAMESPACE" \
  "$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=$DEPLOYMENT,app.kubernetes.io/component=frontend" -o name | head -n 1)" \
  "${LOCAL_PORT}:8000"
```

From a second terminal, verify health and confirm that the exact served model
is registered before sending a streaming chat request:

```bash
export LOCAL_PORT=8000

curl -fsS "http://127.0.0.1:${LOCAL_PORT}/health"
curl -fsS "http://127.0.0.1:${LOCAL_PORT}/v1/models" | jq .

curl -fsS -N "http://127.0.0.1:${LOCAL_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    "messages": [
      {"role": "user", "content": "Reply with exactly: frontend ready"}
    ],
    "max_tokens": 64,
    "temperature": 0,
    "stream": true
  }'
```

Do not start the benchmark unless `/health` succeeds, `/v1/models` contains
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, and the chat request
returns a valid streaming response. Stop the port-forward with `Ctrl-C` after
testing; the in-cluster benchmark uses the frontend Service directly.

## 4. Run the saturation and breaker benchmark

The benchmark Job, high-concurrency sweep, stopping rules, and matched MTP
on/off comparison procedure are in [`perf.md`](../perf.md). The default Job now
sweeps through concurrency 2048, equal to the configured aggregate
`max-num-seqs` ceiling of four workers times 512 sequences, and continues to
8192 to locate the breaker boundary.

Configure the Prometheus and DCGM capture in [`metrics.md`](../metrics.md) before
the first load point. MTP acceptance, engine queue depth, GPU activity, KV-cache
pressure, errors, and pod restarts must be recorded over the same UTC interval
as every AIPerf result.

## 5. Run all four cells automatically

After generating the download and deployment templates above and `perf.yaml`
from the shared benchmark runbook, create the runner below. It runs the model
gate, deploys each cell, waits for the benchmark, archives logs, and removes the
cell before moving on. Complete the manual streaming and repeated-prompt checks
first, including all four cells for disaggregated serving.

```bash
tee "$EXP_DIR/run-experiment.sh" >/dev/null <<'RUNNER_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

umask 027

NAMESPACE="${NAMESPACE:-qwen32-bench}"
RECIPE_ROOT="${RECIPE_ROOT:-/ephemeral/shared/nemotron-3.5-lightning}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$RECIPE_ROOT/model-cache}"
EXP_DIR="${EXP_DIR:-$RECIPE_ROOT/vllm/experiments/03-spec-kv-routing/aggregated}"
DOWNLOAD_JOB="${DOWNLOAD_JOB:-nemotron35-model-download}"
DEPLOYMENT="${DEPLOYMENT:-nemotron35-vllm-e3}"
PERF_JOB="${PERF_JOB:-nemotron35-vllm-e3-perf}"
SETTINGS_CONFIGMAP="${SETTINGS_CONFIGMAP:-nemotron35-vllm-e3-settings}"
GRAPH_LABEL="${GRAPH_LABEL:-nvidia.com/dynamo-graph-deployment-name=$DEPLOYMENT}"
EXPECTED_WORKERS="${EXPECTED_WORKERS:-4}"
DOWNLOAD_TIMEOUT_SECONDS="${DOWNLOAD_TIMEOUT_SECONDS:-43200}"
DEPLOY_TIMEOUT="${DEPLOY_TIMEOUT:-60m}"
PERF_TIMEOUT_SECONDS="${PERF_TIMEOUT_SECONDS:-14700}"
CELL_ORDER="${CELL_ORDER:-A B C D}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-false}"
DOWNLOAD_MANIFEST="${DOWNLOAD_MANIFEST:-$MODEL_CACHE_DIR/model-download.yaml}"
DEPLOY_TEMPLATE="${DEPLOY_TEMPLATE:-$EXP_DIR/deploy.template.yaml}"
PERF_TEMPLATE="${PERF_TEMPLATE:-$EXP_DIR/perf.yaml}"
PUBLIC_DOWNLOAD_MANIFEST="$EXP_DIR/model-download.public.yaml"
PUBLIC_DEPLOY_TEMPLATE="$EXP_DIR/deploy.public.template.yaml"
LOG_ROOT="${LOG_ROOT:-$EXP_DIR/logs}"
RUN_ID="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
SUITE_LOG_FILE="${SUITE_LOG_FILE:-$LOG_ROOT/suite-$RUN_ID.log}"
SUMMARY_FILE="${SUMMARY_FILE:-$LOG_ROOT/suite-$RUN_ID.tsv}"
MODEL_GATE_LOG="${MODEL_GATE_LOG:-$LOG_ROOT/model-cache-gate-$RUN_ID.log}"
CURRENT_CELL=""
CURRENT_PHASE="startup"

[[ "$EXPECTED_WORKERS" =~ ^[1-9][0-9]*$ ]] || {
  printf 'EXPECTED_WORKERS must be a positive integer\n' >&2
  exit 2
}
case "$CONTINUE_ON_ERROR" in
  true|false) ;;
  *) printf 'CONTINUE_ON_ERROR must be true or false\n' >&2; exit 2 ;;
esac

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$SUITE_LOG_FILE") 2>&1

log() {
  local context=""
  if [[ -n "$CURRENT_CELL" ]]; then
    context=" [cell=$CURRENT_CELL]"
  fi
  printf '%s [%s]%s %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$context" "$2"
}

prepare_public_templates() {
  log INFO "rendering credential-free templates for public images and models"
  sed \
    -e '/^[[:space:]]*imagePullSecrets:[[:space:]]*$/{N;d;}' \
    -e '/^[[:space:]]*envFrom:[[:space:]]*$/{N;N;d;}' \
    "$DOWNLOAD_MANIFEST" > "$PUBLIC_DOWNLOAD_MANIFEST"
  sed \
    -e '/imagePullSecrets: &image_pull_secrets/{N;d;}' \
    -e '/imagePullSecrets: \*image_pull_secrets/d' \
    "$DEPLOY_TEMPLATE" > "$PUBLIC_DEPLOY_TEMPLATE"
}

suite_cleanup() {
  local rc=$?
  trap - EXIT ERR INT TERM
  set +e
  log INFO "performing final defensive cleanup"
  kubectl delete job "$PERF_JOB" -n "$NAMESPACE" \
    --ignore-not-found --wait=true --timeout=10m
  kubectl delete dynamographdeployment "$DEPLOYMENT" -n "$NAMESPACE" \
    --ignore-not-found --wait=true --timeout=20m
  kubectl delete configmap "$SETTINGS_CONFIGMAP" -n "$NAMESPACE" \
    --ignore-not-found --wait=true --timeout=5m
  kubectl delete job "$DOWNLOAD_JOB" -n "$NAMESPACE" \
    --ignore-not-found --wait=true --timeout=10m
  log INFO "suite log: $SUITE_LOG_FILE"
  log INFO "suite summary: $SUMMARY_FILE"
  exit "$rc"
}
suite_on_error() {
  local rc=$?
  local line="${BASH_LINENO[0]:-unknown}"
  trap - ERR
  log ERROR "fatal error during $CURRENT_PHASE at line $line (exit code $rc)"
  exit "$rc"
}
trap suite_cleanup EXIT
trap suite_on_error ERR
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_job() {
  local job_name=$1
  local timeout_seconds=$2
  local poll_seconds=${3:-10}
  local deadline=$(( $(date +%s) + timeout_seconds ))
  local previous_state=""

  while (( $(date +%s) < deadline )); do
    local snapshot active succeeded failed failed_condition failed_reason state
    snapshot="$(kubectl get job "$job_name" -n "$NAMESPACE" \
      -o jsonpath='{.status.active}{"|"}{.status.succeeded}{"|"}{.status.failed}{"|"}{range .status.conditions[?(@.type=="Failed")]}{.status}{"|"}{.reason}{end}')"
    IFS='|' read -r active succeeded failed failed_condition failed_reason <<<"$snapshot"
    active="${active:-0}"
    succeeded="${succeeded:-0}"
    failed="${failed:-0}"
    state="active=$active succeeded=$succeeded failed=$failed"
    if [[ "$state" != "$previous_state" ]]; then
      log INFO "$job_name: $state"
      previous_state="$state"
    fi
    if (( succeeded >= 1 )); then
      return 0
    fi
    if [[ "$failed_condition" == "True" ]]; then
      log ERROR "$job_name failed: ${failed_reason:-unknown reason}"
      return 1
    fi
    sleep "$poll_seconds"
  done

  log ERROR "$job_name did not complete within ${timeout_seconds}s"
  return 124
}

wait_for_job_pod() {
  local job_name=$1
  local timeout_seconds=$2
  local deadline=$(( $(date +%s) + timeout_seconds ))
  local snapshot pod_name phase

  while (( $(date +%s) < deadline )); do
    snapshot="$(kubectl get pods -n "$NAMESPACE" -l "job-name=$job_name" \
      -o jsonpath='{.items[0].metadata.name}{"|"}{.items[0].status.phase}')"
    IFS='|' read -r pod_name phase <<<"$snapshot"
    case "$phase" in
      Running|Succeeded|Failed)
        [[ -n "$pod_name" ]] || continue
        printf '%s\n' "$pod_name"
        return 0
        ;;
    esac
    sleep 5
  done
  return 124
}

download_diagnostics() (
  set +e
  log ERROR "collecting model-cache gate diagnostics"
  kubectl get job "$DOWNLOAD_JOB" -n "$NAMESPACE" -o wide
  kubectl get pods -n "$NAMESPACE" -l "job-name=$DOWNLOAD_JOB" -o wide
  kubectl describe job "$DOWNLOAD_JOB" -n "$NAMESPACE"
  kubectl logs -n "$NAMESPACE" -l "job-name=$DOWNLOAD_JOB" \
    --all-containers=true --prefix=true --tail=200
)

run_model_gate() {
  log INFO "checking the three pinned snapshots on the model-cache PVC"
  kubectl apply --dry-run=server -n "$NAMESPACE" \
    -f "$PUBLIC_DOWNLOAD_MANIFEST" >/dev/null
  kubectl delete job "$DOWNLOAD_JOB" -n "$NAMESPACE" \
    --ignore-not-found --wait=true --timeout=10m
  kubectl apply -n "$NAMESPACE" -f "$PUBLIC_DOWNLOAD_MANIFEST"

  if ! wait_for_job "$DOWNLOAD_JOB" "$DOWNLOAD_TIMEOUT_SECONDS" 15; then
    download_diagnostics
    return 1
  fi

  kubectl logs -n "$NAMESPACE" "job/$DOWNLOAD_JOB" --timestamps |
    tee "$MODEL_GATE_LOG"
  for snapshot in \
    '/model-cache/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/snapshots/cc84af2fe71647d87f4486c064f320e1e7535243' \
    '/model-cache/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash/snapshots/7fc1f1ff4b82b917efbd0710df0872c2bb89caa5' \
    '/model-cache/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark/snapshots/d10c6ff40d6e69d1f92e407e027de3eafdb77645'; do
    grep -Fq "$snapshot" "$MODEL_GATE_LOG" || {
      log ERROR "model-cache Job did not report required snapshot: $snapshot"
      return 1
    }
    log INFO "required snapshot is ready: $snapshot"
  done
  log INFO "model-cache gate passed; complete snapshots were not downloaded again"
}

run_cell() (
  set -Eeuo pipefail

  local cell=$1
  local cell_stamp=$2
  local cell_log="$LOG_ROOT/cell-$cell-$cell_stamp.log"
  local kube_log_dir="$LOG_ROOT/kubernetes/cell-$cell-$cell_stamp"
  local deploy_manifest="$EXP_DIR/deploy-$cell.yaml"
  local perf_manifest="$EXP_DIR/perf-$cell.yaml"
  local follow_pid=""

  CURRENT_CELL="$cell"
  exec > >(tee -a "$cell_log") 2>&1

  cell_diagnostics() (
    set +e
    log ERROR "collecting cell failure diagnostics"
    kubectl get dynamographdeployment "$DEPLOYMENT" -n "$NAMESPACE" -o wide
    kubectl describe dynamographdeployment "$DEPLOYMENT" -n "$NAMESPACE"
    kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide
    kubectl get job "$PERF_JOB" -n "$NAMESPACE" -o wide
    kubectl describe job "$PERF_JOB" -n "$NAMESPACE"
    kubectl logs -n "$NAMESPACE" -l "job-name=$PERF_JOB" \
      --all-containers=true --prefix=true --tail=200
    kubectl get events -n "$NAMESPACE" --sort-by=.lastTimestamp | tail -n 80
  )

  archive_kubernetes_logs() {
    local pod pod_log_name
    mkdir -p "$kube_log_dir"
    kubectl get dynamographdeployment "$DEPLOYMENT" -n "$NAMESPACE" -o yaml \
      > "$kube_log_dir/deployment.yaml" 2>&1
    kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide \
      > "$kube_log_dir/graph-pods.txt" 2>&1
    kubectl get job "$PERF_JOB" -n "$NAMESPACE" -o yaml \
      > "$kube_log_dir/perf-job.yaml" 2>&1
    kubectl logs -n "$NAMESPACE" -l "job-name=$PERF_JOB" \
      --all-containers=true --prefix=true --tail=1000 \
      > "$kube_log_dir/perf-pod.log" 2>&1

    while IFS= read -r pod; do
      [[ -n "$pod" ]] || continue
      pod_log_name="${pod#pod/}"
      kubectl logs -n "$NAMESPACE" "$pod" \
        --all-containers=true --prefix=true --tail=1000 \
        > "$kube_log_dir/$pod_log_name.log" 2>&1
    done < <(kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o name)
  }

  cell_cleanup() {
    local rc=$?
    local cleanup_rc=0
    trap - EXIT ERR INT TERM
    set +e

    if [[ -n "$follow_pid" ]] && kill -0 "$follow_pid" 2>/dev/null; then
      kill "$follow_pid" 2>/dev/null
      wait "$follow_pid" 2>/dev/null
    fi

    log INFO "archiving Kubernetes state and final pod logs"
    archive_kubernetes_logs
    log INFO "Kubernetes log archive: $kube_log_dir"
    log INFO "cleaning benchmark job and cell deployment"
    kubectl delete job "$PERF_JOB" -n "$NAMESPACE" \
      --ignore-not-found --wait=true --timeout=10m || cleanup_rc=$?
    kubectl delete dynamographdeployment "$DEPLOYMENT" -n "$NAMESPACE" \
      --ignore-not-found --wait=true --timeout=20m || cleanup_rc=$?
    kubectl delete configmap "$SETTINGS_CONFIGMAP" -n "$NAMESPACE" \
      --ignore-not-found --wait=true --timeout=5m || cleanup_rc=$?

    if (( cleanup_rc != 0 )); then
      log ERROR "cleanup failed with exit code $cleanup_rc; refusing handoff"
      if (( rc == 0 )); then
        rc=$cleanup_rc
      fi
    else
      log INFO "cleanup complete"
    fi
    log INFO "cell log: $cell_log"
    exit "$rc"
  }

  cell_on_error() {
    local rc=$?
    local line="${BASH_LINENO[0]:-unknown}"
    trap - ERR
    log ERROR "cell runner failed at line $line with exit code $rc"
    cell_diagnostics
    exit "$rc"
  }

  trap cell_cleanup EXIT
  trap cell_on_error ERR
  trap 'exit 130' INT
  trap 'exit 143' TERM

  log INFO "removing stale resources before deployment"
  kubectl delete job "$PERF_JOB" -n "$NAMESPACE" \
    --ignore-not-found --wait=true --timeout=10m
  kubectl delete dynamographdeployment "$DEPLOYMENT" -n "$NAMESPACE" \
    --ignore-not-found --wait=true --timeout=20m
  kubectl delete configmap "$SETTINGS_CONFIGMAP" -n "$NAMESPACE" \
    --ignore-not-found --wait=true --timeout=5m

  sed "s/experiment-cell: A/experiment-cell: $cell/" \
    "$PUBLIC_DEPLOY_TEMPLATE" > "$deploy_manifest"
  grep -Fq "experiment-cell: $cell" "$deploy_manifest"

  log INFO "server-validating and deploying $EXPECTED_WORKERS workers"
  kubectl apply --dry-run=server -n "$NAMESPACE" \
    -f "$deploy_manifest" >/dev/null
  kubectl apply -n "$NAMESPACE" -f "$deploy_manifest"
  kubectl wait -n "$NAMESPACE" --for=jsonpath='{.status.state}'=successful \
    "dynamographdeployment/$DEPLOYMENT" --timeout="$DEPLOY_TIMEOUT"
  kubectl wait -n "$NAMESPACE" --for=condition=Ready pod -l "$GRAPH_LABEL" \
    --timeout=10m

  actual_workers="$(kubectl get dynamographdeployment "$DEPLOYMENT" \
    -n "$NAMESPACE" \
    -o jsonpath='{.spec.components[?(@.name=="VllmWorker")].replicas}')"
  [[ "$actual_workers" == "$EXPECTED_WORKERS" ]] || {
    log ERROR "expected $EXPECTED_WORKERS workers, deployment declares ${actual_workers:-none}"
    exit 1
  }
  kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide

  sed \
    -e "s/{name: EXPERIMENT_CELL, value: A}/{name: EXPERIMENT_CELL, value: $cell}/" \
    -e "/^[[:space:]]*- name: EXPERIMENT_CELL[[:space:]]*$/{n;s/value: A/value: $cell/;}" \
    "$PERF_TEMPLATE" > "$perf_manifest"

  grep -A1 'name: EXPERIMENT_CELL' "$perf_manifest" |
    grep -Eq "value: ['\"]?$cell['\"]?([},]|$)"

  log INFO "server-validating and starting benchmark job $PERF_JOB"
  kubectl apply --dry-run=server -n "$NAMESPACE" \
    -f "$perf_manifest" >/dev/null
  kubectl apply -n "$NAMESPACE" -f "$perf_manifest"

  perf_pod="$(wait_for_job_pod "$PERF_JOB" 600)"
  log INFO "streaming benchmark pod $perf_pod"
  kubectl logs -n "$NAMESPACE" -f "$perf_pod" --timestamps &
  follow_pid=$!

  if ! wait_for_job "$PERF_JOB" "$PERF_TIMEOUT_SECONDS" 10; then
    if kill -0 "$follow_pid" 2>/dev/null; then
      kill "$follow_pid" 2>/dev/null || true
    fi
    wait "$follow_pid" 2>/dev/null || true
    follow_pid=""
    cell_diagnostics
    exit 1
  fi

  wait "$follow_pid" || true
  follow_pid=""
  kubectl get job "$PERF_JOB" -n "$NAMESPACE" -o wide
  log INFO "benchmark completed successfully"
)

for command in kubectl sed grep tee; do
  command -v "$command" >/dev/null || {
    log ERROR "required command is missing: $command"
    exit 127
  }
done
for file in "$DOWNLOAD_MANIFEST" "$DEPLOY_TEMPLATE" "$PERF_TEMPLATE"; do
  [[ -r "$file" ]] || {
    log ERROR "required manifest or template is not readable: $file"
    exit 2
  }
done
for cell in $CELL_ORDER; do
  case "$cell" in
    A|B|C|D) ;;
    *) log ERROR "invalid cell in CELL_ORDER: $cell"; exit 2 ;;
  esac
done

printf 'started_utc\tfinished_utc\tcell\tstatus\texit_code\tlog_file\n' \
  > "$SUMMARY_FILE"

CURRENT_PHASE="preflight"
log INFO "preflight for $EXPECTED_WORKERS-worker experiment in namespace $NAMESPACE"
kubectl get crd dynamographdeployments.nvidia.com >/dev/null
kubectl get pvc model-cache perf-cache -n "$NAMESPACE"

CURRENT_PHASE="public-template-render"
prepare_public_templates
CURRENT_PHASE="model-cache-gate"
run_model_gate

failures=0
for cell in $CELL_ORDER; do
  started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cell_stamp="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
  cell_log="$LOG_ROOT/cell-$cell-$cell_stamp.log"
  CURRENT_PHASE="cell-$cell"
  log INFO "starting cell $cell"

  set +e
  run_cell "$cell" "$cell_stamp"
  rc=$?
  set -e
  if (( rc == 0 )); then
    status=passed
  else
    status=failed
    failures=$(( failures + 1 ))
  fi

  finished_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$started_utc" "$finished_utc" "$cell" \
    "$status" "$rc" "$cell_log" >> "$SUMMARY_FILE"

  if (( rc != 0 )); then
    log ERROR "cell $cell failed with exit code $rc"
    if [[ "$CONTINUE_ON_ERROR" == "false" ]]; then
      log ERROR "stopping; set CONTINUE_ON_ERROR=true to continue after cleanup"
      exit "$rc"
    fi
  else
    log INFO "cell $cell passed and was cleaned up"
  fi
done

if (( failures > 0 )); then
  log ERROR "suite completed with $failures failed cell runs"
  exit 1
fi

CURRENT_PHASE="complete"
log INFO "all cell runs completed successfully"
RUNNER_EOF
chmod +x "$EXP_DIR/run-experiment.sh"
bash "$EXP_DIR/run-experiment.sh"
```

## 6. Cleanup

```bash
kubectl delete job "$PERF_JOB" -n "$NAMESPACE" --ignore-not-found
kubectl delete dynamographdeployment "$DEPLOYMENT" -n "$NAMESPACE" --ignore-not-found
kubectl delete configmap nemotron35-vllm-e3-settings -n "$NAMESPACE" --ignore-not-found
kubectl delete job "$DOWNLOAD_JOB" -n "$NAMESPACE" --ignore-not-found
```

Deleting the download Job does not remove the checkpoints stored on the
`model-cache` PVC.
