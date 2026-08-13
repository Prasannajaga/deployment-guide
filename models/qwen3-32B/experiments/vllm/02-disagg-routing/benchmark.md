# Qwen3-32B-FP8 disaggregated deployment: In-Cluster Benchmark

This guide creates the benchmark Pod manifest and executes the AIPerf benchmark against the `qwen3-32b-fp8-vllm-disagg` deployment using the Mooncake trace dataset.

## 1. Create the benchmark manifest

**Run on `gpu05` (Kubernetes control-plane node).**

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3-32b/experiments/vllm/02-disagg-routing

mkdir -p "$EXP_DIR"

tee "$EXP_DIR/benchmark.yaml" >/dev/null <<'EOF'
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
apiVersion: v1
kind: Pod
metadata:
  name: disagg-router-6p-2d-benchmark
  labels:
    app: benchmark
spec:
  containers:
  - name: python
    image: python:3.11
    command:
      - /bin/bash
      - -lc
      - |
        # Verbose Logging Helper
        log() { echo "[$(date '+%H:%M:%S')] $1"; }

        # Setup
        log "Setting ulimits..."
        ulimit -n 1048576
        ulimit -u 65536

        log "Updating packages and installing dependencies..."
        apt update && apt install tmux wget curl jq build-essential -y

        log "Installing AIPerf..."
        pip install aiperf==0.10.0

        log "Waiting for model '${MODEL_NAME}' at http://${FRONTEND}:8000/v1/models..."
        until curl -s "http://${FRONTEND}:8000/v1/models" | jq -e --arg model "${MODEL_NAME}" '.data[]? | select(.id == $model)' >/dev/null 2>&1; do
          log "Model not ready yet, retrying in 5s..."
          sleep 5
        done
        log "Model '${MODEL_NAME}' is ready!"

        # Download Mooncake conversation trace dataset if not already present
        mkdir -p ${BASE_DIR}/traces
        mkdir -p ${BASE_DIR}/artifacts
        export INPUT_FILE="${BASE_DIR}/traces/conversation_trace.jsonl"
        if [ ! -f "${INPUT_FILE}" ]; then
          log "Downloading Mooncake trace dataset..."
          wget --progress=dot:giga -O "${INPUT_FILE}" https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl
          log "Trace dataset download complete. Size: $(du -h "${INPUT_FILE}" | cut -f1)"
        else
          log "Trace dataset already present at ${INPUT_FILE}. Size: $(du -h "${INPUT_FILE}" | cut -f1)"
        fi

        # Setup Paths and Endpoints
        export MODEL_BASE_NAME="${MODEL_NAME##*/}"
        export FRONTEND_LIB="${FRONTEND%-frontend}"
        export ARTIFACT_DIR="${BASE_DIR}/artifacts/${MODEL_BASE_NAME}_${FRONTEND_LIB}"
        mkdir -p "${ARTIFACT_DIR}"
        log "Artifact directory prepared at ${ARTIFACT_DIR}"

        # Run Benchmark using local cached tokenizer path
        log "Launching AIPerf benchmark session inside tmux..."
        export AIPERF_CMD="aiperf profile -m ${MODEL_NAME} --tokenizer ${MODEL_PATH} --input-file ${INPUT_FILE} --custom-dataset-type mooncake_trace --fixed-schedule --url http://${FRONTEND}:8000 --streaming --artifact-dir ${ARTIFACT_DIR} --goodput \"time_to_first_token:2000 inter_token_latency:25\" 2>&1 | tee ${ARTIFACT_DIR}/aiperf.log"

        tmux new-session -d -s benchmark -c "${ARTIFACT_DIR}"
        tmux send-keys -t benchmark "${AIPERF_CMD}" C-m
        log "AIPerf session launched in tmux."
        log "Command: ${AIPERF_CMD}"

        # Tail live AIPerf logs to container stdout
        log "Tailing live AIPerf logs to container output..."
        touch "${ARTIFACT_DIR}/aiperf.log"
        tail -f "${ARTIFACT_DIR}/aiperf.log"
    env:
      - name: MODEL_NAME
        value: Qwen/Qwen3-32B-FP8
      - name: MODEL_PATH
        value: /opt/models/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df
      - name: HF_HOME
        value: /opt/models
      - name: FRONTEND
        value: qwen3-32b-fp8-vllm-disagg-routing-frontend
      - name: BASE_DIR
        value: /perf-cache
    resources:
      requests:
        cpu: "8"
        memory: 16Gi
      limits:
        cpu: "16"
        memory: 32Gi
    volumeMounts:
    - name: model-cache
      mountPath: /opt/models
    - name: perf-cache
      mountPath: /perf-cache
    workingDir: /workspace
  volumes:
  - name: model-cache
    persistentVolumeClaim:
      claimName: model-cache
  - name: perf-cache
    persistentVolumeClaim:
      claimName: perf-cache
  restartPolicy: Never
EOF
```

## 2. Deploy and Run Benchmark Pod

**Run on `gpu05`.**

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3-32b/experiments/vllm/02-disagg-routing

kubectl delete pod disagg-router-6p-2d-benchmark -n "$NAMESPACE" --ignore-not-found
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/benchmark.yaml"

kubectl wait --for=condition=Ready pod/disagg-router-6p-2d-benchmark -n "$NAMESPACE" --timeout=5m || true
```

## 3. Monitor and View Results

### Result Storage Paths
- **Inside Container**: `/perf-cache/artifacts/Qwen3-32B-FP8_qwen3-32b-fp8-vllm-disagg/`
- **Persistent Volume**: Backed by the Kubernetes `perf-cache` PVC.

### Attach to Live Session
To attach to the live benchmarking session inside `tmux`:

```bash
kubectl exec -it disagg-router-6p-2d-benchmark -n "$NAMESPACE" -- tmux attach -t benchmark
```

### Tail Startup Logs
To tail the container startup logs:

```bash
kubectl logs -n "$NAMESPACE" disagg-router-6p-2d-benchmark -f
```

### Retrieve Artifacts to Host
To copy the generated benchmark artifacts from the `perf-cache` PVC to the host machine:

```bash
kubectl cp -n "$NAMESPACE" \
  disagg-router-6p-2d-benchmark:/perf-cache/artifacts \
  "$EXP_DIR/artifacts"
```
