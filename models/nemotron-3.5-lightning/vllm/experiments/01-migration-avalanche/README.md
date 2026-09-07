# Experiment 1: migration avalanche

This experiment runs four TP1 prefill workers and eight TP1 decode workers on
12 H100 GPUs. The frontend enables KV-aware routing and permits three
migrations for sequences no longer than 65,536 tokens. A controlled client
load is held just below saturation; after the two-minute warmup, the hottest
decode pod is killed during the measured interval.

The first pass is deliberately 4P+8D. Once the fault mechanism, metric labels,
and recovery accounting are correct, increase `VllmDecodeWorker.replicas` to
12 for the full 16-GPU topology. Keep the normal-InfiniBand settings in this
recipe; do not substitute NVIDIA's EFA resource claims on this cluster.

## 1. Variables and preflight

Run all commands from a cluster-administration host.

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/nemotron-3.5-lightning/vllm/experiments/01-migration-avalanche
export DEPLOYMENT=nemotron35-vllm-e1-migration
export PERF_JOB=nemotron35-vllm-e1-perf
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=$DEPLOYMENT"
export RUNTIME_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1
export CAPABILITY_POD=nemotron35-vllm-capability
mkdir -p "$EXP_DIR"

kubectl get crd dynamographdeployments.nvidia.com
kubectl get pvc model-cache perf-cache -n "$NAMESPACE"
kubectl get secret hf-token-secret nvcrimagepullsecret -n "$NAMESPACE"
kubectl get network-attachment-definition qwen-roce -n "$NAMESPACE"
kubectl get nodes \
  -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,RDMA:.status.allocatable.rdma/ib'
```

Verify the version pin before occupying the GPUs. This check must print vLLM
0.26.0. If it does not, stop and resolve the runtime image rather than silently
changing experiment software.

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

Populate the pinned model cache using the model-level runbook before
continuing.

## 2. Create the deployment manifest

```bash
tee "$EXP_DIR/deploy.yaml" >/dev/null <<'DEPLOY_EOF'
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: nemotron35-vllm-e1-migration
  labels:
    app.kubernetes.io/name: nemotron35-vllm-e1-migration
    app.kubernetes.io/part-of: nemotron35-research
    research.nvidia.com/experiment: migration-avalanche
spec:
  backendFramework: vllm
  components:
    - name: Frontend
      type: frontend
      replicas: 1
      podTemplate:
        spec:
          imagePullSecrets: &image_pull_secrets
            - name: nvcrimagepullsecret
          containers:
            - name: main
              image: &runtime_image nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1
              imagePullPolicy: IfNotPresent
              command: [python3]
              args:
                - -m
                - dynamo.frontend
                - --router-mode
                - kv
                - --router-kv-events
                - --kv-cache-block-size
                - "64"
                - --trust-remote-code
              env:
                - name: DYN_MIGRATION_LIMIT
                  value: "3"
                - name: DYN_MIGRATION_MAX_SEQ_LEN
                  value: "65536"
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
                  port: 8000
                periodSeconds: 10
                timeoutSeconds: 60
                failureThreshold: 90
              volumeMounts:
                - name: model-cache
                  mountPath: /opt/models
                  readOnly: true
          volumes:
            - name: model-cache
              persistentVolumeClaim:
                claimName: model-cache

    - name: VllmPrefillWorker
      type: prefill
      replicas: 4
      sharedMemorySize: 40Gi
      podTemplate:
        metadata:
          annotations:
            k8s.v1.cni.cncf.io/networks: qwen-roce
        spec:
          imagePullSecrets: *image_pull_secrets
          hostNetwork: false
          dnsPolicy: ClusterFirst
          affinity: &h100_affinity
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
          tolerations: &gpu_tolerations
            - key: nvidia.com/gpu
              operator: Equal
              value: "true"
              effect: NoSchedule
          volumes: &worker_volumes
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
                  exec python3 -m dynamo.vllm \
                    --model "$MODEL_PATH" \
                    --served-model-name "$SERVED_MODEL_NAME" \
                    --trust-remote-code \
                    --tensor-parallel-size 1 \
                    --max-model-len 65536 \
                    --max-num-seqs 128 \
                    --max-num-batched-tokens 32768 \
                    --gpu-memory-utilization 0.90 \
                    --async-scheduling \
                    --enable-prefix-caching \
                    --block-size 64 \
                    --mamba-cache-mode align \
                    --no-disable-hybrid-kv-cache-manager \
                    --mamba-backend flashinfer \
                    --mamba-ssm-cache-dtype float16 \
                    --enable-mamba-cache-stochastic-rounding \
                    --mamba-cache-philox-rounds 5 \
                    --dyn-tool-call-parser nemotron_nano \
                    --dyn-reasoning-parser nemotron_nano \
                    --reasoning-parser nemotron_v3 \
                    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
                    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' \
                    --disaggregation-mode prefill
              env: &worker_env
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
                - name: VLLM_NIXL_SIDE_CHANNEL_HOST
                  valueFrom:
                    fieldRef:
                      fieldPath: status.podIP
                - name: VLLM_NIXL_SIDE_CHANNEL_PORT
                  value: "5600"
                - name: VLLM_SSM_CONV_STATE_LAYOUT
                  value: DS
                - name: NIXL_LOG_LEVEL
                  value: INFO
                - name: NIXL_TELEMETRY_ENABLE
                  value: "y"
                - name: NIXL_TELEMETRY_EXPORTER
                  value: prometheus
                - name: NIXL_TELEMETRY_PROMETHEUS_PORT
                  value: "19090"
                - name: UCX_TLS
                  value: rc_x,rc,cuda_copy,cuda_ipc
                - name: UCX_NET_DEVICES
                  value: mlx5_8:1
                - name: UCX_IB_ADDR_TYPE
                  value: eth
                - name: UCX_RCACHE_MAX_UNRELEASED
                  value: "1024"
                - name: NCCL_IB_DISABLE
                  value: "1"
                - name: PYTHONHASHSEED
                  value: "0"
              resources: &worker_resources
                requests:
                  cpu: "16"
                  memory: 96Gi
                  nvidia.com/gpu: "1"
                  rdma/ib: "1"
                limits:
                  cpu: "32"
                  memory: 128Gi
                  nvidia.com/gpu: "1"
                  rdma/ib: "1"
              securityContext: &worker_security
                runAsUser: 0
                capabilities:
                  add: [IPC_LOCK, SYS_RESOURCE]
              startupProbe: &worker_startup_probe
                httpGet:
                  path: /live
                  port: 9090
                periodSeconds: 30
                timeoutSeconds: 20
                failureThreshold: 120
              ports: &worker_ports
                - name: system
                  containerPort: 9090
                - name: nixl-side
                  containerPort: 5600
                - name: nixl-metrics
                  containerPort: 19090
              volumeMounts: &worker_mounts
                - name: model-cache
                  mountPath: /opt/models
                  readOnly: true
                - name: dshm
                  mountPath: /dev/shm

    - name: VllmDecodeWorker
      type: decode
      replicas: 8
      sharedMemorySize: 40Gi
      podTemplate:
        metadata:
          annotations:
            k8s.v1.cni.cncf.io/networks: qwen-roce
        spec:
          imagePullSecrets: *image_pull_secrets
          hostNetwork: false
          dnsPolicy: ClusterFirst
          affinity: *h100_affinity
          tolerations: *gpu_tolerations
          volumes: *worker_volumes
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
                  exec python3 -m dynamo.vllm \
                    --model "$MODEL_PATH" \
                    --served-model-name "$SERVED_MODEL_NAME" \
                    --trust-remote-code \
                    --tensor-parallel-size 1 \
                    --max-model-len 65536 \
                    --max-num-seqs 128 \
                    --max-num-batched-tokens 32768 \
                    --gpu-memory-utilization 0.90 \
                    --async-scheduling \
                    --no-enable-prefix-caching \
                    --block-size 64 \
                    --mamba-cache-mode align \
                    --no-disable-hybrid-kv-cache-manager \
                    --mamba-backend flashinfer \
                    --mamba-ssm-cache-dtype float16 \
                    --enable-mamba-cache-stochastic-rounding \
                    --mamba-cache-philox-rounds 5 \
                    --dyn-tool-call-parser nemotron_nano \
                    --dyn-reasoning-parser nemotron_nano \
                    --reasoning-parser nemotron_v3 \
                    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' \
                    --disaggregation-mode decode
              env: *worker_env
              resources: *worker_resources
              securityContext: *worker_security
              startupProbe: *worker_startup_probe
              ports: *worker_ports
              volumeMounts: *worker_mounts
DEPLOY_EOF

kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl wait -n "$NAMESPACE" --for=jsonpath='{.status.state}'=successful \
  "dynamographdeployment/$DEPLOYMENT" --timeout=60m
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide
```

Before benchmarking, verify that every worker has the secondary network and
RDMA device, and confirm the frontend exposes the migration metric family.

```bash
export WORKER_POD=$(kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  -o name | awk 'tolower($0) ~ /worker/ {sub("pod/", ""); print; exit}')
kubectl get pod "$WORKER_POD" -n "$NAMESPACE" -o json | \
  jq '{node:.spec.nodeName,networkStatus:.metadata.annotations["k8s.v1.cni.cncf.io/network-status"]}'
kubectl exec -n "$NAMESPACE" "$WORKER_POD" -- \
  sh -lc 'test -d /dev/infiniband && ls /dev/infiniband && printenv UCX_TLS UCX_NET_DEVICES'
kubectl get --raw "/api/v1/namespaces/${NAMESPACE}/services/${DEPLOYMENT}-frontend:8000/proxy/metrics" | \
  rg 'dynamo_frontend_model_migration|dynamo_frontend_worker_active_decode_blocks'
```

## 3. Create the benchmark Job

The checked-in `perf.yaml` is the source of truth for the Job. Create the
runtime copy as a quoted heredoc before applying it. Set the workload and fault
values in the `env` section to one row from `matrix.yaml`; do not change model,
block-size, timing, or random-seed controls between comparisons.

```bash
tee "$EXP_DIR/perf.yaml" >/dev/null <<'PERF_EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: nemotron35-vllm-e1-perf
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 1800
  template:
    metadata:
      labels:
        app: nemotron35-vllm-e1-perf
        research.nvidia.com/experiment: migration-avalanche
    spec:
      restartPolicy: Never
      containers:
        - name: perf
          image: python:3.12-slim
          imagePullPolicy: IfNotPresent
          command: [/bin/bash, -lc]
          args:
            - |
              set -Eeuo pipefail
              apt-get update
              apt-get install -y --no-install-recommends ca-certificates curl jq
              rm -rf /var/lib/apt/lists/*
              python -m pip install --no-cache-dir "aiperf==$AIPERF_VERSION"
              ready_deadline=$(( $(date +%s) + MODEL_READY_TIMEOUT_SECONDS ))
              until curl -fsS --max-time 10 "http://$ENDPOINT/v1/models" | jq -e --arg model "$MODEL_NAME" '.data[]? | select(.id == $model)' >/dev/null; do
                if [ "$(date +%s)" -ge "$ready_deadline" ]; then echo "Model readiness timed out" >&2; exit 1; fi
                sleep 5
              done
              run_id=$(date -u +%Y-%m-%dT%H-%M-%SZ)
              run_dir="$ARTIFACT_ROOT/$WORKLOAD_ID/c${CONCURRENCY}/$FAULT_CASE/$run_id"
              artifact_dir="$run_dir/aiperf"
              trace_file="$run_dir/trace.jsonl"
              mkdir -p "$artifact_dir"
              python3 - "$trace_file" <<'PY'
              import json, math, os, sys
              path = sys.argv[1]
              block_size = int(os.environ["TRACE_BLOCK_SIZE"])
              input_tokens = int(os.environ["INPUT_TOKENS"])
              output_tokens = int(os.environ["OUTPUT_TOKENS"])
              reuse = int(os.environ["PREFIX_REUSE_PERCENT"])
              request_count = int(os.environ["REQUEST_COUNT"])
              block_count = math.ceil(input_tokens / block_size)
              shared_count = math.floor(block_count * reuse / 100)
              shared = list(range(shared_count))
              with open(path, "w", encoding="utf-8") as handle:
                  for request_id in range(request_count):
                      unique_count = block_count - shared_count
                      unique_base = 1_000_000 + request_id * max(unique_count, 1)
                      hashes = shared + list(range(unique_base, unique_base + unique_count))
                      row = {"session_id": f"e1-{request_id:08d}", "timestamp": request_id, "input_length": input_tokens, "output_length": output_tokens, "hash_ids": hashes}
                      handle.write(json.dumps(row, separators=(",", ":")) + "\n")
              PY
              sha256sum "$trace_file" | awk '{print $1}' > "$run_dir/trace.sha256"
              jq -n --arg run_id "$run_id" --arg workload "$WORKLOAD_ID" --arg fault_case "$FAULT_CASE" --arg model "$MODEL_NAME" --arg model_revision "$MODEL_REVISION" --arg aiperf_version "$AIPERF_VERSION" --argjson input_tokens "$INPUT_TOKENS" --argjson output_tokens "$OUTPUT_TOKENS" --argjson prefix_reuse_percent "$PREFIX_REUSE_PERCENT" --argjson concurrency "$CONCURRENCY" --argjson warmup_seconds "$WARMUP_SECONDS" --argjson measured_seconds "$MEASURED_SECONDS" --argjson fault_at_seconds "$FAULT_AT_SECONDS" '{run_id:$run_id,experiment:"01-migration-avalanche",workload:$workload,fault_case:$fault_case,model:$model,model_revision:$model_revision,aiperf_version:$aiperf_version,input_tokens:$input_tokens,output_tokens:$output_tokens,prefix_reuse_percent:$prefix_reuse_percent,concurrency:$concurrency,warmup_seconds:$warmup_seconds,measured_seconds:$measured_seconds,fault_at_measured_seconds:$fault_at_seconds}' > "$run_dir/manifest.json"
              echo "Run directory: $run_dir"
              echo "Inject $FAULT_CASE at measured T+$FAULT_AT_SECONDS seconds."
              aiperf profile --model "$MODEL_NAME" --tokenizer "$TOKENIZER_PATH" --input-file "$trace_file" --custom-dataset-type mooncake_trace --no-fixed-schedule --url "http://$ENDPOINT" --endpoint-type chat --streaming --use-server-token-count --extra-inputs ignore_eos:true --extra-inputs temperature:0.0 --extra-inputs min_tokens:"$OUTPUT_TOKENS" --concurrency "$CONCURRENCY" --request-count "$REQUEST_COUNT" --warmup-duration "$WARMUP_SECONDS" --benchmark-duration "$MEASURED_SECONDS" --random-seed 100 --workers-max 512 --record-processors 32 --artifact-dir "$artifact_dir" --server-metrics-formats json jsonl csv --no-gpu-telemetry --ui simple 2>&1 | tr '\r' '\n' | tee "$run_dir/aiperf.log"
          env:
            - {name: MODEL_NAME, value: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}
            - {name: MODEL_REVISION, value: cc84af2fe71647d87f4486c064f320e1e7535243}
            - {name: TOKENIZER_PATH, value: /opt/models/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/snapshots/cc84af2fe71647d87f4486c064f320e1e7535243}
            - {name: ENDPOINT, value: 'nemotron35-vllm-e1-migration-frontend:8000'}
            - {name: AIPERF_VERSION, value: '0.12.0'}
            - {name: MODEL_READY_TIMEOUT_SECONDS, value: '3600'}
            - {name: WORKLOAD_ID, value: B}
            - {name: INPUT_TOKENS, value: '16384'}
            - {name: OUTPUT_TOKENS, value: '2048'}
            - {name: PREFIX_REUSE_PERCENT, value: '80'}
            - {name: TRACE_BLOCK_SIZE, value: '512'}
            - {name: CONCURRENCY, value: '128'}
            - {name: REQUEST_COUNT, value: '32768'}
            - {name: WARMUP_SECONDS, value: '120'}
            - {name: MEASURED_SECONDS, value: '300'}
            - {name: FAULT_AT_SECONDS, value: '120'}
            - {name: FAULT_CASE, value: F1}
            - {name: ARTIFACT_ROOT, value: /perf-cache/nemotron-3.5-lightning/01-migration-avalanche}
            - {name: HF_HOME, value: /opt/models}
            - {name: HF_HUB_OFFLINE, value: '1'}
            - {name: TRANSFORMERS_OFFLINE, value: '1'}
            - {name: AIPERF_HTTP_CONNECTION_LIMIT, value: '1024'}
            - {name: PYTHONUNBUFFERED, value: '1'}
          resources:
            requests: {cpu: '16', memory: 32Gi}
            limits: {cpu: '32', memory: 64Gi}
          volumeMounts:
            - {name: model-cache, mountPath: /opt/models, readOnly: true}
            - {name: perf-cache, mountPath: /perf-cache}
      volumes:
        - name: model-cache
          persistentVolumeClaim: {claimName: model-cache}
        - name: perf-cache
          persistentVolumeClaim: {claimName: perf-cache}
PERF_EOF
```

The compact mappings in this runtime heredoc are semantically identical to
the expanded checked manifest. Validate them before each run.

## 4. Create the fault helper and run

```bash
tee "$EXP_DIR/fault-inject.sh" >/dev/null <<'FAULT_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
: "${NAMESPACE:?set NAMESPACE}"
: "${DEPLOYMENT:?set DEPLOYMENT}"
: "${EXP_DIR:?set EXP_DIR}"
GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
FAULT_LOG="${EXP_DIR}/faults.jsonl"
mkdir -p "$EXP_DIR"
mapfile -t decode_pods < <(kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o name | sed 's#pod/##' | awk 'tolower($0) ~ /decode/')
if [ "${#decode_pods[@]}" -eq 0 ]; then echo "No decode pods found for $GRAPH_LABEL" >&2; exit 1; fi
hottest_pod=; hottest_score=-1; hottest_running=0; hottest_kv=0
for pod in "${decode_pods[@]}"; do
  metrics=$(kubectl get --raw "/api/v1/namespaces/${NAMESPACE}/pods/${pod}:9090/proxy/metrics" 2>/dev/null || true)
  if ! grep -Eq '^vllm:(num_requests_running|kv_cache_usage_perc|gpu_cache_usage_perc)' <<<"$metrics"; then echo "$pod did not expose the required vLLM load metrics; skipping" >&2; continue; fi
  running=$(awk '/^vllm:num_requests_running([{ ]|$)/ {sum += $NF} END {printf "%.0f", sum + 0}' <<<"$metrics")
  kv=$(awk '/^vllm:(kv_cache_usage_perc|gpu_cache_usage_perc)([{ ]|$)/ {sum += $NF; n += 1} END {if (n) printf "%.9f", sum / n; else print "0"}' <<<"$metrics")
  score=$(awk -v running="$running" -v kv="$kv" 'BEGIN {printf "%.9f", (running * 1000000) + kv}')
  printf '%s running=%s kv_usage=%s score=%s\n' "$pod" "$running" "$kv" "$score"
  if awk -v score="$score" -v hottest="$hottest_score" 'BEGIN {exit !(score > hottest)}'; then hottest_pod=$pod; hottest_score=$score; hottest_running=$running; hottest_kv=$kv; fi
done
if [ -z "$hottest_pod" ]; then echo "Unable to select a decode pod with observable load metrics" >&2; exit 1; fi
fault_ts=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
pod_uid=$(kubectl get pod "$hottest_pod" -n "$NAMESPACE" -o jsonpath='{.metadata.uid}')
printf '{"timestamp":"%s","fault_type":"forced_pod_delete","pod":"%s","pod_uid":"%s","running_requests":%s,"kv_usage":%s}\n' "$fault_ts" "$hottest_pod" "$pod_uid" "$hottest_running" "$hottest_kv" | tee -a "$FAULT_LOG"
kubectl delete pod "$hottest_pod" -n "$NAMESPACE" --grace-period=0 --force
FAULT_EOF
chmod +x "$EXP_DIR/fault-inject.sh"

kubectl delete job "$PERF_JOB" -n "$NAMESPACE" --ignore-not-found
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/perf.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/perf.yaml"
kubectl logs -n "$NAMESPACE" -f "job/$PERF_JOB"
```

In a second terminal, start the timer when the Job prints `Run directory`.
The AIPerf warmup is 120 seconds and the default fault point is measured
T+120, so the default F1 injection occurs 240 seconds after that marker.

```bash
sleep 240
"$EXP_DIR/fault-inject.sh"
```

Run F0 without executing the fault helper. For F2 through F4, invoke the helper
the required number of times and retain every line in `faults.jsonl`. Export
Prometheus and DCGM time series over the same UTC interval as the AIPerf
manifest. Recovery time is the interval until p99 ITL remains within 10% of
the F0 baseline for 30 seconds; RAF and collateral damage use the definitions
in `matrix.yaml`.

## 5. Cleanup

```bash
kubectl delete job "$PERF_JOB" -n "$NAMESPACE" --ignore-not-found
kubectl delete dynamographdeployment "$DEPLOYMENT" -n "$NAMESPACE" --ignore-not-found
```
