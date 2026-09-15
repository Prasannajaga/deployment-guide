# GLM-5.2-FP8 aggregated vLLM recipe

This recipe serves the pinned public checkpoint `zai-org/GLM-5.2-FP8` with
NVIDIA Dynamo 1.3.0 and vLLM 0.23.0 on two 8 x H100 80 GB nodes.

```text
1 model replica x 2 nodes x 8 GPUs per node = 16 GPUs
Tensor parallel size = 16
Pipeline parallel size = 1
Data parallel size = 1
```

`replicas: 1` means one distributed model replica. Because that replica has
`multinode.nodeCount: 2`, Kubernetes creates two worker Pods: one leader and
one follower. Each worker Pod requests eight GPUs.

The checkpoint is approximately 756 GB. TP=8 would require roughly 94.5 GB of
weights per GPU and cannot fit on an 80 GB H100. TP=16 reduces the weight share
to roughly 47.25 GB per GPU before runtime and KV-cache allocations. The
baseline uses a 131,072-token context, FP8 KV cache, 0.80 GPU memory utilization,
and 16 concurrent sequences.

Expert parallelism, pipeline parallelism, MTP speculative decoding, and
prefill/decode disaggregation are intentionally disabled until this baseline
passes. `VLLM_DEEP_GEMM_WARMUP=skip` avoids a long startup precompile; kernels
may instead compile during the first few requests.

## Files

```text
glm-5.2-fp8/
├── README.md
├── model-cache/
│   └── model-download.yaml
└── vllm/agg/
    └── deploy.yaml
```

## 1. Variables

Run these commands from a Kubernetes administrator host:

```bash
export NAMESPACE=dynamo-bench
export RECIPE_ROOT=/ephemeral/shared/glm-5.2-fp8
export MODEL_CACHE_DIR="${RECIPE_ROOT}/model-cache"
export EXP_DIR="${RECIPE_ROOT}/vllm/agg"
export MODEL_DOWNLOAD_JOB=glm52-fp8-download
export DEPLOYMENT=glm52-fp8-vllm-agg-tp16
export OLD_DEPLOYMENT=glm52-fp8-sglang-agg-tp16
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
export FRONTEND_SERVICE="${DEPLOYMENT}-frontend"
export MODEL=zai-org/GLM-5.2-FP8
export MODEL_REVISION=ba978f7d347eaf65d22f1a86833408afdb953541
export LOCAL_PORT=8000
```

## 2. Remove the previous SGLang graph

The old graph holds all 16 GPUs, so remove it before applying the vLLM graph.
This does not delete the model cache PVC.

```bash
kubectl delete dynamographdeployment.nvidia.com "$OLD_DEPLOYMENT" \
  -n "$NAMESPACE" --ignore-not-found

kubectl wait -n "$NAMESPACE" \
  --for=delete pod \
  -l "nvidia.com/dynamo-graph-deployment-name=${OLD_DEPLOYMENT}" \
  --timeout=600s
```

Confirm that the old worker Pods are gone before continuing:

```bash
kubectl get pods -n "$NAMESPACE" \
  -l "nvidia.com/dynamo-graph-deployment-name=${OLD_DEPLOYMENT}"
```

## 3. Preflight

The deployment requires Dynamo multinode orchestration, the shared
`model-cache` PVC, two nodes with eight available GPUs each, the `roce`
network attachment, and eight `rdma/ib` resources per node. Dynamo 1.3.0 also
requires an NVIDIA driver compatible with CUDA 13.

```bash
kubectl get crd \
  dynamographdeployments.nvidia.com \
  podcliquesets.grove.io \
  network-attachment-definitions.k8s.cni.cncf.io

kubectl get pvc model-cache -n "$NAMESPACE"
kubectl get network-attachment-definition roce -n "$NAMESPACE"
kubectl get nodes \
  -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,RDMA:.status.allocatable.rdma/ib'
kubectl get pods -A \
  -o custom-columns='NAMESPACE:.metadata.namespace,POD:.metadata.name,NODE:.spec.nodeName,GPU:.spec.containers[*].resources.requests.nvidia\.com/gpu' \
  | grep -v '<none>' || true
```

## 4. Reuse or populate the model cache

If the `$MODEL_DOWNLOAD_JOB` Job already completed, keep the existing PVC contents
and skip directly to the deployment manifest.

```bash
kubectl get job "$MODEL_DOWNLOAD_JOB" -n "$NAMESPACE"
kubectl logs -n "$NAMESPACE" "job/$MODEL_DOWNLOAD_JOB" --tail=50
```

Otherwise, create the pinned download Job:

```bash
mkdir -p "$MODEL_CACHE_DIR"
tee "$MODEL_CACHE_DIR/model-download.yaml" >/dev/null <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${MODEL_DOWNLOAD_JOB}
spec:
  backoffLimit: 2
  activeDeadlineSeconds: 43200
  template:
    metadata:
      labels:
        app: glm52-fp8-download
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
              hf download "\$MODEL_NAME" --revision "\$MODEL_REVISION"

              SNAPSHOT_DIR="\$HF_HOME/hub/models--zai-org--GLM-5.2-FP8/snapshots/\$MODEL_REVISION"
              test -s "\$SNAPSHOT_DIR/config.json"
              test -s "\$SNAPSHOT_DIR/model.safetensors.index.json"
              test -s "\$SNAPSHOT_DIR/model-00141-of-00141.safetensors"
              du -sh "\$SNAPSHOT_DIR"
          env:
            - name: MODEL_NAME
              value: zai-org/GLM-5.2-FP8
            - name: MODEL_REVISION
              value: ba978f7d347eaf65d22f1a86833408afdb953541
            - name: HF_HOME
              value: /model-store
            - name: HF_XET_HIGH_PERFORMANCE
              value: "1"
            - name: HF_HUB_DOWNLOAD_TIMEOUT
              value: "120"
          envFrom:
            - secretRef:
                name: hf-token-secret
                optional: true
          resources:
            requests:
              cpu: "4"
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

kubectl delete job "$MODEL_DOWNLOAD_JOB" -n "$NAMESPACE" --ignore-not-found
kubectl apply --dry-run=server -n "$NAMESPACE" \
  -f "$MODEL_CACHE_DIR/model-download.yaml"
kubectl apply -n "$NAMESPACE" -f "$MODEL_CACHE_DIR/model-download.yaml"
kubectl wait -n "$NAMESPACE" \
  --for=condition=Complete "job/$MODEL_DOWNLOAD_JOB" \
  --timeout=43200s
kubectl logs -n "$NAMESPACE" "job/$MODEL_DOWNLOAD_JOB" --tail=100
```

## 5. Create the vLLM deployment manifest

Dynamo injects the vLLM multinode flags, including `--nnodes`, `--node-rank`,
`--master-addr`, and `--distributed-executor-backend mp`. Do not add those
flags manually.

```bash
mkdir -p "$EXP_DIR"
tee "$EXP_DIR/deploy.yaml" >/dev/null <<'EOF'
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: glm52-fp8-vllm-agg-tp16
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
      envs:
        - name: HF_HOME
          value: /opt/models
      replicas: 4
      resources:
        requests:
          cpu: "8"
          memory: 64Gi
        limits:
          cpu: "16"
          memory: 128Gi
    VllmWorker:
      componentType: worker
      multinode:
        nodeCount: 2
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 128Gi
      extraPodMetadata:
        annotations:
          k8s.v1.cni.cncf.io/networks: roce
      extraPodSpec:
        hostNetwork: false
        dnsPolicy: ClusterFirst
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          workingDir: /workspace/examples/backends/vllm
          command:
            - /bin/sh
            - -c
          args:
            - |
              set -eu
              ulimit -l unlimited
              exec python3 -m dynamo.vllm \
                --model "$MODEL_PATH" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --tensor-parallel-size 16 \
                --pipeline-parallel-size 1 \
                --data-parallel-size 1 \
                --kv-cache-dtype fp8 \
                --gpu-memory-utilization 0.80 \
                --max-model-len 131072 \
                --max-num-seqs 16 \
                --max-num-batched-tokens 8192 \
                --enable-chunked-prefill \
                --no-enable-prefix-caching \
                --reasoning-parser glm45 \
                --dyn-reasoning-parser glm45 \
                --dyn-tool-call-parser glm47
          env:
            - name: SERVED_MODEL_NAME
              value: zai-org/GLM-5.2-FP8
            - name: MODEL_PATH
              value: /opt/models/hub/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541
            - name: HF_HOME
              value: /opt/models
            - name: PYTHONHASHSEED
              value: "0"
            - name: GLOO_SOCKET_IFNAME
              value: eth0
            - name: NCCL_SOCKET_IFNAME
              value: eth0
            - name: NCCL_IB_DISABLE
              value: "0"
            - name: NCCL_DEBUG
              value: INFO
            - name: VLLM_DEEP_GEMM_WARMUP
              value: skip
          securityContext:
            runAsUser: 0
            capabilities:
              add:
                - IPC_LOCK
                - SYS_RESOURCE
      replicas: 1
      resources:
        requests:
          gpu: "8"
          cpu: "32"
          memory: 256Gi
          custom:
            rdma/ib: "8"
        limits:
          gpu: "8"
          cpu: "64"
          memory: 512Gi
          custom:
            rdma/ib: "8"
EOF
```

The RDMA Shared Device Plugin limits the available RDMA device set to the
`rdma7` rail, so NCCL can discover the selected HCA without a per-manifest pin.
No GID index is hardcoded because the correct GID is determined by the Pod's
secondary network attachment.

## 6. Validate and deploy

```bash
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"

kubectl get pods -n "$NAMESPACE" \
  -l "$GRAPH_LABEL" \
  -L nvidia.com/dynamo-component-type \
  -o wide -w
```

Expected topology: one frontend Pod and two vLLM worker Pods. The leader may
remain `0/1` while both Pods load weights and initialize all 16 ranks.

```bash
kubectl wait -n "$NAMESPACE" \
  --for=condition=Ready pod \
  -l "$GRAPH_LABEL" \
  --timeout=7200s

kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  -o custom-columns='POD:.metadata.name,NODE:.spec.nodeName,READY:.status.containerStatuses[*].ready,RESTARTS:.status.containerStatuses[*].restartCount,GPU:.spec.containers[*].resources.requests.nvidia\.com/gpu,RDMA:.spec.containers[*].resources.requests.rdma/ib'
```

Inspect leader and follower logs independently if readiness stalls:

```bash
export LEADER="$(kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  -o name | grep -- '-ldr-' | cut -d/ -f2)"
export FOLLOWER="$(kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  -o name | grep -- '-wkr-' | cut -d/ -f2)"

kubectl logs -n "$NAMESPACE" "$LEADER" -c main \
  --timestamps --tail=300
kubectl logs -n "$NAMESPACE" "$FOLLOWER" -c main \
  --timestamps --tail=300
```

## 7. Smoke test

Forward the frontend in one terminal:

```bash
kubectl port-forward -n "$NAMESPACE" \
  service/"$FRONTEND_SERVICE" "$LOCAL_PORT":8000
```

In another terminal:

```bash
curl -fsS --max-time 30 \
  "http://127.0.0.1:${LOCAL_PORT}/v1/models"

curl -fsS --max-time 600 \
  "http://127.0.0.1:${LOCAL_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON
{
  "model": "$MODEL",
  "messages": [
    {"role": "user", "content": "Reply with exactly: ready"}
  ],
  "chat_template_kwargs": {"enable_thinking": false},
  "temperature": 0,
  "max_tokens": 16,
  "stream": false
}
JSON
```

Pass when the model is listed, the completion succeeds without a 4xx/5xx
response, both worker Pods are ready, and all restart counts remain zero. The
first request can be slower because DeepGEMM warm-up was skipped at startup.

## 8. Cleanup

Stop the port-forward with `Ctrl-C`, then remove only this graph:

```bash
kubectl delete dynamographdeployment.nvidia.com "$DEPLOYMENT" \
  -n "$NAMESPACE" --ignore-not-found

kubectl wait -n "$NAMESPACE" \
  --for=delete pod \
  -l "$GRAPH_LABEL" \
  --timeout=600s
```

Remove the completed download Job after inspecting its logs:

```bash
kubectl delete job "$MODEL_DOWNLOAD_JOB" \
  -n "$NAMESPACE" --ignore-not-found
```

Cleanup intentionally preserves the shared `model-cache` PVC and the pinned
model snapshot.

## References

- [GLM-5.2-FP8 checkpoint](https://huggingface.co/zai-org/GLM-5.2-FP8)
- [Official vLLM GLM-5.2 recipe](https://github.com/vllm-project/recipes/blob/main/models/zai-org/GLM-5.2.yaml)
- [Dynamo 1.3.0 runtime artifacts](https://docs.nvidia.com/dynamo/latest/resources/release-artifacts)
- [Dynamo multinode deployments](https://docs.nvidia.com/dynamo/dev/knowledge-base/kubernetes/multinode/multinode-deployments)


## 9. Performance benchmark

The benchmark supports fixed and mixed sequence distributions. It mounts the
shared model cache for the pinned tokenizer and writes results to the
`perf-cache` PVC.

```bash
tee "$EXP_DIR/perf.yaml" >/dev/null <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: glm52-fp8-vllm-agg-tp16-perf
spec:
  backoffLimit: 0
  completions: 1
  parallelism: 1
  activeDeadlineSeconds: 14400
  template:
    metadata:
      labels:
        app: glm52-fp8-vllm-agg-tp16-perf
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
                  aiperf:"0.10.0", dynamo:"1.3.0", backend:"vllm"}' \
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
              value: zai-org/GLM-5.2-FP8
            - name: TOKENIZER
              value: /opt/models/hub/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541
            - name: ENDPOINT
              value: glm52-fp8-vllm-agg-tp16-frontend.dynamo-bench.svc.cluster.local:8000
            - name: ISL
              value: "8000"
            - name: OSL
              value: "1024"
            - name: WORKLOAD_MODE
              value: fixed
            - name: SEQUENCE_DISTRIBUTION
              value: "1024,256:20;4096,512:10;8192,1024:20;16384,1024:20;32768,1024:15;65536,1024:5;98304,1024:5;126976,1024:5"
            - name: CONCURRENCIES
              value: "1 2 4 8 16 32 64 128"
            - name: BENCHMARK_DURATION
              value: "180"
            - name: WARMUP_REQUESTS
              value: "32"
            - name: RANDOM_SEED
              value: "42"
            - name: ARTIFACT_ROOT
              value: /perf-cache/aiperf/glm52-fp8-vllm-agg-tp16
            - name: TOPOLOGY_NAME
              value: agg-tp16
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

### Run the requested mixed workload

Resolve the generated frontend Service instead of hardcoding a deployment
instance suffix. The fully qualified DNS name also works when the benchmark
Pod and frontend are in different namespaces.

```bash
export FRONTEND_SERVICE="$(
  kubectl get services -n "$NAMESPACE" -l "$GRAPH_LABEL" -o json |
    jq -r '.items[] | select(any(.spec.ports[]?; .port == 8000)) | .metadata.name' |
    head -n 1
)"

test -n "$FRONTEND_SERVICE"
export FRONTEND_ENDPOINT="${FRONTEND_SERVICE}.${NAMESPACE}.svc.cluster.local:8000"
export PERF_JOB_NAME="$(
  kubectl create --dry-run=client --validate=false \
    -f "$EXP_DIR/perf.yaml" \
    -o jsonpath='{.metadata.name}'
)"

test -n "$PERF_JOB_NAME"
kubectl delete job -n "$NAMESPACE" "$PERF_JOB_NAME" \
  --ignore-not-found --wait=true
kubectl set env --local -f "$EXP_DIR/perf.yaml" -o yaml \
  ENDPOINT="$FRONTEND_ENDPOINT" \
  WORKLOAD_MODE=mixed \
  WORKLOAD_NAME=baseline-glm-5.2 \
  SEQUENCE_DISTRIBUTION='1024,256:20;4096,512:10;8192,1024:20;16384,1024:20;32768,1024:15;65536,1024:5;98304,1024:5;126976,1024:5' \
  PREFIX_MODE=isolated \
  CONCURRENCIES='32 64 128 512' \
  BENCHMARK_DURATION=60 \
  WARMUP_REQUESTS=16 \
  ARTIFACT_ROOT="/perf-cache/aiperf/glm52-fp8-vllm-agg-tp16-final" |
  kubectl create -n "$NAMESPACE" -f -

kubectl logs -n "$NAMESPACE" -f "job/$PERF_JOB_NAME"
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  "job/$PERF_JOB_NAME" --timeout=14400s
```
