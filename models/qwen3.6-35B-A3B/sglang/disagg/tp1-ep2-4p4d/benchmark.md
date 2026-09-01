# Benchmark

Run this benchmark only after the deployment passes the startup, real-request,
and NIXL/UCX transfer acceptance gates in [README.md](README.md).

Set the cluster-host variables used by every command in this runbook:

```bash
export NAMESPACE=dynamo-bench
export EXP_DIR=/ephemeral/shared/qwen3.6-35b-a3b/sglang/disagg/tp1-ep2-4p4d
export DEPLOYMENT=q36-sgl-pd-tp1ep2-4p4d
export PERF_JOB_NAME=qwen36-sglang-tp1ep2-4p4d-perf
export PLOT_JOB_NAME=qwen36-sglang-kv-ab-plot
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
```

The namespace must already contain the running deployment plus the
`model-cache` and `perf-cache` PVCs described by the parent runbook. Benchmark
artifacts are written to the `perf-cache` PVC.

## Comparison goal and controls

This runbook compares two variants of the same topology:

- `baseline-kv-aware`: `deploy.yaml`, with KV-aware routing and no CPU KV
  offload.
- `treatment-kv-aware-cpu-offload`: `deploy-kv-offloading.yaml`, with the same
  routing and prefill-only CPU HiCache offload.

Use one UTC run ID for both variants. Keep the model, topology, workload
values, random seed, workload order, and AIPerf version unchanged:

```bash
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export RESULT_ROOT="/perf-cache/aiperf/qwen36-sglang-kv-ab/${RUN_ID}"
```

Run only one variant at a time because the manifests use the same
`DynamoGraphDeployment` name. Before each variant, select and apply its
manifest, wait for all Pods, and repeat the acceptance gates in
[README.md](README.md):

```bash
# Baseline pass.
export VARIANT=baseline-kv-aware
export DEPLOY_FILE="$EXP_DIR/deploy.yaml"

# Treatment pass. Use these two lines instead for the second pass.
# export VARIANT=treatment-kv-aware-cpu-offload
# export DEPLOY_FILE="$EXP_DIR/deploy-kv-offloading.yaml"

kubectl apply --dry-run=server -n "$NAMESPACE" -f "$DEPLOY_FILE"
kubectl apply -n "$NAMESPACE" -f "$DEPLOY_FILE"
```

Confirm that the old worker Pods have been replaced before starting the next
variant; otherwise GPU and CPU cache state from the previous variant can bias
the comparison. Run at least three complete A/B repetitions, alternate which
variant runs first, and use a new `RUN_ID` for every repetition.

## Create perf.yaml

The benchmark supports two sequence-length modes:

- `WORKLOAD_MODE=fixed` uses only `ISL` and `OSL`, with zero variance.
- `WORKLOAD_MODE=mixed` ignores `ISL` and `OSL` and passes
  `SEQUENCE_DISTRIBUTION` to AIPerf's native
  `--sequence-distribution` option.

Mixed mode requires `PREFIX_MODE=isolated`. The Job validates the distribution
format, requires positive lengths and weights totaling 100, and records the
selected mode and distribution in `input-config.json`. It uses SGLang's
server-reported usage counts for output-length metrics. Fixed mode additionally
sets `min_tokens` and `max_tokens` to the requested `OSL`; all modes set
`ignore_eos=true`.

```bash
tee "$EXP_DIR/perf.yaml" >/dev/null <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: qwen36-sglang-tp1ep2-4p4d-perf
spec:
  backoffLimit: 0
  completions: 1
  parallelism: 1
  activeDeadlineSeconds: 14400
  template:
    metadata:
      labels:
        app: qwen36-sglang-tp1ep2-4p4d-perf
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
              warmup_args=()

              if [ "$WARMUP_REQUESTS" -gt 0 ]; then
                warmup_args=(--warmup-request-count "$WARMUP_REQUESTS")
              elif [ "$WARMUP_REQUESTS" -lt 0 ]; then
                echo "WARMUP_REQUESTS must be zero or positive" >&2
                exit 2
              fi

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
                --arg experiment_variant "$EXPERIMENT_VARIANT" \
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
                  experiment_variant:$experiment_variant,
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
                  --use-server-token-count \
                  --concurrency "$concurrency" \
                  --benchmark-duration "$BENCHMARK_DURATION" \
                  --benchmark-grace-period "$GRACE_PERIOD" \
                  "${warmup_args[@]}" \
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
              value: q36-sgl-pd-tp1ep2-4p4d-frontend:8000
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
              value: tp1-ep2-4p4d
            - name: WORKLOAD_NAME
              value: balanced
            - name: EXPERIMENT_VARIANT
              value: unset
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

## Capture KV-transfer and CPU-offload metrics

Capture the worker metric endpoints immediately before and after every preset.
The NIXL counters are on port 19090 and the SGLang CPU HiCache gauges are on
port 9090:

```bash
export METRIC_SNAPSHOT_DIR="$EXP_DIR/metrics/kv-ab/$RUN_ID"
mkdir -p "$METRIC_SNAPSHOT_DIR"

capture_worker_metrics() {
  phase="$1"
  output="$METRIC_SNAPSHOT_DIR/${VARIANT}-${phase}.prom"

  kubectl get pods -n "$NAMESPACE" \
    -l "$GRAPH_LABEL,nvidia.com/dynamo-component-type=worker" \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' |
  while IFS= read -r pod; do
    for port in 9090 19090; do
      echo "# pod=$pod port=$port"
      kubectl exec -n "$NAMESPACE" "$pod" -- \
        python3 -c '
import sys
import urllib.request

metrics = urllib.request.urlopen(
    f"http://127.0.0.1:{sys.argv[1]}/metrics", timeout=10
).read().decode()

prefixes = (
    "agent_tx_",
    "agent_rx_",
    "agent_xfer_",
    "agent_errors_",
    "sglang:hicache_host_",
)
print("\n".join(
    line for line in metrics.splitlines()
    if line and not line.startswith("#") and line.startswith(prefixes)
))
' "$port"
    done
  done | tee "$output"
}
```

For each before/after pair, require an increase in
`agent_tx_bytes_total` and `agent_tx_requests_num_total` on at least one worker
and no increase in `agent_errors_total`. The CPU-offload variant must also show
positive `sglang:hicache_host_total_tokens` and
`sglang:hicache_host_used_tokens` on at least one prefill rank. Those HiCache
series should be absent from the baseline.

## Preset 1: prefill-heavy shared-prefix

This is the primary CPU-offload comparison. It uses a 32K total input, 256
output tokens, 64 shared-prefix groups, and 75% prefix reuse. The large prefix
pool is intended to exceed the useful GPU-resident cache working set over time,
while the long reused prefixes create heavy prefill work and keep decode work
small.

```bash
export PRESET=prefill-heavy
export WORKLOAD_NAME="${VARIANT}-${PRESET}"
capture_worker_metrics "${PRESET}-before"

kubectl delete job -n "$NAMESPACE" "$PERF_JOB_NAME" --ignore-not-found
kubectl set env --local -f "$EXP_DIR/perf.yaml" -o yaml \
  EXPERIMENT_VARIANT="$VARIANT" \
  WORKLOAD_MODE=fixed WORKLOAD_NAME="$WORKLOAD_NAME" \
  ISL=32768 OSL=256 PREFIX_MODE=shared \
  PREFIX_GROUPS=64 PREFIX_REUSE_PERCENT=75 \
  CONCURRENCIES='1 4 8 16 32' BENCHMARK_DURATION=300 \
  WARMUP_REQUESTS=16 RANDOM_SEED=42 NUM_DATASET_ENTRIES=4096 \
  ARTIFACT_ROOT="${RESULT_ROOT}/${VARIANT}" |
  kubectl apply -n "$NAMESPACE" -f -
kubectl logs -n "$NAMESPACE" -f "job/$PERF_JOB_NAME"
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  "job/$PERF_JOB_NAME" --timeout=14400s

capture_worker_metrics "${PRESET}-after"
```

## Preset 2: decode-heavy control

This 1K-input/4K-output preset makes generation dominant. It is a control for
the cost of enabling CPU offload: HiCache should provide little benefit when
prompts are short and isolated, so compare TPOT and output-token throughput.

```bash
export PRESET=decode-heavy
export WORKLOAD_NAME="${VARIANT}-${PRESET}"
capture_worker_metrics "${PRESET}-before"

kubectl delete job -n "$NAMESPACE" "$PERF_JOB_NAME" --ignore-not-found
kubectl set env --local -f "$EXP_DIR/perf.yaml" -o yaml \
  EXPERIMENT_VARIANT="$VARIANT" \
  WORKLOAD_MODE=fixed WORKLOAD_NAME="$WORKLOAD_NAME" \
  ISL=1024 OSL=4096 PREFIX_MODE=isolated \
  CONCURRENCIES='1 4 8 16 32' BENCHMARK_DURATION=300 \
  WARMUP_REQUESTS=16 RANDOM_SEED=42 NUM_DATASET_ENTRIES=4096 \
  ARTIFACT_ROOT="${RESULT_ROOT}/${VARIANT}" |
  kubectl apply -n "$NAMESPACE" -f -
kubectl logs -n "$NAMESPACE" -f "job/$PERF_JOB_NAME"
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  "job/$PERF_JOB_NAME" --timeout=14400s

capture_worker_metrics "${PRESET}-after"
```

## Preset 3: mixed production shape

This isolated-prefix distribution combines decode-heavy and prefill-heavy
requests. The weights total 100. It measures the overall tradeoff without
giving either variant shared-prefix reuse.

```bash
export PRESET=mixed
export WORKLOAD_NAME="${VARIANT}-${PRESET}"
capture_worker_metrics "${PRESET}-before"

kubectl delete job -n "$NAMESPACE" "$PERF_JOB_NAME" --ignore-not-found
kubectl set env --local -f "$EXP_DIR/perf.yaml" -o yaml \
  EXPERIMENT_VARIANT="$VARIANT" \
  WORKLOAD_MODE=mixed WORKLOAD_NAME="$WORKLOAD_NAME" \
  SEQUENCE_DISTRIBUTION='1024,4096:20;4096,2048:25;8192,1024:25;16384,512:20;32768,256:10' \
  PREFIX_MODE=isolated CONCURRENCIES='1 4 8 16 32' \
  BENCHMARK_DURATION=300 WARMUP_REQUESTS=16 \
  RANDOM_SEED=42 NUM_DATASET_ENTRIES=4096 \
  ARTIFACT_ROOT="${RESULT_ROOT}/${VARIANT}" |
  kubectl apply -n "$NAMESPACE" -f -
kubectl logs -n "$NAMESPACE" -f "job/$PERF_JOB_NAME"
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  "job/$PERF_JOB_NAME" --timeout=14400s

capture_worker_metrics "${PRESET}-after"
```

Run all three presets for `VARIANT=baseline-kv-aware`, switch to
`VARIANT=treatment-kv-aware-cpu-offload`, recreate the deployment so its caches
are cold, repeat the acceptance gates, and run the same presets without
changing `RUN_ID` or any workload value other than its variant prefix.

## Plot the A/B comparison

Create a CPU-only Job that mounts `perf-cache` and generates separate AIPerf
comparison plots for the prefill-heavy, decode-heavy, and mixed results:

```bash
tee "$EXP_DIR/plot.yaml" >/dev/null <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: qwen36-sglang-kv-ab-plot
spec:
  backoffLimit: 0
  template:
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
        - name: plot
          image: python:3.12-slim
          command:
            - /bin/bash
            - -lc
          args:
            - |
              set -euo pipefail
              apt-get update
              apt-get install -y --no-install-recommends chromium findutils tar
              rm -rf /var/lib/apt/lists/*
              python -m pip install --no-cache-dir "aiperf==0.10.0"

              result_root="/perf-cache/aiperf/qwen36-sglang-kv-ab/${RUN_ID}"
              test -d "$result_root/baseline-kv-aware"
              test -d "$result_root/treatment-kv-aware-cpu-offload"
              mkdir -p "$result_root/plots"

              tee "$result_root/plot-config.yaml" >/dev/null <<'PLOT_EOF'
              experiment_classification:
                baselines:
                  - "*baseline-kv-aware*"
                treatments:
                  - "*treatment-kv-aware-cpu-offload*"
                default: treatment
              visualization:
                multi_run_defaults:
                  - ttft_vs_request_throughput
                  - tpot_vs_output_throughput
                  - latency_vs_request_throughput
                multi_run_plots:
                  ttft_vs_request_throughput:
                    type: scatter_line
                    x:
                      metric: request_throughput
                      stat: avg
                    y:
                      metric: time_to_first_token
                      stat: p95
                    labels: [concurrency]
                    groups: [experiment_group]
                    x_label: "Request Throughput (requests/sec)"
                    y_label: "P95 TTFT (ms)"
                    title: "P95 TTFT vs Request Throughput"
                  tpot_vs_output_throughput:
                    type: scatter_line
                    x:
                      metric: output_token_throughput
                      stat: avg
                    y:
                      metric: inter_token_latency
                      stat: p95
                    labels: [concurrency]
                    groups: [experiment_group]
                    x_label: "Output Token Throughput (tokens/sec)"
                    y_label: "P95 TPOT (ms/token)"
                    title: "P95 TPOT vs Output Token Throughput"
                  latency_vs_request_throughput:
                    type: scatter_line
                    x:
                      metric: request_throughput
                      stat: avg
                    y:
                      metric: request_latency
                      stat: p95
                    labels: [concurrency]
                    groups: [experiment_group]
                    x_label: "Request Throughput (requests/sec)"
                    y_label: "P95 Request Latency (ms)"
                    title: "P95 Request Latency vs Request Throughput"
              PLOT_EOF

              for workload in prefill-heavy decode-heavy mixed; do
                mapfile -t paths < <(
                  find "$result_root/baseline-kv-aware" \
                    "$result_root/treatment-kv-aware-cpu-offload" \
                    -type d -name "*-$workload" | sort
                )
                if [ "${#paths[@]}" -ne 2 ]; then
                  echo "Expected two result paths for $workload, got ${#paths[@]}" >&2
                  exit 1
                fi
                mkdir -p "$result_root/plots/$workload"
                aiperf plot --paths "${paths[@]}" \
                  --config "$result_root/plot-config.yaml" \
                  --output "$result_root/plots/$workload"
              done
          env:
            - name: RUN_ID
              value: SET_ME
          resources:
            requests:
              cpu: "2"
              memory: 4Gi
            limits:
              cpu: "8"
              memory: 16Gi
          volumeMounts:
            - name: perf-cache
              mountPath: /perf-cache
      volumes:
        - name: perf-cache
          persistentVolumeClaim:
            claimName: perf-cache
EOF

kubectl delete job -n "$NAMESPACE" "$PLOT_JOB_NAME" --ignore-not-found
kubectl set env --local -f "$EXP_DIR/plot.yaml" -o yaml RUN_ID="$RUN_ID" |
  kubectl apply -n "$NAMESPACE" -f -
kubectl logs -n "$NAMESPACE" -f "job/$PLOT_JOB_NAME"
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  "job/$PLOT_JOB_NAME" --timeout=1800s
```

The plots remain under `${RESULT_ROOT}/plots` on `perf-cache`. Copy them to the
cluster host's ephemeral experiment directory if needed:

```bash
plot_pod="$(kubectl get pods -n "$NAMESPACE" \
  -l "job-name=$PLOT_JOB_NAME" \
  -o jsonpath='{.items[0].metadata.name}')"
mkdir -p "$EXP_DIR/plots/$RUN_ID"
kubectl cp -n "$NAMESPACE" \
  "$plot_pod:${RESULT_ROOT}/plots/." "$EXP_DIR/plots/$RUN_ID"
```

Use the same concurrency point when comparing variants:

| Preset | Primary comparison | CPU offload is better when |
| --- | --- | --- |
| Prefill-heavy | p50/p95 TTFT and request throughput | TTFT is lower or throughput is higher with healthy NIXL deltas and positive HiCache use |
| Decode-heavy | p50/p95 TPOT and output-token throughput | Decode performance is not materially worse; a large gain is not expected |
| Mixed | p95 request latency, TTFT, TPOT, and throughput | End-to-end latency/throughput improves without more failures |

Treat prefill-heavy as the primary result and decode-heavy as the regression
guard. Declare a winner only when the direction holds across at least three
A/B repetitions and is larger than the run-to-run variation. Any increase in
failed requests or `agent_errors_total` disqualifies that run.

## Initial cold-burst check

Run this short diagnostic immediately after a freshly recreated variant passes
the acceptance gates. It starts 64 concurrent 32K-input requests with no
warmup across 64 shared-prefix groups with 90% prefix reuse. This creates an
initial closed-loop burst that is useful for watching KV transfer and CPU
HiCache fill. Run it once for each variant with fresh worker Pods; do not
include it in the steady-state winner decision.

```bash
export PRESET=initial-burst
export WORKLOAD_NAME="${VARIANT}-${PRESET}"
capture_worker_metrics "${PRESET}-before"

kubectl delete job -n "$NAMESPACE" "$PERF_JOB_NAME" --ignore-not-found
kubectl set env --local -f "$EXP_DIR/perf.yaml" -o yaml \
  EXPERIMENT_VARIANT="$VARIANT" \
  WORKLOAD_MODE=fixed WORKLOAD_NAME="$WORKLOAD_NAME" \
  ISL=32768 OSL=128 PREFIX_MODE=shared \
  PREFIX_GROUPS=64 PREFIX_REUSE_PERCENT=90 \
  CONCURRENCIES=64 BENCHMARK_DURATION=45 WARMUP_REQUESTS=0 \
  RANDOM_SEED=42 NUM_DATASET_ENTRIES=1024 \
  ARTIFACT_ROOT="${RESULT_ROOT}/${VARIANT}/diagnostic" |
  kubectl apply -n "$NAMESPACE" -f -
kubectl logs -n "$NAMESPACE" -f "job/$PERF_JOB_NAME"
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  "job/$PERF_JOB_NAME" --timeout=14400s

capture_worker_metrics "${PRESET}-after"
```

This checks cold behavior only. Recreate all graph Pods and repeat the
acceptance gates before starting the three measured presets so the burst does
not pre-warm the worker prefix caches or leave stale frontend routing state:

```bash
kubectl delete pods -n "$NAMESPACE" -l "$GRAPH_LABEL"
kubectl wait -n "$NAMESPACE" --for=condition=Ready pod \
  -l "$GRAPH_LABEL" --timeout=1800s
```
