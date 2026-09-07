# High-load MTP scaling benchmark

This runbook drives the four-worker aggregated deployment from useful load
through saturation and then beyond its configured scheduling capacity. It is
intentionally capable of causing request timeouts, failed Jobs, frontend
disconnects, worker restarts, and engine OOM termination. Run it only in the
dedicated experiment namespace, verify that no production traffic uses the
frontend Service, and configure the capture in [`metrics.md`](./metrics.md)
before starting.

The deployment allows 512 sequences per worker and has four workers, so 2048 is
the aggregate configured `max-num-seqs` ceiling. It is not a promise that 2048
long sequences fit simultaneously in KV cache. Concurrency above 2048 is
deliberate queue and failure pressure. vLLM may remain healthy by queueing work;
that is a valid overload result and is preferable to an artificial OOM.

| Stage | Concurrency | Purpose |
| :--- | :--- | :--- |
| Screening | 64–1024, preset-dependent | Establish the rising throughput curve |
| Saturation | 1536–4096 for balanced/decode | Cross the scheduler ceiling and locate the throughput knee |
| Breaker | 6144–8192 balanced/decode; 1536–2048 prefill | Locate the first timeout, error, restart, or throughput-collapse boundary |

Start with the balanced preset and keep temperature=0, `ignore_eos=true`, and
the generated trace constant between cells. Then use `decode-stress`, whose
8192-token output is eight times its input, to expose the MTP verification and
acceptance bottleneck under sustained decode. Run A versus B for the round-robin
MTP comparison and C versus D for the KV-routed comparison. Do not compare A
directly with D as an MTP-only effect because routing changes too.

## 1. Variables and benchmark template

Use the same administration-host variables and deployed `CELL` described in
the main README. These paths are cluster-host paths, not repository paths.

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/nemotron-3.5-lightning/vllm/experiments/03-spec-kv-routing
export DEPLOYMENT=nemotron35-vllm-e3
export PERF_JOB=nemotron35-vllm-e3-perf
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=$DEPLOYMENT"
mkdir -p "$EXP_DIR"
```

Create the Job definition once. Its default is the complete balanced
concurrency sweep. Every point warms up over 120 seconds before measurement.
The AIPerf process limit is 48 for the 64-CPU load-generator limit; high
concurrency does not require one process per request.

```bash
tee "$EXP_DIR/perf.yaml" >/dev/null <<'PERF_EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: nemotron35-vllm-e3-perf
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 14400
  template:
    metadata:
      labels:
        app: nemotron35-vllm-e3-perf
        research.nvidia.com/experiment: spec-kv-routing
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

              case "$EXPERIMENT_CELL" in
                A) routing=round_robin; speculation=off ;;
                B) routing=round_robin; speculation=mtp ;;
                C) routing=kv; speculation=off ;;
                D) routing=kv; speculation=mtp ;;
                *) echo "EXPERIMENT_CELL must be A, B, C, or D" >&2; exit 2 ;;
              esac

              apt-get update
              apt-get install -y --no-install-recommends ca-certificates curl jq
              rm -rf /var/lib/apt/lists/*
              python -m pip install --no-cache-dir "aiperf==$AIPERF_VERSION"
              aiperf profile --help > /tmp/aiperf-profile-help.txt
              for required_option in \
                --warmup-duration \
                --benchmark-grace-period \
                --request-timeout-seconds; do
                grep -Fq -- "$required_option" /tmp/aiperf-profile-help.txt || {
                  echo "AIPerf $AIPERF_VERSION lacks $required_option" >&2
                  exit 2
                }
              done

              ready_deadline=$(( $(date +%s) + MODEL_READY_TIMEOUT_SECONDS ))
              until curl -fsS --max-time 10 "http://$ENDPOINT/v1/models" |
                jq -e --arg model "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4" \
                  '.data[]? | select(.id == $model)' >/dev/null; do
                if [ "$(date +%s)" -ge "$ready_deadline" ]; then
                  echo "Model readiness timed out" >&2
                  exit 1
                fi
                sleep 5
              done

              run_point() {
              details="isl-${INPUT_TOKENS}_osl-${OUTPUT_TOKENS}_c-${CONCURRENCY}_reuse-${PREFIX_REUSE_PERCENT}_preset-${PRESET}"
              run_id="cell-$EXPERIMENT_CELL-$details"
              run_dir="$ARTIFACT_ROOT/cell-$EXPERIMENT_CELL/$details"
              results_dir="$run_dir/results"
              artifact_dir="$results_dir/aiperf"
              trace_file="$results_dir/trace.jsonl"
              if [ -e "$results_dir" ]; then
                echo "Result path already exists: $results_dir" >&2
                echo "Remove that result explicitly before rerunning this point" >&2
                exit 3
              fi
              mkdir -p "$artifact_dir"

              python3 - "$trace_file" <<'PY'
              import json
              import math
              import os
              import sys

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
                      row = {
                          "session_id": f"e3-{request_id:08d}",
                          "timestamp": request_id,
                          "input_length": input_tokens,
                          "output_length": output_tokens,
                          "hash_ids": hashes,
                      }
                      handle.write(json.dumps(row, separators=(",", ":")) + "\n")
              PY

              trace_sha256=$(sha256sum "$trace_file" | awk '{print $1}')
              benchmark_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
              benchmark_started_epoch=$(date -u +%s)
              jq -n \
                --arg run_id "$run_id" \
                --arg preset "$PRESET" \
                --arg load_stage "$LOAD_STAGE" \
                --arg cell "$EXPERIMENT_CELL" \
                --arg routing "$routing" \
                --arg speculation "$speculation" \
                --arg model "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4" \
                --arg model_revision "$MODEL_REVISION" \
                --arg aiperf_version "$AIPERF_VERSION" \
                --arg trace_sha256 "$trace_sha256" \
                --arg benchmark_started_utc "$benchmark_started_utc" \
                --argjson benchmark_started_epoch "$benchmark_started_epoch" \
                --argjson input_tokens "$INPUT_TOKENS" \
                --argjson output_tokens "$OUTPUT_TOKENS" \
                --argjson prefix_reuse_percent "$PREFIX_REUSE_PERCENT" \
                --argjson concurrency "$CONCURRENCY" \
                '{
                  run_id: $run_id,
                  experiment: "03-spec-kv-routing",
                  preset: $preset,
                  load_stage: $load_stage,
                  cell: $cell,
                  routing: $routing,
                  speculation: $speculation,
                  model: $model,
                  model_revision: $model_revision,
                  aiperf_version: $aiperf_version,
                  trace_sha256: $trace_sha256,
                  benchmark_started_utc: $benchmark_started_utc,
                  benchmark_started_epoch: $benchmark_started_epoch,
                  input_tokens: $input_tokens,
                  output_tokens: $output_tokens,
                  prefix_reuse_percent: $prefix_reuse_percent,
                  concurrency: $concurrency
                }' > "$results_dir/manifest.json"

              set +e
              aiperf profile \
                --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
                --tokenizer-revision "$MODEL_REVISION" \
                --input-file "$trace_file" \
                --custom-dataset-type mooncake_trace \
                --no-fixed-schedule \
                --url "http://$ENDPOINT" \
                --endpoint-type chat \
                --streaming \
                --use-server-token-count \
                --extra-inputs ignore_eos:true \
                --extra-inputs temperature:0.0 \
                --extra-inputs min_tokens:"$OUTPUT_TOKENS" \
                --concurrency "$CONCURRENCY" \
                --request-count "$REQUEST_COUNT" \
                --request-timeout-seconds "$REQUEST_TIMEOUT_SECONDS" \
                --warmup-duration "$WARMUP_SECONDS" \
                --benchmark-duration "$MEASURED_SECONDS" \
                --benchmark-grace-period "$GRACE_SECONDS" \
                --random-seed 100 \
                --workers-max "$WORKERS_MAX" \
                --record-processors "$RECORD_PROCESSORS" \
                --artifact-dir "$artifact_dir" \
                --server-metrics-formats json jsonl csv \
                --no-gpu-telemetry \
                --ui simple \
                2>&1 | tr '\r' '\n' | tee "$results_dir/aiperf.log"
              aiperf_rc=${PIPESTATUS[0]}
              set -e

              benchmark_finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
              benchmark_finished_epoch=$(date -u +%s)
              jq \
                --arg benchmark_finished_utc "$benchmark_finished_utc" \
                --argjson benchmark_finished_epoch "$benchmark_finished_epoch" \
                --argjson aiperf_exit_code "$aiperf_rc" \
                '. + {
                  benchmark_finished_utc: $benchmark_finished_utc,
                  benchmark_finished_epoch: $benchmark_finished_epoch,
                  aiperf_exit_code: $aiperf_exit_code
                }' "$results_dir/manifest.json" > "$results_dir/manifest.json.tmp"
              mv "$results_dir/manifest.json.tmp" "$results_dir/manifest.json"

              echo "Run artifacts: $results_dir"
              return "$aiperf_rc"
              }

              for concurrency in $CONCURRENCIES; do
                case "$concurrency" in
                  ''|*[!0-9]*|0) echo "Invalid concurrency: $concurrency" >&2; exit 2 ;;
                esac
                export CONCURRENCY="$concurrency"
                echo "Starting preset=$PRESET cell=$EXPERIMENT_CELL concurrency=$CONCURRENCY"
                run_point
              done
          env:
            - name: EXPERIMENT_CELL
              value: A
            - name: PRESET
              value: balanced
            - name: LOAD_STAGE
              value: saturation
            - name: MODEL_REVISION
              value: cc84af2fe71647d87f4486c064f320e1e7535243
            - name: ENDPOINT
              value: nemotron35-vllm-e3-frontend:8000
            - name: AIPERF_VERSION
              value: "0.12.0"
            - name: MODEL_READY_TIMEOUT_SECONDS
              value: "3600"
            - name: INPUT_TOKENS
              value: "8192"
            - name: OUTPUT_TOKENS
              value: "2048"
            - name: PREFIX_REUSE_PERCENT
              value: "90"
            - name: TRACE_BLOCK_SIZE
              value: "512"
            - name: CONCURRENCIES
              value: "256 512 1024 1536 2048 3072 4096 6144 8192"
            - name: REQUEST_COUNT
              value: "262144"
            - name: WARMUP_SECONDS
              value: "120"
            - name: MEASURED_SECONDS
              value: "300"
            - name: GRACE_SECONDS
              value: "120"
            - name: REQUEST_TIMEOUT_SECONDS
              value: "900"
            - name: WORKERS_MAX
              value: "48"
            - name: RECORD_PROCESSORS
              value: "32"
            - name: ARTIFACT_ROOT
              value: /perf-cache/specrouting
            - name: HF_HOME
              value: /opt/models
            - name: HF_HUB_OFFLINE
              value: "1"
            - name: TRANSFORMERS_OFFLINE
              value: "1"
            - name: AIPERF_HTTP_CONNECTION_LIMIT
              value: "16384"
            - name: PYTHONUNBUFFERED
              value: "1"
          resources:
            requests:
              cpu: "32"
              memory: 64Gi
            limits:
              cpu: "64"
              memory: 128Gi
          volumeMounts:
            - name: model-cache
              mountPath: /opt/models
              readOnly: true
            - name: perf-cache
              mountPath: /perf-cache
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache
        - name: perf-cache
          persistentVolumeClaim:
            claimName: perf-cache
PERF_EOF
```

The Job deliberately disables AIPerf GPU telemetry because DCGM and Prometheus
provide one authoritative GPU time series for all cells. A second sampler would
not improve engine observability and could produce timestamps that are harder
to align. The run manifest records start and finish epochs even when AIPerf
returns a nonzero exit code.

Artifacts use one deterministic path and no timestamp directory:

```text
/perf-cache/specrouting/cell-A/isl-8192_osl-2048_c-2048_reuse-90_preset-balanced/results
```

`results` contains `manifest.json`, `trace.jsonl`, `aiperf.log`, and the AIPerf
artifact subdirectory. If that result already exists, the Job stops before
writing anything. Remove the existing result explicitly before rerunning the
same point.

## 2. Run a preset

Set `CELL` to the cell currently deployed. Each Job runs the complete
concurrency list in order and writes one `results` directory per concurrency.
Delete the previous Job before applying another preset or cell.

### Balanced

```bash
export CELL=A
kubectl delete job "$PERF_JOB" -n "$NAMESPACE" --ignore-not-found --wait=true
kubectl set env --local -f "$EXP_DIR/perf.yaml" -o yaml \
  EXPERIMENT_CELL="$CELL" \
  PRESET=balanced LOAD_STAGE=sweep \
  INPUT_TOKENS=8192 OUTPUT_TOKENS=2048 PREFIX_REUSE_PERCENT=90 \
  CONCURRENCIES='256 512 1024 1536 2048 3072 4096 6144 8192' \
  WARMUP_SECONDS=120 MEASURED_SECONDS=300 \
  REQUEST_TIMEOUT_SECONDS=900 ARTIFACT_ROOT=/perf-cache/specrouting |
  kubectl apply -n "$NAMESPACE" -f -
kubectl logs -n "$NAMESPACE" -f "job/$PERF_JOB"
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  "job/$PERF_JOB" --timeout=14400s
```

### Decode stress

This is the primary MTP preset: output is eight times longer than input.

```bash
export CELL=A
kubectl delete job "$PERF_JOB" -n "$NAMESPACE" --ignore-not-found --wait=true
kubectl set env --local -f "$EXP_DIR/perf.yaml" -o yaml \
  EXPERIMENT_CELL="$CELL" \
  PRESET=decode-stress LOAD_STAGE=sweep \
  INPUT_TOKENS=1024 OUTPUT_TOKENS=8192 PREFIX_REUSE_PERCENT=50 \
  CONCURRENCIES='256 512 1024 1536 2048 3072 4096 6144 8192' \
  WARMUP_SECONDS=120 MEASURED_SECONDS=300 \
  REQUEST_TIMEOUT_SECONDS=1800 ARTIFACT_ROOT=/perf-cache/specrouting |
  kubectl apply -n "$NAMESPACE" -f -
kubectl logs -n "$NAMESPACE" -f "job/$PERF_JOB"
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  "job/$PERF_JOB" --timeout=14400s
```

### Prefill and KV stress

```bash
export CELL=A
kubectl delete job "$PERF_JOB" -n "$NAMESPACE" --ignore-not-found --wait=true
kubectl set env --local -f "$EXP_DIR/perf.yaml" -o yaml \
  EXPERIMENT_CELL="$CELL" \
  PRESET=prefill-kv LOAD_STAGE=sweep \
  INPUT_TOKENS=32768 OUTPUT_TOKENS=128 PREFIX_REUSE_PERCENT=90 \
  CONCURRENCIES='64 128 256 512 768 1024 1536 2048' \
  WARMUP_SECONDS=120 MEASURED_SECONDS=300 \
  REQUEST_TIMEOUT_SECONDS=900 ARTIFACT_ROOT=/perf-cache/specrouting |
  kubectl apply -n "$NAMESPACE" -f -
kubectl logs -n "$NAMESPACE" -f "job/$PERF_JOB"
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  "job/$PERF_JOB" --timeout=14400s
```

The Job stops on the first AIPerf failure, so later breaker values are not run
after a failed point. Preserve the Job and worker logs before redeploying.

## 3. Sweep order and breaker rule

For each preset, deploy A and run its Job sweep, followed by fresh B, C,
and D deployments. Recreating the graph between cells prevents an old
ConfigMap, cache state, or worker restart from crossing into the paired result.
Each cell and concurrency point has one deterministic result directory.

The breaker boundary for a cell is the first concurrency at which any of these
occurs: AIPerf exits nonzero or reports more than 1% failed requests; any graph
pod UID changes or restart count increases; a container terminates with
`OOMKilled` or a nonzero exit; `/health` or `/v1/models` fails after the Job;
p99 TTFT exceeds the selected preset's request timeout; or output-token
throughput drops by more than 10% from the prior point while queue depth grows.
Do not increase that cell beyond its first boundary. Recover it with a fresh
deployment, verify health, and run the paired MTP state at the same concurrency
once so the relative boundary is known.

At each boundary, preserve the evidence before redeploying:

```bash
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  -o custom-columns='POD:.metadata.name,UID:.metadata.uid,RESTARTS:.status.containerStatuses[*].restartCount,PHASE:.status.phase,REASON:.status.containerStatuses[*].lastState.terminated.reason'
kubectl get events -n "$NAMESPACE" --sort-by=.lastTimestamp | tail -n 100
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --all-containers=true --prefix=true --timestamps --tail=2000
```

If the AIPerf log reports a high-load worker CPU warning, blocked HTTP
connections, or sustained client CPU saturation, the point measured the load
generator rather than the LLM engine. Do not call it an engine ceiling. Increase
load-generator CPU or distribute the load before repeating that point.

## 4. Decide where MTP scales better

For each matched routing mode and concurrency, calculate MTP throughput lift as
`100 × (TPS_on / TPS_off - 1)`. Use B/A for round robin and D/C for KV routing.
Also compare p99 TTFT, p99 inter-token latency, error rate, stable concurrency
ceiling, and the breaker boundary. MTP scales better only when it raises
throughput or the stable concurrency ceiling without an unacceptable latency or
error regression. Acceptance rate explains the result but is not itself a
performance result.

The interaction term remains `(T_D - T_C - T_B + T_A) / T_A` at each matched
load. Positive values mean KV routing and MTP together outperform the sum of
their isolated effects; negative values show interference. Compute the term
only from successful, client-unconstrained runs that share the same trace
SHA-256 and workload shape.

After finding each cell's stable ceiling, repeat at 50%, 75%, 90%, 100%, and
110% of the lowest ceiling shared by the compared pair. Run `decode-stress` at
the local knee for the decisive MTP comparison, then use `prefill-kv` as the
control that shows whether any advantage survives a prefill-dominated shape.

## 5. Stop and cleanup

Deleting the load Job stops new traffic. If deletion hangs because the client
is draining thousands of streams, use the bounded grace period first; do not
delete the graph until the Job pod has stopped sending requests.

```bash
kubectl delete job "$PERF_JOB" -n "$NAMESPACE" \
  --ignore-not-found --wait=true --timeout=10m
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide
curl --fail --silent --show-error \
  "http://127.0.0.1:8000/health"
```

The final health command assumes the frontend port-forward from the main README
is active. A failed health check or changed worker UID requires a clean graph
redeployment before another measurement.

## References

NVIDIA's AIPerf documentation defines concurrency as the number of active
sessions and recommends crossing the engine batch ceiling to find saturation:
[load-generator options](https://docs.nvidia.com/aiperf/benchmark-modes/load-generator-options-reference)
and [benchmark parameter guidance](https://docs.nvidia.com/nim/benchmarking/llm/latest/parameters.html).
The configured HTTP connection limit and worker-process behavior are documented
in the [AIPerf environment variables](https://github.com/ai-dynamo/aiperf/blob/main/docs/environment-variables.md)
and [CLI reference](https://docs.nvidia.com/aiperf/reference/command-line-options).
