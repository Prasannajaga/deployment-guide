# Real-world benchmarking for NVIDIA Dynamo

This guide benchmarks the bare-metal Dynamo deployment created in
[`setup.md`](setup.md). It assumes:

- Dynamo `v1.3.0` and `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0`.
- Frontend: `http://10.18.96.143:8090` on `gpu05`.
- Model: `meta-llama/Llama-3.1-8B-Instruct`.
- One vLLM replica on each of `gpu05` and `gpu06`, discovered through etcd.
- Eight H100 GPUs per replica. Change the values below if the deployed
  topology, model, or frontend port differs.

The primary tool is
[NVIDIA AIPerf](https://docs.nvidia.com/aiperf/reference/command-line-options).
Dynamo v1.3.0 pins AIPerf `0.10.0`. Keep that version fixed while comparing
deployments.

## 1. Decide what success means

Do not optimize only for maximum requests per second. Define service-level
objectives (SLOs) before testing. Example starting targets are:

| Metric | Example SLO | Meaning |
| --- | ---: | --- |
| Error rate | `< 0.1%` | Transport and HTTP/model errors |
| P95 TTFT | `< 1,000 ms` | Time from request submission to first token |
| P95 ITL | `< 50 ms` | Delay between streamed output tokens |
| P95 request latency | Product-specific | End-to-end completion time |
| Goodput | Maximize | Requests/second satisfying all SLOs |

Replace the example latency limits with requirements from the actual product.
Report P50, P95, and P99. Do not trust P99 from a run with only a few dozen
completed requests; collect at least hundreds, preferably thousands, for a
production tail-latency result.

## 2. Use a separate benchmark client

For credible results, run AIPerf from a third CPU machine on the same private
network. The load generator must not share CPU, memory, disk, or GPUs with the
Dynamo frontend or workers.

Running AIPerf on `gpu05` is acceptable for a smoke test, but it can compete
with the frontend and understate capacity. Never benchmark through an SSH
tunnel, public Internet path, VPN, or Kubernetes port-forward when measuring
server capacity.

Verify connectivity from the benchmark client:

```bash
ping -c 4 10.18.96.143
curl -fsS http://10.18.96.143:8090/health
curl -fsS http://10.18.96.143:8090/v1/models
```

Keep port `8090` private. If the benchmark client cannot reach it, update the
private firewall/security group rather than exposing the unauthenticated API
publicly.

## 3. Install the pinned benchmark client

On the dedicated benchmark client:

```bash
python3 -m venv .venv-aiperf
source .venv-aiperf/bin/activate
python -m pip install --upgrade pip
python -m pip install 'aiperf==0.10.0'
aiperf --version
aiperf profile --help
```

The Dynamo runtime image also contains AIPerf, but using the lightweight
virtual environment avoids downloading a large GPU runtime image onto a CPU
load-generator machine.

The benchmark HTTP request does not require Hugging Face access. Synthetic
prompt generation does: AIPerf loads the model tokenizer locally so it can
create exact token lengths and calculate token-based metrics. Because Llama
3.1 is gated, authenticate the AIPerf process or give it a local tokenizer
directory.

If AIPerf is running on `gpu05` and the shared login from `setup.md` is already
present, reuse the credential **path** without printing or exporting its value:

```bash
export HF_HOME=/ephemeral/shared/huggingface
export HF_TOKEN_PATH=/ephemeral/shared/huggingface/token

test -r "$HF_TOKEN_PATH" \
  && echo "Hugging Face token file is readable"

hf auth whoami
```

`export` is a shell built-in: use `export ...` directly, never `sudo export`.
Run `aiperf` as the same user, not with `sudo`, so it inherits these variables
and the active virtual environment.

If the readability test fails, inspect only metadata:

```bash
id
sudo stat -c 'path=%n uid=%u gid=%g mode=%a' \
  /ephemeral/shared/huggingface \
  /ephemeral/shared/huggingface/token
```

Prefer a per-user access-control list (ACL). It preserves the existing owner
used by the Dynamo containers and does not make the credential world-readable:

```bash
sudo setfacl -m "u:$(id -un):rx" /ephemeral/shared/huggingface
sudo setfacl -m "u:$(id -un):r" /ephemeral/shared/huggingface/token

test -r "$HF_TOKEN_PATH" \
  && echo "Hugging Face token file is readable"
```

If `setfacl` is unavailable, and `id -u` confirms that the host user is UID
`1000` like the Dynamo runtime, use this fallback:

```bash
test "$(id -u)" -eq 1000 \
  || { echo "Current user is not UID 1000; do not change token ownership" >&2; exit 1; }

sudo chown "$(id -u):$(id -g)" /ephemeral/shared/huggingface/token
sudo chmod 600 /ephemeral/shared/huggingface/token
```

Never use `chmod 644`, `chmod 777`, or print the file to diagnose access.

On a dedicated benchmark client without the shared cache, choose one option:

1. Run `hf auth login` interactively using the authorized Hugging Face account.
   Do not put the token in a command argument or shell-history entry.
2. Securely provision only the non-secret tokenizer files and set
   `BENCH_TOKENIZER=/path/to/local/tokenizer`. Do not copy the token.

`--use-server-token-count` does not fix synthetic tests because AIPerf still
needs a tokenizer to generate the synthetic prompts. `--tokenizer builtin` is
acceptable for an API connectivity smoke test, but it uses a different
tokenization scheme and must not be used for Llama capacity comparisons.

Never copy SSH keys or store credentials in this repository or benchmark
artifacts.

## 4. Record an immutable test manifest

Use a new result directory for every deployment or configuration. On the
benchmark client:

```bash
export BENCH_URL=http://10.18.96.143:8090
export MODEL=meta-llama/Llama-3.1-8B-Instruct
unset AIPERF_TOKENIZER
export BENCH_TOKENIZER="${BENCH_TOKENIZER:-$MODEL}"
export TOPOLOGY=two-replicas-tp8
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export BENCH_RESULT_BASE="${BENCH_RESULT_BASE:-$PWD/dynamo-benchmarks}"
export RESULT_ROOT="${BENCH_RESULT_BASE}/${RUN_ID}-${TOPOLOGY}"

mkdir -p "$RESULT_ROOT"

printf '%s\n' \
  "run_id=$RUN_ID" \
  "model=$MODEL" \
  "tokenizer=$BENCH_TOKENIZER" \
  "url=$BENCH_URL" \
  "topology=$TOPOLOGY" \
  "aiperf=$(aiperf --version 2>&1 | head -n 1)" \
  > "$RESULT_ROOT/manifest.txt"
```

On both GPU nodes, record the non-secret software and hardware state. Save the
output alongside the benchmark results:

```bash
hostname
date -u --iso-8601=seconds
nvidia-smi --query-gpu=name,driver_version,memory.total,pstate,clocks.sm,power.limit --format=csv
git -C /ephemeral/shared/dynamo rev-parse HEAD
sudo docker image inspect nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0 \
  --format 'image={{.RepoDigests}} id={{.Id}}'
```

Also record all performance-affecting settings manually in `manifest.txt`:

- Number of replicas and GPUs per replica.
- Tensor, pipeline, and data-parallel sizes.
- Model dtype or quantization.
- `--gpu-memory-utilization`, `--max-model-len`, `--max-num-seqs`, and
  `--max-num-batched-tokens` if explicitly set.
- Routing mode and whether prefix caching or KV offloading is enabled.
- Any changes to clocks, power limits, network interface, or RDMA settings.

Do not dump the complete container environment because it may contain
credentials. Record only known non-secret performance settings.

## 5. Confirm that the deployment is ready

On `gpu05`:

```bash
curl -fsS http://127.0.0.1:8090/health
curl -fsS http://127.0.0.1:8090/v1/models
sudo docker ps --filter name=dynamo-frontend
sudo docker logs --tail 100 dynamo-frontend
```

On each GPU node:

```bash
sudo docker ps --filter name=dynamo-replica
sudo docker logs --tail 100 dynamo-replica
nvidia-smi
```

Both replicas must be registered and stable before the test. There must be no
model-download errors, container restarts, NCCL errors, or unrelated GPU
processes.

## 6. Run a streamed smoke benchmark

On the benchmark client:

```bash
aiperf profile \
  --model "$MODEL" \
  --tokenizer "$BENCH_TOKENIZER" \
  --url "$BENCH_URL" \
  --endpoint-type chat \
  --streaming \
  --concurrency 1 \
  --request-count 10 \
  --warmup-request-count 2 \
  --synthetic-input-tokens-mean 128 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 32 \
  --output-tokens-stddev 0 \
  --extra-inputs max_tokens:32 \
  --extra-inputs min_tokens:32 \
  --extra-inputs ignore_eos:true \
  --extra-inputs temperature:0.0 \
  --random-seed 42 \
  --artifact-dir "$RESULT_ROOT/smoke"
```

Continue only if all measured requests succeed and AIPerf reports TTFT, ITL,
request latency, request throughput, and output-token throughput.

Streaming must stay enabled. Without streaming, TTFT and ITL cannot represent
the experience of a chat user.

## 7. Establish a single-request latency baseline

This measures the unloaded latency floor. It is not a capacity result.

```bash
aiperf profile \
  --model "$MODEL" \
  --tokenizer "$BENCH_TOKENIZER" \
  --url "$BENCH_URL" \
  --endpoint-type chat \
  --streaming \
  --concurrency 1 \
  --request-count 100 \
  --warmup-request-count 5 \
  --synthetic-input-tokens-mean 2000 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 256 \
  --output-tokens-stddev 0 \
  --extra-inputs max_tokens:256 \
  --extra-inputs min_tokens:256 \
  --extra-inputs ignore_eos:true \
  --extra-inputs temperature:0.0 \
  --random-seed 42 \
  --goodput 'time_to_first_token:1000 inter_token_latency:50' \
  --artifact-dir "$RESULT_ROOT/fixed-i2000-o256/c1"
```

For configuration comparisons, keep prompt tokens, generated tokens, random
seed, sampling parameters, and AIPerf version identical. Forcing exactly 256
output tokens prevents early end-of-sequence responses from making one run
look artificially faster.

## 8. Find the concurrency saturation point

A concurrency test is closed-loop: AIPerf maintains a target number of
in-flight requests. `--concurrency 100` therefore represents 100 simultaneous
requests, not 100 human users who occasionally send prompts.

Run a three-minute discovery sweep:

```bash
for CONCURRENCY in 1 2 4 8 16 32 64 100 128; do
  aiperf profile \
    --model "$MODEL" \
    --tokenizer "$BENCH_TOKENIZER" \
    --url "$BENCH_URL" \
    --endpoint-type chat \
    --streaming \
    --concurrency "$CONCURRENCY" \
    --benchmark-duration 180 \
    --benchmark-grace-period 60 \
    --warmup-duration 30 \
    --synthetic-input-tokens-mean 2000 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 256 \
    --output-tokens-stddev 0 \
    --extra-inputs max_tokens:256 \
    --extra-inputs min_tokens:256 \
    --extra-inputs ignore_eos:true \
    --extra-inputs temperature:0.0 \
    --random-seed 42 \
    --goodput 'time_to_first_token:1000 inter_token_latency:50' \
    --artifact-dir "$RESULT_ROOT/fixed-i2000-o256/c${CONCURRENCY}"
done
```

The saturation point is where one or more of the following begins:

- Completed request throughput stops increasing materially.
- P95/P99 TTFT rises sharply.
- ITL exceeds the product SLO.
- Goodput falls even while raw throughput rises.
- Requests queue, time out, or fail.
- vLLM reports KV-cache preemption or recomputation.

Do not assume the highest concurrency is best. Production capacity is the
highest load that still meets error-rate and latency SLOs.

## 9. Test an open-loop production arrival rate

Human traffic is normally open-loop: new requests arrive independently of
whether earlier ones finished. A concurrency-only test can hide queue buildup.
Use a request-rate sweep to discover the sustainable arrival rate.

```bash
for REQUEST_RATE in 1 2 4 8 16 24 32 48 64; do
  aiperf profile \
    --model "$MODEL" \
    --tokenizer "$BENCH_TOKENIZER" \
    --url "$BENCH_URL" \
    --endpoint-type chat \
    --streaming \
    --request-rate "$REQUEST_RATE" \
    --benchmark-duration 300 \
    --benchmark-grace-period 120 \
    --warmup-duration 30 \
    --synthetic-input-tokens-mean 2000 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 256 \
    --output-tokens-stddev 0 \
    --extra-inputs max_tokens:256 \
    --extra-inputs min_tokens:256 \
    --extra-inputs ignore_eos:true \
    --extra-inputs temperature:0.0 \
    --random-seed 42 \
    --goodput 'time_to_first_token:1000 inter_token_latency:50' \
    --artifact-dir "$RESULT_ROOT/rate-i2000-o256/rps${REQUEST_RATE}"
done
```

Adjust the rate list after the first sweep so that several points lie below
saturation, several surround it, and one or two exceed it. Stop an overload
run if pending work grows without recovering or the client machine itself
becomes saturated.

## 10. Run a mixed real-world sequence-length workload

Fixed input/output lengths make A/B comparisons clean, but production traffic
is heterogeneous. This example represents short chat, medium chat/RAG, coding,
and long-context requests:

```text
50%:  512 input,   128 output tokens
30%: 2048 input,   256 output tokens
15%: 8192 input,   512 output tokens
 5%: 32768 input,  256 output tokens
```

Run it at a concurrency below the saturation point discovered in step 8:

```bash
export MIXED_CONCURRENCY=64

aiperf profile \
  --model "$MODEL" \
  --tokenizer "$BENCH_TOKENIZER" \
  --url "$BENCH_URL" \
  --endpoint-type chat \
  --streaming \
  --concurrency "$MIXED_CONCURRENCY" \
  --benchmark-duration 600 \
  --benchmark-grace-period 180 \
  --warmup-duration 60 \
  --sequence-distribution '512,128:50;2048,256:30;8192,512:15;32768,256:5' \
  --extra-inputs ignore_eos:true \
  --extra-inputs temperature:0.0 \
  --random-seed 42 \
  --goodput 'time_to_first_token:1000 inter_token_latency:50' \
  --artifact-dir "$RESULT_ROOT/mixed/c${MIXED_CONCURRENCY}"
```

Replace this distribution with production measurements when available. The
input sequence length must include the full conversation history sent on each
request, not just the user's newest message.

## 11. Benchmark multi-turn conversations or a production trace

Independent synthetic prompts do not measure conversation growth, prefix
reuse, or user think time. AIPerf can run a public multi-turn dataset:

```bash
aiperf profile \
  --model "$MODEL" \
  --tokenizer "$BENCH_TOKENIZER" \
  --url "$BENCH_URL" \
  --endpoint-type chat \
  --streaming \
  --public-dataset sharegpt \
  --num-sessions 100 \
  --concurrency 20 \
  --random-seed 42 \
  --artifact-dir "$RESULT_ROOT/multiturn-sharegpt"
```

The public dataset may require Internet access on the benchmark client. A
sanitized production trace is more representative. After converting it to an
AIPerf-supported trace format, preserve its recorded arrival times:

```bash
export TRACE_FILE=/path/to/sanitized-production-trace.jsonl

aiperf profile \
  --model "$MODEL" \
  --tokenizer "$BENCH_TOKENIZER" \
  --url "$BENCH_URL" \
  --endpoint-type chat \
  --streaming \
  --input-file "$TRACE_FILE" \
  --fixed-schedule \
  --random-seed 42 \
  --artifact-dir "$RESULT_ROOT/production-trace"
```

Remove personal data, credentials, private source code, access tokens, and
customer content before creating or sharing a trace. Never store a raw
production trace in this repository.

## 12. Run the explicit 100-concurrent-request test

After the discovery sweeps, run a longer test for the stated 100-concurrent
request target:

```bash
aiperf profile \
  --model "$MODEL" \
  --tokenizer "$BENCH_TOKENIZER" \
  --url "$BENCH_URL" \
  --endpoint-type chat \
  --streaming \
  --concurrency 100 \
  --benchmark-duration 900 \
  --benchmark-grace-period 180 \
  --warmup-duration 60 \
  --synthetic-input-tokens-mean 2000 \
  --synthetic-input-tokens-stddev 500 \
  --output-tokens-mean 256 \
  --output-tokens-stddev 64 \
  --extra-inputs temperature:0.0 \
  --random-seed 42 \
  --goodput 'time_to_first_token:1000 inter_token_latency:50' \
  --artifact-dir "$RESULT_ROOT/production-c100"
```

This variable-length test intentionally does not force `min_tokens` or
`ignore_eos`; it models natural early stopping. Keep the deterministic test in
step 8 as the apples-to-apples capacity comparison.

## 13. Monitor both replicas during every load test

In separate terminals on both `gpu05` and `gpu06`:

```bash
watch -n 1 nvidia-smi
```

```bash
sudo docker logs -f dynamo-replica
```

On `gpu05`, also follow the router:

```bash
sudo docker logs -f dynamo-frontend
```

Check for:

- Both nodes showing GPU activity. If only one node is busy, do not report the
  result as a two-replica benchmark.
- Stable GPU clocks, temperature, and power without thermal throttling.
- Growing request queues or timeouts.
- KV-cache preemption/recomputation warnings.
- NCCL, TCP request-plane, or container-restart errors.
- CPU saturation on the benchmark client or frontend node.

If workers were launched with `DYN_SYSTEM_PORT=8081`, inspect their actual
Prometheus metric names on each node:

```bash
curl -fsS http://127.0.0.1:8081/metrics \
  | grep -Ei 'vllm:|dynamo_.*(request|queue|cache|token|latency)'
```

The system port is disabled unless explicitly configured. Client-side AIPerf
results remain the source of truth for end-to-end latency. Server metrics
explain why a result changed.

## 14. Generate plots and retain raw results

```bash
aiperf plot "$RESULT_ROOT/fixed-i2000-o256"
aiperf plot "$RESULT_ROOT/rate-i2000-o256"
aiperf plot "$RESULT_ROOT"
```

Each AIPerf artifact directory normally contains:

- `profile_export_aiperf.json`: structured summary metrics.
- `profile_export.jsonl`: per-request raw results.
- `profile_export_aiperf.csv`: summary in CSV form.

Retain raw per-request output so percentiles and error classifications can be
audited later. Do not keep only screenshots or averaged numbers.

## 15. Determine production capacity

For each workload, create a table containing:

| Offered load | Completed RPS | Output tok/s | P50/P95/P99 TTFT | P50/P95/P99 ITL | P95 latency | Goodput | Errors |
| ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |

Choose the production operating point as the highest load where:

1. Error rate remains below the target.
2. P95 and P99 latency remain inside their SLOs.
3. Goodput still increases with offered load.
4. Queue depth remains bounded and drains after a burst.
5. No replica, GPU, network link, frontend CPU, or benchmark client is an
   accidental bottleneck.

Keep capacity headroom for bursts and failures. A service that meets SLOs only
at 100% steady-state utilization has no safe production capacity.

## 16. Repeat before comparing topologies

Run each final benchmark at least three times and report the median plus the
range. Use the same:

- Model and model revision.
- Container image digest.
- AIPerf version and random seed.
- Input/output distribution and sampling settings.
- Benchmark-client machine and network path.
- Warmup, duration, and SLO thresholds.
- GPU clocks, power limits, and other tenants.

When comparing the current `2 × TP=8` deployment against `16 × TP=1`, change
only the topology. Report:

- Aggregate output tokens/second.
- Output tokens/second/GPU.
- Goodput/GPU.
- P95/P99 TTFT and ITL.
- Error and preemption rates.

For an 8B model, this comparison is more useful than the raw memory percentage
shown by `nvidia-smi`.

## 17. Soak and burst tests

After selecting a capacity point:

- Run a 30-60 minute steady-state soak at approximately 70-80% of sustainable
  request rate.
- Run short bursts at 120-150% of sustainable rate, then verify that queues
  drain and latency returns to baseline.
- Confirm GPU memory is stable and containers do not restart.
- Repeat with one long-context-heavy workload because KV-cache pressure may be
  the limiting resource even when compute is not saturated.

Do not introduce worker termination or other failure injection until the
steady-state benchmark is repeatable. Failure testing is a separate exercise
and should be performed only in a disposable deployment.

## 18. Common benchmarking mistakes

- Benchmarking from the frontend/GPU node and measuring the load generator.
- Reporting a single request or only average latency.
- Comparing different token counts or allowing uncontrolled early EOS.
- Calling 100 sequential requests "100 users."
- Ignoring warmup, model compilation, cache state, or thermal state.
- Increasing concurrency after throughput has plateaued and calling the larger
  queue "higher capacity."
- Looking only at `nvidia-smi` memory allocation instead of KV-cache occupancy,
  queueing, goodput, and errors.
- Comparing two topologies with different model revisions, quantization, or
  client paths.
- Exposing the unauthenticated frontend publicly for convenience.
- Saving credentials or unsanitized prompts in benchmark artifacts.

## References

- [NVIDIA Dynamo benchmarking guide](https://docs.nvidia.com/dynamo/latest/user-guides/benchmarking)
- [NVIDIA AIPerf command-line reference](https://docs.nvidia.com/aiperf/reference/command-line-options)
- [NVIDIA AIPerf server metrics](https://docs.nvidia.com/aiperf/server-metrics/server-metrics-collection)
- [NVIDIA Dynamo vLLM observability](https://docs.nvidia.com/dynamo/backends/v-llm/observability)
- [vLLM metrics](https://docs.vllm.ai/en/stable/design/metrics.html)
