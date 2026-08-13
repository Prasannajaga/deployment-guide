# vLLM disaggregated TP1 versus TP2 benchmark

This recipe compares horizontal replication with tensor parallelism while
holding the total allocation at eight H100 GPUs. Both variants use Dynamo's
KV-aware frontend, vLLM prefill/decode disaggregation, NIXL over the existing
pod-native RoCE network, and the same pinned FP8 checkpoint.

## Purpose and architecture

```text
                         AIPerf Job
                             |
                             v
                     Dynamo Frontend
                       router-mode=kv
                        /           \
                       v             v
                 Prefill pool --> Decode pool
                         NIXL/UCX KV transfer
```

## Comparison and concurrency

| Setting | `tp1-4p4d` | `tp2-2p2d` |
|---|---:|---:|
| Prefill replicas x GPUs | 4 x 1 | 2 x 2 |
| Decode replicas x GPUs | 4 x 1 | 2 x 2 |
| Prefill GPUs | 4 | 4 |
| Decode GPUs | 4 | 4 |
| Total GPUs | 8 | 8 |
| TP per worker | 1 | 2 |

The canonical AIPerf Job has Kubernetes `parallelism: 1` and runs
concurrencies `1 2 4 8 16 32 64 128` sequentially. Concurrency here means
closed-loop in-flight requests, not eight simultaneous Jobs. Add `256` and
optionally `512` only when throughput is still increasing materially at 128
without an error or latency collapse.

## Fixed server settings

The two deployment manifests differ only in worker replica count, TP size,
GPUs per worker, and proportional CPU/RAM/RDMA/shared-memory resources. They
otherwise pin:

- checkpoint `95a723d08a9490559dae23d0cff1d9466213d989`;
- Dynamo `1.3.0`, vLLM `0.23.0`, FP8 model weights, and automatic KV dtype;
- `max-model-len=131072`, `gpu-memory-utilization=0.85`, and block size 128;
- FCFS scheduling, async scheduling disabled, chunked prefill,
  `max-num-seqs=128`, and
  `max-num-batched-tokens=32768`;
- vLLM prefix caching with `mamba-cache-mode=align` and the hybrid cache
  manager explicitly enabled;
- producer/consumer `NixlConnector` roles with fail-on-load-error;
- prefill ZMQ KV events, deterministic `PYTHONHASHSEED=0`, and frontend
  `--router-mode kv`;
- no expert parallelism, EPLB, DeepEP, speculative decoding/MTP, KV
  offloading, or weight/KV quantization variation.

Block size 128 follows the repository's vLLM KV-aware recipe and vLLM's
hybrid NIXL validation. The SGLang recipe's 64-token page is a different
backend setting and is not used as a vLLM block-size precedent.

## Hybrid-cache compatibility gate

Dynamo 1.3.0 ships vLLM 0.23.0, whose release includes NIXL Mamba prefix
caching. That is why both roles explicitly use:

```text
--enable-prefix-caching
--mamba-cache-mode align
--no-disable-hybrid-kv-cache-manager
```

Do not benchmark merely because Pods are `Running`. Require model readiness,
a completed P-to-D request, and NIXL/UCX transfer evidence. Abort if logs say
the hybrid KV cache manager was disabled, cache specs could not be unified,
or the NIXL connector does not support HMA. The older failure is tracked in
[Dynamo issue 10741](https://github.com/ai-dynamo/dynamo/issues/10741); the
vLLM 0.23 work is in
[vLLM PR 42554](https://github.com/vllm-project/vllm/pull/42554).

The router indexes the full-attention KV groups it receives as events. Its
reported cache hit rate is not proof that every Gated-DeltaNet recurrent state
was reused. Conversely, a near-zero router hit rate on known shared prompts
requires event/log investigation before declaring KV routing broken.

## Workloads

| Workload | `WORKLOAD_NAME` | ISL | OSL | Primary signal |
|---|---|---:|---:|---|
| Prefill heavy | `prefill-heavy` | 32000 | 256 | TTFT and prefill throughput |
| Balanced | `balanced` | 8000 | 1024 | General goodput, TTFT, and ITL |
| Decode heavy | `decode-heavy` | 2000 | 4096 | ITL and output-token throughput |

ISL and OSL standard deviations are zero. The Job sends `min_tokens`,
`max_tokens`, and `ignore_eos=true`, uses deterministic seed 42, streams all
responses, warms up with 32 requests, measures for 180 seconds, and allows a
3600-second per-request timeout. Use 300 seconds for final publication runs.

`PREFIX_MODE=isolated` creates 4,096 deterministic independent synthetic
entries and does not deliberately introduce shared prefixes.
`PREFIX_MODE=shared` uses eight prefix groups and assigns 75% of the target
ISL to the shared prefix. For the balanced case this is a 6,000-token prefix
plus about 2,000 unique tokens, keeping the total near 8,000.

## Files and entry points

| File | Purpose |
|---|---|
| [`tp1-4p4d/deploy.yaml`](tp1-4p4d/deploy.yaml) | Four TP1 prefill plus four TP1 decode workers |
| [`tp1-4p4d/perf.yaml`](tp1-4p4d/perf.yaml) | In-cluster TP1 AIPerf sweep |
| [`tp2-2p2d/deploy.yaml`](tp2-2p2d/deploy.yaml) | Two TP2 prefill plus two TP2 decode workers |
| [`tp2-2p2d/perf.yaml`](tp2-2p2d/perf.yaml) | In-cluster TP2 AIPerf sweep |

Run only one deployment and one performance Job at a time. The topology
READMEs contain copy-pasteable setup, smoke, benchmark, artifact, and cleanup
commands:

- [TP1 / 4P+4D runbook](tp1-4p4d/README.md)
- [TP2 / 2P+2D runbook](tp2-2p2d/README.md)

## Common runnable workflow

Select exactly one topology. This block derives all other variables and keeps
the service and Job names aligned with the manifests:

```bash
export NAMESPACE=qwen32-bench
export RECIPE_ROOT=/ephemeral/shared/qwen3.6-35b-a3b
export TOPOLOGY=tp1-4p4d # change to tp2-2p2d for configuration B
case "$TOPOLOGY" in
  tp1-4p4d)
    export DEPLOYMENT=qwen36-35b-a3b-vllm-disagg-kv-tp1-4p4d
    export PERF_JOB_NAME=qwen36-vllm-tp1-4p4d-perf
    ;;
  tp2-2p2d)
    export DEPLOYMENT=qwen36-35b-a3b-vllm-disagg-kv-tp2-2p2d
    export PERF_JOB_NAME=qwen36-vllm-tp2-2p2d-perf
    ;;
  *) echo "Unknown topology: $TOPOLOGY" >&2; exit 2 ;;
esac
export EXP_DIR="${RECIPE_ROOT}/vllm/disagg/${TOPOLOGY}"
export MODEL_CACHE_DIR="${RECIPE_ROOT}/model-cache"
export MODEL=Qwen/Qwen3.6-35B-A3B-FP8
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
export FRONTEND_SERVICE="${DEPLOYMENT}-frontend"
```

### Prerequisites, model cache, and optional secret

```bash
kubectl get crd dynamographdeployments.nvidia.com
kubectl get pvc model-cache perf-cache -n "$NAMESPACE"
kubectl get network-attachment-definition qwen-roce -n "$NAMESPACE"
kubectl get nodes -L qwen.nvidia.com/role \
  -o custom-columns='NODE:.metadata.name,ROLE:.metadata.labels.qwen\.nvidia\.com/role,GPU:.status.allocatable.nvidia\.com/gpu,RDMA:.status.allocatable.rdma/ib'
```

The public checkpoint normally needs no secret. If authentication is required,
create the optional secret from a protected file outside the repository; never
print or commit its contents:

```bash
kubectl create secret generic hf-token-secret -n "$NAMESPACE" \
  --from-file=HF_TOKEN=/secure/path/to/hf-token \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n "$NAMESPACE" -f "$MODEL_CACHE_DIR/model-download.yaml"
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  job/qwen36-35b-a3b-fp8-download --timeout=3600s
```

### Manifest validation, apply, readiness, Pods, and logs

The checked-in files are the canonical manifest creation output. Validate them
against the installed CRD before applying:

```bash
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl wait -n "$NAMESPACE" --for=condition=Ready pod \
  -l "$GRAPH_LABEL" --timeout=1800s
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  -L nvidia.com/dynamo-component-type -o wide
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --all-containers --prefix --tail=1000 | tee "/tmp/qwen36-${TOPOLOGY}-startup.log"
```

Apply the hybrid compatibility gate described above and require a real NIXL
transfer after the smoke request.

### Internal smoke test

```bash
kubectl run qwen-smoke --rm -i --restart=Never -n "$NAMESPACE" \
  --image=curlimages/curl -- \
  curl -fsS "http://${FRONTEND_SERVICE}:8000/v1/models"
kubectl run qwen-smoke --rm -i --restart=Never -n "$NAMESPACE" \
  --image=curlimages/curl -- \
  curl -fsS -H 'Content-Type: application/json' \
  --data-binary "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: smoke-test-ok\"}],\"chat_template_kwargs\":{\"enable_thinking\":false},\"temperature\":0,\"max_tokens\":32,\"stream\":false}" \
  "http://${FRONTEND_SERVICE}:8000/v1/chat/completions"
```

For optional manual access use `kubectl port-forward`, but never measure
performance through it.

### AIPerf quick check and full benchmark

Quick validation:

```bash
kubectl delete job -n "$NAMESPACE" "$PERF_JOB_NAME" --ignore-not-found
kubectl set env --local -f "$EXP_DIR/perf.yaml" -o yaml \
  WORKLOAD_NAME=quick-balanced ISL=8000 OSL=1024 PREFIX_MODE=isolated \
  CONCURRENCIES=4 BENCHMARK_DURATION=30 WARMUP_REQUESTS=4 \
  ARTIFACT_ROOT="/perf-cache/aiperf/qwen36-${TOPOLOGY}-quick" | \
  kubectl apply -n "$NAMESPACE" -f -
kubectl logs -n "$NAMESPACE" -f "job/${PERF_JOB_NAME}"
```

Canonical balanced sweep:

```bash
kubectl delete job -n "$NAMESPACE" "$PERF_JOB_NAME" --ignore-not-found
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/perf.yaml"
kubectl logs -n "$NAMESPACE" -f "job/${PERF_JOB_NAME}"
```

The topology runbooks provide copy-pasteable commands for all three workloads
and the balanced shared-prefix run. They also show this `tee <<EOF` one-off
manifest pattern; the checked-in Job remains canonical:

```bash
mkdir -p "/tmp/qwen36-${TOPOLOGY}-oneoff"
cp "$EXP_DIR/perf.yaml" "/tmp/qwen36-${TOPOLOGY}-oneoff/perf.yaml"
tee "/tmp/qwen36-${TOPOLOGY}-oneoff/kustomization.yaml" >/dev/null <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources: [perf.yaml]
nameSuffix: -oneoff
patches:
  - target: {kind: Job}
    patch: |-
      apiVersion: batch/v1
      kind: Job
      metadata: {name: ${PERF_JOB_NAME}}
      spec:
        template:
          spec:
            containers:
              - name: perf
                env:
                  - {name: ISL, value: "8000"}
                  - {name: OSL, value: "1024"}
                  - {name: CONCURRENCIES, value: "1 2 4 8 16 32 64 128"}
                  - {name: BENCHMARK_DURATION, value: "180"}
                  - {name: ARTIFACT_ROOT, value: "/perf-cache/aiperf/qwen36-${TOPOLOGY}-oneoff"}
EOF
kubectl apply -n "$NAMESPACE" -k "/tmp/qwen36-${TOPOLOGY}-oneoff"
```

### Artifact inspection and cleanup

Inspect the `perf-cache` PVC using the temporary inspector Pod in either
topology runbook. Require `input-config.json`, `matrix-status.tsv`, AIPerf
summaries, `profile_export_raw.jsonl`, and server metrics for every concurrency
point. Cleanup preserves both PVCs:

```bash
kubectl delete job -n "$NAMESPACE" "$PERF_JOB_NAME" --ignore-not-found
kubectl delete dynamographdeployment.nvidia.com -n "$NAMESPACE" \
  "$DEPLOYMENT" --wait=false --ignore-not-found
kubectl delete pod -n "$NAMESPACE" qwen-smoke qwen-perf-inspector \
  --ignore-not-found
```

### Troubleshooting

- Pending workers: check role labels, free GPUs, `rdma/ib`, and `qwen-roce`.
- Hybrid/HMA or cache-unification error: stop; do not benchmark with HMA
  disabled.
- No KV hits with shared prefixes: inspect ZMQ events, router logs, and the
  full-attention versus recurrent-state limitation.
- AIPerf install failure: allow egress or use an internal image pinned to
  AIPerf 0.10.0.
- Timeout saturation: increase `REQUEST_TIMEOUT`/`GRACE_PERIOD` and rerun.
- Existing raw artifact directory: choose a new `ARTIFACT_ROOT`.

## Result interpretation

Compare the same workload, prefix mode, concurrency, duration, and seed. At
each point retain request/output/total-token throughput; TTFT, ITL, and request
latency distributions; actual ISL/OSL; errors; and timeouts. AIPerf 0.10.0
keeps summary files and raw `profile_export_raw.jsonl` records in every `cN`
directory.

Use the highest concurrency that still satisfies the chosen latency/error SLO,
not automatically the largest throughput point. TP2 is favorable only if its
latency/NVLink gains compensate for halving independent worker count. Correlate
results with DCGM GPU utilization, HBM use, queue metrics, and, when already
exported by the cluster, NVLink/NCCL counters. Do not add a separate telemetry
stack for this experiment.

Discover metric names from the deployed version instead of assuming them:

```bash
kubectl run qwen-metrics --rm -i --restart=Never -n "$NAMESPACE" \
  --image=curlimages/curl -- \
  sh -c "curl -fsS http://${FRONTEND_SERVICE}:8000/metrics | sort"
```

Look for the installed version's router decisions, KV/shared-cache hit rates,
prefill events, worker queues, and vLLM cache metrics. `perf.yaml` also saves
frontend metrics before and after the matrix and enables AIPerf server-metric
artifacts. Set `SERVER_METRICS_URLS` to space-separated existing Prometheus or
DCGM exporter URLs to collect additional endpoints.

## Fairness checklist before publishing

1. Save both applied manifests and image digests with the artifacts.
2. Confirm all eight GPUs are idle before each deployment and allocated as
   four prefill plus four decode GPUs afterward.
3. Confirm worker logs report Dynamo 1.3.0, vLLM 0.23.0, HMA enabled, prefix
   caching enabled, block size 128, and successful NIXL/UCX initialization.
4. Run TP1 and TP2 in alternating order or repeat both to reduce thermal and
   temporal bias.
5. Run isolated mode for all three workloads, then the balanced shared-prefix
   test on both topologies.
6. Publish no point with endpoint failures, client saturation, wrong actual
   lengths, or request timeouts.

No benchmark numbers are included here; they must come from successful cluster
runs.
