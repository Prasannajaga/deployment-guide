# Qwen3-32B-FP8 vLLM serving experiments

Five experiments to compare worker layout, router, and CPU KV-cache offloading under the same workload ([Mooncake FAST25 fixed-schedule trace](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/traces/conversation_trace.jsonl)).

Every experiment uses Qwen3-32B-FP8 on 16 x H100 GPUs with a 40,960-token context limit. Goodput thresholds are 2,000 ms TTFT and 25 ms ITL. More details on the benchmark setup and experiment differences are available in the [Appendix](#appendix).


| Recipe | Worker layout | Router | CPU KV offload |
|---|---|---|---|
| [`01-agg-routing`](./01-agg-routing/) | 8 aggregate engines, TP2 | round-robin | no |
| [`02-disagg-routing`](./02-disagg-routing/) | 6 prefill + 2 decode, TP2 | round-robin | no |
| [`03-disagg-routing-4p4d`](./03-disagg-routing-4p4d/) | 4 prefill + 4 decode, TP2 | round-robin | no |
| [`04-disagg-routing-kv-aware`](./04-disagg-routing-kv-aware/) | 4 prefill + 4 decode, TP2 | KV-aware | no |
| [`05-disagg-routing-kv-aware-offloading`](./05-disagg-routing-kv-aware-offloading/) | 4 prefill + 4 decode, TP2 | KV-aware | 32 GiB/engine |

## How to run the experiments

Each experiment directory contains a checked-in `deploy.yaml` and `perf.yaml`.

Check the required PVCs before starting an experiment:

```bash
kubectl get pvc -n qwen32-bench model-cache compilation-cache perf-cache
```

Select one recipe and run it until its benchmark finishes. It takes about 1H~2H per experiment:

```bash
export NAMESPACE=qwen32-bench
export VLLM_EXP=/ephemeral/shared/qwen3-32b/vllm
export RECIPE=04-disagg-routing-kv-aware
export EXP_DIR="$VLLM_EXP/$RECIPE"

export DGD=$(awk '/^metadata:/{m=1; next} m && /^  name:/{print $2; exit}' "$EXP_DIR/deploy.yaml")
export PERF_JOB=$(awk '/^metadata:/{m=1; next} m && /^  name:/{print $2; exit}' "$EXP_DIR/perf.yaml")

kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/perf.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl wait -n "$NAMESPACE" --for=jsonpath='{.status.state}'=successful \
  "dynamographdeployment/$DGD" --timeout=45m

kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/perf.yaml"
kubectl logs -n "$NAMESPACE" -f "job/$PERF_JOB"
```

After preserving the artifact directory printed by the Job, release the GPUs:

```bash
kubectl delete job "$PERF_JOB" -n "$NAMESPACE" --ignore-not-found
kubectl delete dynamographdeployment "$DGD" -n "$NAMESPACE" --ignore-not-found
```

# Appendix

## Comparability

The following configurations remain aligned across all five experiments:

- 2 nodes (8 GPUs each), 16 GPUs total
- Qwen3-32B-FP8 revision `aa55da1`
- Dynamo v1.3.0
- 8 workers, TP2 per worker
- 40,960-token maximum context length
  - The original trace contains 12,031 requests. After filtering 574 over-context requests, the same 11,457 requests were used in all five manifests.
- GPU-memory utilization 0.90
- 128-token GPU blocks
- synchronous scheduling
- Mooncake FAST25 fixed schedule, streaming, `ignore_eos: true`
- Goodput thresholds: 2,000 ms TTFT / 25 ms ITL

The previous implementation split between Exp 1-3 and Exp 4-5 is deprecated. The current recipes use the same manifest conventions and benchmark wrapper, with only the experiment-specific settings below changed.

### Exp 1. Aggregate baseline

- Serves as the baseline for the disaggregated experiments.
- Runs 8x aggregate TP2 workers with round-robin routing.

### Exp 2. 6P+2D disaggregated

- Splits the 8 TP2 workers into 6 Prefill(TP2) and 2 Decode(TP2) workers.
- round-robin routing
- NIXL/RDMA KV transfer

### Exp 3. 4P+4D disaggregated

- Changes Exp 2's worker balance to 4P(TP2) and 4D(TP2) workers while retaining round-robin routing.

### Exp 4. 4P+4D with KV-aware routing

- Replaces round-robin routing with KV-aware routing.

### Exp 5. 4P+4D with KV-aware routing and CPU KV offload

- Extends Exp 4 with `MultiConnector` and a 32 GiB/engine CPU KV-cache tier.

## References

- [Dynamo Qwen3-32B vLLM recipe](https://github.com/ai-dynamo/dynamo/tree/main/recipes/qwen3-32b/vllm/disagg-kv-router)
- [Dynamo native vLLM offloading](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/backends/v-llm/native-kv-offloading)
