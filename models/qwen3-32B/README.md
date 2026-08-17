# Qwen3-32B serving experiments

This directory contains one comparable experiment matrix for vLLM and SGLang.
Every experiment uses Qwen3-32B-FP8, Dynamo 1.3.0, TP=2 workers unless noted,
the existing `model-cache` PVC, and namespace `qwen32-bench`.

## Experiment matrix

| Folder | Topology | Routing | KV-cache tier |
| --- | --- | --- | --- |
| `agg-routing` | 8 nodes × 1 aggregated TP=2 worker | default | GPU only |
| `agg-routing-kv-offloading` | 8 nodes × 1 aggregated TP=2 worker | KV-aware | GPU + host RAM |
| `disagg-routing` | 2 prefill × TP=2, 1 decode × TP=4 | default | GPU only |
| `disagg-routing-kv-aware` | 6 prefill × TP=2, 2 decode × TP=2 | KV-aware | GPU only |

The same four folder names exist below `vllm/` and `sglang/`. Framework-specific
offloading is intentionally different:

- vLLM uses native `OffloadingConnector` with a 32 GiB CPU tier per engine.
- SGLang uses HiCache with a host tier twice the GPU KV-cache pool.

## Common workflow

Run from the repository checkout:

```bash
export NAMESPACE=qwen32-bench
export EXPERIMENT_ROOT=/ephemeral/shared/qwen3-32b
```

Before each run, verify the shared prerequisites:

```bash
kubectl get crd dynamographdeployments.nvidia.com
kubectl get pvc -n "$NAMESPACE" model-cache
kubectl get nodes -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
```

The aggregated manifests require eight GPU nodes because worker anti-affinity
allows exactly one worker per node. Only one 16-GPU experiment should run at a
time. Apply the selected manifest:

```bash
export BACKEND=vllm
export EXPERIMENT=agg-routing
export EXP_DIR="$EXPERIMENT_ROOT/$BACKEND/$EXPERIMENT"

kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl get pods -n "$NAMESPACE" -o wide -w
```

To inspect pod logs across any experiment:

```bash
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"

# Tail recent logs
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" --all-containers --prefix --tail=500

# Stream live logs
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" --all-containers --prefix -f
```

Each experiment README gives its deployment name and frontend Service. Use
those values for logs, smoke tests, and cleanup. Keep benchmark inputs identical
across backends; compare P50/P95/P99 TTFT, TPOT, output-token throughput,
goodput, errors, and GPU utilization.

## Directory layout

```text
├── vllm/
│   ├── agg-routing/
│   ├── agg-routing-kv-offloading/
│   ├── disagg-routing/
│   └── disagg-routing-kv-aware/
└── sglang/
    ├── agg-routing/
    ├── agg-routing-kv-offloading/
    ├── disagg-routing/
    └── disagg-routing-kv-aware/
```

## References

- [Dynamo vLLM KV-cache offloading](https://docs.nvidia.com/dynamo/backends/v-llm/kv-cache-offloading)
- [Dynamo native vLLM offloading](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/backends/v-llm/native-kv-offloading)
- [Dynamo SGLang backend](https://docs.nvidia.com/dynamo/backends/sg-lang)
- [SGLang HiCache design](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/hicache_design.mdx)
