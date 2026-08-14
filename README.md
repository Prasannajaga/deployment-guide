# 16x H100 LLM Inference Cluster

This repository tracks LLM deployment experiments, setup runbooks, and benchmark results on a 2-node cluster with **16x NVIDIA H100 GPUs** (8x H100 per node).

## Repository Structure

* **`models/`**: Deployment guides and setup documentation for individual models (e.g., [`models/llama-8B/setup.md`](models/llama-8B/setup.md)).
* **`progress.md`**: daily progress update on deployments.
* **`benchmarks/`**: Benchmark runs, input datasets, export metrics, and performance plots.
* **`setup.md`**: Master NVIDIA Dynamo v1.3.0 cluster setup guide.

## Experiments

### Qwen-3-32B-FP8

| Recipe | Worker layout | Router | CPU KV offload |
|---|---|---|---|
| [`01-agg-routing`](models/qwen3-32B/experiments/vllm/01-agg-routing/) | 8 aggregate engines, TP2 | round-robin | no |
| [`02-disagg-routing`](models/qwen3-32B/experiments/vllm/02-disagg-routing/) | 6 prefill + 2 decode, TP2 | round-robin | no |
| [`03-disagg-routing-4p4d`](models/qwen3-32B/experiments/vllm/03-disagg-routing-4p4d/) | 4 prefill + 4 decode, TP2 | round-robin | no |
| [`04-disagg-routing-kv-aware`](models/qwen3-32B/experiments/vllm/04-disagg-routing-kv-aware/) | 4 prefill + 4 decode, TP2 | KV-aware | no |
| [`05-disagg-routing-kv-aware-offloading`](models/qwen3-32B/experiments/vllm/05-disagg-routing-kv-aware-offloading/) | 4 prefill + 4 decode, TP2 | KV-aware | 32 GiB/engine |

See the [experiment details and runbook](models/qwen3-32B/experiments/README.md) and [benchmark artifacts](models/qwen3-32B/artifacts/README.md).
