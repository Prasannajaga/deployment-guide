# 16x H100 LLM Inference Cluster

This repository tracks LLM deployment experiments, setup runbooks, and benchmark results on a 2-node cluster with **16x NVIDIA H100 GPUs** (8x H100 per node).

> **PS**: I was running an LLM inference series on X that ended up gaining a lot of traction. After posting to ask if anyone could volunteer cluster access, the team at Lambda reached out and gave us access to a 16x H100 cluster for a week which was crazy! We did a ton of speedrunning here.
>
> Starting with a plain Ubuntu server, we set up all the K8s adapters and environment prerequisites from scratch for our experimentation.

## Cluster Overview & Environment Specifications

| Component | Specification | Details |
| :--- | :--- | :--- |
| **Compute Nodes** | `gpu05`, `gpu06` | 2 × 8-way H100 80GB SXM5 Nodes (16 GPUs total) |
| **Orchestration** | Kubernetes & Dynamo 1.3.0 | `DynamoGraphDeployment` CRD, PodCliqueSets |
| **Serving Engines** | vLLM `0.23.0` & SGLang `0.5.14` | Docker image: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0` / `sglang-runtime:1.3.0` |
| **Networking & RDMA** | RoCE v2 / InfiniBand | Network Operator (`qwen-roce-pool`), `mlx5_8:1` HCA, UCX/NIXL inter-pod transfer |
| **Benchmarking** | NVIDIA AIPerf `0.10.0` | Kubernetes Job template (`perf.yaml`), `/perf-cache` PVC storage |

---

## Available Model Experiments & Recipes

| Experiment / Recipe Name | Model | Engine | Status | Topology | Recipe Link |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Llama-3.1-8B-Instruct** | Llama-3.1-8B | vLLM | Working | Cross-node TP=16 | [`models/llama-8B`](models/llama-8B/setup.md) |
| **Qwen3-32B FP8 Aggregated** | Qwen3-32B-FP8 | vLLM / SGLang | Working | 8 aggregated workers × TP=2 (with/without CPU KV offload) | [`models/qwen3-32B/vllm/agg-routing`](models/qwen3-32B/vllm/agg-routing/README.md) |
| **Qwen3-32B FP8 Disaggregated** | Qwen3-32B-FP8 | vLLM / SGLang | Working | 6 prefill × TP=2 + 2 decode × TP=2 (KV-aware routing) | [`models/qwen3-32B/vllm/disagg-routing`](models/qwen3-32B/vllm/disagg-routing/README.md) |
| **Qwen3.6-35B-A3B FP8 Aggregated** | Qwen3.6-35B-A3B | SGLang | Working | Aggregated TP=2, KEDA autoscaling 1–8 workers | [`models/qwen3.6-35B-A3B/sglang/agg-autoscaling`](models/qwen3.6-35B-A3B/sglang/agg-autoscaling/README.md) |
| **Qwen3.6-35B-A3B FP8 Disaggregated** | Qwen3.6-35B-A3B | SGLang | Working | 4 prefill × TP=1 + 4 decode × TP=1 (EP=2, NIXL state transfer) | [`models/qwen3.6-35B-A3B/sglang/disagg/tp1-ep2-4p4d`](models/qwen3.6-35B-A3B/sglang/disagg/tp1-ep2-4p4d/README.md) |
| **Qwen3-235B-A22B FP8** | Qwen3-235B-A22B | SGLang | Working | 4 aggregated workers × TP=4 | [`models/qwen3-235B-A22B`](models/qwen3-235B-A22B/sglang/agg/deploy.yaml) |
| **GLM-5.2-FP8** | GLM-5.2-FP8 | vLLM | Working | 1 two-node replica, TP=16 | [`models/glm-5.2-fp8`](models/glm-5.2-fp8/vllm/agg/README.md) |
| **DeepSeek-V4-Flash FP8** | DeepSeek-V4-Flash | SGLang | Experimental | 4 aggregated workers × TP=4 | [`models/deepseek-v4-flash-fp8`](models/deepseek-v4-flash-fp8/sglang/agg/README.md) |

---

## Repository Structure

```text
├── setup.md          # Master environment & K8s operations guide
├── benchmark.md      # Kubernetes-native AIPerf benchmark runbook
├── progress.md       # Experiment tracking logs & active status
├── README.md         # Master repository overview (this file)
├── assets/           # Performance plots, diagrams, and Grafana exports
└── models/           # Individual model recipes, manifests, and runbooks
    ├── deepseek-v4-flash-fp8/
    ├── glm-5.2-fp8/
    ├── llama-8B/
    ├── qwen3-32B/
    ├── qwen3-235B-A22B/
    └── qwen3.6-35B-A3B/
```
