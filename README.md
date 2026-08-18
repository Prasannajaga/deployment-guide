# 16x H100 LLM Inference Cluster

This repository tracks LLM deployment experiments, setup runbooks, and benchmark results on a 2-node cluster with **16x NVIDIA H100 GPUs** (8x H100 per node).

> **PS**: I was running an LLM inference series on X that ended up gaining a lot of traction. After posting to ask if anyone could volunteer cluster access, the team at Lambda reached out and gave us access to a 16x H100 cluster for a week which was crazy! We did a ton of speedrunning here. starting with a plain Ubuntu server, we set up all the K8s adapters and environment prerequisites from scratch for our experimentation

## Cluster Overview & Environment Specifications

| Component | Specification | Details |
| :--- | :--- | :--- |
| **Compute Nodes** | `gpu05`, `gpu06` | 2 × 8-way H100 80GB SXM5 Nodes (16 GPUs total) |
| **Orchestration** | Kubernetes & Dynamo 1.3.0 | `DynamoGraphDeployment` CRD, PodCliqueSets |
| **Serving Engines** | vLLM `0.23.0` & SGLang `0.5.14` | Docker image: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0` / `sglang-runtime:1.3.0` |
| **Networking & RDMA** | RoCE v2 over Ethernet | Network Operator, RDMA Shared Device Plugin, Multus/MacVLAN, NV-IPAM pool `qwen-roce-pool`, `mlx5_8:1` HCA, UCX/NIXL inter-pod transfer |
| **Benchmarking** | NVIDIA AIPerf `0.10.0` | Kubernetes Job template (`perf.yaml`), `/perf-cache` PVC storage |

---

## What we deployed

| Name | Model | Engine | Status | GPUs Used | Topology | Recipe Link |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **Llama-3.1-8B-Instruct** | Llama-3.1-8B | vLLM | Working | 16 | Cross-node TP=16 | [`models/llama-8B`](models/llama-8B/setup.md) |
| **Qwen3-32B FP8 Aggregated** | Qwen3-32B-FP8 | vLLM / SGLang | Working | 16 | 8 aggregated workers × TP=2; four workers per node when all GPUs are allocated | [vLLM](models/qwen3-32B/vllm/agg-routing/README.md) / [SGLang](models/qwen3-32B/sglang/agg-routing/README.md) |
| **Qwen3-32B FP8 Disaggregated Baseline** | Qwen3-32B-FP8 | vLLM | Working | 16 | 6 prefill × TP=2 + 2 decode × TP=2; default routing | [`models/qwen3-32B/vllm/disagg-routing`](models/qwen3-32B/vllm/disagg-routing/README.md) |
| **Qwen3-32B FP8 Disaggregated Baseline** | Qwen3-32B-FP8 | SGLang | Working | 8 | 2 prefill × TP=2 + 1 decode × TP=4; default routing | [`models/qwen3-32B/sglang/disagg-routing`](models/qwen3-32B/sglang/disagg-routing/README.md) |
| **Qwen3-32B FP8 KV-Aware Disaggregated** | Qwen3-32B-FP8 | vLLM / SGLang | Working | 16 | 6 prefill × TP=2 + 2 decode × TP=2; Dynamo KV-aware routing | [vLLM](models/qwen3-32B/vllm/disagg-routing-kv-aware/README.md) / [SGLang](models/qwen3-32B/sglang/disagg-routing-kv-aware/README.md) |
| **Qwen3.6-35B-A3B FP8 Aggregated** | Qwen3.6-35B-A3B | SGLang | Working | 2–16 | 1–8 aggregated workers × TP=2 with KEDA; maximum requires all 16 GPUs | [`models/qwen3.6-35B-A3B/sglang/agg-autoscaling`](models/qwen3.6-35B-A3B/sglang/agg-autoscaling/README.md) |
| **Qwen3.6-35B-A3B FP8 Disaggregated** | Qwen3.6-35B-A3B | SGLang | Working | 16 | 4 prefill + 4 decode, 2 GPUs per worker; `tp-size=2`, `dp-size=2`, `ep-size=2`, DP attention with effective TP=1, with prefill CPU KV offload | [`models/qwen3.6-35B-A3B/sglang/disagg/tp1-ep2-4p4d`](models/qwen3.6-35B-A3B/sglang/disagg/tp1-ep2-4p4d/README.md) |
| **Qwen3-235B-A22B FP8** | Qwen3-235B-A22B | SGLang | Working | 16 | 4 aggregated workers × TP=4 | [`models/qwen3-235B-A22B/sglang/agg`](models/qwen3-235B-A22B/sglang/agg/README.md) |
| **GLM-5.2-FP8** | GLM-5.2-FP8 | vLLM | Working | 16 | 1 two-node replica, TP=16 | [`models/glm-5.2-fp8`](models/glm-5.2-fp8/vllm/agg/README.md) |
| **DeepSeek-V4-Flash FP8** | DeepSeek-V4-Flash | SGLang | Experimental | 16 | 2 aggregated workers × TP=8 | [`models/deepseek-v4-flash-fp8`](models/deepseek-v4-flash-fp8/sglang/agg/README.md) |

---

## What broke & How we fixed it

When we first deployed disaggregation, two separate issues blocked the workers. The RDMA Shared Device Plugin initially could not advertise `rdma/ib` until the host RDMA subsystem used shared network-namespace mode. After that was fixed, ordinary Calico Pods could see the verbs devices but still lacked a usable RoCE interface and Pod-specific GID, so UCX failed and NIXL could not transfer KV cache across prefill and decode workers. We fixed the Pod data path with a dedicated NV-IPAM pool (`qwen-roce-pool`) and pod-native RoCE interfaces using Multus and MacVLAN while keeping `hostNetwork: false`.

This was one of the hardest problem & big efforts we pulled off during the entire benchmark series. Once we cracked this, everything went smooth you can check out the full step-by-step breakdown in [`pod-native-roce.md`](pod-native-roce.md) and [`progress.md`](progress.md).

We only had one week of access to this 16x H100 cluster, so I'm incredibly glad we got this connection set up by Day 2. Resolving this early gave us the full runway to experiment with disaggregated KV offloading, KEDA autoscaling, cross-node parallelism, and a ton more.

## What we benchmarked

> **Blog Series Incoming**: Detailed technical write-ups and benchmark deep-dives for these experiments are coming soon!

* **Aggregated vs. Disaggregated Scaling**: Decoupled Prefill & Decode (P/D) vs. aggregated serving across vLLM and SGLang.
* **KV-Aware Routing & CPU Offloading**: Evaluated prompt prefix caching vs. CPU KV offloading under high concurrency.
* **Parallelism Bottlenecks (TP / DP / EP)**: Analyzed cross-node network stalls (TP=16 across nodes) vs. EP/DP scalability.
* **Event-Driven Autoscaling (KEDA)**: Dynamic pod scaling (1–8 workers) based on queue depth and GPU load metrics.
* **NIXL RDMA Latency Profiling**: Measured KV transfer latency growth over RoCE v2 as context length scales. See the reusable [NIXL Prometheus and Grafana runbook](NIXL-grafana.md).

We are currently processing and extracting all raw AIPerf benchmark artifacts, DCGM GPU utilization metrics, and Grafana performance dashboards. We'll be updating this section with full visual plots and benchmark graphs shortly!

## Repository Structure

```text
├── README.md             # Master repository overview (this file)
├── cluster.md            # Base environment & K8s deployment runbook
├── pod-native-roce.md     # Multus/MacVLAN & NV-IPAM RoCE networking guide
├── NIXL-grafana.md       # NIXL Prometheus telemetry & Grafana dashboard guide
├── benchmark.md          # Kubernetes-native AIPerf benchmark runbook
├── progress.md           # Experiment tracking logs & active status
├── setup.md              # Master environment & operations guide
├── assets/               # Performance plots, diagrams, and Grafana exports
└── models/               # Individual model recipes, manifests, and runbooks
    ├── deepseek-v4-flash-fp8/
    ├── glm-5.2-fp8/
    ├── llama-8B/
    ├── qwen3-32B/
    ├── qwen3-235B-A22B/
    ├── qwen3.6-35B-A3B/
    └── qwen3.8-27B/
```
