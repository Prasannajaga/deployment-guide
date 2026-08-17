# 16x H100 LLM Inference Cluster

This repository tracks LLM deployment experiments, setup runbooks, and benchmark results on a 2-node cluster with **16x NVIDIA H100 GPUs** (8x H100 per node).

> **PS**: I was running an LLM inference series on X that ended up gaining a lot of traction. After posting to ask if anyone could volunteer cluster access, the team at Lambda reached out and gave us access to a 16x H100 cluster for a week which was crazy! We did a ton of speedrunning here.
>
> Starting with a plain Ubuntu server, we set up all the K8s adapters and environment prerequisites from scratch for our experimentation.

## Repository Structure

* **`models/`**: Deployment configurations, runbooks, and benchmarks for individual models (e.g., [`models/llama-8B/setup.md`](models/llama-8B/setup.md), DeepSeek, GLM, Qwen).
* **`models/<model>/artifacts/`**: benchmark result exports, AI Perf profiling traces, server metrics, and summary plots.
* **`setup.md`**: Master cl*uster setup guide for NVIDIA Dynamo v1.3.0, Kubernetes adapters, and GPU nodes.
* **`progress.md`**: Daily experiment logs, active deployment progress, and status updates.
* **`benchmark.md`**: Overall cluster benchmark results, throughput metrics, and performance analysis.
* **`assets/`**: Performance plots, Grafana dashboards, and architecture diagrams.
