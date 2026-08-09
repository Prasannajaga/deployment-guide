# Goal & Benchmarking Specification: Qwen 32B Disaggregated Serving

This document defines the architectural goals, experimental matrix, and deep-dive benchmarking methodology for **Qwen 32B** (`Qwen/Qwen2.5-32B-Instruct`) deployed on a 2-node 8xH100 cluster (`gpu05` and `gpu06`) using NVIDIA Dynamo v1.3.0 with SGLang backend.

---

## 1. Goal Overview

The primary goal is to evaluate and compare **three progressive Prefill/Decode (P/D) disaggregation architectures** to identify the optimal serving setup for long-context, agentic, and high-concurrency workloads.

Instead of running a monolithic model, we disaggregate serving across nodes:
* **`gpu05` (8x H100):** Dedicated Prefill Node (4 Replicas @ TP=2)
* **`gpu06` (8x H100):** Dedicated Decode Node (4 Replicas @ TP=2)

---

## 2. Experimental Matrix: The 3 Experiments Under Test

We execute three structured experiments, building capabilities incrementally from a control baseline to full event-driven routing and multi-tier memory offloading.

```
+-----------------------------------------------------------------------------------+
| Experiment 1: Baseline P/D Disaggregation                                         |
| 4 Prefill (gpu05) + 4 Decode (gpu06) | Round-Robin Routing | No KV Events/Offload   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Experiment 2: P/D + Event-Driven KV-Aware Routing                                 |
| ZMQ Event Bus | DYN_ROUTER_MODE=kv | Prefix-Cache Aware Request Steering          |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Experiment 3: P/D + KV-Aware Routing + SGLang HiCache (Host RAM Offload)          |
| Multi-tier KV Cache (HBM L1 -> Pinned Host RAM L2) | Write-Through Policy          |
+-----------------------------------------------------------------------------------+
```

### Experiment 1: Baseline P/D Disaggregation (Control)
* **Objective:** Establish baseline P/D disaggregation latency and throughput without cache awareness.
* **Topology:** 4 Prefill replicas on `gpu05` (TP=2) + 4 Decode replicas on `gpu06` (TP=2).
* **Transport:** NVIDIA NIXL (Inter-process eXchange Library) over UCX / GPUDirect RDMA.
* **Routing:** Frontend naive round-robin.
* **Cache State:** Local GPU HBM only; no event publishing or offloading.

### Experiment 2: P/D Disaggregation + KV-Aware Routing
* **Objective:** Evaluate prefix-cache reuse gains under multi-turn agentic traffic.
* **Configuration Changes:** 
  * Frontend enables `DYN_ROUTER_MODE=kv` and `DYN_ROUTER_USE_KV_EVENTS=true`.
  * SGLang workers publish ZMQ events over port 5557 when KV blocks are computed/freed.
* **Routing Logic:** Requests sharing prompt prefixes (e.g. system prompts, agent instructions) are routed to prefill workers already holding cached KV blocks.

### Experiment 3: P/D + KV-Aware Routing + SGLang HiCache (Host KV Offloading)
* **Objective:** Expand KV cache capacity beyond GPU HBM limits using host CPU DRAM to prevent KV cache evictions under high concurrency.
* **Configuration Changes:** Prefill workers enable SGLang HiCache:
  ```text
  --enable-hierarchical-cache
  --hicache-ratio 2
  --hicache-write-policy write_through
  --hicache-io-backend kernel
  ```
* **Memory Hierarchy:** L1 (GPU HBM VRAM) $\rightarrow$ L2 (Pinned Host RAM on `gpu05`). Frontend uses `DYN_ROUTER_HOST_CACHE_HIT_WEIGHT=0.75` for credit scoring.

---

## 3. Deep-Dive Benchmarking Objectives

Our benchmarking suite evaluates performance across five critical dimensions:

### A. Agentic Workflow Performance
Agentic workloads feature multi-turn conversations, tool-calling loops, system prompt reuse, and iterative chain-of-thought expansion.
* **What We Benchmark:** 
  * Re-use latency of shared system prompts and tool definition schemas.
  * Multi-turn chat completion latency degradation over 4–16 turn conversations.
  * Turn-by-turn Time-to-First-Token (TTFT) acceleration under KV-aware routing (Exp 2 & Exp 3 vs Exp 1).

### B. Context Awareness & Prefix Cache Hit Rate
* **What We Benchmark:**
  * **Cache Hit Ratio (%):** Percentage of prompt tokens served directly from pre-computed KV blocks.
  * **TTFT vs. Prefix Overlap:** TTFT reduction as prompt prefix overlap increases (0%, 25%, 50%, 75%, 90% overlap).
  * **Inter-Node KV Transfer Overhead:** Time required by NIXL to transfer prefill KV states to decode workers over RDMA.

### C. GPU & Host Resource Utilization
* **What We Benchmark:**
  * **VRAM Occupancy & HBM Usage:** GPU memory allocation per rank under varying batch sizes.
  * **Host RAM Allocation (HiCache):** Memory footprint and allocation stability on `gpu05` under `--hicache-ratio 2`.
  * **Streaming Compute Utilization (SM %):** GPU SM activity during heavy prefill bursts vs continuous decode generation.
  * **Interconnect Bandwidth:** RDMA throughput over the `10.18.96.x` fabric network during peak NIXL KV transfers.

### D. Concurrency Scaling & SLO-Bound Goodput
* **What We Benchmark:**
  * **Concurrency Sweeps:** Load testing across 1, 8, 16, 32, 64, 128, and 256 concurrent client streams.
  * **SLO Goodput (Requests/sec):** Maximum request rate achieved while satisfying strict Service Level Objectives:
    * **P95 TTFT:** `< 1,000 ms`
    * **P95 Inter-Token Latency (ITL):** `< 50 ms`
    * **Error Rate:** `< 0.1%`

---

## 4. Benchmarking Stack & Tooling

We use an enterprise-grade, deterministic benchmarking stack to ensure reproducible results.

| Component | Tool / Technology | Version / Configuration | Purpose |
|---|---|---|---|
| **Load Generator** | **NVIDIA AIPerf** | `v0.10.0` (pinned) | Generates synthetic & dataset-driven traffic, measures exact TTFT, ITL, latency, and goodput. |
| **Serving Backend** | **SGLang** | vLLM-compatible runtime in Dynamo | Executes prefill & decode passes, manages radix-tree KV caching and HiCache offload. |
| **Orchestrator** | **NVIDIA Dynamo** | `v1.3.0` (`DynamoGraphDeployment`) | Disaggregates P/D components, manages NIXL state transfers, and hosts the HTTP frontend router. |
| **Cluster Management** | **Kubernetes** | K8s Operator (`qwen32-bench` namespace) | Manages pod lifecycles, memory limits, and PVC model caches (`/model-store`). |
| **Transport Layer** | **NVIDIA NIXL + UCX / RDMA** | InfiniBand / RoCE GPUDirect | Enables low-latency GPU-to-GPU KV tensor transfers between `gpu05` prefill and `gpu06` decode nodes. |
| **Event Bus** | **ZeroMQ (ZMQ)** | Port `5557` per worker pod | Real-time event publishing of KV block allocation/eviction to the Dynamo frontend router. |
| **Observability** | **`nvidia-smi` / DCGM & PyTorch Profiler** | Native CUDA / DCGM Exporter | Tracks GPU VRAM, power, PCIe/NVLink bandwidth, and kernel execution metrics. |

---

## 5. Comparative Evaluation Matrix

Upon completing all three experiment runs, results will be compiled into the following comparative format:

| Metric | Exp 1: Baseline P/D | Exp 2: KV-Aware Routing | Exp 3: KV-Aware + HiCache |
|---|---|---|---|
| **Prefix Cache Hit Rate (%)** | N/A (Round-Robin) | High (ZMQ-driven) | Highest (HBM + Host RAM) |
| **P95 TTFT (Shared Prefix)** | Baseline | Reduced by ~60–80% | Reduced under heavy load |
| **P95 ITL (Decode Speed)** | Baseline | Same | Same |
| **Max SLO Goodput (Req/sec)** | Baseline | Higher under repeat prompts | Highest at high concurrency |
| **GPU HBM Memory (Prefill)** | ~82% limit | ~82% limit | ~82% HBM + 2x Host RAM |
| **Host System Memory Usage** | Minimal | Minimal | Heavy (Pinned Host KV Pool) |

---

## 6. Execution Instructions

Refer to individual experiment guides in `models/qwen3-32B/experiments/` for step-by-step launch commands:
* [`01-pd-disaggregation.md`](experiments/01-pd-disaggregation.md)
* [`02-pd-kv-aware-routing.md`](experiments/02-pd-kv-aware-routing.md)
* [`03-pd-kv-aware-routing-kv-offloading.md`](experiments/03-pd-kv-aware-routing-kv-offloading.md)
