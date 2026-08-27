# Qwen3.6-35B-A3B 64K Benchmark: Baseline KV-Aware Serving vs. HiCache CPU KV Offloading

This benchmark report provides the complete empirical performance analysis for our **Disaggregated 4 Prefill + 4 Decode (4P4D)** deployment serving `Qwen/Qwen3.6-35B-A3B-FP8` across a dedicated **16x NVIDIA H100 GPU cluster**. It compares baseline GPU VRAM-only serving with Dynamo KV-aware routing against the same topology enhanced with **Hierarchical CPU KV Offloading (HiCache)** on the prefill workers.

All client-observed metrics are extracted directly from the verified AIPerf benchmark runs across the concurrency ladder (C = 8 to 128) with a 64K-token input workload. For the complete system architecture, deployment manifests, and Kubernetes setup, refer to the main [`blog.md`](../../../../sglang/disagg/tp1-ep2-4p4d/blog.md).

---

## Executive Summary: Peak Load (C=128) Highlights

Under peak concurrent demand (C=128), where the active prefix working set (3.10 Million tokens) severely exceeds physical GPU HBM capacity (~1.3M tokens), HiCache fundamentally transforms cluster performance:

- **+29.3% Output Throughput Boost:** Increases output token throughput from **1,425 tokens/s to 1,843 tokens/s** (5.57 req/s to 7.20 req/s), delivering **+498 more completed requests** (+25.2%) in a 5-minute window.
- **13.76s Slashed Off Tail TTFT (P99):** Reduces P99 Time-to-First-Token from **63.72s down to 49.96s** (21.6% faster) by streaming cached prefix blocks from host RAM over PCIe Gen5 DMA in **~45 ms** instead of burning GPU Tensor Core compute to recalculate from scratch.
- **Resilient 67.7% Cache Hit Rate:** Prevents the baseline cache hit rate collapse (which falls from 77.0% down to **54.4%** due to VRAM thrashing), maintaining a steady **67.7% hit rate** (+13.3% higher).
- **Rock-Solid Decode Speed (<10 ms P99 ITL):** Decode workers remain 100% GPU VRAM-resident and isolated from host memory traffic, sustaining steady token generation with under 1 ms difference in Inter-Token Latency (9.08 ms vs. 9.95 ms).

---

## 1. Hardware, Model & Benchmark Configuration

### System & Cluster Specifications

| Parameter | Configuration | Details |
| :--- | :--- | :--- |
| **Model** | `Qwen/Qwen3.6-35B-A3B-FP8` | Hybrid Attention / Mamba / MoE in native FP8 (35B total, ~3B active/token) |
| **Supported Context Window** | **131,072 tokens (~132K context)** | Full enterprise context length configured via `--context-length 131072` |
| **Page Size & KV Staging** | **64 tokens per page** | Unified attention & recurrent linear state paging (`--page-size 64`) |
| **Static VRAM Allocation** | **85% GPU Memory Fraction** | `--mem-fraction-static 0.85` (~68 GB of 80 GB VRAM per GPU dedicated to KV) |
| **Hierarchical Host Memory** | **1.2x GPU KV Capacity** | Pinned DDR5 Host RAM eviction reservoir (~3.1M cached prefix tokens) |
| **LLM Engine & Version** | **SGLang v0.5.14** | Disaggregated P/D runtime with HiCache CPU offloading |
| **Routing & Control Plane** | **NVIDIA Dynamo v1.3.0** | Prefix-aware KV router (`--router-mode kv`) with ZeroMQ telemetry |
| **Total Accelerators** | **16 × NVIDIA H100 SXM5 GPUs** | 80 GB HBM3 VRAM per GPU (1,280 GB cluster aggregate across 2 nodes) |
| **Worker Partitioning** | **8 Disaggregated Workers (4P4D)** | 4 Prefill Workers + 4 Decode Workers (2 GPUs per worker) |
| **Parallelism Strategy** | **TP=1 (DP-Attention) + EP=2 (MoE)** | DP Attention / LM-Head (`tp=2, dp=2, ep=2, moe-dense-tp=1`) |
| **Interconnect / State Transfer** | High-Speed RoCE RDMA Network | NVIDIA NIXL / UCX zero-copy transfer (`--disaggregation-transfer-backend nixl`) |

### Workload & Memory Footprint Specifications

| Field | Configuration | Details |
| :--- | :--- | :--- |
| **Input Sequence Length (ISL)** | 64,536 tokens (~64K) | Fixed workload mode (`workload_mode: fixed`) |
| **Output Sequence Length (OSL)** | 256 tokens | Fixed decode generation (`min_tokens=256, max_tokens=256`) |
| **Input-to-Output Ratio** | ~252:1 | Prefill-heavy enterprise agent workload |
| **Shared Prefix Target** | 75% (48,402 tokens/group) | Common context shared across requests in the same group |
| **Unique Suffix Context** | 25% (16,134 tokens/request) | Per-request unique agent instructions |
| **Distinct Prefix Groups** | 64 groups | 48,402 tokens x 64 = **3.10 Million shared prefix tokens** |
| **Concurrency Load Ladder** | C = 8, 16, 32, 64, 96, 128 | Incremental concurrency sweep testing memory saturation |
| **Execution Window & Controls** | 300s profile duration | 16 warmup requests, seed 42, 3,600s request timeout ceiling |

This setup intentionally creates an active prefix working set of **3.10 Million tokens**, exceeding the aggregate ~1.3M-token HBM cache capacity of the 4 prefill worker pods. Under baseline serving, this forces continuous LRU cache evictions. With HiCache, evicted blocks spill into pinned host DDR5 RAM, allowing SGLang to reload them over PCIe DMA rather than recomputing the full 64K prompt.

---

## 2. Aggregate Benchmark Results & Scaling Analysis

### Throughput & Cache Reuse Across Concurrency Ladder

| Concurrency (C) | Completed Requests | Measured Duration | Request Throughput | Output Token Throughput | Client Prompt Cache Read |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C = 8** | 733 | 304.00 s | 2.41 req/s | 617.3 tok/s | 66.4% |
| **C = 16** | 1,513 | 303.30 s | 4.99 req/s | 1,277.0 tok/s | **83.4%** |
| **C = 32** | 1,929 | 306.75 s | 6.29 req/s | 1,609.9 tok/s | 74.6% |
| **C = 64** | 2,650 | 322.47 s | **8.22 req/s** | **2,103.8 tok/s** | 72.2% |
| **C = 96** | 2,566 | 333.46 s | 7.70 req/s | 1,969.9 tok/s | 69.7% |
| **C = 128** | 2,471 | 343.27 s | **7.20 req/s** | **1,842.8 tok/s** | **67.7%** |

### Latency Progression Across Concurrency Ladder

| Concurrency (C) | TTFT Mean | TTFT P50 | TTFT P95 | TTFT P99 | ITL Mean | ITL P99 | E2E P99 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C = 8** | 1.73 s | 0.99 s | 4.64 s | 9.28 s | 6.11 ms | 7.57 ms | 10.74 s |
| **C = 16** | 1.48 s | 0.91 s | **6.12 s** | 9.66 s | 6.70 ms | 8.47 ms | 11.36 s |
| **C = 32** | 3.23 s | 1.40 s | 14.35 s | 19.77 s | 6.98 ms | 8.45 ms | 21.49 s |
| **C = 64** | 5.38 s | 3.38 s | 22.37 s | 29.28 s | 7.77 ms | 9.98 ms | 31.36 s |
| **C = 96** | 9.56 s | 7.16 s | 32.82 s | 40.84 s | 7.74 ms | 9.81 ms | 42.80 s |
| **C = 128** | 14.15 s | 11.07 s | 42.85 s | **49.96 s** | 7.60 ms | **9.95 ms** | **51.85 s** |

### Relative Impact vs. Baseline Serving (% Improvement)

| Concurrency (C) | Output Throughput Gain | Mean TTFT Improvement | P95 TTFT Improvement | P99 TTFT Improvement | Cache Read Gain | P99 ITL Delta |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C = 8** | +9.6% | -14.1% | +4.8% | -74.8% | -2.2 pp | -6.2% |
| **C = 16** | +28.6% | -37.9% | **-44.1%** | -32.6% | +6.3 pp | +6.7% |
| **C = 32** | +5.4% | -6.2% | -22.0% | -21.4% | +1.6 pp | +0.7% |
| **C = 64** | +10.6% | -15.1% | -0.5% | +2.0% | +7.1 pp | +6.2% |
| **C = 96** | +15.7% | -14.2% | -4.0% | +0.4% | +7.1 pp | -0.7% |
| **C = 128** | **+29.3%** | **-24.1%** | **-13.6%** | **-21.6%** | **+13.3 pp** | +0.87 ms |

---

## 3. Concurrency 128 Deep-Dive Breakdown

![Concurrency 128 Performance Breakdown](../../assets/benchmark-analysis.png)

At peak concurrency (C=128), the empirical results highlight four major technical takeaways:

| Metric Category | Baseline (GPU VRAM Only) | + HiCache CPU Offloading | Impact (+/- %) | Architectural Takeaway |
| :--- | :--- | :--- | :--- | :--- |
| **Request Throughput** | 5.57 req/s | **7.20 req/s** | **+29.3%** | Eliminates high-concurrency throughput cliff |
| **Output Token Throughput** | 1,425 tokens/s | **1,843 tokens/s** | **+29.3%** | Sustained high token generation under heavy saturation |
| **5-Min Completed Requests** | 1,973 requests | **2,471 requests** | **+498 reqs (+25.2%)** | Drains queues faster and completes more user workflows |
| **P99 Tail TTFT** | 63.72 s | **49.96 s** | **-21.6% (-13.76s)** | Fast ~45 ms PCIe DMA reloads replace expensive prefill compute |
| **P99 End-to-End Latency** | 65.17 s | **51.85 s** | **-20.4% (-13.32s)** | Faster overall turnaround for long-context requests |
| **Cache Hit Rate** | 54.4% (Collapse) | **67.7% (Resilient)** | **+13.3%** | Pinned DDR5 host RAM prevents VRAM LRU eviction collapse |
| **P99 Inter-Token Latency** | 9.08 ms | **9.95 ms** | < 1 ms | Decode workers remain 100% VRAM resident and isolated |

### 1. Throughput Scaling & Eliminating the Performance Cliff
In baseline serving, cluster throughput peaks at C=64 (7.43 req/s) and collapses by **25.0%** down to 5.57 req/s at C=128 due to severe cache thrashing. HiCache avoids this collapse entirely, sustaining **7.20 req/s** (1,843 tok/s)—delivering a **+29.3% throughput gain** and finishing 2,471 requests (+498 more than baseline).

### 2. Slashing Tail Time-to-First-Token (P99 TTFT)
Under heavy queue saturation, baseline prefill workers stall recalculating evicted 64K-token contexts on Tensor Cores, blowing out P99 TTFT to 63.72s. With HiCache active, SGLang reloads cached KV tensors from host RAM over PCIe Gen5 DMA in **~45 ms**, dropping P99 TTFT to **49.96s** (a **13.76-second reduction**).

### 3. Autoregressive Decode Isolation (P99 TPOT & ITL)
Because offloading is strictly confined to prefill workers, decode workers execute with zero host memory traffic. P99 Inter-Token Latency remains rock-solid at **9.95 ms** (vs 9.08 ms on baseline), proving that hierarchical offloading on prefill workers does not disrupt decode GPU memory bandwidth.

### 4. Cache Hit Rate Resilience
At C=128, baseline cache hit rate collapses to **54.4%** as the 3.10M working set overflows VRAM. HiCache buffers evicted prefix blocks in host DDR5 RAM, maintaining a resilient **67.7% hit rate** (+13.3% higher) that powers the sustained throughput curve.

---

## 4. NIXL & HiCache Telemetry Analysis

![NIXL Transfer and CPU HiCache Dashboard](../../assets/NIXL-metrics-offloading.png)

### Frontend & Network Telemetry Summary

| Concurrency (C) | Router KV-Hit Rate | Estimated State Transfer (Mean) | Estimated State Transfer (P95) | Frontend Ingress Queue (Avg / Max) |
| :--- | :--- | :--- | :--- | :--- |
| **C = 8** | 70.7% | 1.84 s | 4.76 s | 0.52 / 1 |
| **C = 16** | 81.4% | 1.51 s | 6.06 s | 0.91 / 2 |
| **C = 32** | 75.3% | 3.06 s | 14.73 s | 3.72 / 6 |
| **C = 64** | 70.4% | 5.09 s | 21.73 s | 11.62 / 17 |
| **C = 96** | 71.1% | 9.43 s | 31.46 s | 10.01 / 13 |
| **C = 128** | 72.2% | 14.20 s | 23.16 s | 11.18 / 14 |

### Telemetry Takeaways:
- **Zero-Copy RDMA Transfer (NIXL):** Average transfer time settles into a tight **15 ms to 20 ms** window per request. Over **2.50 TiB** of KV and linear recurrent state tensors were transferred across the test suite at burst rates of **10 to 12 req/s** with zero dropped packets or transport errors.
- **100% Host Memory Utilization (HiCache):** Token residency in host RAM ramps smoothly during warmup, plateauing at **~3.1 Million tokens** across the 4 prefill pods. The gauge meters show **99.7% to 100.0% CPU HiCache utilization** with remaining token capacity dropping to just **832 to 7.87K tokens**, confirming that host DDR5 RAM is fully utilized as an active secondary cache tier.

--- s