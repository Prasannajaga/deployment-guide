# Qwen3.6-35B-A3B Benchmark: SGLang Disaggregated TP1 (4P4D) vs. TP2 (2P2D)

This benchmark report provides the complete empirical performance analysis for our **Disaggregated SGLang** deployment serving `Qwen/Qwen3.6-35B-A3B-FP8` across a dedicated **8x NVIDIA H100 SXM5 GPU cluster**. It evaluates the architectural trade-offs between intra-pod Tensor Parallelism (**TP2-2P2D**: 2 Prefill + 2 Decode pods, 2 GPUs per pod) and inter-pod replica parallelism (**TP1-4P4D**: 4 Prefill + 4 Decode pods, 1 GPU per pod) under a production-grade mixed sequence workload.

For a detailed breakdown, check out: [read deep-dive blog](https://x.com/jaga_prasanna/status/2094419634489549223?s=20)

---

### Workload & Memory Footprint Specifications

| Field | Configuration | Details |
| :--- | :--- | :--- |
| **Model** | `Qwen/Qwen3.6-35B-A3B-FP8` | Hybrid Gated DeltaNet / MoE in native FP8 (35B total, ~3B active/token) |
| **Supported Context Window** | 131,072 tokens (~132K) | Full context length configured via `--context-length 131072` |
| **Page Size & Memory Budget** | 64 tokens per page | Unified paging (`--page-size 64`), static VRAM fraction `0.85` (~68 GB / 80 GB) |
| **Cluster Accelerators** | 8 × NVIDIA H100 SXM5 GPUs | 80 GB HBM3 per GPU (640 GB cluster total) connected via 900 GB/s NVLink |
| **Prefill Tier Layout** | TP1: 4 Pods (1 GPU/pod)<br/>TP2: 2 Pods (2 GPUs/pod) | Dedicated prompt ingestion and attention prefill compute engines |
| **Decode Tier Layout** | TP1: 4 Pods (1 GPU/pod)<br/>TP2: 2 Pods (2 GPUs/pod) | Dedicated autoregressive generation engines fed via RoCE RDMA |
| **Workload Distribution** | 5-tier mixed sequence | 1K (35%), 4K (30%), 8K (20%), 16K (10%), 32K (5%) context lengths |
| **Prefix Sharing & Reuse** | 75% target reuse | 8 distinct prefix groups (`--target-prefix-token-percent 75`) |
| **Concurrency Load Ladder** | C = 1, 4, 8, 16, 32, 64, 128 | Incremental concurrency sweep testing latency crossover and saturation |
| **Execution Window & Controls** | 60s profile duration / point | 16 warmup requests, seed 42, 3,600s request timeout ceiling |

In native FP8 precision, model weights occupy approximately 35 GB of VRAM. Because each NVIDIA H100 accelerator provides 80 GB of high-bandwidth memory (HBM3), a single GPU hosts the entire parameter set with over 45 GB of unallocated headroom dedicated to the static KV cache and Mamba linear recurrent state pool (`--mem-fraction-static 0.85`), providing capacity for 380,000 to 450,000 active tokens per device. Because physical memory capacity is not constrained, Tensor Parallelism is not required to fit the model, transforming the choice between TP=1 and TP=2 into an architectural trade-off between per-token arithmetic latency and cluster-wide replica throughput.

---

## 2. Aggregate Benchmark Results & Scaling Analysis

![Concurrency Scaling Sweep (c1 to c128)](../../../blogs/assets/parallelism-conccurency-sweep-dashboard.png)

### Throughput & Latency Across Concurrency Ladder

| Concurrency (C) | TP1 Output Tok/s | TP2 Output Tok/s | TP1 Req/s | TP2 Req/s | TP1 P50 TTFT | TP2 P50 TTFT | TP1 P95 TTFT | TP2 P95 TTFT | Empirical Regime Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C = 1** | 214.6 tok/s | 247.8 tok/s | 0.42 req/s | 0.51 req/s | 138.5 ms | 146.4 ms | 931.2 ms | 791.4 ms | TP2 faster single-stream decode |
| **C = 4** | 819.6 tok/s | 931.4 tok/s | 1.66 req/s | 1.87 req/s | 134.6 ms | 143.8 ms | 717.1 ms | 550.2 ms | TP2 leads under light utilization |
| **C = 8** | 1,529.9 tok/s | 1,634.6 tok/s | 3.13 req/s | 3.30 req/s | 134.4 ms | 145.0 ms | 591.6 ms | 536.8 ms | Parity regime; near-zero queuing |
| **C = 16** | 2,573.5 tok/s | 2,709.5 tok/s | 5.03 req/s | 5.25 req/s | 137.1 ms | 238.6 ms | 591.9 ms | 885.0 ms | TP2 prefill queuing begins |
| **C = 32** | 4,180.0 tok/s | 4,302.1 tok/s | 8.11 req/s | 8.38 req/s | 224.8 ms | 304.6 ms | 748.7 ms | 1,168.1 ms | Crossover boundary zone |
| **C = 64** | **6,470.5 tok/s** | 6,226.0 tok/s | **12.65 req/s** | 12.16 req/s | **239.0 ms** | 445.9 ms | **944.7 ms** | 3,150.2 ms | **TP1 takes throughput lead (+3.9%)** |
| **C = 128** | **9,435.7 tok/s** | 7,201.8 tok/s | **18.83 req/s** | 14.26 req/s | **474.9 ms** | 1,073.3 ms | **1,529.3 ms** | 12,520.0 ms | **TP1 dominates (+31.0% tok/s, 8.2x TTFT)** |

---

## 3. Concurrency 128 Deep-Dive Breakdown

![Concurrency 128 Performance Breakdown](../../../blogs/assets/parallelism-comparison-dashboard.png)

At peak concurrency (C=128), the empirical results highlight four major technical takeaways:

| Metric Category | TP1-4P4D (4P + 4D, TP=1) | TP2-2P2D (2P + 2D, TP=2) | Impact (+/- %) | Architectural Takeaway |
| :--- | :--- | :--- | :--- | :--- |
| **Output Token Throughput** | **9,435.67 tok/s** | 7,201.80 tok/s | **+31.0%** | Sustained high generation without decode starvation |
| **Request Throughput** | **18.83 req/s** | 14.26 req/s | **+32.0%** | Drains concurrent agent queues substantially faster |
| **5-Min Completed Workflows** | **5,649 requests** | 4,278 requests | **+1,371 reqs (+32.0%)** | Higher total workflow completions under saturation |
| **Median TTFT (P50)** | **474.9 ms (0.47 s)** | 1,073.3 ms (1.07 s) | **-55.8% (2.3x faster)** | Rapid prompt ingestion across 4 independent prefill pods |
| **Tail TTFT (P95)** | **1,529.3 ms (1.53 s)** | 12,520.0 ms (12.52 s) | **-87.8% (8.2x faster)** | Eradicates prefill head-of-line blocking on long prompts |
| **Worst-Case TTFT (P99)** | **2,549.7 ms (2.55 s)** | 17,124.7 ms (17.12 s) | **-85.1% (6.7x faster)** | Consistent prompt turnaround under extreme burst load |
| **End-to-End Latency (P95)** | **12.60 s** | 16.20 s | **-22.2% (-3.60 s)** | Faster overall response completion for complex agent queries |
| **Median ITL (P50 TPOT)** | 11.44 ms / tok | **8.85 ms / tok** | +2.59 ms (+22.6%) | TP2 achieves faster per-token generation forward passes |
| **Tail ITL (P99 TPOT)** | 13.06 ms / tok | **10.97 ms / tok** | +2.09 ms (+16.0%) | Memory-bandwidth bound decode benefits from weight sharding |

### 1. Throughput Scaling & Eliminating Decode Starvation
At peak concurrency (C=128), TP1-4P4D sustains 9,435.67 output tokens/sec and 18.83 req/s, delivering a **+31.0% throughput advantage** over TP2-2P2D (7,201.80 tok/s). In TP2, funneling 128 simultaneous client streams through only two prefill pods creates an acute prefill bottleneck. Because prefill workers cannot process long contexts fast enough to populate downstream queues, decode workers periodically experience starvation, sitting idle while waiting for state transfers over RDMA. By provisioning four independent prefill workers, TP1 keeps all four decode workers continuously saturated with requests.

### 2. Eradicating Head-of-Line Blocking and Slashing Tail TTFT
Under heavy concurrency, long prompts (up to 32K tokens) monopolize GPU Tensor Cores during prefill computation. In TP2-2P2D, arriving requests queue behind these active jobs, causing tail latency to explode to 12.52 seconds at P95 and 17.12 seconds at P99. Doubling the prefill worker count to four in TP1-4P4D cuts the average queue depth per worker from 64 to 32 requests, reducing P95 TTFT to **1.53 seconds** (an **8.2x speedup**) and P99 TTFT to **2.55 seconds** (a **6.7x speedup**).

### 3. The Concurrency Inversion: Low-Load Latency vs. High-Load Throughput
At light loads ($C \le 16$), TP2 delivers superior per-token generation speeds (8.85 ms vs. 11.44 ms P50 ITL) because sharding the 35 GB model weights across two GPUs cuts the per-device memory bandwidth read requirement in half (~17.5 GB per GPU). When prefill arrival rates are low and queues remain empty, this arithmetic speedup translates into faster end-to-end turnaround. Once concurrency exceeds 32, however, queueing delays eclipse the per-token latency advantage, flattening TP2 throughput while TP1 scales smoothly.

### 4. Zero NVLink All-Reduce Overhead and Compute Efficiency
Every transformer layer in TP2 requires two synchronous `all-reduce` communication barriers across NVLink to synchronize attention and MoE projections, introducing 128 barrier synchronizations per generated token. TP1 executes all linear projections, DeltaNet recurrent updates, and expert routings entirely within local GPU registers and HBM3 memory. By eliminating communication barriers, TP1 allows each H100 to operate at 100% Tensor Core duty cycle.

---

## 4. Architectural Decision Guide & Deployment Recipes

Platform teams evaluating Tensor Parallelism versus Disaggregated Replica Parallelism should apply the following criteria when model weights fit comfortably in single-accelerator memory:

When model weights fit within single-GPU VRAM (such as 35B FP8 occupying 35 GB on an 80 GB H100), allocating the remaining memory to KV cache at TP=1 maximizes total serving capacity. Intra-pod Tensor Parallelism (TP2) is advantageous only for low-concurrency, single-user interactive deployments where minimizing per-token generation latency is the sole optimization target and prefill queueing delays are absent. Under multi-tenant production traffic ($C \ge 64$), Disaggregated TP1 (4P4D) delivers superior aggregate throughput, lower tail latency, and greater resilience against prefill head-of-line blocking.

The complete deployment manifests and benchmarking configurations are available in the cluster repository:
- **TP1-4P4D Serving Manifest:** [`deploy.yaml`](../../../models/qwen3.6-35B-A3B/sglang/disagg/tp1-4p4d/deploy.yaml)
- **TP2-2P2D Serving Manifest:** [`deploy.yaml`](../../../models/qwen3.6-35B-A3B/sglang/disagg/tp2-2p2d/deploy.yaml)
- **AIPerf Benchmark Recipe:** [`perf.yaml`](../../../models/qwen3.6-35B-A3B/sglang/disagg/tp1-4p4d/perf.yaml)
- **Interactive Jupyter Notebook:** [`play.ipynb`](../../../models/qwen3.6-35B-A3B/sglang/disagg/tp1-4p4d/play.ipynb)

---

