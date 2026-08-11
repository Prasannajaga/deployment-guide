# Qwen3-32B-FP8 experiment goal

Evaluate aggregated serving, prefill/decode disaggregation, KV-aware routing,
and host-memory KV-cache offloading on the same two-node 16×H100 cluster using
Dynamo 1.3.0. Run every topology with both vLLM and SGLang.

## Controlled variables

- Model: `Qwen/Qwen3-32B-FP8`
- Model snapshot: `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`
- Runtime images: Dynamo `1.3.0`
- Namespace: `qwen32-bench`
- Primary worker parallelism: TP=2
- Benchmark ISL: `1K 4K 8K 16K 32K`
- Benchmark OSL: `256`
- Concurrency: `1 8 16 32 64 128 256`
- AIPerf: `0.10.0`

## Questions to answer

1. At each input context, where does output throughput plateau?
2. At what concurrency do TTFT and TPOT rise sharply?
3. Does disaggregation outperform aggregation after NIXL transfer cost?
4. Does KV-aware routing reduce repeated-prefix TTFT?
5. Does host KV-cache offloading improve goodput under cache pressure without
   causing host-memory instability?
6. How do vLLM and SGLang compare under identical inputs and topology?

## Required measurements

- P50, P95, and P99 TTFT
- P50, P95, and P99 TPOT/ITL
- request and output-token throughput
- goodput under TTFT `2000 ms` and TPOT `50 ms/token` SLOs
- request error rate
- per-worker DCGM GPU utilization
- host RAM for offloading experiments
- NIXL transfer health for disaggregated experiments

## Acceptance rules

- No worker restarts or failed benchmark cells.
- Disaggregated runs show successful NIXL/UCX initialization.
- KV-aware runs publish and apply KV events.
- Offloading runs demonstrate host-tier allocation and hits without swap or OOM.
- Compare only runs using the same model, ISL, OSL, concurrency, duration, and
  warmup configuration.

See [experiments/README.md](experiments/README.md) for the complete symmetric
vLLM/SGLang experiment matrix and launch workflow.
