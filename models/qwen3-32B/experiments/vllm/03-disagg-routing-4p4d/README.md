# Exp 3: disaggregated 4P+4D, round-robin

4P + 4D, keeping round-robin routing and GPU-only KV cache.

| Pool | Replicas | TP | GPUs |
|---|---:|---:|---:|
| Prefill | 4 | 2 | 8 |
| Decode | 4 | 2 | 8 |

Artifacts are written below
`/perf-cache/artifacts/03-disagg-routing-4p4d/<run-id>/`.
