# Qwen3-32B benchmark artifacts

The experiment directories at this level contain the runs used by the current analysis. Earlier and failed runs are preserved under [`deprecated/`](./deprecated/).

## Includes

- benchmark job configuration, description, and logs
- deployment snapshots
- AIPerf request-level traces (`profile_export.jsonl.gz`)
- AIPerf result summaries and exported server metrics
- DCGM GPU utilization, memory, power, PCIe, and NVLink metrics
- NIXL transfer and exposer metrics for disaggregated runs
- metric-name inventories and exact query windows

Request traces can be inspected with `gzip -cd profile_export.jsonl.gz`.
