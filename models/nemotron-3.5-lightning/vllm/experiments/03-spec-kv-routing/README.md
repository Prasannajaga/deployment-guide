# Experiment 3: speculative decoding × KV routing

Choose a topology, then follow its complete cluster-host runbook. The benchmark
and observability instructions remain shared in [perf.md](./perf.md) and
[metrics.md](./metrics.md).

| Recipe | GPU layout | Purpose |
| :--- | :--- | :--- |
| [Aggregated](./aggregated/README.md) | Four workers × TP=1, four H100s | Original MTP × KV-routing experiment |
| [Disaggregated](./disaggregated/README.md) | TP=1: one prefill + three decode (or 2P2D), four H100s | P/D transfer and MTP comparison |

Each directory owns its deployment, matrix, benchmark Job, and suite runner.
Cluster artifacts live under the corresponding `aggregated` or `disaggregated`
subdirectory of `/ephemeral/shared/nemotron-3.5-lightning/vllm/experiments/03-spec-kv-routing`.
Benchmark results likewise use separate topology directories on the perf PVC.
The recipes deliberately reuse the graph, ConfigMap, and Job names: run one
topology at a time in a namespace and clean up before switching. The downloaded
model snapshots are shared.

The default 1P3D layout dedicates three GPUs to decode stress. Use the optional
2P2D layout to evaluate selection among multiple prefill replicas. Preserve the
same workload and timing when comparing either layout with aggregated TP=1.
