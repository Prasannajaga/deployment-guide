# Exp 4: disaggregated KV-aware routing

4P(TP2)+4D(TP2) with KV aware routing.

The frontend uses `--router-mode kv`, and
all workers publish KV events.

Artifacts are written under
`/perf-cache/artifacts/04-disagg-routing-kv-aware/<run-id>/` using the same
retention policy as exp 3.
