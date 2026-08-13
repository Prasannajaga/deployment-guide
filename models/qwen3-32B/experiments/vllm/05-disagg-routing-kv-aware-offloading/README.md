# Exp 5: disaggregated KV-aware routing with CPU KV offload

On top of Exp 4, each TP2 engine adds a 32 GiB pinned CPU cache through `MultiConnector(NixlConnector + OffloadingConnector)`.

Run the compatibility gate in [`../README.md`](../README.md) before applying the deployment.

The offload pool is 32 GiB **per TP2 engine**, not per GPU rank. Across eight engines this reserves 256 GiB of host cache. per-node consumption depends on the worker placement selected by Kubernetes.

Artifacts are written under `/perf-cache/artifacts/05-disagg-routing-kv-aware-offloading/<run-id>/` using the same retention policy as exp 3.