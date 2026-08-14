# Qwen3-32B serving experiments

Five experiments to compare worker layout, router, and CPU KV-cache offloading under the same workload (Mooncake FAST25 fixed-schedule trace).

Every experiment uses Qwen3-32B-FP8 model on 16 x H100 GPUs, have a 40,960-token context limit.

| Recipe | Worker layout | Router | CPU KV offload |
|---|---|---|---|
| [`01-agg-routing`](./vllm/01-agg-routing/) | 8 aggregate engines, TP2 | round-robin | no |
| [`02-disagg-routing`](./vllm/02-disagg-routing/) | 6 prefill + 2 decode, TP2 | round-robin | no |
| [`03-disagg-routing-4p4d`](./vllm/03-disagg-routing-4p4d/) | 4 prefill + 4 decode, TP2 | round-robin | no |
| [`04-disagg-routing-kv-aware`](./vllm/04-disagg-routing-kv-aware/) | 4 prefill + 4 decode, TP2 | KV-aware | no |
| [`05-disagg-routing-kv-aware-offloading`](./vllm/05-disagg-routing-kv-aware-offloading/) | 4 prefill + 4 decode, TP2 | KV-aware | 32 GiB/engine |

## Comparability and known differences

The following configurations remain aligned across all five experiments:

- 2 nodes (8 GPUs each), 16 GPUs total
- Qwen3-32B-FP8 revision `aa55da1`
- Dynamo v1.3.0
- 8 workers, TP2 per worker
- 40,960-token maximum context length
  - The original trace contains 12,031 requests. After filtering 574 over-context requests, the same 11,457 requests were used in all five manifests.
- GPU-memory utilization 0.90
- 128-token GPU blocks
- synchronous scheduling
- Mooncake FAST25 fixed schedule, streaming, `ignore_eos: true`
- Goodput thresholds: 2,000 ms TTFT / 25 ms ITL

There are some known differences in `deploy.yaml` / `perf.yaml` of the experiments.

### Exp 1. aggregate(TP2 per worker) baseline

- Serves as the aggregate reference
- Uses 8 frontend replicas and the first-generation benchmark wrapper.
  - this was to avoid frontend crash but turns out 8 replicas were too much.
  - We reduced the frontend replicas in exp4 and exp5.

### Exp 2. 6P+2D(TP2 per worker) disaggregated

- Explicitly adds the NIXL/RDMA path, producer/consumer KV-transfer roles, and explicit UCX settings.

### Exp 3. 4P+4D(TP2 per worker) disaggregated

- Keeps Exp 2's frontend, transfer path, worker flags, and benchmark wrapper and only the prefill/decode balance and per-node role allocation materially change.


### Exp 4. 4P+4D(TP2 per worker) with KV-aware routing

- Uses kv-aware routing instead of Round-Robin routing.
- Changes the frontend from eight large replicas to two replicas.
- Allows scheduler placement instead of node pinning, adjusts KV roles, events, prefix caching, and UCX settings, and simplifies the benchmark wrapper (this may have slightly influenced the performance).
  - These operational differences were not isolated, but are expected to have limited influence on the overall trajectory

### Exp 5. 4P+4D(TP2 per worker) with KV-aware routing and CPU KV offload

- Adds `MultiConnector`, a 32 GiB/engine CPU tier, host-cache-aware routing, KV events and prefix caching on both roles, and explicit worker memory limits.
- Everything else is same as exp 4


## one time cache setup

[`vllm/setup/cache.yaml`](./vllm/setup/cache.yaml) creates the shared model, vLLM compilation, and benchmark artifact PV/PVC pairs. It is specific to this
two-node lab and assumes `/ephemeral/shared` is available on both nodes. Apply it only during initial setup or after `qwen32-bench` namespace is created:

```bash
kubectl apply -f /ephemeral/shared/qwen3-32b/experiments/vllm/setup/cache.yaml
kubectl get pvc -n qwen32-bench model-cache compilation-cache perf-cache
```

All three PVCs must become `Bound`. The PVs use the `Retain` policy, so deleting the namespace preserves their data but can leave a PV in `Released`. Before
reusing PV, verify that it belongs to the deleted cache PVC, remove its stale claim reference, and apply the setup manifest again:

```bash
kubectl get pv
export RELEASED_PV=qwen32-vllm-compilation-cache-pv
kubectl get pv "$RELEASED_PV" -o jsonpath='{.status.phase}{"\n"}{.spec.claimRef.namespace}{"/"}{.spec.claimRef.name}{"\n"}'
kubectl patch pv "$RELEASED_PV" --type=json \
  -p='[{"op":"remove","path":"/spec/claimRef"}]'
kubectl apply -f /ephemeral/shared/qwen3-32b/experiments/vllm/setup/cache.yaml
```

## How to run the experiment

Run the following commands on the cluster host. Change only `RECIPE` when moving to another experiment.

e.g.
```bash
export NAMESPACE=qwen32-bench
export VLLM_EXP=/ephemeral/shared/qwen3-32b/experiments/vllm
export RECIPE=04-disagg-routing-kv-aware

export EXP_DIR="$VLLM_EXP/$RECIPE"
export DGD=$(awk '/^metadata:/{m=1; next} m && /^  name:/{print $2; exit}' "$EXP_DIR/deploy.yaml")
export PERF_JOB=$(awk '/^metadata:/{m=1; next} m && /^  name:/{print $2; exit}' "$EXP_DIR/perf.yaml")
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=$DGD"

printf 'recipe=%s\ndeployment=%s\nbenchmark=%s\n' "$RECIPE" "$DGD" "$PERF_JOB"
```

These variables are local to the current shell. So you should run this block again after opening a terminal window.

### 1. Preflight

```bash
kubectl get pvc -n "$NAMESPACE" model-cache compilation-cache perf-cache
kubectl get secret -n "$NAMESPACE" hf-token-secret
kubectl get dynamographdeployments -A
```

All three PVCs must be `Bound`, hf secret must exist, and no other experiment should be using the 16 GPUs.

For disaggregated setup(Experiments 2-5) it requires  RDMA attachment:

```bash
kubectl get network-attachment-definition qwen-roce -n "$NAMESPACE"
```

### 2. Deploy

```bash
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"

kubectl get dynamographdeployment "$DGD" -n "$NAMESPACE" -w
```

Stop the watch with `Ctrl+C` after `READY=True`, then verify every Pod is running and ready:

```bash
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide
```

If the deployment remains `False`, print the controller's reason:

```bash
kubectl get dynamographdeployment "$DGD" -n "$NAMESPACE" \
  -o jsonpath='state={.status.state}{"\n"}message={.status.conditions[0].message}{"\n"}'
```

### 3. Benchmark

```bash
kubectl delete job "$PERF_JOB" -n "$NAMESPACE" --ignore-not-found
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/perf.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/perf.yaml"

kubectl wait -n "$NAMESPACE" --for=condition=Ready \
  pod -l "app=$PERF_JOB" --timeout=5m
kubectl logs -n "$NAMESPACE" -f --tail=30 "job/$PERF_JOB"
```

The benchmark takes about an hour. This is because we're running fixed traces for an hour. After it finishes:

```bash
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  "job/$PERF_JOB" --timeout=3h

export RUN_DIR=$(kubectl logs -n "$NAMESPACE" "job/$PERF_JOB" |
  sed -n 's/^Run artifacts: //p' | tail -1)
export HOST_RUN_DIR="/ephemeral/shared/qwen3-32b/perf-cache/${RUN_DIR#/perf-cache/}"

test -f "$HOST_RUN_DIR/manifest.json"
gzip -t "$HOST_RUN_DIR/aiperf/profile_export.jsonl.gz"
printf 'artifacts=%s\n' "$HOST_RUN_DIR"
```

### 4. Collect metrics

Keep this port-forward running in another terminal:

```bash
kubectl port-forward -n monitoring \
  service/monitoring-kube-prometheus-prometheus 9090:9090
```

Then collect the benchmark window's DCGM and NIXL metrics:

```bash
PROMETHEUS_URL=http://127.0.0.1:9090 \
  bash "$VLLM_EXP/collect-metrics.sh" "$HOST_RUN_DIR"
```

The files are saved under `$HOST_RUN_DIR/metrics/dcgm/` and, for disaggregated experiments, `$HOST_RUN_DIR/metrics/nixl/`. Stop the port-forward with `Ctrl+C` after collection.

### 5. Release the GPUs

Delete the benchmark Job and its Dynamo deployment.

```bash
kubectl delete job "$PERF_JOB" -n "$NAMESPACE" --ignore-not-found
kubectl delete dynamographdeployment "$DGD" -n "$NAMESPACE" --ignore-not-found
kubectl wait -n "$NAMESPACE" --for=delete \
  pod -l "$GRAPH_LABEL" --timeout=15m
```

Final check:

```bash
kubectl get dynamographdeployments -A
kubectl get pods -n "$NAMESPACE"
```


## Copy completed results

Run this from the local workstation after the benchmark has finished:

```bash
export RECIPE=04-disagg-routing-kv-aware
rsync -avhP \
  "<remote ssh name>:/ephemeral/shared/qwen3-32b/perf-cache/artifacts/$RECIPE/" \
  "models/qwen3-32B/artifacts/$RECIPE/"
```

## References

- [Dynamo Qwen3-32B vLLM recipe](https://github.com/ai-dynamo/dynamo/tree/main/recipes/qwen3-32b/vllm/disagg-kv-router)
- [Dynamo native vLLM offloading](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/backends/v-llm/native-kv-offloading)
