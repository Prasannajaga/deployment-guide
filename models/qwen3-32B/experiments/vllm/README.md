# Qwen3-32B vLLM experiments

This README covers all five 16-GPU vLLM experiments.
For each numbered directory, `deploy.yaml` and `perf.yaml` are the source of
truth; some directories also retain optional operational helpers.
Use the common procedure below to run any row.

| Experiment | Worker layout | Router | CPU KV offload |
|---|---|---|---|
| [`01-agg-routing`](./01-agg-routing/) | 8 aggregate engines, TP2 | round-robin | no |
| [`02-disagg-routing`](./02-disagg-routing/) | 6P + 2D, TP2 | round-robin | no |
| [`03-disagg-routing-4p4d`](./03-disagg-routing-4p4d/) | 4P + 4D, TP2 | round-robin | no |
| [`04-disagg-routing-kv-aware`](./04-disagg-routing-kv-aware/) | 4P + 4D, TP2 | KV-aware | no |
| [`05-disagg-routing-kv-aware-offloading`](./05-disagg-routing-kv-aware-offloading/) | 4P + 4D, TP2 | KV-aware | 32 GiB/engine |

All rows use Qwen3-32B-FP8 at the same pinned revision, 40,960-token context,
the Mooncake FAST25 fixed schedule, TTFT 2,000 ms, and ITL 25 ms.

## How to run

Run these commands on the cluster host. Each experiment uses all 16 GPUs, so
finish cleanup before selecting the next `RECIPE`.

### 1. Select and validate

Change only `RECIPE`:

```bash
export NAMESPACE=qwen32-bench
export VLLM_EXP=/ephemeral/shared/qwen3-32b/experiments/vllm
export RECIPE=01-agg-routing
# 02-disagg-routing
# 03-disagg-routing-4p4d
# 04-disagg-routing-kv-aware
# 05-disagg-routing-kv-aware-offloading

export EXP_DIR="$VLLM_EXP/$RECIPE"
export DGD=$(awk '/^metadata:/{m=1; next} m && /^  name:/{print $2; exit}' "$EXP_DIR/deploy.yaml")
export PERF_JOB=$(awk '/^metadata:/{m=1; next} m && /^  name:/{print $2; exit}' "$EXP_DIR/perf.yaml")
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=$DGD"

printf 'recipe=%s\ndeployment=%s\nbenchmark=%s\n' "$RECIPE" "$DGD" "$PERF_JOB"

kubectl get pvc -n "$NAMESPACE" model-cache compilation-cache perf-cache
kubectl get secret -n "$NAMESPACE" hf-token-secret
kubectl get dynamographdeployments -n "$NAMESPACE"
```

All PVCs must be `Bound`, the secret must exist, and no other DGD may be
running. For Exp 2-5, also verify and enable the existing RDMA observability
path:

```bash
if [[ "$RECIPE" != "01-agg-routing" ]]; then
  kubectl get network-attachment-definition qwen-roce -n "$NAMESPACE"
  kubectl get nodes \
    -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,RDMA:.status.allocatable.rdma/ib'
  kubectl apply -n monitoring -f "$VLLM_EXP/nixl-podmonitor.yaml"
fi
```

Exp 5 alone requires this runtime gate:

```bash
if [[ "$RECIPE" == "05-disagg-routing-kv-aware-offloading" ]]; then
  kubectl run -n "$NAMESPACE" vllm-offload-version-check \
    --rm -it --restart=Never \
    --image=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0 -- \
    python3 -c 'import vllm; from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector; from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector; print(vllm.__version__)'
fi
```

### 2. Deploy

```bash
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/perf.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"

kubectl wait -n "$NAMESPACE" \
  --for=jsonpath='{.status.state}'=successful \
  "dynamographdeployment/$DGD" --timeout=45m

kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  -o custom-columns='POD:.metadata.name,NODE:.spec.nodeName,PHASE:.status.phase,READY:.status.containerStatuses[*].ready,RESTARTS:.status.containerStatuses[*].restartCount'
```

Do not benchmark until every deployment Pod is running and ready with zero
unexpected restarts. For Exp 2-5, confirm all eight workers expose NIXL metrics:

```bash
if [[ "$RECIPE" != "01-agg-routing" ]]; then
  kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o json |
    jq -e '[.items[] |
      select(.metadata.labels["nvidia.com/dynamo-sub-component-type"] == "prefill" or
             .metadata.labels["nvidia.com/dynamo-sub-component-type"] == "decode") |
      .spec.containers[].ports[]? |
      select(.name == "nixl-metrics")
    ] | length == 8'
fi
```

### 3. Benchmark

```bash
kubectl delete job "$PERF_JOB" -n "$NAMESPACE" --ignore-not-found
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/perf.yaml"

kubectl wait -n "$NAMESPACE" --for=condition=Ready \
  pod -l "app=$PERF_JOB" --timeout=5m
kubectl logs -n "$NAMESPACE" -f "job/$PERF_JOB"
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  "job/$PERF_JOB" --timeout=3h

export RUN_DIR=$(kubectl logs -n "$NAMESPACE" "job/$PERF_JOB" |
  sed -n 's/^Run artifacts: //p' | tail -1)
export HOST_RUN_DIR="/ephemeral/shared/qwen3-32b/perf-cache/${RUN_DIR#/perf-cache/}"

test -f "$HOST_RUN_DIR/manifest.json"
gzip -t "$HOST_RUN_DIR/aiperf/profile_export.jsonl.gz"
printf 'artifacts=%s\n' "$HOST_RUN_DIR"
```

### 4. Metrics, copy, and cleanup

Before deleting the deployment, keep this running in a second terminal:

```bash
kubectl port-forward -n monitoring \
  service/monitoring-kube-prometheus-prometheus 9090:9090
```

Then collect the post-run DCGM/NIXL series:

```bash
PROMETHEUS_URL=http://127.0.0.1:9090 \
  bash "$VLLM_EXP/collect-metrics.sh" "$HOST_RUN_DIR"
```

From the local workstation, copy the compact run artifacts:

```bash
export RECIPE=01-agg-routing  # use the recipe that just finished
rsync -az \
  "lambda-gpu05:/ephemeral/shared/qwen3-32b/perf-cache/artifacts/$RECIPE/" \
  "models/qwen3-32B/artifacts/$RECIPE/"
```

Finally release the GPUs:

```bash
kubectl delete job "$PERF_JOB" -n "$NAMESPACE" --ignore-not-found
kubectl delete dynamographdeployment "$DGD" -n "$NAMESPACE" --ignore-not-found
kubectl wait -n "$NAMESPACE" --for=delete \
  pod -l "$GRAPH_LABEL" --timeout=15m
```

Repeat from step 1 with the next recipe. Raw request results are retained as
`aiperf/profile_export.jsonl.gz`; processed data and plots are generated
locally, not by `perf.yaml`.
