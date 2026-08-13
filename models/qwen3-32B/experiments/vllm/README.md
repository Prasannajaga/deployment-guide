# Qwen3-32B vLLM experiments

This README covers 16-GPU vLLM experiments: Exp 3, 4, and 5.
Each experiment directory contains only its deployment, benchmark, and
experiment-specific notes. Use the common procedure below to run any row.

| Experiment | Topology | Router | CPU KV offload |
|---|---|---|---|
| [`03-disagg-routing-4p4d`](./03-disagg-routing-4p4d/) | 4P + 4D, TP2 | round-robin | no |
| [`04-disagg-routing-kv-aware`](./04-disagg-routing-kv-aware/) | 4P + 4D, TP2 | KV-aware | no |
| [`05-disagg-routing-kv-aware-offloading`](./05-disagg-routing-kv-aware-offloading/) | 4P + 4D, TP2 | KV-aware | 32 GiB/engine |

All rows use Qwen3-32B-FP8 at the same pinned revision, 40,960-token context,
the Mooncake FAST25 fixed schedule, TTFT 2,000 ms, and ITL 25 ms.

## One-time setup

Set the directory used on the benchmark cluster and verify the shared
prerequisites:

```bash
export NAMESPACE=qwen32-bench
export VLLM_EXP=/ephemeral/shared/qwen3-32b/experiments/vllm

kubectl get crd dynamographdeployments.nvidia.com
kubectl get pvc model-cache perf-cache -n "$NAMESPACE"
kubectl get nodes \
  -L kubernetes.io/hostname \
  -o custom-columns=NAME:.metadata.name,HOSTNAME:.metadata.labels.kubernetes\.io/hostname,GPU:.status.allocatable.nvidia\.com/gpu,RDMA:.status.allocatable.rdma/ib
```

Apply the shared NIXL PodMonitor once before running experiments 3-5:

```bash
kubectl apply -n "$NAMESPACE" -f "$VLLM_EXP/nixl-podmonitor.yaml"
```

## Select an experiment

Set only `RECIPE`. The remaining names are read from the selected manifests,
so they do not need to be copied manually from a README:

```bash
export RECIPE=03-disagg-routing-4p4d
export EXP_DIR="$VLLM_EXP/$RECIPE"

export DGD=$(
  awk '/^metadata:/{metadata=1; next} metadata && /^  name:/{print $2; exit}' \
    "$EXP_DIR/deploy.yaml"
)
export PERF_JOB=$(
  awk '/^metadata:/{metadata=1; next} metadata && /^  name:/{print $2; exit}' \
    "$EXP_DIR/perf.yaml"
)
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=$DGD"

printf 'recipe=%s\ndeployment=%s\nbenchmark=%s\n' \
  "$RECIPE" "$DGD" "$PERF_JOB"
```

Valid `RECIPE` values are the three directory names in the table above.

## Deploy

Validate the manifest on the API server, apply it, and wait for the Dynamo
operator to report success:

```bash
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl wait -n "$NAMESPACE" \
  --for=jsonpath='{.status.state}'=successful \
  "dynamographdeployment/$DGD" --timeout=45m
```

Before benchmarking, verify readiness, restarts, placement, and GPU requests:

```bash
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  -L nvidia.com/dynamo-component-type \
  -L nvidia.com/dynamo-sub-component-type \
  -o custom-columns=POD:.metadata.name,ROLE:.metadata.labels.nvidia\.com/dynamo-sub-component-type,NODE:.spec.nodeName,PHASE:.status.phase,READY:.status.containerStatuses[*].ready,RESTARTS:.status.containerStatuses[*].restartCount,GPU:.spec.containers[*].resources.requests.nvidia\.com/gpu
```

Inspect recent logs if a Pod is not healthy:

```bash
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --all-containers --prefix --tail=300
```

## Smoke test

Verify that all eight workers expose the NIXL metrics
port:

```bash
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o json |
  jq -e '[.items[] |
    select(
      .metadata.labels["nvidia.com/dynamo-sub-component-type"] == "prefill" or
      .metadata.labels["nvidia.com/dynamo-sub-component-type"] == "decode"
    ) |
    .spec.containers[].ports[]? |
    select(.name == "nixl-metrics")
  ] | length == 8'
```

For a KV-aware experiment, also confirm after a few requests that the frontend
has started applying stored KV events. A zero value means the KV router index
is still empty even if serving itself works.

## Run the benchmark

The selected `perf.yaml` defines the workload, SLOs, trace filtering, and
artifact path:

```bash
kubectl delete job "$PERF_JOB" -n "$NAMESPACE" --ignore-not-found
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/perf.yaml"
kubectl wait -n "$NAMESPACE" --for=condition=Ready \
  pod -l "app=$PERF_JOB" --timeout=5m
kubectl logs -n "$NAMESPACE" -f "job/$PERF_JOB"
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  "job/$PERF_JOB" --timeout=3h
```

The Job prints the final run directory. Results are stored below
`/perf-cache/artifacts/<experiment-id>/<run-id>/`; retain `manifest.json` and
the AIPerf exports. The manifest start/end timestamps define the Prometheus
query window.

For interactive Prometheus queries:

```bash
kubectl port-forward -n monitoring \
  service/monitoring-kube-prometheus-prometheus 9090:9090
```

Open `http://127.0.0.1:9090` and use the queries in [`metrics.promql`](./metrics.promql).

## Clean up

Remove the benchmark and deployment before selecting the next row:

```bash
kubectl delete job "$PERF_JOB" -n "$NAMESPACE" --ignore-not-found
kubectl delete dynamographdeployment "$DGD" \
  -n "$NAMESPACE" --ignore-not-found
kubectl wait -n "$NAMESPACE" --for=delete \
  pod -l "$GRAPH_LABEL" --timeout=15m
```

Deploy the next experiment only after until the previous worker Pods have gone and GPUs are free.

## Exp 5 compatibility gate

Experiment 5 uses vLLM `MultiConnector(NixlConnector + OffloadingConnector)`.
Before applying it, inspect the actual runtime image:

```bash
kubectl run -n "$NAMESPACE" vllm-offload-version-check \
  --rm -it --restart=Never \
  --image=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0 -- \
  python3 -c 'import vllm; from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector; from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector; print(vllm.__version__)'
```

The run might be blocked if the required connector imports fail.
