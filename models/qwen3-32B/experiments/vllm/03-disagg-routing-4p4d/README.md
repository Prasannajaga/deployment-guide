# Exp 3: disaggregated 4P+4D, round-robin

4P+4D topology follow-up to Exp 2.

| Pool | Replicas | TP | GPUs |
|---|---:|---:|---:|
| Prefill | 4 | 2 | 8 |
| Decode | 4 | 2 | 8 |

## How to run

```bash
export NAMESPACE=qwen32-bench
export VLLM_EXP=/ephemeral/shared/qwen3-32b/experiments/vllm
export RECIPE=03-disagg-routing-4p4d
export DGD=qwen3-32b-fp8-vllm-disagg-routing-4p4d
export PERF_JOB=${DGD}-perf

kubectl apply -n "$NAMESPACE" -f "$VLLM_EXP/$RECIPE/deploy.yaml"
kubectl wait -n "$NAMESPACE" \
  --for=jsonpath='{.status.state}'=successful \
  "dynamographdeployment/$DGD" --timeout=45m
kubectl get pods -n "$NAMESPACE" \
  -l "nvidia.com/dynamo-graph-deployment-name=$DGD" \
  -o wide

kubectl get --raw \
  "/api/v1/namespaces/$NAMESPACE/services/http:$DGD-frontend:8000/proxy/v1/models" |
  jq -e '.data[] | select(.id == "Qwen/Qwen3-32B-FP8")'

kubectl delete job "$PERF_JOB" -n "$NAMESPACE" --ignore-not-found
kubectl apply -n "$NAMESPACE" -f "$VLLM_EXP/$RECIPE/perf.yaml"
kubectl wait -n "$NAMESPACE" --for=condition=Ready \
  pod -l "app=$PERF_JOB" --timeout=5m
kubectl logs -n "$NAMESPACE" -f "job/$PERF_JOB"
kubectl get job "$PERF_JOB" -n "$NAMESPACE"
```

Deployment: `qwen3-32b-fp8-vllm-disagg-routing-4p4d`

Artifacts: `03-disagg-routing-4p4d`

The Job writes AIPerf output and a small `manifest.json` under `/perf-cache/artifacts/03-disagg-routing-4p4d/<run-id>/`.

The manifest records the exact Prometheus query window. DCGM and NIXL are
already scraped by Prometheus. Inspect them after the run with the shared
[`metrics.promql`](../metrics.promql).
