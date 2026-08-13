# TP1-attention + EP2 KV-transfer metrics

This recipe exposes three independent Prometheus endpoints:

| Endpoint | Port | Expected families |
|---|---:|---|
| Dynamo frontend | 8000 | `dynamo_frontend_*`, KV-router metrics |
| Dynamo/SGLang worker | 9090 | `dynamo_component_*`, `sglang:*` |
| NIXL telemetry | 19090 | `nixl_*` transfer metrics |

NIXL metrics are populated lazily. An empty NIXL endpoint before the first
prefill-to-decode transfer is not a failure. Pod readiness is also not proof
that KV data crossed NIXL; require a completed request and increasing transfer
counters.

## Variables

```bash
export NAMESPACE=qwen32-bench
export DEPLOYMENT=q36-sgl-pd-tp1ep2-4p4d
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
export FRONTEND_SERVICE="${DEPLOYMENT}-frontend"
export MODEL=Qwen/Qwen3.6-35B-A3B-FP8
```

## Verify discovery

Dynamo Operator 1.3.0 creates the application PodMonitor and injects the
`system` and `nixl` container ports. Confirm the Prometheus Operator CRDs,
monitor, labels, and ports:

```bash
kubectl api-resources | grep -E 'podmonitors|servicemonitors'
kubectl get podmonitor -A
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  -o custom-columns='POD:.metadata.name,TYPE:.metadata.labels.nvidia\.com/dynamo-component-type,METRICS:.metadata.labels.nvidia\.com/metrics-enabled,PORTS:.spec.containers[0].ports[*].name'
```

Every managed Pod should have `nvidia.com/metrics-enabled=true`. Workers
should expose `system` and `nixl`; the frontend should expose its HTTP
metrics endpoint.

If the monitor exists but Prometheus does not select it, inspect selectors:

```bash
kubectl get prometheus -A -o yaml |
  grep -A8 -E 'podMonitorNamespaceSelector|podMonitorSelector'
```

Do not add a second PodMonitor until confirming the operator-created monitor
is absent. Duplicate monitors double-scrape counters and distort rates.

## Scrape every endpoint directly

The commands below execute inside each Pod, avoiding temporary Services and
verifying the actual listener.

```bash
frontend_pod="$(
  kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
    -l 'nvidia.com/dynamo-component-type=frontend' \
    -o jsonpath='{.items[0].metadata.name}'
)"

kubectl exec -n "$NAMESPACE" "$frontend_pod" -- \
  python3 -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=5).read().decode())' |
  grep -E '^dynamo_' | head -40

for pod in $(kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  -l 'nvidia.com/dynamo-component-type=worker' \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'); do
  echo "===== $pod: 9090 ====="
  kubectl exec -n "$NAMESPACE" "$pod" -- \
    python3 -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:9090/metrics", timeout=5).read().decode())' |
    grep -E '^(sglang:|dynamo_)' | head -40

  echo "===== $pod: 19090 ====="
  kubectl exec -n "$NAMESPACE" "$pod" -- \
    python3 -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:19090/metrics", timeout=5).read().decode())' |
    grep -E '^nixl_' | head -40
done
```

Expected worker evidence includes SGLang metrics on port 9090. After traffic,
the NIXL endpoint should include transfer families such as
`nixl_bytes_transferred_count`; the failed-transfer counter
`nixl_num_failed_transfers_total` must remain zero. Inspect the endpoint
instead of assuming every NIXL build exposes an identical complete family set.

## Record before/after NIXL snapshots

```bash
export METRIC_SNAPSHOT_DIR="/tmp/${DEPLOYMENT}-metrics"
mkdir -p "$METRIC_SNAPSHOT_DIR"

for phase in before after; do
  if [ "$phase" = after ]; then
    kubectl run qwen-kv-transfer-smoke --rm -i --restart=Never \
      --namespace "$NAMESPACE" --image=curlimages/curl -- \
      curl -fsS -H 'Content-Type: application/json' \
      --data-binary "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain in 256 tokens why a KV-aware prefill/decode router benefits repeated prefixes.\"}],\"temperature\":0,\"max_tokens\":256,\"stream\":false}" \
      "http://${FRONTEND_SERVICE}:8000/v1/chat/completions"
  fi

  for pod in $(kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
    -l 'nvidia.com/dynamo-component-type=worker' \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'); do
    kubectl exec -n "$NAMESPACE" "$pod" -- \
      python3 -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:19090/metrics", timeout=5).read().decode())' \
      > "${METRIC_SNAPSHOT_DIR}/${pod}-nixl-${phase}.prom"
  done
done

grep -H -E '^nixl_(bytes_transferred|num_failed_transfers)' \
  "$METRIC_SNAPSHOT_DIR"/*.prom
```

The transfer count must increase on at least the participating prefill/decode
pair. Any increase in failed transfers fails the acceptance gate. Preserve
the snapshots with the benchmark artifacts.

## Query cluster Prometheus

The existing cluster normally exposes Prometheus as:

```bash
export PROM_NAMESPACE=monitoring
export PROM_SERVICE=monitoring-kube-prometheus-prometheus
export PROM_SERVICE_PORT=9090
```

Reconfirm before use:

```bash
kubectl get svc -A \
  -o custom-columns='NAMESPACE:.metadata.namespace,NAME:.metadata.name,PORTS:.spec.ports[*].port' |
  grep -i prometheus
```

Port-forward it in a separate terminal:

```bash
kubectl port-forward -n "$PROM_NAMESPACE" "service/$PROM_SERVICE" \
  "9095:$PROM_SERVICE_PORT" --address 127.0.0.1
```

Then validate readiness and scrape targets:

```bash
export PROM=http://127.0.0.1:9095
curl -fsS "$PROM/-/ready"

curl -fsS "$PROM/api/v1/targets" |
jq -r --arg namespace "$NAMESPACE" --arg deployment "$DEPLOYMENT" '
  .data.activeTargets[]
  | select(
      (.labels.namespace // "") == $namespace
      and (
        ((.labels.pod // "") | contains($deployment))
        or ((.discoveredLabels.__meta_kubernetes_pod_label_nvidia_com_dynamo_graph_deployment_name // "") == $deployment)
      )
    )
  | [
      (.health // ""),
      (.labels.pod // ""),
      (.scrapeUrl // ""),
      (.lastError // "")
    ]
  | @tsv
'
```

Require every 8000, 9090, and 19090 target to report `up` with an empty
`lastError`.

Discover the exact NIXL and SGLang series retained by this Prometheus:

```bash
curl -fsS -G "$PROM/api/v1/series" \
  --data-urlencode 'match[]={__name__=~"nixl_.*|sglang:.*|dynamo_.*"}' \
  --data-urlencode "start=$(date -u -d '15 minutes ago' +%s)" \
  --data-urlencode "end=$(date -u +%s)" |
jq -r --arg namespace "$NAMESPACE" '
  .data[]
  | select((.namespace // "") == $namespace)
  | [.__name__, (.pod // ""), (.job // "")]
  | @tsv
' | sort -u
```

Example instant checks:

```bash
curl -fsS -G "$PROM/api/v1/query" \
  --data-urlencode "query=sum by (pod) (nixl_bytes_transferred_count{namespace=\"${NAMESPACE}\"})" |
jq

curl -fsS -G "$PROM/api/v1/query" \
  --data-urlencode "query=sum by (pod) (nixl_num_failed_transfers_total{namespace=\"${NAMESPACE}\"})" |
jq

curl -fsS -G "$PROM/api/v1/query" \
  --data-urlencode "query=sum by (pod) (sglang:prompt_tokens_total{namespace=\"${NAMESPACE}\"})" |
jq
```

If an instant query is empty, use the series-discovery output to confirm the
metric name and labels emitted by the pinned image. Do not rename metrics in
the recipe to make a guessed query work.

## Export a benchmark window

```bash
export END_EPOCH="$(date -u +%s)"
export START_EPOCH="$(date -u -d '30 minutes ago' +%s)"
export OUT_DIR="/ephemeral/shared/qwen3.6-35b-a3b/sglang/disagg/tp1-ep2-4p4d/metrics"
mkdir -p "$OUT_DIR"

for metric in \
  nixl_bytes_transferred_count \
  nixl_num_failed_transfers_total \
  'sglang:prompt_tokens_total' \
  'sglang:generation_tokens_total'; do
  safe_name="$(printf '%s' "$metric" | tr ':/' '__')"
  curl -fsS -G "$PROM/api/v1/query_range" \
    --data-urlencode "query=${metric}{namespace=\"${NAMESPACE}\"}" \
    --data-urlencode "start=$START_EPOCH" \
    --data-urlencode "end=$END_EPOCH" \
    --data-urlencode 'step=15s' \
    -o "$OUT_DIR/${safe_name}.json"
  jq -e '.status == "success"' "$OUT_DIR/${safe_name}.json" >/dev/null
done
```

Also preserve frontend `/metrics`, worker `9090/metrics`, NIXL
`19090/metrics`, startup logs, and the exact query window alongside AIPerf
results.

References:

- [Dynamo metrics and NIXL telemetry](https://docs.nvidia.com/dynamo/latest/user-guides/observability-local/metrics)
- [Dynamo SGLang observability](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/backends/sg-lang/observability)
- [Dynamo Kubernetes observability](https://docs.nvidia.com/dynamo/latest/kubernetes-deployment/operate/observability/metrics)
