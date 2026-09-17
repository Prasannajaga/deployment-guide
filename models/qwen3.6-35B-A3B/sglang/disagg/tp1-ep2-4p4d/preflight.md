# SGLang KV Offload & NIXL Preflight Runbook

This 4-GPU canary verifies SGLang prefill CPU KV-offloading, NIXL state transfer over UCX/RDMA, and Prometheus/Grafana metric telemetry before applying the 16-GPU production [`deploy-kv-offloading.yaml`](deploy-kv-offloading.yaml).

The verified configuration enables CPU KV offload only on the prefill worker. Decode-side KV offload and the NIXL HiCache storage backend are not supported for this hybrid attention/Mamba model in the pinned SGLang runtime.

---

## 1. Shared Variables & Preflight Configuration

Run these commands from a Kubernetes administrator host:

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3.6-35b-a3b/sglang/disagg/tp1-ep2-4p4d
export PREFLIGHT_DEPLOYMENT=q36-sgl-pd-tp1ep2-4p4d-pf
export PREFLIGHT_GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${PREFLIGHT_DEPLOYMENT}"
export MODEL=Qwen/Qwen3.6-35B-A3B-FP8
export MONITOR_NAMESPACE=monitoring
export PREFLIGHT_PODMONITOR=q36-sgl-pd-tp1ep2-4p4d-pf-metrics
```

---

## 2. Preflight Manifest Generation & Resource Checks

### 2.1 Generate the 4-GPU Canary Manifest (`preflight.yaml`)

Generate `$EXP_DIR/preflight.yaml` from `$EXP_DIR/deploy-kv-offloading.yaml` by scaling down replica counts (1 Frontend + 1 Prefill + 1 Decode worker = 4 GPUs total):

```bash
mkdir -p "$EXP_DIR"

sed \
  -e 's/^  name: q36-sgl-pd-tp1ep2-4p4d$/  name: q36-sgl-pd-tp1ep2-4p4d-pf/' \
  -e 's/^      replicas: 6$/      replicas: 1/' \
  -e 's/^      replicas: 4$/      replicas: 1/' \
  "$EXP_DIR/deploy-kv-offloading.yaml" |
tee "$EXP_DIR/preflight.yaml" >/dev/null
```

### 2.2 Verify Storage, RoCE Network & Node Capacity

Confirm that shared storage, RoCE network attachment, GPU nodes, and the preflight manifest exist before allocating GPUs:

```bash
kubectl get pvc model-cache perf-cache -n "$NAMESPACE"
kubectl get network-attachment-definition qwen-roce -n "$NAMESPACE"
kubectl get nodes -L qwen.nvidia.com/role \
  -o custom-columns='NODE:.metadata.name,ROLE:.metadata.labels.qwen\.nvidia\.com/role,GPU:.status.allocatable.nvidia\.com/gpu,RDMA:.status.allocatable.rdma/ib'
kubectl get dynamographdeployments.nvidia.com -A

kubectl apply --dry-run=server -n "$NAMESPACE" \
  -f "$EXP_DIR/preflight.yaml"
```

> [!IMPORTANT]
> Both PVCs must be `Bound`, `qwen-roce` must exist, and 2 available GPUs + 2 `rdma/ib` resources must be allocatable per role node.

---

## 3. Deploy Canary & Verify Pod Startup

### 3.1 Apply Single-Replica Canary

```bash
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/preflight.yaml"

kubectl wait -n "$NAMESPACE" --for=condition=Ready pod \
  -l "$PREFLIGHT_GRAPH_LABEL" --timeout=1800s

kubectl get pods -n "$NAMESPACE" -l "$PREFLIGHT_GRAPH_LABEL" \
  -L nvidia.com/dynamo-component-type,nvidia.com/dynamo-sub-component-type \
  -o wide
```

If readiness times out, collect diagnostics and stop before sending traffic:
```bash
kubectl get pods -n "$NAMESPACE" -l "$PREFLIGHT_GRAPH_LABEL" -o wide
kubectl describe pods -n "$NAMESPACE" -l "$PREFLIGHT_GRAPH_LABEL"
kubectl logs -n "$NAMESPACE" -l "$PREFLIGHT_GRAPH_LABEL" \
  --all-containers --prefix --tail=1500
```

### 3.2 Startup Log Gate Check

Reject known unsupported hybrid-pool, scheduler, NIXL, and OOM failure signatures before sending requests:

```bash
kubectl logs -n "$NAMESPACE" -l "$PREFLIGHT_GRAPH_LABEL" \
  --all-containers --prefix --tail=1500 |
  tee "$EXP_DIR/qwen36-sglang-preflight-startup.log"

grep -Ei 'traceback|assertionerror|notimplementederror|nixl_err|out of memory|does not support|unsupported' \
  "$EXP_DIR/qwen36-sglang-preflight-startup.log"
```
*(No output from the final `grep` is the expected result. If it prints a fatal signature, stop).*

### 3.3 Resolve Canary Pod Variables

```bash
frontend_pod="$(kubectl get pods -n "$NAMESPACE" \
  -l "$PREFLIGHT_GRAPH_LABEL,nvidia.com/dynamo-component-type=frontend" \
  -o jsonpath='{.items[0].metadata.name}')"
prefill_pod="$(kubectl get pods -n "$NAMESPACE" \
  -l "$PREFLIGHT_GRAPH_LABEL,nvidia.com/dynamo-sub-component-type=prefill" \
  -o jsonpath='{.items[0].metadata.name}')" 
worker_pods="$(kubectl get pods -n "$NAMESPACE" \
  -l "$PREFLIGHT_GRAPH_LABEL" \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' |
  grep -Ei 'sglang(prefill|decode)worker')"

printf 'FRONTEND=%s\nPREFILL=%s\nWORKERS=%s\n' \
  "$frontend_pod" "$prefill_pod" "$worker_pods"
```

All three variables must be nonempty before continuing.

---

## 4. Functional Validation & Direct Metric Scrapes

### 4.1 In-Cluster NIXL State Transfer Test

Send a long prompt request to trigger prefill state transfer to the decode worker:

```bash
timeout 330s kubectl exec -n "$NAMESPACE" "$frontend_pod" -- \
  env "MODEL=$MODEL" \
  python3 -c '
import json
import os
import urllib.request

body = json.dumps({
    "model": os.environ["MODEL"],
    "messages": [{
        "role": "user",
        "content": (
            "This is a long NIXL prefill-to-decode transfer verification prefix. "
            * 256
        ) + "\nReply with exactly: nixl-transfer-ok"
    }],
    "temperature": 0,
    "max_tokens": 16,
    "stream": False
}).encode()

request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urllib.request.urlopen(request, timeout=300) as response:
    result = json.load(response)

print(json.dumps({
    "response": result["choices"][0]["message"]["content"],
    "usage": result.get("usage", {})
}, indent=2))
'
```

### 4.2 Direct Worker NIXL Metric Scrape (Port 19090)

```bash
for pod in $worker_pods; do
  echo "===== NIXL TRANSFER METRICS: $pod ====="

  kubectl exec -n "$NAMESPACE" "$pod" -- python3 -c '
import urllib.request

metrics = urllib.request.urlopen(
    "http://127.0.0.1:19090/metrics", timeout=10
).read().decode()

prefixes = (
    "agent_tx_bytes",
    "agent_rx_bytes",
    "agent_tx_requests_num",
    "agent_rx_requests_num",
    "agent_xfer_time",
    "agent_xfer_post_time",
    "agent_errors",
)

selected = [
    line for line in metrics.splitlines()
    if line and not line.startswith("#") and line.startswith(prefixes)
]

print("\n".join(selected) if selected else "NO_AGENT_TRANSFER_METRICS")
'
done
```
*(Accept NIXL transfer when at least one worker reports both `agent_tx_bytes_total > 0` and `agent_tx_requests_num_total > 0` with zero `agent_errors_total`).*

### 4.3 Record Before & After NIXL Metric Snapshots

```bash
export METRIC_SNAPSHOT_DIR="$EXP_DIR/metrics/${PREFLIGHT_DEPLOYMENT}-snapshots"
mkdir -p "$METRIC_SNAPSHOT_DIR"

# Snapshot BEFORE request
for pod in $worker_pods; do
  kubectl exec -n "$NAMESPACE" "$pod" -- python3 -c '
import urllib.request
print(urllib.request.urlopen(
    "http://127.0.0.1:19090/metrics", timeout=10
).read().decode())
' > "$METRIC_SNAPSHOT_DIR/${pod}-nixl-before.prom"
done

# Send verification request
timeout 330s kubectl exec -n "$NAMESPACE" "$frontend_pod" -- \
  env "MODEL=$MODEL" \
  python3 -c '
import json
import os
import urllib.request

body = json.dumps({
    "model": os.environ["MODEL"],
    "messages": [{
        "role": "user",
        "content": (
            "NIXL before and after transfer snapshot verification prefix. " * 256
        ) + "\nReply with exactly: nixl-snapshot-ok"
    }],
    "temperature": 0,
    "max_tokens": 16,
    "stream": False
}).encode()

request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urllib.request.urlopen(request, timeout=300) as response:
    result = json.load(response)

print(json.dumps({
    "response": result["choices"][0]["message"]["content"],
    "usage": result.get("usage", {})
}, indent=2))
'

# Snapshot AFTER request
for pod in $worker_pods; do
  kubectl exec -n "$NAMESPACE" "$pod" -- python3 -c '
import urllib.request
print(urllib.request.urlopen(
    "http://127.0.0.1:19090/metrics", timeout=10
).read().decode())
' > "$METRIC_SNAPSHOT_DIR/${pod}-nixl-after.prom"
done

grep -H -E '^agent_(tx|rx|xfer|errors)' \
  "$METRIC_SNAPSHOT_DIR"/*.prom
```

### 4.4 CPU HiCache Offload Verification (Port 9090)

```bash
kubectl exec -n "$NAMESPACE" "$prefill_pod" -- python3 -c '
import urllib.request

metrics = urllib.request.urlopen(
    "http://127.0.0.1:9090/metrics", timeout=10
).read().decode()

selected = [
    line for line in metrics.splitlines()
    if line.startswith("sglang:hicache_host_total_tokens")
    or line.startswith("sglang:hicache_host_used_tokens")
]

print("\n".join(selected) if selected else "NO_HICACHE_HOST_METRICS")
'
```
*(Accept CPU KV offload when at least one rank reports `sglang:hicache_host_total_tokens > 0` and `sglang:hicache_host_used_tokens > 0`).*

### 4.5 Prompt Cache Reporting Verification

```bash
timeout 330s kubectl exec -n "$NAMESPACE" "$frontend_pod" -- \
  env "MODEL=$MODEL" \
  python3 -c '
import json
import os
import urllib.request

endpoint = "http://127.0.0.1:8000/v1/chat/completions"

def send(messages):
    body = json.dumps({
        "model": os.environ["MODEL"],
        "messages": messages,
        "temperature": 0,
        "max_tokens": 32,
        "stream": False,
        "return_cached_tokens_details": True,
    }).encode()
    request = urllib.request.Request(
        endpoint, body, {"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)

prefix = "KV offload verification prefix with stable content. " * 128
turn1 = [
    {"role": "system", "content": prefix},
    {"role": "user", "content": "Reply with exactly: cache-warm-ok"},
]
first = send(turn1)
turn2 = turn1 + [
    {"role": "assistant", "content": first["choices"][0]["message"]["content"]},
    {"role": "user", "content": "Reply with exactly: cache-report-ok"},
]
second = send(turn2)
details = second.get("usage", {}).get("prompt_tokens_details") or {}

print(json.dumps({
    "status": "CACHE_REPORT_OK" if details.get("cached_tokens", 0) > 0 else "CACHE_REPORT_MISSING",
    "cached_tokens": details.get("cached_tokens", 0),
    "cache_sources": ((second.get("sglext") or {}).get("cached_tokens_details") or {}),
}, indent=2))
'
```
*(Require `status` to be `CACHE_REPORT_OK` and `cached_tokens > 0`).*

---

## 5. Prometheus & Grafana Telemetry Validation

### 5.1 Check PodMonitor Discovery & Deploy Canary PodMonitor

```bash
export PROMETHEUS_NAME=monitoring-kube-prometheus-prometheus

# List worker ports
kubectl get pods -n "$NAMESPACE" -l "$PREFLIGHT_GRAPH_LABEL" \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{range .spec.containers[*].ports[*]}{"  "}{.name}{": "}{.containerPort}{"\n"}{end}{end}'

# Inspect Prometheus selector
kubectl get prometheus -n "$MONITOR_NAMESPACE" "$PROMETHEUS_NAME" \
  -o json |
jq '{podMonitorSelector: .spec.podMonitorSelector, podMonitorNamespaceSelector: .spec.podMonitorNamespaceSelector}'

# List active PodMonitors
kubectl get podmonitor -A -o json |
jq -r '
  .items[]
  | [
      .metadata.namespace,
      .metadata.name,
      (.metadata.labels | tostring),
      (.spec.namespaceSelector | tostring),
      (.spec.selector | tostring),
      ([.spec.podMetricsEndpoints[]? | {port, path, interval}] | tostring)
    ]
  | @tsv
'

kubectl get svc -A \
  -o custom-columns='NAMESPACE:.metadata.namespace,NAME:.metadata.name,PORTS:.spec.ports[*].port' |
  grep -Ei 'grafana|prometheus'
```

If no existing monitor matches, deploy the canary PodMonitor `$EXP_DIR/preflight-podmonitor.yaml`:

```bash
tee "$EXP_DIR/preflight-podmonitor.yaml" <<'EOF'
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: q36-sgl-pd-tp1ep2-4p4d-pf-metrics
  namespace: monitoring
  labels:
    release: monitoring
spec:
  namespaceSelector:
    matchNames:
      - qwen32-bench
  selector:
    matchExpressions:
      - key: nvidia.com/dynamo-graph-deployment-name
        operator: In
        values:
          - q36-sgl-pd-tp1ep2-4p4d-pf
      - key: nvidia.com/dynamo-sub-component-type
        operator: In
        values:
          - prefill
          - decode
  podMetricsEndpoints:
    - port: nixl
      path: /metrics
      interval: 5s
      scrapeTimeout: 4s
    - port: system
      path: /metrics
      interval: 5s
      scrapeTimeout: 4s
EOF

kubectl apply --dry-run=server -f "$EXP_DIR/preflight-podmonitor.yaml"
kubectl apply -f "$EXP_DIR/preflight-podmonitor.yaml"
```

### 5.2 Port-Forward Grafana & PromQL Dashboard Queries

```bash
export GRAFANA_NAMESPACE=monitoring
export GRAFANA_SERVICE=monitoring-grafana
export GRAFANA_SERVICE_PORT="$(kubectl get service -n "$GRAFANA_NAMESPACE" \
  "$GRAFANA_SERVICE" -o jsonpath='{.spec.ports[0].port}')"

printf 'Grafana service: %s/%s port %s\n' \
  "$GRAFANA_NAMESPACE" "$GRAFANA_SERVICE" "$GRAFANA_SERVICE_PORT"

# Keep port-forward running in a background / separate terminal
kubectl port-forward -n "$GRAFANA_NAMESPACE" \
  "service/$GRAFANA_SERVICE" "3000:$GRAFANA_SERVICE_PORT" \
  --address 127.0.0.1
```

#### Recommended Grafana Dashboard Panels

1. **Panel 1: NIXL Transmitted Bytes** (Unit: bytes IEC)
   ```promql
   sum by (pod) (
     agent_tx_bytes_total{
       namespace="qwen32-bench",
       pod=~"q36-sgl-pd-tp1ep2-4p4d-pf.*"
     }
   )
   ```

2. **Panel 2: NIXL Transmit Throughput** (Unit: bytes/sec IEC)
   ```promql
   sum by (pod) (
     rate(agent_tx_bytes_total{
       namespace="qwen32-bench",
       pod=~"q36-sgl-pd-tp1ep2-4p4d-pf.*"
     }[1m])
   )
   ```

3. **Panel 3: NIXL Transfer Requests** (Unit: requests/sec)
   ```promql
   sum by (pod) (
     rate(agent_tx_requests_num_total{
       namespace="qwen32-bench",
       pod=~"q36-sgl-pd-tp1ep2-4p4d-pf.*"
     }[1m])
   )
   ```

4. **Panel 4: NIXL Average Transfer Time** (Unit: microseconds)
   ```promql
   sum by (pod) (
     rate(agent_xfer_time_total{
       namespace="qwen32-bench",
       pod=~"q36-sgl-pd-tp1ep2-4p4d-pf.*"
     }[5m])
   )
   /
   sum by (pod) (
     rate(agent_tx_requests_num_total{
       namespace="qwen32-bench",
       pod=~"q36-sgl-pd-tp1ep2-4p4d-pf.*"
     }[5m])
   )
   ```

5. **Panel 5: NIXL Errors** (Stat panel)
   ```promql
   sum by (pod) (
     agent_errors_total{
       namespace="qwen32-bench",
       pod=~"q36-sgl-pd-tp1ep2-4p4d-pf.*"
     }
   )
   ```

6. **Panel 6: CPU HiCache Capacity** (Stat panel)
   ```promql
   sum by (pod) (
     sglang:hicache_host_total_tokens{
       namespace="qwen32-bench",
       pod=~"q36-sgl-pd-tp1ep2-4p4d-pf.*sglangprefillworker.*"
     }
   )
   ```

7. **Panel 7: CPU HiCache Remaining Tokens** (Stat panel)
   ```promql
   sum by (pod) (
     sglang:hicache_host_total_tokens{
       namespace="qwen32-bench",
       pod=~"q36-sgl-pd-tp1ep2-4p4d-pf.*sglangprefillworker.*"
     }
   )
   -
   sum by (pod) (
     sglang:hicache_host_used_tokens{
       namespace="qwen32-bench",
       pod=~"q36-sgl-pd-tp1ep2-4p4d-pf.*sglangprefillworker.*"
     }
   )
   ```

8. **Panel 8: CPU HiCache Utilization** (Gauge panel, Unit: percent 0-100)
   ```promql
   100 *
   sum by (pod) (
     sglang:hicache_host_used_tokens{
       namespace="qwen32-bench",
       pod=~"q36-sgl-pd-tp1ep2-4p4d-pf.*sglangprefillworker.*"
     }
   )
   /
   clamp_min(
     sum by (pod) (
       sglang:hicache_host_total_tokens{
         namespace="qwen32-bench",
         pod=~"q36-sgl-pd-tp1ep2-4p4d-pf.*sglangprefillworker.*"
       }
     ),
     1
   )
   ```

### 5.3 Prometheus In-Cluster API Direct Scrape Check

Verify Prometheus scrape targets and query engine directly:

```bash
export PROM_NAMESPACE=monitoring
export PROM_SERVICE=monitoring-kube-prometheus-prometheus
export PROM_SERVICE_PORT="$(kubectl get service -n "$PROM_NAMESPACE" \
  "$PROM_SERVICE" -o jsonpath='{.spec.ports[0].port}')"

printf 'Prometheus service: %s/%s port %s\n' \
  "$PROM_NAMESPACE" "$PROM_SERVICE" "$PROM_SERVICE_PORT"

# Keep port-forward running in separate terminal
kubectl port-forward -n "$PROM_NAMESPACE" \
  "service/$PROM_SERVICE" "9095:$PROM_SERVICE_PORT" \
  --address 127.0.0.1
```

```bash
export PROM=http://127.0.0.1:9095
curl -fsS "$PROM/-/ready"

# Check Prometheus targets for canary deployment
curl -fsS "$PROM/api/v1/targets?state=any" |
jq -r --arg deployment "$PREFLIGHT_DEPLOYMENT" '
  (
    .data.activeTargets[]?,
    .data.droppedTargets[]?
  )
  | select(
      ((.labels.pod // "") | contains($deployment))
      or
      ((.discoveredLabels.__meta_kubernetes_pod_name // "") | contains($deployment))
    )
  | [
      (.health // "dropped"),
      (.labels.pod // .discoveredLabels.__meta_kubernetes_pod_name // ""),
      (.labels.endpoint // ""),
      (.discoveredLabels.__meta_kubernetes_pod_container_port_name // ""),
      (.scrapeUrl // ""),
      (.lastError // "")
    ]
  | @tsv
'

# Discover active metric series
curl -fsS -G "$PROM/api/v1/series" \
  --data-urlencode 'match[]={__name__=~"agent_.*|sglang:.*|dynamo_.*"}' \
  --data-urlencode "start=$(date -u -d '15 minutes ago' +%s)" \
  --data-urlencode "end=$(date -u +%s)" |
jq -r \
  --arg namespace "$NAMESPACE" \
  --arg deployment "$PREFLIGHT_DEPLOYMENT" '
    .data[]
    | select(
        (.namespace // "") == $namespace
        and ((.pod // "") | contains($deployment))
      )
    | [.__name__, (.pod // ""), (.job // "")]
    | @tsv
  ' |
sort -u

# Run instant queries
export PREFLIGHT_POD_REGEX="${PREFLIGHT_DEPLOYMENT}.*"

curl -fsS -G "$PROM/api/v1/query" \
  --data-urlencode "query=sum by (pod) (agent_tx_bytes_total{namespace=\"${NAMESPACE}\",pod=~\"${PREFLIGHT_POD_REGEX}\"})" | jq

curl -fsS -G "$PROM/api/v1/query" \
  --data-urlencode "query=sum by (pod) (agent_tx_requests_num_total{namespace=\"${NAMESPACE}\",pod=~\"${PREFLIGHT_POD_REGEX}\"})" | jq

curl -fsS -G "$PROM/api/v1/query" \
  --data-urlencode "query=sum by (pod) (agent_errors_total{namespace=\"${NAMESPACE}\",pod=~\"${PREFLIGHT_POD_REGEX}\"})" | jq

curl -fsS -G "$PROM/api/v1/query" \
  --data-urlencode "query=sum by (pod) (sglang:hicache_host_used_tokens{namespace=\"${NAMESPACE}\",pod=~\"${PREFLIGHT_POD_REGEX}\"})" | jq
```

### 5.4 Export Prometheus Query Window Artifacts

```bash
export END_EPOCH="$(date -u +%s)"
export START_EPOCH="$(date -u -d '30 minutes ago' +%s)"
export PROM_OUT_DIR="$EXP_DIR/metrics/prometheus-preflight"
mkdir -p "$PROM_OUT_DIR"

for metric in \
  agent_tx_bytes_total \
  agent_rx_bytes_total \
  agent_tx_requests_num_total \
  agent_rx_requests_num_total \
  agent_xfer_time_total \
  agent_xfer_post_time_total \
  agent_errors_total \
  'sglang:hicache_host_total_tokens' \
  'sglang:hicache_host_used_tokens'; do
  safe_name="$(printf '%s' "$metric" | tr ':/' '__')"
  curl -fsS -G "$PROM/api/v1/query_range" \
    --data-urlencode "query=${metric}{namespace=\"${NAMESPACE}\",pod=~\"${PREFLIGHT_POD_REGEX}\"}" \
    --data-urlencode "start=$START_EPOCH" \
    --data-urlencode "end=$END_EPOCH" \
    --data-urlencode 'step=15s' \
    -o "$PROM_OUT_DIR/${safe_name}.json"
  jq -e '.status == "success"' \
    "$PROM_OUT_DIR/${safe_name}.json" >/dev/null
done

printf 'START_EPOCH=%s\nEND_EPOCH=%s\n' "$START_EPOCH" "$END_EPOCH" \
  > "$PROM_OUT_DIR/query-window.txt"
```

### 5.5 Production PodMonitor Creation

Generate the production PodMonitor (`production-podmonitor.yaml`) for the main deployment before cleaning up the canary:

```bash
sed 's/q36-sgl-pd-tp1ep2-4p4d-pf/q36-sgl-pd-tp1ep2-4p4d/g' \
  "$EXP_DIR/preflight-podmonitor.yaml" |
tee "$EXP_DIR/production-podmonitor.yaml" >/dev/null

kubectl apply --dry-run=server -f "$EXP_DIR/production-podmonitor.yaml"
kubectl apply -f "$EXP_DIR/production-podmonitor.yaml"
```

---

## 6. Final Acceptance Verification & Preflight Cleanup

### 6.1 Final Log Verification

Inspect logs collected after traffic generation:

```bash
kubectl logs -n "$NAMESPACE" -l "$PREFLIGHT_GRAPH_LABEL" \
  --all-containers --prefix --since=15m |
  tee "$EXP_DIR/qwen36-sglang-preflight-after-traffic.log"

grep -Ei 'traceback|assertionerror|notimplementederror|nixl_err|transfer.*fail|does not support|unsupported' \
  "$EXP_DIR/qwen36-sglang-preflight-after-traffic.log"
```

> [!IMPORTANT]
> Production deployment is approved **ONLY** when:
> 1. In-cluster test request completes with `200 OK`.
> 2. NIXL `agent_tx_bytes_total > 0` on prefill worker.
> 3. CPU HiCache `sglang:hicache_host_used_tokens > 0` on prefill worker.
> 4. Prompt cache report returns `cached_tokens > 0`.
> 5. Log output contains **ZERO** fatal error signatures.

### 6.2 Preflight Resource Cleanup

Remove canary deployment graph and canary PodMonitor:

```bash
kubectl delete dynamographdeployment.nvidia.com -n "$NAMESPACE" \
  "$PREFLIGHT_DEPLOYMENT" --wait=true

kubectl delete podmonitor -n "$MONITOR_NAMESPACE" \
  "$PREFLIGHT_PODMONITOR" --ignore-not-found
```
