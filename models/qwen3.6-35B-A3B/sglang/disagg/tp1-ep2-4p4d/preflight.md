# SGLang KV offload + NIXL preflight

Run this four-GPU canary before applying the optional production
`deploy-kv-offloading.yaml`. It uses one prefill worker and one decode worker
with the same model, parallelism, CPU-HiCache, cache-reporting, UCX, and NIXL
telemetry settings as that variant. The baseline `deploy.yaml` intentionally
does not allocate CPU HiCache and therefore does not use this offload-specific
acceptance gate.

The verified configuration intentionally enables CPU KV offload only on the
prefill worker. Decode-side KV offload and the NIXL HiCache storage backend are
not supported for this hybrid attention/Mamba model in the pinned SGLang
runtime.

## Variables

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3.6-35b-a3b/sglang/disagg/tp1-ep2-4p4d
export PREFLIGHT_DEPLOYMENT=q36-sgl-pd-tp1ep2-4p4d-pf
export PREFLIGHT_GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${PREFLIGHT_DEPLOYMENT}"
export MODEL=Qwen/Qwen3.6-35B-A3B-FP8
export MONITOR_NAMESPACE=monitoring
export PREFLIGHT_PODMONITOR=q36-sgl-pd-tp1ep2-4p4d-pf-metrics
```

## Create the preflight manifest

First create `$EXP_DIR/deploy-kv-offloading.yaml` with the complete quoted
heredoc in the [production README](README.md). Generate the canary copy in the
ephemeral directory by changing its deployment name, the frontend replica
count, and the two worker replica counts:

```bash
mkdir -p "$EXP_DIR"

sed \
  -e 's/^  name: q36-sgl-pd-tp1ep2-4p4d$/  name: q36-sgl-pd-tp1ep2-4p4d-pf/' \
  -e 's/^      replicas: 6$/      replicas: 1/' \
  -e 's/^      replicas: 4$/      replicas: 1/' \
  "$EXP_DIR/deploy-kv-offloading.yaml" |
tee "$EXP_DIR/preflight.yaml" >/dev/null
```

This preserves all cache, UCX, NIXL, resource, and worker arguments from the
production manifest while reducing the canary from 16 GPUs to four GPUs.

## Resource and manifest checks

These checks confirm that the shared storage, RoCE network, GPU nodes, and
preflight manifest exist before allocating GPUs:

```bash
kubectl get pvc model-cache perf-cache -n "$NAMESPACE"
kubectl get network-attachment-definition qwen-roce -n "$NAMESPACE"
kubectl get nodes -L qwen.nvidia.com/role \
  -o custom-columns='NODE:.metadata.name,ROLE:.metadata.labels.qwen\.nvidia\.com/role,GPU:.status.allocatable.nvidia\.com/gpu,RDMA:.status.allocatable.rdma/ib'
kubectl get dynamographdeployments.nvidia.com -A

kubectl apply --dry-run=server -n "$NAMESPACE" \
  -f "$EXP_DIR/preflight.yaml"
```

Require both PVCs to be `Bound`, `qwen-roce` to exist, and two available GPUs
plus two `rdma/ib` resources on each selected role node.

## Deploy the single-replica canary

This creates one two-GPU prefill worker, one two-GPU decode worker, and one
frontend. It does not modify the production deployment:

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

## Startup compatibility check

This rejects the known unsupported hybrid-pool, scheduler, NIXL, and OOM
failure signatures before the request test:

```bash
kubectl logs -n "$NAMESPACE" -l "$PREFLIGHT_GRAPH_LABEL" \
  --all-containers --prefix --tail=1500 |
  tee "$EXP_DIR/qwen36-sglang-preflight-startup.log"

grep -Ei 'traceback|assertionerror|notimplementederror|nixl_err|out of memory|does not support|unsupported' \
  "$EXP_DIR/qwen36-sglang-preflight-startup.log"
```

No output from the final `grep` is the expected result. If it prints a fatal
signature, do not continue to production.

## Resolve the canary Pods

These variables are reused by the request and metric checks. The request runs
inside the frontend Pod, so no port-forward or generated Service DNS name is
required:

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

All three values must be nonempty before continuing.

## Send one NIXL transfer request

This long prompt forces meaningful prefill work followed by transfer to the
decode worker. The outer timeout bounds a stalled request and does not close
the SSH session:

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

Require a successful response before checking counters.

## Verify NIXL transfer metrics

NIXL exports `agent_tx_*`, `agent_rx_*`, and `agent_xfer_*` series on port
19090. The counters persist after the request, so this scrape does not need to
run concurrently with traffic:

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

Accept NIXL transfer when at least one worker reports both:

- `agent_tx_bytes_total > 0`
- `agent_tx_requests_num_total > 0`

The verified canary transferred `215288960` bytes across seven requests on the
prefill exporter. The decode exporter can show zero because this runtime uses a
single-process Prometheus exporter in a multi-process worker Pod. Any nonzero
`agent_errors_total` is a failure.

## Record before/after NIXL snapshots

This captures every worker's raw NIXL endpoint before and after one request.
The request executes inside the already-resolved frontend Pod and calls
`127.0.0.1:8000`; it does not depend on a generated Service name or cluster
DNS.

```bash
export METRIC_SNAPSHOT_DIR="$EXP_DIR/metrics/${PREFLIGHT_DEPLOYMENT}-snapshots"
mkdir -p "$METRIC_SNAPSHOT_DIR"

for pod in $worker_pods; do
  kubectl exec -n "$NAMESPACE" "$pod" -- python3 -c '
import urllib.request
print(urllib.request.urlopen(
    "http://127.0.0.1:19090/metrics", timeout=10
).read().decode())
' > "$METRIC_SNAPSHOT_DIR/${pod}-nixl-before.prom"
done

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

Compare the numeric values in the before and after files. At least one worker's
`agent_tx_bytes_total` and `agent_tx_requests_num_total` must increase. Any
increase in `agent_errors_total` fails the check. The snapshots remain under
`$EXP_DIR` for collection with the benchmark artifacts.

## Verify Prometheus discovery and Grafana

Direct scraping proves that NIXL emitted the counters. Grafana verification
additionally proves that the cluster Prometheus discovered, scraped, retained,
and can query those counters.

### 1. Confirm monitoring discovery and exact port names

List the worker ports, Prometheus PodMonitor selector, existing monitors, and
monitoring Services:

```bash
export PROMETHEUS_NAME=monitoring-kube-prometheus-prometheus

kubectl get pods -n "$NAMESPACE" -l "$PREFLIGHT_GRAPH_LABEL" \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{range .spec.containers[*].ports[*]}{"  "}{.name}{": "}{.containerPort}{"\n"}{end}{end}'

kubectl get prometheus -n "$MONITOR_NAMESPACE" "$PROMETHEUS_NAME" \
  -o json |
jq '{podMonitorSelector: .spec.podMonitorSelector, podMonitorNamespaceSelector: .spec.podMonitorNamespaceSelector}'

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

Both workers must declare `system: 9090` and `nixl: 19090`. PodMonitor endpoint
names must exactly match those declared names: `nixl-metrics` does not select a
Pod port named `nixl`, even when both refer to TCP 19090. In this cluster,
Prometheus selects PodMonitors labeled `release: monitoring`.

An existing PodMonitor is relevant only if its namespace selector includes
`qwen32-bench`, its Pod selector matches this exact canary deployment, and its
endpoints select `nixl` and `system`. Do not create a second monitor when an
existing monitor already produces healthy active targets for the same Pods and
ports; an actual duplicate would double-scrape the series.

If no monitor matches, generate a canary-specific one under the ephemeral
recipe directory. It scrapes NIXL counters on 19090 and SGLang CPU HiCache
metrics on 9090:

Use the complete two-endpoint manifest only when neither endpoint already has
a healthy active target. If an existing monitor already scrapes one endpoint,
extend that monitor when it is owned by this deployment, or remove the already
covered endpoint from the new manifest. This avoids duplicate samples.

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

If the Prometheus selector printed a different required metadata label, change
only that label before applying. Do not modify a monitor owned by another
deployment.

### 2. Port-forward Grafana

Set the namespace and Service name from the previous command. The common
kube-prometheus-stack values are shown below, but confirm them in this cluster:

```bash
export GRAFANA_NAMESPACE=monitoring
export GRAFANA_SERVICE=monitoring-grafana
export GRAFANA_SERVICE_PORT="$(kubectl get service -n "$GRAFANA_NAMESPACE" \
  "$GRAFANA_SERVICE" -o jsonpath='{.spec.ports[0].port}')"

printf 'Grafana service: %s/%s port %s\n' \
  "$GRAFANA_NAMESPACE" "$GRAFANA_SERVICE" "$GRAFANA_SERVICE_PORT"
```

Keep this command running in its own terminal:

```bash
kubectl port-forward -n "$GRAFANA_NAMESPACE" \
  "service/$GRAFANA_SERVICE" "3000:$GRAFANA_SERVICE_PORT" \
  --address 127.0.0.1
```

If `kubectl` is running on the same machine as the browser, open
`http://127.0.0.1:3000`. If it is running on a remote cluster host, create a
local SSH tunnel from the workstation using an SSH agent, then open the same
URL locally:

```bash
ssh -N -L 3000:127.0.0.1:3000 USER@CLUSTER_HOST
```

Sign in with the cluster's existing Grafana SSO or administrator credentials.
Do not expose Grafana with `--address 0.0.0.0` merely for this check.

### 3. Confirm the NIXL series in Explore

In Grafana:

1. Open **Explore**.
2. Select the cluster Prometheus data source.
3. Switch the query editor from **Builder** to **Code**.
4. Enable an **Instant** query and first run this filter-free source check:

```promql
count(agent_tx_bytes_total)
```

The result must be positive; it can exceed `2` when the selected Prometheus
also scrapes other NIXL deployments. Confirm that exactly two retained series
belong to this canary:

```promql
count(
  agent_tx_bytes_total{
    namespace="qwen32-bench",
    pod=~"q36-sgl-pd-tp1ep2-4p4d-pf.*"
  }
)
```

Then verify the current counter values without Kubernetes-label filters:

```promql
sum by (pod, hostname) (
  agent_tx_bytes_total
)
```

The transmitting prefill worker must be positive. `kube_pod_info` returning
data only proves that the selected source contains kube-state-metrics; it does
not prove that it is the Prometheus instance that scraped the NIXL endpoints.
If the direct Prometheus API returns `agent_tx_bytes_total` but Grafana does
not, inspect the selected data source. It must point to the in-cluster Service,
commonly:

```text
http://monitoring-kube-prometheus-prometheus.monitoring.svc:9090
```

Do not use `http://127.0.0.1:9095` as the Grafana data-source URL; that
port-forward exists on the cluster host, not inside the Grafana Pod. Prefer a
dedicated debug data source over changing a shared one, and use Grafana's Query
Inspector to compare the data-source UID, request, and response.

5. Set the time range to **Last 15 minutes**.
6. Enable a five-second refresh while generating verification traffic.
7. Run this discovery query:

```promql
count by (__name__) (
  {namespace="qwen32-bench", __name__=~"agent_(tx|rx|xfer|errors).*"}
)
```

Expected names include `agent_tx_bytes_total`,
`agent_tx_requests_num_total`, `agent_xfer_time_total`, and
`agent_xfer_post_time_total`. `agent_errors_total` might be absent when this
NIXL build has never emitted that family.

If discovery is empty while direct port-19090 scraping is positive, the NIXL
exporter works but either Prometheus is not scraping it or Grafana selected a
different data source. Complete the cluster-Prometheus checks below before
creating panels.

### 4. Create the NIXL dashboard panels

Open **Dashboards -> New -> New dashboard -> Add visualization**, choose the
verified Prometheus data source, switch each query to **Code**, and use
`{{pod}}` for the legend.

Panel 1, **NIXL Transmitted Bytes**: Time series, unit **bytes (IEC)**.

```promql
sum by (pod) (
  agent_tx_bytes_total{
    namespace="qwen32-bench",
    pod=~"q36-sgl-pd-tp1ep2-4p4d-pf.*"
  }
)
```

Panel 2, **NIXL Transmit Throughput**: Time series, unit
**bytes/sec (IEC)**.

```promql
sum by (pod) (
  rate(agent_tx_bytes_total{
    namespace="qwen32-bench",
    pod=~"q36-sgl-pd-tp1ep2-4p4d-pf.*"
  }[1m])
)
```

Panel 3, **NIXL Transfer Requests**: Time series, unit **requests/sec**.

```promql
sum by (pod) (
  rate(agent_tx_requests_num_total{
    namespace="qwen32-bench",
    pod=~"q36-sgl-pd-tp1ep2-4p4d-pf.*"
  }[1m])
)
```

Panel 4, **NIXL Average Transfer Time**: Time series, unit
**microseconds**.

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

Panel 5, **NIXL Errors**: Stat with **Last not null**, integer unit, green base
threshold, and red threshold at `1`.

```promql
sum by (pod) (
  agent_errors_total{
    namespace="qwen32-bench",
    pod=~"q36-sgl-pd-tp1ep2-4p4d-pf.*"
  }
)
```

An empty error panel can mean that this exporter has never emitted the family.
Do not map no data to zero; retain the direct scrape and log checks as the
authoritative error gates.

### 5. Create the CPU HiCache offload panels

CPU KV offload is enabled only on prefill. These panels use the `system`/9090
endpoint.

Panel 6, **CPU HiCache Capacity**: Stat with **Last not null**.

```promql
sum by (pod) (
  sglang:hicache_host_total_tokens{
    namespace="qwen32-bench",
    pod=~"q36-sgl-pd-tp1ep2-4p4d-pf.*sglangprefillworker.*"
  }
)
```

Panel 7, **CPU HiCache Remaining Tokens**: Stat with **Last not null**.

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

Panel 8, **CPU HiCache Utilization**: Gauge, unit **percent (0-100)**, minimum
`0`, maximum `100`, yellow threshold `70`, and red threshold `90`.

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

Capacity and remaining values are token counts, not bytes. Grafana's **Short**
unit can render `3.10 Mil`, meaning 3.10 million tokens. Select **Locale
format** to show full comma-separated counts. `agent_memory_registered` is a
NIXL memory-registration metric, not proof of CPU KV offload.

### 6. Generate traffic and observe the panels

Keep Grafana open with the five-second refresh enabled, then rerun **Send one
NIXL transfer request** from this guide in the cluster terminal. No frontend
port-forward is required for that request.

Accept the Grafana check when:

- cumulative transmitted bytes are greater than zero;
- the throughput and request-rate panels spike during the request;
- transfer time has observations for the transmitting worker;
- CPU HiCache total and used tokens are positive on prefill;
- no nonzero NIXL error counter appears; and
- the series is labeled with a canary worker Pod.

It is acceptable for the decode Pod panels to remain at zero in this pinned
multi-process runtime, provided the prefill exporter has positive transmit
bytes and requests and the inference request completed. A five-second Grafana
refresh does not change Prometheus's scrape interval; a one-minute `rate()`
window displays a short traffic burst as an approximately one-minute-wide
bump.

Save the dashboard only after the canary queries return data. Use a name such
as `Qwen3.6 SGLang P-D NIXL and CPU HiCache` and record the deployment,
namespace, and time window with the benchmark evidence.

### 7. Create production monitoring before deleting the canary

The canary PodMonitor selects only the `-pf` deployment. Generate a separate
production monitor instead of changing it in place:

```bash
sed 's/q36-sgl-pd-tp1ep2-4p4d-pf/q36-sgl-pd-tp1ep2-4p4d/g' \
  "$EXP_DIR/preflight-podmonitor.yaml" |
tee "$EXP_DIR/production-podmonitor.yaml" >/dev/null

kubectl apply --dry-run=server -f "$EXP_DIR/production-podmonitor.yaml"
kubectl apply -f "$EXP_DIR/production-podmonitor.yaml"
```

For production dashboard panels, use this Pod regex so the similarly prefixed
canary cannot match:

```promql
pod=~"q36-sgl-pd-tp1ep2-4p4d-[0-9]+-.*"
```

Production creates new Pods, so its counters begin at zero. Generate fresh
traffic and require healthy `nixl` and `system` targets before accepting the
production dashboard.

## Verify NIXL through cluster Prometheus

Use this when direct worker scraping succeeds but Grafana has no NIXL data. It
separates a Prometheus discovery/scrape problem from a Grafana panel problem.

### 1. Discover and port-forward Prometheus

Find the existing Service instead of assuming that its name is identical in
every cluster:

```bash
kubectl get service -A \
  -o custom-columns='NAMESPACE:.metadata.namespace,NAME:.metadata.name,PORTS:.spec.ports[*].port' |
  grep -i prometheus
```

Set the values from that output. These are the common kube-prometheus-stack
defaults:

```bash
export PROM_NAMESPACE=monitoring
export PROM_SERVICE=monitoring-kube-prometheus-prometheus
export PROM_SERVICE_PORT="$(kubectl get service -n "$PROM_NAMESPACE" \
  "$PROM_SERVICE" -o jsonpath='{.spec.ports[0].port}')"

printf 'Prometheus service: %s/%s port %s\n' \
  "$PROM_NAMESPACE" "$PROM_SERVICE" "$PROM_SERVICE_PORT"
```

Keep this command running in a separate cluster terminal:

```bash
kubectl port-forward -n "$PROM_NAMESPACE" \
  "service/$PROM_SERVICE" "9095:$PROM_SERVICE_PORT" \
  --address 127.0.0.1
```

Run the **Variables** block from the top of this guide again in another
terminal on the same cluster host, then set the Prometheus URL:

```bash
export PROM=http://127.0.0.1:9095
curl -fsS "$PROM/-/ready"
```

The readiness response must be successful before continuing.

### 2. Check Prometheus scrape targets

This prints active and dropped candidates belonging to the canary deployment:

```bash
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
```

Require four active worker targets to show `up` with an empty `lastError`:
prefill and decode for both `nixl`/19090 and `system`/9090. Dropped candidates
are expected when the same Pods are evaluated against unrelated monitors; they
do not negate the required active targets. If direct port-19090 scraping works
but its active target is absent, fix the PodMonitor selector or exact named-port
match. If the target exists but is down, use its `lastError` and scrape URL to
diagnose connectivity.

### 3. Discover retained metric names and labels

Do this before writing PromQL so the query uses the names and labels actually
retained by this Prometheus:

```bash
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
```

Expected NIXL families include `agent_tx_bytes_total`,
`agent_tx_requests_num_total`, `agent_xfer_time_total`, and
`agent_xfer_post_time_total`.

### 4. Run acceptance queries

These queries use the preflight Pod-name prefix so other deployments cannot
satisfy the acceptance gate:

```bash
export PREFLIGHT_POD_REGEX="${PREFLIGHT_DEPLOYMENT}.*"

curl -fsS -G "$PROM/api/v1/query" \
  --data-urlencode "query=sum by (pod) (agent_tx_bytes_total{namespace=\"${NAMESPACE}\",pod=~\"${PREFLIGHT_POD_REGEX}\"})" |
jq

curl -fsS -G "$PROM/api/v1/query" \
  --data-urlencode "query=sum by (pod) (agent_tx_requests_num_total{namespace=\"${NAMESPACE}\",pod=~\"${PREFLIGHT_POD_REGEX}\"})" |
jq

curl -fsS -G "$PROM/api/v1/query" \
  --data-urlencode "query=sum by (pod) (agent_errors_total{namespace=\"${NAMESPACE}\",pod=~\"${PREFLIGHT_POD_REGEX}\"})" |
jq

curl -fsS -G "$PROM/api/v1/query" \
  --data-urlencode "query=sum by (pod) (sglang:hicache_host_used_tokens{namespace=\"${NAMESPACE}\",pod=~\"${PREFLIGHT_POD_REGEX}\"})" |
jq
```

Accept the Prometheus path when transmit bytes and request counts are positive
for at least one canary worker and no positive error series is returned. An
empty `agent_errors_total` result can mean the family is not emitted, so keep
the direct endpoint and log checks as the error authority.

### 5. Export the verification window

Preserve the exact Prometheus query window under the ephemeral recipe
directory:

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

Also retain the raw before/after NIXL snapshots and the startup/after-traffic
logs from this runbook alongside these exports.

## Verify CPU KV offload

This checks that the prefill CPU HiCache tier was allocated and contains pages
offloaded from GPU memory:

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

Accept CPU KV offload when at least one rank reports both
`sglang:hicache_host_total_tokens > 0` and
`sglang:hicache_host_used_tokens > 0`. A zero on another rank does not negate a
positive value on the rank that handled the request.

## Verify cache reporting

This sends two related turns and requires cached-token details in the second
response:

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

Require `status` to be `CACHE_REPORT_OK` and `cached_tokens` to be greater than
zero.

## Final log check and acceptance

This catches failures that occur only while transferring or offloading state:

```bash
kubectl logs -n "$NAMESPACE" -l "$PREFLIGHT_GRAPH_LABEL" \
  --all-containers --prefix --since=15m |
  tee "$EXP_DIR/qwen36-sglang-preflight-after-traffic.log"

grep -Ei 'traceback|assertionerror|notimplementederror|nixl_err|transfer.*fail|does not support|unsupported' \
  "$EXP_DIR/qwen36-sglang-preflight-after-traffic.log"
```

No output from the final `grep` is expected. Production deployment is approved
only when the request succeeds, NIXL transmit counters are positive, CPU
HiCache has positive total and used tokens, cache reporting succeeds, and the
logs contain no fatal signature.

## Cleanup

Remove the canary and its canary-specific PodMonitor after all evidence has
been captured. Do not delete the production PodMonitor created during the
handoff step:

```bash
kubectl delete dynamographdeployment.nvidia.com -n "$NAMESPACE" \
  "$PREFLIGHT_DEPLOYMENT" --wait=true

kubectl delete podmonitor -n "$MONITOR_NAMESPACE" \
  "$PREFLIGHT_PODMONITOR" --ignore-not-found
```
