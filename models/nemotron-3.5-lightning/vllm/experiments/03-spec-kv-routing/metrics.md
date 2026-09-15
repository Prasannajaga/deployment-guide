# Prometheus, MTP, KV-routing, and GPU metrics

This runbook records the signals needed to explain every high-load point in
[`perf.md`](./perf.md). Prometheus is used for time-series metrics; it is not a
distributed-trace store. Correlation is done with the exact start and finish
epochs in each benchmark `manifest.json`, the cell name, concurrency, pod UID,
and restart count. Do not enable per-request timing or detailed speculative
response fields during the benchmark: the server-aggregated counters are enough
to calculate MTP acceptance and avoid adding per-request CPU work at the load
point being measured.

The experiment has two metric endpoints. The frontend exposes Dynamo request,
router, and KV-indexer families at `/metrics` on port 8000. Each vLLM worker
exposes Dynamo component and pass-through `vllm:*` engine families at `/metrics`
on the system port 9090. DCGM Exporter is a separate cluster-level target and
adds Kubernetes `namespace`, `pod`, and `container` labels when pod-resource
mapping is available.

## 1. Variables and observability preflight

Run from a cluster-administration host. Keep the application and monitoring
namespaces distinct.

```bash
export NAMESPACE=qwen32-bench
export MONITORING_NAMESPACE=monitoring
export TOPOLOGY="${TOPOLOGY:-aggregated}"
case "$TOPOLOGY" in aggregated|disaggregated) ;; *) echo 'Invalid TOPOLOGY' >&2; exit 2 ;; esac
export EXP_DIR="/ephemeral/shared/nemotron-3.5-lightning/vllm/experiments/03-spec-kv-routing/$TOPOLOGY"
export ARTIFACT_ROOT="/perf-cache/specrouting/$TOPOLOGY"
if [ "$TOPOLOGY" = disaggregated ]; then
  export PD_LAYOUT="${PD_LAYOUT:-tp1-1p3d}"
  export ARTIFACT_ROOT="$ARTIFACT_ROOT/$PD_LAYOUT"
fi
export DEPLOYMENT=nemotron35-vllm-e3
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=$DEPLOYMENT"
export PROMETHEUS_URL=http://127.0.0.1:9090
mkdir -p "$EXP_DIR/metrics"

kubectl get crd podmonitors.monitoring.coreos.com \
  servicemonitors.monitoring.coreos.com
kubectl get prometheus -A
kubectl auth can-i create podmonitors.monitoring.coreos.com \
  -n "$MONITORING_NAMESPACE"
kubectl get podmonitor,servicemonitor -A | grep -E 'dynamo|dcgm|gpu|NAME'
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide
kubectl get pods -A -l app=nvidia-dcgm-exporter -o wide
```

This preflight is successful when Prometheus and DCGM Exporter are healthy even
if no Dynamo monitor is listed yet. The absence of a Dynamo monitor means the
fallback resources in section 3 still need to be created. The authorization
check must print `yes`; otherwise a monitoring administrator must grant access
to create `PodMonitor` objects in the monitoring namespace.

The GPU Operator label can differ by installation. If the last command returns
nothing, locate the exporter without assuming a label. Do not install a second
DCGM Exporter alongside an existing one.

```bash
kubectl get daemonset,pod,service -A | grep -i dcgm
```

Resolve the live Prometheus resource and Pod instead of assuming a Helm release
or Service name. Start a dedicated port-forward and leave it running.

```bash
export PROMETHEUS_NAME=$(kubectl get prometheus -n "$MONITORING_NAMESPACE" \
  -o jsonpath='{.items[0].metadata.name}')
export PROMETHEUS_POD=$(kubectl get pods -n "$MONITORING_NAMESPACE" \
  -l "prometheus=$PROMETHEUS_NAME" \
  -o jsonpath='{.items[0].metadata.name}')
test -n "$PROMETHEUS_NAME" && test -n "$PROMETHEUS_POD"

kubectl port-forward -n "$MONITORING_NAMESPACE" \
  "pod/$PROMETHEUS_POD" 9090:9090
```

From a second terminal, verify the API and inspect which Prometheus resource
selectors are active. A `podMonitorSelector` commonly requires a Helm release
label, so never guess it before creating a fallback monitor.

```bash
export NAMESPACE=qwen32-bench
export MONITORING_NAMESPACE=monitoring
export DEPLOYMENT=nemotron35-vllm-e3
export PROMETHEUS_URL=http://127.0.0.1:9090
export PROMETHEUS_NAME=$(kubectl get prometheus -n "$MONITORING_NAMESPACE" \
  -o jsonpath='{.items[0].metadata.name}')

curl -fsS "$PROMETHEUS_URL/-/ready"
kubectl get prometheus "$PROMETHEUS_NAME" -n "$MONITORING_NAMESPACE" -o json \
  | jq '{
      name: .metadata.name,
      podMonitorSelector: .spec.podMonitorSelector,
      podMonitorNamespaceSelector: .spec.podMonitorNamespaceSelector
    }'
```

## 2. Inventory the live metric names

Metric names can change across engine releases even when the runtime image is
pinned. Treat the live exposition as the contract for this experiment. The pod
labels and named ports are part of the deployment template in the main README.

```bash
for frontend in $(kubectl get pods -n "$NAMESPACE" \
  -l 'app.kubernetes.io/name=nemotron35-vllm-e3,app.kubernetes.io/component=frontend' \
  -o name); do
  pod=${frontend#pod/}
  kubectl get --raw \
    "/api/v1/namespaces/${NAMESPACE}/pods/${pod}:8000/proxy/metrics" \
    > "$EXP_DIR/metrics/$pod.metrics.txt"
done

for worker in $(kubectl get pods -n "$NAMESPACE" \
  -l 'app.kubernetes.io/name=nemotron35-vllm-e3,app.kubernetes.io/component=worker' \
  -o name); do
  pod=${worker#pod/}
  kubectl get --raw \
    "/api/v1/namespaces/${NAMESPACE}/pods/${pod}:9090/proxy/metrics" \
    > "$EXP_DIR/metrics/$pod.metrics.txt"
done

grep -hE '^# (HELP|TYPE) (dynamo_frontend_|dynamo_component_router_|dynamo_router_)' \
  "$EXP_DIR"/metrics/*-frontend-*.metrics.txt | sort -u
grep -hE '^# (HELP|TYPE) (vllm:spec_decode_|vllm:num_requests_|vllm:kv_cache_|vllm:prefix_cache_|dynamo_component_)' \
  "$EXP_DIR"/metrics/*.metrics.txt | sort -u
```

Labeled families may not create a series until a request has exercised that
path. Run one short request before declaring a metric missing. In B and D, all
three MTP aggregate counters must appear after that request. In A and C they
should be absent or remain zero. C and D must populate the router hit-rate and
KV-event families at the frontend.

```bash
kubectl port-forward -n "$NAMESPACE" \
  "service/${DEPLOYMENT}-frontend" 8000:8000
```

In another terminal:

```bash
curl -fsS "http://127.0.0.1:8000/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    "messages": [{"role": "user", "content": "Return one short sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }' | jq .
```

## 3. Verify discovery before adding a monitor

Dynamo's Kubernetes installation can create application monitors by default.
First query Prometheus for this deployment. If the query returns two frontend
and the expected worker targets (four aggregated or four disaggregated) with value `1`, use the generated monitors and skip the
fallback manifest. Adding a second monitor would double-scrape the same counters
and can make unscoped `sum()` queries look twice as large.

```bash
curl -fsSG "$PROMETHEUS_URL/api/v1/query" \
  --data-urlencode \
  'query=up{namespace="qwen32-bench",pod=~"nemotron35-vllm-e3-.*"}' \
  | jq '.data.result[] | {pod: .metric.pod, job: .metric.job, value: .value[1]}'
```

Only when those targets are absent, create two `PodMonitor` resources. The
frontend and worker selectors are disjoint, and the 2-second interval preserves
short queue and utilization spikes without placing a second-by-second scrape
on every pod.

```bash
tee "$EXP_DIR/metrics-podmonitors.yaml" >/dev/null <<'MONITOR_EOF'
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: nemotron35-vllm-e3-frontend
  namespace: MONITORING_NAMESPACE_PLACEHOLDER
  labels:
    research.nvidia.com/experiment: spec-kv-routing
spec:
  namespaceSelector:
    matchNames: [WORKLOAD_NAMESPACE_PLACEHOLDER]
  selector:
    matchLabels:
      app.kubernetes.io/name: nemotron35-vllm-e3
      app.kubernetes.io/component: frontend
  podMetricsEndpoints:
    - port: http
      path: /metrics
      interval: 2s
      scrapeTimeout: 1s
---
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: nemotron35-vllm-e3-workers
  namespace: MONITORING_NAMESPACE_PLACEHOLDER
  labels:
    research.nvidia.com/experiment: spec-kv-routing
spec:
  namespaceSelector:
    matchNames: [WORKLOAD_NAMESPACE_PLACEHOLDER]
  selector:
    matchLabels:
      app.kubernetes.io/name: nemotron35-vllm-e3
      app.kubernetes.io/component: worker
  podMetricsEndpoints:
    - port: system
      path: /metrics
      interval: 2s
      scrapeTimeout: 1s
MONITOR_EOF

sed \
  -e "s/MONITORING_NAMESPACE_PLACEHOLDER/$MONITORING_NAMESPACE/g" \
  -e "s/WORKLOAD_NAMESPACE_PLACEHOLDER/$NAMESPACE/g" \
  "$EXP_DIR/metrics-podmonitors.yaml" \
  > "$EXP_DIR/metrics-podmonitors.rendered.yaml"

kubectl apply --dry-run=server \
  -f "$EXP_DIR/metrics-podmonitors.rendered.yaml"
kubectl apply -f "$EXP_DIR/metrics-podmonitors.rendered.yaml"
```

Prometheus selects `PodMonitor` objects separately from the Pods selected by
each monitor. Copy every live `podMonitorSelector.matchLabels` entry onto both
new monitors instead of guessing the Helm release label. An empty selector
requires no labels. This command deliberately stops when the selector contains
`matchExpressions`, which requires an administrator to choose labels satisfying
the displayed expression.

```bash
PROMETHEUS_JSON=$(kubectl get prometheus "$PROMETHEUS_NAME" \
  -n "$MONITORING_NAMESPACE" -o json)

if ! jq -e '.spec.podMonitorSelector != null' \
  >/dev/null <<<"$PROMETHEUS_JSON"; then
  echo "Prometheus has a null podMonitorSelector and selects no PodMonitors" >&2
  exit 1
fi
if ! jq -e '((.spec.podMonitorSelector.matchExpressions // []) | length) == 0' \
  >/dev/null <<<"$PROMETHEUS_JSON"; then
  jq '.spec.podMonitorSelector' <<<"$PROMETHEUS_JSON" >&2
  echo "podMonitorSelector.matchExpressions must be satisfied manually" >&2
  exit 1
fi

mapfile -t PROMETHEUS_MONITOR_LABELS < <(
  jq -r '.spec.podMonitorSelector.matchLabels // {} |
    to_entries[] | "\(.key)=\(.value)"' <<<"$PROMETHEUS_JSON"
)
if ((${#PROMETHEUS_MONITOR_LABELS[@]})); then
  kubectl label podmonitor -n "$MONITORING_NAMESPACE" \
    nemotron35-vllm-e3-frontend nemotron35-vllm-e3-workers \
    "${PROMETHEUS_MONITOR_LABELS[@]}" --overwrite
fi

kubectl get podmonitor -n "$MONITORING_NAMESPACE" \
  nemotron35-vllm-e3-frontend nemotron35-vllm-e3-workers \
  -o custom-columns='NAME:.metadata.name,LABELS:.metadata.labels'
```

Re-run the `up` query and require six unique healthy pods for either topology. Allow approximately
two scrape intervals after applying the monitors. Also verify that the
DCGM target and samples are present. DCGM Exporter emits `namespace`, `pod`, and
`container`, but a Prometheus scrape configuration that does not honor exporter
labels can rename them to `exported_namespace`, `exported_pod`, and
`exported_container`. Either form is valid when it identifies all worker
pods (four aggregated or four disaggregated); unattributed node-wide GPUs are not sufficient for this experiment.

```bash
curl -fsSG "$PROMETHEUS_URL/api/v1/query" \
  --data-urlencode \
  'query=count by (pod,container,UUID) (DCGM_FI_DEV_GPU_UTIL{namespace="qwen32-bench",pod=~"nemotron35-vllm-e3-.*"})' \
  | jq '.data.result'
```

If that returns no attributed samples, inspect the effective labels and repeat
with the exported names. The default GPU identity label is `UUID`; older
exporter namespaces can use `uuid`.

```bash
curl -fsSG "$PROMETHEUS_URL/api/v1/query" \
  --data-urlencode 'query=DCGM_FI_DEV_GPU_UTIL' \
  | jq '.data.result[0:8] | map(.metric)'

curl -fsSG "$PROMETHEUS_URL/api/v1/query" \
  --data-urlencode \
  'query=count by (exported_pod,exported_container,UUID) (DCGM_FI_DEV_GPU_UTIL{exported_namespace="qwen32-bench",exported_pod=~"nemotron35-vllm-e3-.*"})' \
  | jq '.data.result'
```

For disaggregated runs, keep prefill and decode series separate using the Pod
names (`vllmprefillworker` and `vllmdecodeworker`) and the Pod label
`research.nvidia.com/worker-role`. There are four TP=1 worker Pods and four GPU UUIDs. Evaluate MTP acceptance on decode,
and require successful NIXL transfers plus valid repeated-prompt output before
interpreting cache-hit metrics. With 1P3D there is no prefill replica choice; use 2P2D to
evaluate cache-aware selection among prefill workers.

## 4. Required MTP and engine measurements

Use counter deltas or `rate()` over the benchmark interval. Never divide the
instantaneous raw counter values from two cells: counters reset on every fresh
deployment. The authoritative MTP formulas are:

| Measurement | PromQL calculation | Interpretation |
| :--- | :--- | :--- |
| Draft acceptance rate | accepted draft tokens / proposed draft tokens | Fraction of proposed MTP tokens accepted, range 0–1 |
| Mean acceptance length | 1 + accepted draft tokens / verification steps | Tokens emitted per target verification, range 1–8 for seven speculative tokens |
| Mean effective draft length | proposed draft tokens / verification steps | Actual proposals per step; can be below seven |
| Accepted tokens by position | rate of `spec_decode_num_accepted_tokens_per_pos` grouped by position | Shows where the seven-token draft starts failing |

The following PromQL is suitable for a 2-second scrape interval. Use a 30-second
window for the curve and `increase(metric[benchmark_duration])` when producing
one final scalar.

```promql
sum(rate({__name__="vllm:spec_decode_num_accepted_tokens_total",namespace="qwen32-bench",pod=~"nemotron35-vllm-e3-.*"}[30s]))
/
clamp_min(sum(rate({__name__="vllm:spec_decode_num_draft_tokens_total",namespace="qwen32-bench",pod=~"nemotron35-vllm-e3-.*"}[30s])), 1e-9)
```

```promql
1 +
sum(rate({__name__="vllm:spec_decode_num_accepted_tokens_total",namespace="qwen32-bench",pod=~"nemotron35-vllm-e3-.*"}[30s]))
/
clamp_min(sum(rate({__name__="vllm:spec_decode_num_drafts_total",namespace="qwen32-bench",pod=~"nemotron35-vllm-e3-.*"}[30s])), 1e-9)
```

```promql
sum by (pod) (rate({__name__="vllm:spec_decode_num_accepted_tokens_per_pos",namespace="qwen32-bench",pod=~"nemotron35-vllm-e3-.*"}[30s]))
```

Inspect the live labels on the per-position metric and add its position label to
the final `sum by`; the pinned runtime's exposition is authoritative. Do not
invent a label name if the family is absent.

The engine saturation dashboard should plot these core series by worker:

| Signal | Metric or expression |
| :--- | :--- |
| Running and queued work | `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:num_requests_waiting_by_reason` |
| KV-cache pressure | `vllm:kv_cache_usage_perc` and `dynamo_component_gpu_cache_usage_percent` |
| Preemption | `rate(vllm:num_preemptions_total[30s])` |
| Engine throughput | `rate(vllm:prompt_tokens_total[30s])`, `rate(vllm:generation_tokens_total[30s])` |
| Prefix reuse | `rate(vllm:prefix_cache_hits_total[30s]) / rate(vllm:prefix_cache_queries_total[30s])` |
| Latency | p50/p95/p99 of TTFT, queue, prefill, decode, and inter-token histograms |
| Completion outcomes | `rate(vllm:request_success_total[30s])` grouped by finish-reason labels |

Some vLLM documentation displays a counter family without the Prometheus
client's `_total` suffix. Confirm the stored name with the inventory query and
use exactly what Prometheus has ingested.

Example p99 queue and TTFT queries are:

```promql
histogram_quantile(0.99,
  sum by (le) (
    rate({__name__="vllm:request_queue_time_seconds_bucket",namespace="qwen32-bench",pod=~"nemotron35-vllm-e3-.*"}[1m])
  )
)
```

```promql
histogram_quantile(0.99,
  sum by (le) (
    rate({__name__="vllm:time_to_first_token_seconds_bucket",namespace="qwen32-bench",pod=~"nemotron35-vllm-e3-.*"}[1m])
  )
)
```

## 5. KV-routing and frontend measurements

The router's hit rate is predicted overlap at scheduling time, not the worker's
observed prefix-cache hit rate. Track both. A high predicted value with low
worker prefix hits points to stale or incorrectly applied KV events. C and D
must show KV events; A and B are the negative controls.

```promql
histogram_quantile(0.50,
  sum by (pod, le) (
    rate(dynamo_component_router_kv_hit_rate_bucket{namespace="qwen32-bench",pod=~"nemotron35-vllm-e3-.*"}[1m])
  )
)
```

```promql
sum by (pod, status, event_type) (
  rate(dynamo_component_kv_cache_events_applied{namespace="qwen32-bench",pod=~"nemotron35-vllm-e3-.*"}[30s])
)
```

```promql
avg(
  sum by (pod) (
    rate(dynamo_component_kv_cache_events_applied{namespace="qwen32-bench",pod=~"nemotron35-vllm-e3-.*-frontend-.*",status="success"}[1m])
  )
)
/
clamp_min(sum(rate({__name__="vllm:generation_tokens_total",namespace="qwen32-bench",pod=~"nemotron35-vllm-e3-.*"}[1m])), 1e-9)
```

Plot `dynamo_frontend_active_requests` and the sum of
`dynamo_frontend_stage_requests` for `preprocess|route|dispatch` beside vLLM
waiting requests. This distinguishes frontend or router backlog from the engine
queue. For C and D, graph p99 `dynamo_router_overhead_total_ms` and its block
hashing, indexer lookup, sequence hashing, and scheduling components. Also plot
`dynamo_frontend_worker_active_decode_blocks` and
`dynamo_frontend_worker_active_prefill_tokens` by frontend `pod` and `worker_id`
to catch load imbalance without double-counting the same worker state across
the two frontend replicas.

## 6. Required GPU and failure measurements

DCGM `DEV_*_UTIL` gauges use percent units while `PROF_*_ACTIVE` gauges use a
0–1 ratio. Do not put them on the same axis without converting one. Profiling
families may be absent when an exporter collector file omits them or the active
DCGM profiling group cannot support them concurrently. Record missing families
in the run manifest instead of silently treating them as zero.

| Area | Required series | What it explains |
| :--- | :--- | :--- |
| Compute | `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_PROF_GR_ENGINE_ACTIVE`, `DCGM_FI_PROF_SM_ACTIVE` when available | Whether the target engine is actually saturated |
| Tensor and memory pipelines | `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`, `DCGM_FI_PROF_DRAM_ACTIVE` | Whether MTP shifts tensor activity or memory pressure |
| Framebuffer | `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_FB_FREE`, `DCGM_FI_DEV_FB_RESERVED` | Weight plus KV/cache memory and OOM proximity |
| Power and clocks | `DCGM_FI_DEV_POWER_USAGE`, `DCGM_FI_DEV_SM_CLOCK`, `DCGM_FI_DEV_MEM_CLOCK`, `DCGM_FI_DEV_GPU_TEMP` | Power or thermal throttling behind throughput collapse |
| PCIe | `DCGM_FI_PROF_PCIE_TX_BYTES`, `DCGM_FI_PROF_PCIE_RX_BYTES`, `DCGM_FI_DEV_PCIE_REPLAY_COUNTER` | Host/device transfer pressure or link retries |
| Hard errors | `DCGM_FI_DEV_XID_ERRORS`, `DCGM_EXP_XID_ERRORS_TOTAL`, ECC and row-remap counters when enabled | GPU/driver failure rather than ordinary overload |

The key utilization views retain the `pod` and GPU identity dimensions. A
four-GPU average alone can hide one dead or under-routed worker.

```promql
avg by (pod, UUID) (
  DCGM_FI_DEV_GPU_UTIL{namespace="qwen32-bench",pod=~"nemotron35-vllm-e3-.*"}
)
```

```promql
avg by (pod, UUID) (
  DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{namespace="qwen32-bench",pod=~"nemotron35-vllm-e3-.*"}
)
```

```promql
max by (pod, UUID) (
  DCGM_FI_DEV_FB_USED{namespace="qwen32-bench",pod=~"nemotron35-vllm-e3-.*"}
)
```

Correlate those series with Kubernetes health. A breaker point is an engine
failure when restart count increases, readiness falls, or a last termination
reason becomes `OOMKilled`; it is not merely a slow successful response.

```promql
increase(kube_pod_container_status_restarts_total{namespace="qwen32-bench",pod=~"nemotron35-vllm-e3-.*"}[10m])
```

```promql
max_over_time(kube_pod_container_status_last_terminated_reason{namespace="qwen32-bench",pod=~"nemotron35-vllm-e3-.*",reason="OOMKilled"}[10m])
```

```promql
min_over_time(kube_pod_container_status_ready{namespace="qwen32-bench",pod=~"nemotron35-vllm-e3-.*",condition="true"}[10m])
```

## 7. Export the exact benchmark interval

Read `benchmark_started_epoch` and `benchmark_finished_epoch` from the AIPerf
run's `manifest.json` on the `perf-cache` PVC. Set the cell and concurrency from
the same manifest. Add a 15-second margin on each side so the ramp transition
and post-failure state are retained.

The benchmark no longer creates timestamp directories. For example, the
balanced B result at concurrency 2048 is stored at:

```text
/perf-cache/specrouting/<topology>/cell-B/isl-8192_osl-2048_c-2048_reuse-90_preset-balanced/results/manifest.json
```

```bash
export CELL=B
export CONCURRENCY=2048
export START_EPOCH=REPLACE_FROM_MANIFEST
export END_EPOCH=REPLACE_FROM_MANIFEST
export QUERY_START=$(( START_EPOCH - 15 ))
export QUERY_END=$(( END_EPOCH + 15 ))
export QUERY_STEP=2
export EXPORT_DIR="$EXP_DIR/metrics/cell-$CELL-c$CONCURRENCY"
mkdir -p "$EXPORT_DIR"
```

Use this helper to preserve the raw Prometheus API response for reproducible
analysis. It fails if Prometheus reports an error.

```bash
query_range() {
  name=$1
  query=$2
  curl -fsSG "$PROMETHEUS_URL/api/v1/query_range" \
    --data-urlencode "query=$query" \
    --data-urlencode "start=$QUERY_START" \
    --data-urlencode "end=$QUERY_END" \
    --data-urlencode "step=$QUERY_STEP" \
    > "$EXPORT_DIR/$name.json"
  jq -e '.status == "success"' "$EXPORT_DIR/$name.json" >/dev/null
}

export WORKER_SCOPE="namespace=\"$NAMESPACE\",pod=~\"$DEPLOYMENT-.*\""
export DCGM_SCOPE="$WORKER_SCOPE"
# If Prometheus renamed exporter labels, use this instead:
# export DCGM_SCOPE="exported_namespace=\"$NAMESPACE\",exported_pod=~\"$DEPLOYMENT-.*\""
query_range mtp-accepted \
  "sum(rate({__name__=\"vllm:spec_decode_num_accepted_tokens_total\",$WORKER_SCOPE}[30s]))"
query_range mtp-proposed \
  "sum(rate({__name__=\"vllm:spec_decode_num_draft_tokens_total\",$WORKER_SCOPE}[30s]))"
query_range mtp-steps \
  "sum(rate({__name__=\"vllm:spec_decode_num_drafts_total\",$WORKER_SCOPE}[30s]))"
query_range engine-running \
  "sum by (pod) ({__name__=\"vllm:num_requests_running\",$WORKER_SCOPE})"
query_range engine-waiting \
  "sum by (pod) ({__name__=\"vllm:num_requests_waiting\",$WORKER_SCOPE})"
query_range kv-cache \
  "max by (pod) ({__name__=\"vllm:kv_cache_usage_perc\",$WORKER_SCOPE})"
query_range generation-rate \
  "sum by (pod) (rate({__name__=\"vllm:generation_tokens_total\",$WORKER_SCOPE}[30s]))"
query_range gpu-util \
  "avg by (pod,exported_pod,UUID) (DCGM_FI_DEV_GPU_UTIL{$DCGM_SCOPE})"
query_range gpu-memory \
  "max by (pod,exported_pod,UUID) (DCGM_FI_DEV_FB_USED{$DCGM_SCOPE})"
query_range gpu-power \
  "avg by (pod,exported_pod,UUID) (DCGM_FI_DEV_POWER_USAGE{$DCGM_SCOPE})"
query_range gpu-tensor \
  "avg by (pod,exported_pod,UUID) (DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{$DCGM_SCOPE})"
query_range gpu-dram \
  "avg by (pod,exported_pod,UUID) (DCGM_FI_PROF_DRAM_ACTIVE{$DCGM_SCOPE})"
query_range pod-restarts \
  "increase(kube_pod_container_status_restarts_total{$WORKER_SCOPE}[10m])"

sha256sum "$EXPORT_DIR"/*.json > "$EXPORT_DIR/SHA256SUMS"
```

Export the frontend/router queries from section 5 and the latency histogram
buckets as separate JSON files using the same helper. Keep buckets, `_sum`, and
`_count`, not only a precomputed p99, so quantiles can be recomputed later.

Before accepting the capture, require four aggregated or four disaggregated worker pods for vLLM and
DCGM, two frontend series for frontend metrics, no unexpected target gaps, and
no duplicate jobs scraping the same endpoint. Preserve the rendered Job, AIPerf
manifest, AIPerf export, Prometheus JSON, pod-state JSON, and SHA-256 files as
one result bundle.

## 8. Interpretation gates

An MTP-on result is valid only when the accepted, proposed, and verification
step counters all increase throughout the measured phase. An MTP-off result is
valid only when those counters are absent or do not increase. A KV-routing
result is valid only when the frontend applies successful KV events and emits a
nonempty hit-rate histogram. A GPU comparison is valid only when all four
worker GPUs are attributed and no load-generator saturation warning appears.

At the throughput knee, compare MTP acceptance rate and mean acceptance length
with tensor-pipe activity, DRAM activity, generation-token rate, queue depth,
and p99 ITL. If MTP acceptance falls while tensor activity rises and throughput
stalls, verification overhead is dominating. If acceptance remains high but
waiting requests and KV usage spike, scheduling or memory capacity is the
limiter. If one worker has lower GPU activity and fewer active blocks, routing
imbalance is the likely limit. If XID, ECC, OOM, or readiness signals change,
classify the point as a failure boundary rather than a performance sample.

## 9. Cleanup

Delete only the fallback monitors created by this runbook. Do not remove shared
Prometheus, DCGM Exporter, or cluster dashboards.

```bash
kubectl delete podmonitor -n "$MONITORING_NAMESPACE" \
  nemotron35-vllm-e3-frontend nemotron35-vllm-e3-workers \
  --ignore-not-found
```

## References

The metric locations and Dynamo router/KV semantics come from the official
[Dynamo metrics catalog](https://docs.nvidia.com/dynamo/dev/reference/observability/metrics-catalog)
and [Kubernetes observability guide](https://docs.nvidia.com/dynamo/kubernetes/operations/observability).
The MTP formulas and their exact aggregate counters are defined by the official
[vLLM acceptance metrics guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/acceptance_metrics/),
with general engine families in the [vLLM production metrics reference](https://docs.vllm.ai/en/latest/usage/metrics/).
GPU names, units, and optional profiling-field behavior follow the official
[DCGM Exporter metric catalog](https://docs.nvidia.com/datacenter/dcgm/latest/reference/dcgm-exporter-metrics.html)
and [default collector list](https://github.com/NVIDIA/dcgm-exporter/blob/main/etc/default-counters.csv).
