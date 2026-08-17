# Fetch DCGM Metrics for the Last Eight Hours

This procedure exports DCGM GPU utilization for only the Qwen3-32B
disaggregated prefill and decode workers. The query window is the eight hours
immediately before the export command is run.

NIXL telemetry is intentionally excluded. It will be enabled and collected in
the next benchmark round.

## 1. Confirm the Prometheus Service

The cluster currently exposes the Prometheus server through this Service:

```text
namespace: monitoring
service:   monitoring-kube-prometheus-prometheus
port:      9090
```

Reconfirm it if the monitoring stack has been changed:

```bash
kubectl get svc -A \
  -o custom-columns='NAMESPACE:.metadata.namespace,NAME:.metadata.name,PORTS:.spec.ports[*].port' |
grep -i prometheus
```

Use the following values for this cluster:

```bash
export PROM_NAMESPACE=monitoring
export PROM_SERVICE=monitoring-kube-prometheus-prometheus
export PROM_SERVICE_PORT=9090
```

Do not use `prometheus-operated` for this procedure. It is the internal
governing Service for the Prometheus StatefulSet; the
`monitoring-kube-prometheus-prometheus` Service is the normal client-facing
endpoint.

## 2. Open Prometheus on Local Port 9095

Run this in terminal 1 and keep it running:

```bash
kubectl port-forward \
  -n "$PROM_NAMESPACE" \
  "service/$PROM_SERVICE" \
  "9095:$PROM_SERVICE_PORT" \
  --address 127.0.0.1
```

Expected output:

```text
Forwarding from 127.0.0.1:9095 -> 9090
```

## 3. Configure the Export

Run the remaining commands in terminal 2:

```bash
export PROM=http://127.0.0.1:9095
export NAMESPACE=qwen32-bench
export RESULT_ROOT=/ephemeral/shared/dynamo/aiperf-results
export OUT_DIR="$RESULT_ROOT/dcgm-last-8h"

mkdir -p "$OUT_DIR"
touch "$OUT_DIR/.write-test"
rm "$OUT_DIR/.write-test"

curl -fsS "$PROM/-/ready"
echo
echo "Metrics will be saved under $OUT_DIR"
```

The readiness request must print:

```text
Prometheus Server is Ready.
```

## 4. Set the Exact Eight-Hour Window

```bash
export END_EPOCH="$(date -u +%s)"
export START_EPOCH="$(date -u -d '8 hours ago' +%s)"

printf 'START_UTC=%s\nEND_UTC=%s\n' \
  "$(date -u -d "@$START_EPOCH" +%Y-%m-%dT%H:%M:%SZ)" \
  "$(date -u -d "@$END_EPOCH" +%Y-%m-%dT%H:%M:%SZ)"
```

Save the selected interval with the exported data:

```bash
printf 'start_epoch\tend_epoch\tstart_utc\tend_utc\n%s\t%s\t%s\t%s\n' \
  "$START_EPOCH" \
  "$END_EPOCH" \
  "$(date -u -d "@$START_EPOCH" +%Y-%m-%dT%H:%M:%SZ)" \
  "$(date -u -d "@$END_EPOCH" +%Y-%m-%dT%H:%M:%SZ)" \
  > "$OUT_DIR/query-window.tsv"
```

## 5. Verify Worker GPU Series Exist

DCGM Exporter runs in namespace `gpu-operator`, so its target labels are
`namespace="gpu-operator"` and `pod="nvidia-dcgm-exporter-..."`. The labels
that identify the GPU-owning workload are `exported_namespace` and
`exported_pod`. The following expression matches both `vllmprefillworker` and
`vllmdecodeworker` workloads:

```bash
export GPU_QUERY='DCGM_FI_DEV_GPU_UTIL{exported_namespace="qwen32-bench",exported_pod=~".*vllm(prefill|decode)worker.*"}'
```

List the worker GPU series that existed during the selected interval:

```bash
curl -fsS -G "$PROM/api/v1/series" \
  --data-urlencode "match[]=$GPU_QUERY" \
  --data-urlencode "start=$START_EPOCH" \
  --data-urlencode "end=$END_EPOCH" |
jq -r '
  ["hostname", "worker_pod", "gpu", "uuid"],
  (.data[] | [(.Hostname // ""), (.exported_pod // ""), (.gpu // ""), (.UUID // "")])
  | @tsv
'
```

If only the header is printed, follow the diagnostics in section 9 before
continuing.

## 6. Export the Raw Eight-Hour Time Series

A 30-second step keeps the file manageable while retaining useful benchmark
resolution:

```bash
curl -fsS -G "$PROM/api/v1/query_range" \
  --data-urlencode "query=$GPU_QUERY" \
  --data-urlencode "start=$START_EPOCH" \
  --data-urlencode "end=$END_EPOCH" \
  --data-urlencode 'step=30s' \
  -o "$OUT_DIR/dcgm-gpu-util-last-8h.json"

jq -e '.status == "success"' \
  "$OUT_DIR/dcgm-gpu-util-last-8h.json" >/dev/null

export SERIES_COUNT="$(jq '.data.result | length' \
  "$OUT_DIR/dcgm-gpu-util-last-8h.json")"

echo "DCGM worker GPU series found: $SERIES_COUNT"
```

Do not continue if `SERIES_COUNT` is `0`; use section 9 to determine whether
the worker filter or Prometheus scraping is the problem.

## 7. Create the Sample-Level CSV

```bash
jq -r '
(
  ["timestamp_utc", "hostname", "worker_pod", "gpu", "uuid", "gpu_util_percent"],
  (
    .data.result[]
    | .metric as $m
    | .values[]
    | [
        (.[0] | tonumber | strftime("%Y-%m-%dT%H:%M:%SZ")),
        ($m.Hostname // ""),
        ($m.exported_pod // ""),
        ($m.gpu // ""),
        ($m.UUID // ""),
        .[1]
      ]
  )
) | @csv
' "$OUT_DIR/dcgm-gpu-util-last-8h.json" \
  > "$OUT_DIR/dcgm-gpu-util-last-8h.csv"

head -n 20 "$OUT_DIR/dcgm-gpu-util-last-8h.csv" | column -s, -t
```

## 8. Create the Per-Worker GPU Summary

The average below is the average of the regularly scraped samples. The file
also includes minimum and maximum observed utilization.

```bash
jq -r '
(
  [
    "hostname",
    "worker_pod",
    "gpu",
    "uuid",
    "sample_count",
    "average_gpu_util_percent",
    "minimum_gpu_util_percent",
    "maximum_gpu_util_percent"
  ],
  (
    .data.result[]
    | .metric as $m
    | [.values[][1] | tonumber] as $v
    | [
        ($m.Hostname // ""),
        ($m.exported_pod // ""),
        ($m.gpu // ""),
        ($m.UUID // ""),
        ($v | length),
        (($v | add) / ($v | length)),
        ($v | min),
        ($v | max)
      ]
  )
) | @csv
' "$OUT_DIR/dcgm-gpu-util-last-8h.json" \
  > "$OUT_DIR/dcgm-gpu-util-summary-last-8h.csv"

column -s, -t "$OUT_DIR/dcgm-gpu-util-summary-last-8h.csv"
```

The completed export contains:

```text
query-window.tsv
dcgm-gpu-util-last-8h.json
dcgm-gpu-util-last-8h.csv
dcgm-gpu-util-summary-last-8h.csv
```

List the saved files and their sizes:

```bash
find "$OUT_DIR" -maxdepth 1 -type f -printf '%f\t%k KiB\n' | sort
```

## 9. Empty-Result Diagnostics

First check whether Prometheus has any current DCGM GPU utilization series:

```bash
curl -fsS -G "$PROM/api/v1/query" \
  --data-urlencode 'query=count(DCGM_FI_DEV_GPU_UTIL)' |
jq
```

Then list all DCGM GPU utilization label sets from the eight-hour interval,
without the namespace and worker filters:

```bash
curl -fsS -G "$PROM/api/v1/series" \
  --data-urlencode 'match[]=DCGM_FI_DEV_GPU_UTIL' \
  --data-urlencode "start=$START_EPOCH" \
  --data-urlencode "end=$END_EPOCH" |
jq
```

Interpretation:

- If the unfiltered query returns data, compare its `exported_namespace` and
  `exported_pod` labels with `GPU_QUERY` and adjust only the filter. The plain
  `namespace` and `pod` labels identify DCGM Exporter itself.
- If the unfiltered query is also empty, Prometheus did not scrape or retain
  DCGM data for the interval. The current values on port `9400` cannot
  reconstruct historical utilization.
- If Prometheus is unavailable, return to terminal 1 and restart the port
  forward.

Inspect DCGM scrape targets when necessary:

```bash
curl -fsS "$PROM/api/v1/targets" |
jq -r '
  .data.activeTargets[]
  | select((.scrapeUrl // "") | test("9400|dcgm"; "i"))
  | [
      (.health // ""),
      (.scrapeUrl // ""),
      (.lastScrape // ""),
      (.lastError // "")
    ]
  | @tsv
'
```

## 10. Close the Local Port

After checking the saved files, return to terminal 1 and press `Ctrl-C` to
stop the port-forward process.
