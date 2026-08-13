#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: PROMETHEUS_URL=http://127.0.0.1:9090 $0 RUN_DIR" >&2
  exit 2
fi

run_dir=${1%/}
manifest="$run_dir/manifest.json"
prometheus_url=${PROMETHEUS_URL:-http://127.0.0.1:9090}
namespace=${NAMESPACE:-qwen32-bench}

[[ -f "$manifest" ]] || {
  echo "Missing manifest: $manifest" >&2
  exit 2
}

start=$(jq -er '.benchmark_start_epoch | select(type == "number")' "$manifest")
end=$(jq -er '.benchmark_end_epoch | select(type == "number")' "$manifest")
experiment=$(jq -er '.experiment_id | select(type == "string")' "$manifest")

mkdir -p "$run_dir/metrics/dcgm" "$run_dir/metrics/nixl"

query_range() {
  local output=$1
  local query=$2
  local tmp="${output}.tmp"

  curl -fsSG "$prometheus_url/api/v1/query_range" \
    --data-urlencode "query=$query" \
    --data-urlencode "start=$start" \
    --data-urlencode "end=$end" \
    --data-urlencode "step=15" |
    jq -e '
      if .status != "success" then
        error(.error // "Prometheus query failed")
      elif (.data.result | length) == 0 then
        error("Prometheus query returned no series")
      else
        .
      end
    ' > "$tmp"
  mv "$tmp" "$output"
  echo "saved $output"
}

worker_selector="exported_namespace=\"$namespace\",exported_pod=~\"qwen3-32b-fp8-vllm-.*worker.*\""

query_range "$run_dir/metrics/dcgm/gpu-util.json" \
  "DCGM_FI_DEV_GPU_UTIL{$worker_selector}"
query_range "$run_dir/metrics/dcgm/framebuffer-used.json" \
  "DCGM_FI_DEV_FB_USED{$worker_selector}"
query_range "$run_dir/metrics/dcgm/power.json" \
  "DCGM_FI_DEV_POWER_USAGE{$worker_selector}"
query_range "$run_dir/metrics/dcgm/pcie-rx.json" \
  "DCGM_FI_PROF_PCIE_RX_BYTES{$worker_selector}"
query_range "$run_dir/metrics/dcgm/pcie-tx.json" \
  "DCGM_FI_PROF_PCIE_TX_BYTES{$worker_selector}"

if [[ "$experiment" != "01-agg-routing" ]]; then
  query_range "$run_dir/metrics/nixl/tx-bytes-per-second.json" \
    "sum by (pod) (rate(agent_tx_bytes_total{namespace=\"$namespace\"}[1m]))"
  query_range "$run_dir/metrics/nixl/rx-bytes-per-second.json" \
    "sum by (pod) (rate(agent_rx_bytes_total{namespace=\"$namespace\"}[1m]))"
else
  rmdir "$run_dir/metrics/nixl"
fi
