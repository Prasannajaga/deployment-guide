#!/usr/bin/env bash
set -Eeuo pipefail

: "${NAMESPACE:?set NAMESPACE}"
: "${DEPLOYMENT:?set DEPLOYMENT}"
: "${EXP_DIR:?set EXP_DIR}"

GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
FAULT_LOG="${EXP_DIR}/faults.jsonl"
mkdir -p "$EXP_DIR"

mapfile -t decode_pods < <(
  kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o name |
    sed 's#pod/##' |
    awk 'tolower($0) ~ /decode/'
)

if [ "${#decode_pods[@]}" -eq 0 ]; then
  echo "No decode pods found for $GRAPH_LABEL" >&2
  exit 1
fi

hottest_pod=
hottest_score=-1
hottest_running=0
hottest_kv=0

for pod in "${decode_pods[@]}"; do
  metrics=$(kubectl get --raw \
    "/api/v1/namespaces/${NAMESPACE}/pods/${pod}:9090/proxy/metrics" 2>/dev/null || true)
  if ! grep -Eq '^vllm:(num_requests_running|kv_cache_usage_perc|gpu_cache_usage_perc)' \
    <<<"$metrics"; then
    echo "$pod did not expose the required vLLM load metrics; skipping" >&2
    continue
  fi
  running=$(awk '
    /^vllm:num_requests_running([{ ]|$)/ {sum += $NF}
    END {printf "%.0f", sum + 0}
  ' <<<"$metrics")
  kv=$(awk '
    /^vllm:(kv_cache_usage_perc|gpu_cache_usage_perc)([{ ]|$)/ {sum += $NF; n += 1}
    END {if (n) printf "%.9f", sum / n; else print "0"}
  ' <<<"$metrics")
  score=$(awk -v running="$running" -v kv="$kv" \
    'BEGIN {printf "%.9f", (running * 1000000) + kv}')
  printf '%s running=%s kv_usage=%s score=%s\n' "$pod" "$running" "$kv" "$score"
  if awk -v score="$score" -v hottest="$hottest_score" \
    'BEGIN {exit !(score > hottest)}'; then
    hottest_pod=$pod
    hottest_score=$score
    hottest_running=$running
    hottest_kv=$kv
  fi
done

if [ -z "$hottest_pod" ]; then
  echo "Unable to select a decode pod with observable load metrics" >&2
  exit 1
fi

fault_ts=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
pod_uid=$(kubectl get pod "$hottest_pod" -n "$NAMESPACE" -o jsonpath='{.metadata.uid}')
printf '{"timestamp":"%s","fault_type":"forced_pod_delete","pod":"%s","pod_uid":"%s","running_requests":%s,"kv_usage":%s}\n' \
  "$fault_ts" "$hottest_pod" "$pod_uid" "$hottest_running" "$hottest_kv" |
  tee -a "$FAULT_LOG"

kubectl delete pod "$hottest_pod" -n "$NAMESPACE" --grace-period=0 --force
