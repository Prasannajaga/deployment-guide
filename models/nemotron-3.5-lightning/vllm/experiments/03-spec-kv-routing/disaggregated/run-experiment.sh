#!/usr/bin/env bash
set -Eeuo pipefail

umask 027

NAMESPACE="${NAMESPACE:-qwen32-bench}"
RECIPE_ROOT="${RECIPE_ROOT:-/ephemeral/shared/nemotron-3.5-lightning}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$RECIPE_ROOT/model-cache}"
EXP_DIR="${EXP_DIR:-$RECIPE_ROOT/vllm/experiments/03-spec-kv-routing/disaggregated}"
DOWNLOAD_JOB="${DOWNLOAD_JOB:-nemotron35-model-download}"
DEPLOYMENT="${DEPLOYMENT:-nemotron35-vllm-e3}"
PERF_JOB="${PERF_JOB:-nemotron35-vllm-e3-perf}"
SETTINGS_CONFIGMAP="${SETTINGS_CONFIGMAP:-nemotron35-vllm-e3-settings}"
GRAPH_LABEL="${GRAPH_LABEL:-nvidia.com/dynamo-graph-deployment-name=$DEPLOYMENT}"
EXPECTED_WORKERS=4
PREFILL_WORKERS="${PREFILL_WORKERS:-1}"
DECODE_WORKERS="${DECODE_WORKERS:-3}"
DOWNLOAD_TIMEOUT_SECONDS="${DOWNLOAD_TIMEOUT_SECONDS:-43200}"
DEPLOY_TIMEOUT="${DEPLOY_TIMEOUT:-60m}"
PERF_TIMEOUT_SECONDS="${PERF_TIMEOUT_SECONDS:-14700}"
CELL_ORDER="${CELL_ORDER:-A B C D}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-false}"
DOWNLOAD_MANIFEST="${DOWNLOAD_MANIFEST:-$MODEL_CACHE_DIR/model-download.yaml}"
DEPLOY_TEMPLATE="${DEPLOY_TEMPLATE:-$EXP_DIR/deploy.template.yaml}"
PERF_TEMPLATE="${PERF_TEMPLATE:-$EXP_DIR/perf.yaml}"
PUBLIC_DOWNLOAD_MANIFEST="$EXP_DIR/model-download.public.yaml"
PUBLIC_DEPLOY_TEMPLATE="$EXP_DIR/deploy.public.template.yaml"
LOG_ROOT="${LOG_ROOT:-$EXP_DIR/logs}"
RUN_ID="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
SUITE_LOG_FILE="${SUITE_LOG_FILE:-$LOG_ROOT/suite-$RUN_ID.log}"
SUMMARY_FILE="${SUMMARY_FILE:-$LOG_ROOT/suite-$RUN_ID.tsv}"
MODEL_GATE_LOG="${MODEL_GATE_LOG:-$LOG_ROOT/model-cache-gate-$RUN_ID.log}"
CURRENT_CELL=""
CURRENT_PHASE="startup"

case "$PREFILL_WORKERS:$DECODE_WORKERS" in
  1:3|2:2) ;;
  *) printf 'Use PREFILL_WORKERS=1 DECODE_WORKERS=3 or 2 and 2\n' >&2; exit 2 ;;
esac

[[ "$EXPECTED_WORKERS" =~ ^[1-9][0-9]*$ ]] || {
  printf 'EXPECTED_WORKERS must be a positive integer\n' >&2
  exit 2
}
case "$CONTINUE_ON_ERROR" in
  true|false) ;;
  *) printf 'CONTINUE_ON_ERROR must be true or false\n' >&2; exit 2 ;;
esac

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$SUITE_LOG_FILE") 2>&1

log() {
  local context=""
  if [[ -n "$CURRENT_CELL" ]]; then
    context=" [cell=$CURRENT_CELL]"
  fi
  printf '%s [%s]%s %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$context" "$2"
}

prepare_public_templates() {
  log INFO "rendering credential-free templates for public images and models"
  sed \
    -e '/^[[:space:]]*imagePullSecrets:[[:space:]]*$/{N;d;}' \
    -e '/^[[:space:]]*envFrom:[[:space:]]*$/{N;N;d;}' \
    "$DOWNLOAD_MANIFEST" > "$PUBLIC_DOWNLOAD_MANIFEST"
  sed \
    -e '/imagePullSecrets: &image_pull_secrets/{N;d;}' \
    -e '/imagePullSecrets: \*image_pull_secrets/d' \
    "$DEPLOY_TEMPLATE" > "$PUBLIC_DEPLOY_TEMPLATE"
}

suite_cleanup() {
  local rc=$?
  trap - EXIT ERR INT TERM
  set +e
  log INFO "performing final defensive cleanup"
  kubectl delete job "$PERF_JOB" -n "$NAMESPACE" \
    --ignore-not-found --wait=true --timeout=10m
  kubectl delete dynamographdeployment "$DEPLOYMENT" -n "$NAMESPACE" \
    --ignore-not-found --wait=true --timeout=20m
  kubectl delete configmap "$SETTINGS_CONFIGMAP" -n "$NAMESPACE" \
    --ignore-not-found --wait=true --timeout=5m
  kubectl delete job "$DOWNLOAD_JOB" -n "$NAMESPACE" \
    --ignore-not-found --wait=true --timeout=10m
  log INFO "suite log: $SUITE_LOG_FILE"
  log INFO "suite summary: $SUMMARY_FILE"
  exit "$rc"
}
suite_on_error() {
  local rc=$?
  local line="${BASH_LINENO[0]:-unknown}"
  trap - ERR
  log ERROR "fatal error during $CURRENT_PHASE at line $line (exit code $rc)"
  exit "$rc"
}
trap suite_cleanup EXIT
trap suite_on_error ERR
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_job() {
  local job_name=$1
  local timeout_seconds=$2
  local poll_seconds=${3:-10}
  local deadline=$(( $(date +%s) + timeout_seconds ))
  local previous_state=""

  while (( $(date +%s) < deadline )); do
    local snapshot active succeeded failed failed_condition failed_reason state
    snapshot="$(kubectl get job "$job_name" -n "$NAMESPACE" \
      -o jsonpath='{.status.active}{"|"}{.status.succeeded}{"|"}{.status.failed}{"|"}{range .status.conditions[?(@.type=="Failed")]}{.status}{"|"}{.reason}{end}')"
    IFS='|' read -r active succeeded failed failed_condition failed_reason <<<"$snapshot"
    active="${active:-0}"
    succeeded="${succeeded:-0}"
    failed="${failed:-0}"
    state="active=$active succeeded=$succeeded failed=$failed"
    if [[ "$state" != "$previous_state" ]]; then
      log INFO "$job_name: $state"
      previous_state="$state"
    fi
    if (( succeeded >= 1 )); then
      return 0
    fi
    if [[ "$failed_condition" == "True" ]]; then
      log ERROR "$job_name failed: ${failed_reason:-unknown reason}"
      return 1
    fi
    sleep "$poll_seconds"
  done

  log ERROR "$job_name did not complete within ${timeout_seconds}s"
  return 124
}

wait_for_job_pod() {
  local job_name=$1
  local timeout_seconds=$2
  local deadline=$(( $(date +%s) + timeout_seconds ))
  local snapshot pod_name phase

  while (( $(date +%s) < deadline )); do
    snapshot="$(kubectl get pods -n "$NAMESPACE" -l "job-name=$job_name" \
      -o jsonpath='{.items[0].metadata.name}{"|"}{.items[0].status.phase}')"
    IFS='|' read -r pod_name phase <<<"$snapshot"
    case "$phase" in
      Running|Succeeded|Failed)
        [[ -n "$pod_name" ]] || continue
        printf '%s\n' "$pod_name"
        return 0
        ;;
    esac
    sleep 5
  done
  return 124
}

download_diagnostics() (
  set +e
  log ERROR "collecting model-cache gate diagnostics"
  kubectl get job "$DOWNLOAD_JOB" -n "$NAMESPACE" -o wide
  kubectl get pods -n "$NAMESPACE" -l "job-name=$DOWNLOAD_JOB" -o wide
  kubectl describe job "$DOWNLOAD_JOB" -n "$NAMESPACE"
  kubectl logs -n "$NAMESPACE" -l "job-name=$DOWNLOAD_JOB" \
    --all-containers=true --prefix=true --tail=200
)

run_model_gate() {
  log INFO "checking the three pinned snapshots on the model-cache PVC"
  kubectl apply --dry-run=server -n "$NAMESPACE" \
    -f "$PUBLIC_DOWNLOAD_MANIFEST" >/dev/null
  kubectl delete job "$DOWNLOAD_JOB" -n "$NAMESPACE" \
    --ignore-not-found --wait=true --timeout=10m
  kubectl apply -n "$NAMESPACE" -f "$PUBLIC_DOWNLOAD_MANIFEST"

  if ! wait_for_job "$DOWNLOAD_JOB" "$DOWNLOAD_TIMEOUT_SECONDS" 15; then
    download_diagnostics
    return 1
  fi

  kubectl logs -n "$NAMESPACE" "job/$DOWNLOAD_JOB" --timestamps |
    tee "$MODEL_GATE_LOG"
  for snapshot in \
    '/model-cache/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/snapshots/cc84af2fe71647d87f4486c064f320e1e7535243' \
    '/model-cache/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash/snapshots/7fc1f1ff4b82b917efbd0710df0872c2bb89caa5' \
    '/model-cache/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark/snapshots/d10c6ff40d6e69d1f92e407e027de3eafdb77645'; do
    grep -Fq "$snapshot" "$MODEL_GATE_LOG" || {
      log ERROR "model-cache Job did not report required snapshot: $snapshot"
      return 1
    }
    log INFO "required snapshot is ready: $snapshot"
  done
  log INFO "model-cache gate passed; complete snapshots were not downloaded again"
}

run_cell() (
  set -Eeuo pipefail

  local cell=$1
  local cell_stamp=$2
  local cell_log="$LOG_ROOT/cell-$cell-$cell_stamp.log"
  local kube_log_dir="$LOG_ROOT/kubernetes/cell-$cell-$cell_stamp"
  local deploy_manifest="$EXP_DIR/deploy-$cell.yaml"
  local perf_manifest="$EXP_DIR/perf-$cell.yaml"
  local follow_pid=""

  CURRENT_CELL="$cell"
  exec > >(tee -a "$cell_log") 2>&1

  cell_diagnostics() (
    set +e
    log ERROR "collecting cell failure diagnostics"
    kubectl get dynamographdeployment "$DEPLOYMENT" -n "$NAMESPACE" -o wide
    kubectl describe dynamographdeployment "$DEPLOYMENT" -n "$NAMESPACE"
    kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide
    kubectl get job "$PERF_JOB" -n "$NAMESPACE" -o wide
    kubectl describe job "$PERF_JOB" -n "$NAMESPACE"
    kubectl logs -n "$NAMESPACE" -l "job-name=$PERF_JOB" \
      --all-containers=true --prefix=true --tail=200
    kubectl get events -n "$NAMESPACE" --sort-by=.lastTimestamp | tail -n 80
  )

  archive_kubernetes_logs() {
    local pod pod_log_name
    mkdir -p "$kube_log_dir"
    kubectl get dynamographdeployment "$DEPLOYMENT" -n "$NAMESPACE" -o yaml \
      > "$kube_log_dir/deployment.yaml" 2>&1
    kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide \
      > "$kube_log_dir/graph-pods.txt" 2>&1
    kubectl get job "$PERF_JOB" -n "$NAMESPACE" -o yaml \
      > "$kube_log_dir/perf-job.yaml" 2>&1
    kubectl logs -n "$NAMESPACE" -l "job-name=$PERF_JOB" \
      --all-containers=true --prefix=true --tail=1000 \
      > "$kube_log_dir/perf-pod.log" 2>&1

    while IFS= read -r pod; do
      [[ -n "$pod" ]] || continue
      pod_log_name="${pod#pod/}"
      kubectl logs -n "$NAMESPACE" "$pod" \
        --all-containers=true --prefix=true --tail=1000 \
        > "$kube_log_dir/$pod_log_name.log" 2>&1
    done < <(kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o name)
  }

  cell_cleanup() {
    local rc=$?
    local cleanup_rc=0
    trap - EXIT ERR INT TERM
    set +e

    if [[ -n "$follow_pid" ]] && kill -0 "$follow_pid" 2>/dev/null; then
      kill "$follow_pid" 2>/dev/null
      wait "$follow_pid" 2>/dev/null
    fi

    log INFO "archiving Kubernetes state and final pod logs"
    archive_kubernetes_logs
    log INFO "Kubernetes log archive: $kube_log_dir"
    log INFO "cleaning benchmark job and cell deployment"
    kubectl delete job "$PERF_JOB" -n "$NAMESPACE" \
      --ignore-not-found --wait=true --timeout=10m || cleanup_rc=$?
    kubectl delete dynamographdeployment "$DEPLOYMENT" -n "$NAMESPACE" \
      --ignore-not-found --wait=true --timeout=20m || cleanup_rc=$?
    kubectl delete configmap "$SETTINGS_CONFIGMAP" -n "$NAMESPACE" \
      --ignore-not-found --wait=true --timeout=5m || cleanup_rc=$?

    if (( cleanup_rc != 0 )); then
      log ERROR "cleanup failed with exit code $cleanup_rc; refusing handoff"
      if (( rc == 0 )); then
        rc=$cleanup_rc
      fi
    else
      log INFO "cleanup complete"
    fi
    log INFO "cell log: $cell_log"
    exit "$rc"
  }

  cell_on_error() {
    local rc=$?
    local line="${BASH_LINENO[0]:-unknown}"
    trap - ERR
    log ERROR "cell runner failed at line $line with exit code $rc"
    cell_diagnostics
    exit "$rc"
  }

  trap cell_cleanup EXIT
  trap cell_on_error ERR
  trap 'exit 130' INT
  trap 'exit 143' TERM

  log INFO "removing stale resources before deployment"
  kubectl delete job "$PERF_JOB" -n "$NAMESPACE" \
    --ignore-not-found --wait=true --timeout=10m
  kubectl delete dynamographdeployment "$DEPLOYMENT" -n "$NAMESPACE" \
    --ignore-not-found --wait=true --timeout=20m
  kubectl delete configmap "$SETTINGS_CONFIGMAP" -n "$NAMESPACE" \
    --ignore-not-found --wait=true --timeout=5m

  sed "s/experiment-cell: A/experiment-cell: $cell/" \
    "$PUBLIC_DEPLOY_TEMPLATE" > "$deploy_manifest"
  grep -Fq "experiment-cell: $cell" "$deploy_manifest"

  log INFO "server-validating and deploying $EXPECTED_WORKERS workers"
  kubectl apply --dry-run=server -n "$NAMESPACE" \
    -f "$deploy_manifest" >/dev/null
  kubectl apply -n "$NAMESPACE" -f "$deploy_manifest"
  kubectl wait -n "$NAMESPACE" --for=jsonpath='{.status.state}'=successful \
    "dynamographdeployment/$DEPLOYMENT" --timeout="$DEPLOY_TIMEOUT"
  kubectl wait -n "$NAMESPACE" --for=condition=Ready pod -l "$GRAPH_LABEL" \
    --timeout=10m

  for component in VllmPrefillWorker VllmDecodeWorker; do
    expected_replicas="$PREFILL_WORKERS"
    [[ "$component" != VllmDecodeWorker ]] || expected_replicas="$DECODE_WORKERS"
    actual_workers="$(kubectl get dynamographdeployment "$DEPLOYMENT" \
      -n "$NAMESPACE" \
      -o "jsonpath={.spec.components[?(@.name==\"$component\")].replicas}")"
    [[ "$actual_workers" == "$expected_replicas" ]] || {
      log ERROR "expected $expected_replicas $component, deployment declares ${actual_workers:-none}"
      exit 1
    }
  done
  kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide

  sed \
    -e "s/{name: EXPERIMENT_CELL, value: A}/{name: EXPERIMENT_CELL, value: $cell}/" \
    -e "/^[[:space:]]*- name: EXPERIMENT_CELL[[:space:]]*$/{n;s/value: A/value: $cell/;}" \
    "$PERF_TEMPLATE" > "$perf_manifest"

  grep -A1 'name: EXPERIMENT_CELL' "$perf_manifest" |
    grep -Eq "value: ['\"]?$cell['\"]?([},]|$)"

  log INFO "server-validating and starting benchmark job $PERF_JOB"
  kubectl apply --dry-run=server -n "$NAMESPACE" \
    -f "$perf_manifest" >/dev/null
  kubectl apply -n "$NAMESPACE" -f "$perf_manifest"

  perf_pod="$(wait_for_job_pod "$PERF_JOB" 600)"
  log INFO "streaming benchmark pod $perf_pod"
  kubectl logs -n "$NAMESPACE" -f "$perf_pod" --timestamps &
  follow_pid=$!

  if ! wait_for_job "$PERF_JOB" "$PERF_TIMEOUT_SECONDS" 10; then
    if kill -0 "$follow_pid" 2>/dev/null; then
      kill "$follow_pid" 2>/dev/null || true
    fi
    wait "$follow_pid" 2>/dev/null || true
    follow_pid=""
    cell_diagnostics
    exit 1
  fi

  wait "$follow_pid" || true
  follow_pid=""
  kubectl get job "$PERF_JOB" -n "$NAMESPACE" -o wide
  log INFO "benchmark completed successfully"
)

for command in kubectl sed grep tee; do
  command -v "$command" >/dev/null || {
    log ERROR "required command is missing: $command"
    exit 127
  }
done
for file in "$DOWNLOAD_MANIFEST" "$DEPLOY_TEMPLATE" "$PERF_TEMPLATE"; do
  [[ -r "$file" ]] || {
    log ERROR "required manifest or template is not readable: $file"
    exit 2
  }
done
if grep -q '__RUNTIME_IMAGE__' "$DEPLOY_TEMPLATE"; then
  log ERROR "render __RUNTIME_IMAGE__ with the built vLLM 0.27.1 image before running"
  exit 2
fi
for cell in $CELL_ORDER; do
  case "$cell" in
    A|B|C|D) ;;
    *) log ERROR "invalid cell in CELL_ORDER: $cell"; exit 2 ;;
  esac
done

printf 'started_utc\tfinished_utc\tcell\tstatus\texit_code\tlog_file\n' \
  > "$SUMMARY_FILE"

CURRENT_PHASE="preflight"
log INFO "preflight for $EXPECTED_WORKERS-worker experiment in namespace $NAMESPACE"
kubectl get crd dynamographdeployments.nvidia.com >/dev/null
kubectl get pvc model-cache perf-cache -n "$NAMESPACE"
kubectl get network-attachment-definition qwen-roce -n "$NAMESPACE"

CURRENT_PHASE="public-template-render"
prepare_public_templates
CURRENT_PHASE="model-cache-gate"
run_model_gate

failures=0
for cell in $CELL_ORDER; do
  started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cell_stamp="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
  cell_log="$LOG_ROOT/cell-$cell-$cell_stamp.log"
  CURRENT_PHASE="cell-$cell"
  log INFO "starting cell $cell"

  set +e
  run_cell "$cell" "$cell_stamp"
  rc=$?
  set -e
  if (( rc == 0 )); then
    status=passed
  else
    status=failed
    failures=$(( failures + 1 ))
  fi

  finished_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$started_utc" "$finished_utc" "$cell" \
    "$status" "$rc" "$cell_log" >> "$SUMMARY_FILE"

  if (( rc != 0 )); then
    log ERROR "cell $cell failed with exit code $rc"
    if [[ "$CONTINUE_ON_ERROR" == "false" ]]; then
      log ERROR "stopping; set CONTINUE_ON_ERROR=true to continue after cleanup"
      exit "$rc"
    fi
  else
    log INFO "cell $cell passed and was cleaned up"
  fi
done

if (( failures > 0 )); then
  log ERROR "suite completed with $failures failed cell runs"
  exit 1
fi

CURRENT_PHASE="complete"
log INFO "all cell runs completed successfully"
