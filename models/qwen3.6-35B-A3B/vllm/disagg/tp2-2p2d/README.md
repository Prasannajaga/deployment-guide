# TP2 tensor-parallel scaling: 2 prefill + 2 decode

## Purpose

This is configuration B of the controlled comparison: two two-GPU TP2
prefill workers and two two-GPU TP2 decode workers. It consumes exactly eight
GPUs and uses the Dynamo KV-aware frontend plus NIXL/UCX transfer.

Read the [comparison README](../README.md) before publishing results, especially
the hybrid-cache compatibility gate and fairness checklist.

## Architecture and GPU topology

```text
AIPerf -> Dynamo frontend (KV router)
                    |-> 2 x prefill (TP2, 2 GPUs) --NIXL--> 2 x decode (TP2, 2 GPUs)
```

The prefill node selector consumes four GPUs in two co-located two-GPU Pods on the node labeled `prefill`;
the decode selector consumes four GPUs in two co-located two-GPU Pods on the node labeled `decode`.

## Variables

Run commands from this directory and set:

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3.6-35b-a3b/vllm/disagg/tp2-2p2d
export MODEL_CACHE_DIR=/ephemeral/shared/qwen3.6-35b-a3b/model-cache
export DEPLOYMENT=qwen36-35b-a3b-vllm-disagg-kv-tp2-2p2d
export PERF_JOB_NAME=qwen36-vllm-tp2-2p2d-perf
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
export FRONTEND_SERVICE="${DEPLOYMENT}-frontend"
export MODEL=Qwen/Qwen3.6-35B-A3B-FP8
export MODEL_REVISION=95a723d08a9490559dae23d0cff1d9466213d989
```

Change `EXP_DIR` and `MODEL_CACHE_DIR` only if the shared checkout is elsewhere.

## Prerequisites

```bash
kubectl get crd dynamographdeployments.nvidia.com
kubectl get pvc model-cache perf-cache -n "$NAMESPACE"
kubectl get network-attachment-definition qwen-roce -n "$NAMESPACE"
kubectl get nodes -L qwen.nvidia.com/role \
  -o custom-columns='NODE:.metadata.name,ROLE:.metadata.labels.qwen\.nvidia\.com/role,GPU:.status.allocatable.nvidia\.com/gpu,RDMA:.status.allocatable.rdma/ib'
kubectl get dynamographdeployments.nvidia.com -n "$NAMESPACE"
```

Require a Bound shared `model-cache`, a Bound writable `perf-cache`, one
prefill and one decode node, at least four GPUs and four `rdma/ib` resources on
each role node, and the `qwen-roce` attachment. Run only one eight-GPU recipe.

The public checkpoint needs no secret after it is cached. If the environment
requires a Hugging Face token, put it in a root/user-readable-only file outside
the repository and create the optional secret without printing its contents:

```bash
kubectl create secret generic hf-token-secret -n "$NAMESPACE" \
  --from-file=HF_TOKEN=/secure/path/to/hf-token \
  --dry-run=client -o yaml | kubectl apply -f -
```

Never put the token in this manifest, shell history, or benchmark artifacts.

## Populate the model cache

```bash
kubectl apply -n "$NAMESPACE" -f "$MODEL_CACHE_DIR/model-download.yaml"
kubectl wait -n "$NAMESPACE" \
  --for=condition=Complete job/qwen36-35b-a3b-fp8-download \
  --timeout=3600s
kubectl logs -n "$NAMESPACE" job/qwen36-35b-a3b-fp8-download --tail=100
```

The download Job pins both model ID and revision.

## Validate and apply the deployment manifest

The repository's `deploy.yaml` is canonical. Validate against the installed
CRD, then apply it:

```bash
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide -w
```

Wait for all five Pods (one frontend plus four workers):

```bash
kubectl wait -n "$NAMESPACE" --for=condition=Ready pod \
  -l "$GRAPH_LABEL" --timeout=1800s
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  -L nvidia.com/dynamo-component-type -o wide
```

## Logs and required compatibility acceptance

Inspect every container and fail the run if hybrid HMA was disabled or cache
specs could not be unified:

```bash
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --all-containers --prefix --tail=1000 | tee /tmp/qwen36-tp2-startup.log

if grep -E 'Hybrid KV cache manager is disabled|failed to convert the KV cache specs|does not support HMA' \
  /tmp/qwen36-tp2-startup.log; then
  echo 'Hybrid NIXL compatibility gate failed' >&2
  exit 1
fi

grep -Ei 'vllm|nixl|ucx|mamba|prefix|kv.event|block.size' \
  /tmp/qwen36-tp2-startup.log | tail -200
```

Require vLLM 0.23.0, successful NIXL/UCX initialization, prefix caching,
Mamba `align` mode, HMA enabled, and block size 128. A real request in the next
step must produce P-to-D transfer evidence.

## Internal smoke test

Verify the exact served model and then issue a deterministic completion:

```bash
kubectl run qwen-smoke --rm -i --restart=Never \
  --namespace "$NAMESPACE" --image=curlimages/curl -- \
  curl -fsS "http://${FRONTEND_SERVICE}:8000/v1/models"

kubectl run qwen-smoke --rm -i --restart=Never \
  --namespace "$NAMESPACE" --image=curlimages/curl -- \
  curl -fsS -H 'Content-Type: application/json' \
  --data-binary "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: smoke-test-ok\"}],\"chat_template_kwargs\":{\"enable_thinking\":false},\"temperature\":0,\"max_tokens\":32,\"stream\":false}" \
  "http://${FRONTEND_SERVICE}:8000/v1/chat/completions"
```

Reinspect prefill and decode logs and require a successful transfer. For an
optional workstation smoke test:

```bash
kubectl port-forward -n "$NAMESPACE" "service/${FRONTEND_SERVICE}" 8000:8000
curl -fsS http://127.0.0.1:8000/v1/models
```

Do not use port-forwarding for AIPerf measurements.

## Canonical AIPerf Job

`perf.yaml` runs inside Kubernetes against the internal frontend service. Its
default is the balanced, isolated-prefix, eight-point sweep. It pins AIPerf
0.10.0, prints AIPerf/Python versions, checks `/v1/models`, preserves raw
records, and saves frontend/server metrics.

```bash
kubectl delete job -n "$NAMESPACE" "$PERF_JOB_NAME" --ignore-not-found
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/perf.yaml"
kubectl logs -n "$NAMESPACE" -f "job/${PERF_JOB_NAME}"
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  "job/${PERF_JOB_NAME}" --timeout=14400s
```

## Quick single-point validation

Use a separate artifact root so this does not collide with a full run:

```bash
kubectl delete job -n "$NAMESPACE" "$PERF_JOB_NAME" --ignore-not-found
kubectl set env --local -f "$EXP_DIR/perf.yaml" -o yaml \
  ISL=8000 OSL=1024 WORKLOAD_NAME=quick-balanced \
  PREFIX_MODE=isolated CONCURRENCIES=4 BENCHMARK_DURATION=30 \
  WARMUP_REQUESTS=4 ARTIFACT_ROOT=/perf-cache/aiperf/qwen36-quick | \
  kubectl apply -n "$NAMESPACE" -f -
kubectl logs -n "$NAMESPACE" -f "job/${PERF_JOB_NAME}"
```

Continue only if the actual input/output lengths are near 8000/1024, all
requests succeed, streaming TTFT/ITL exist, and raw artifacts are present.

## Full isolated-prefix workloads

This helper deletes only the previous client Job; it does not touch the model
deployment or PVC artifacts:

```bash
run_workload () {
  workload="$1" isl="$2" osl="$3"
  kubectl delete job -n "$NAMESPACE" "$PERF_JOB_NAME" --ignore-not-found
  kubectl set env --local -f "$EXP_DIR/perf.yaml" -o yaml \
    "WORKLOAD_NAME=${workload}" "ISL=${isl}" "OSL=${osl}" \
    PREFIX_MODE=isolated CONCURRENCIES='1 2 4 8 16 32 64 128' \
    BENCHMARK_DURATION=180 WARMUP_REQUESTS=32 | \
    kubectl apply -n "$NAMESPACE" -f -
  kubectl logs -n "$NAMESPACE" -f "job/${PERF_JOB_NAME}"
  kubectl wait -n "$NAMESPACE" --for=condition=Complete \
    "job/${PERF_JOB_NAME}" --timeout=14400s
}

run_workload prefill-heavy 32000 256
run_workload balanced 8000 1024
run_workload decode-heavy 2000 4096
```

For publication reruns change `BENCHMARK_DURATION=300`. To extend a discovery
sweep after observing continued scaling, set
`CONCURRENCIES='1 2 4 8 16 32 64 128 256'`.

## Controlled shared-prefix workload

This creates eight deterministic prefix groups with 75% of the total 8K input
assigned to a shared prefix:

```bash
kubectl delete job -n "$NAMESPACE" "$PERF_JOB_NAME" --ignore-not-found
kubectl set env --local -f "$EXP_DIR/perf.yaml" -o yaml \
  WORKLOAD_NAME=balanced-kv-reuse ISL=8000 OSL=1024 \
  PREFIX_MODE=shared PREFIX_GROUPS=8 PREFIX_REUSE_PERCENT=75 \
  CONCURRENCIES='1 2 4 8 16 32 64 128' BENCHMARK_DURATION=180 | \
  kubectl apply -n "$NAMESPACE" -f -
kubectl logs -n "$NAMESPACE" -f "job/${PERF_JOB_NAME}"
```

During the run, inspect the installed metric names and router/worker logs. The
router-visible full-attention hit ratio is not complete recurrent-state reuse.

## One-off manifest with `tee <<EOF`

The checked-in `perf.yaml` remains canonical. This Kustomize overlay creates a
temporary variant without editing it:

```bash
export RUN_SUFFIX=oneoff
export ISL=8000 OSL=1024
export CONCURRENCIES='1 2 4 8 16 32 64 128'
export BENCHMARK_DURATION=180
mkdir -p /tmp/qwen36-tp2-oneoff
cp "$EXP_DIR/perf.yaml" /tmp/qwen36-tp2-oneoff/perf.yaml
tee /tmp/qwen36-tp2-oneoff/kustomization.yaml >/dev/null <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - perf.yaml
nameSuffix: -${RUN_SUFFIX}
patches:
  - target:
      kind: Job
    patch: |-
      apiVersion: batch/v1
      kind: Job
      metadata:
        name: ${PERF_JOB_NAME}
      spec:
        template:
          spec:
            containers:
              - name: perf
                env:
                  - name: ISL
                    value: "${ISL}"
                  - name: OSL
                    value: "${OSL}"
                  - name: CONCURRENCIES
                    value: "${CONCURRENCIES}"
                  - name: BENCHMARK_DURATION
                    value: "${BENCHMARK_DURATION}"
                  - name: ARTIFACT_ROOT
                    value: "/perf-cache/aiperf/qwen36-${RUN_SUFFIX}"
EOF
kubectl apply -n "$NAMESPACE" -k /tmp/qwen36-tp2-oneoff
kubectl logs -n "$NAMESPACE" -f "job/${PERF_JOB_NAME}-${RUN_SUFFIX}"
```

## Artifact inspection

Artifacts use:

```text
/perf-cache/aiperf/qwen36/tp2-2p2d/<prefix-mode>/<workload>/isl-<isl>_osl-<osl>/c<concurrency>/
```

Every `cN` retains AIPerf summaries, JSONL records, raw JSONL records, and
server metrics. The workload directory also contains `input-config.json`,
`models.json`, before/after frontend metrics, and `matrix-status.tsv`.
Create a temporary PVC inspector after the Job completes:

```bash
kubectl apply -n "$NAMESPACE" -f - <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: qwen-perf-inspector
spec:
  restartPolicy: Never
  containers:
    - name: shell
      image: busybox:1.36
      command: [sh, -c, 'sleep 3600']
      volumeMounts:
        - name: perf-cache
          mountPath: /perf-cache
  volumes:
    - name: perf-cache
      persistentVolumeClaim:
        claimName: perf-cache
EOF
kubectl wait -n "$NAMESPACE" --for=condition=Ready pod/qwen-perf-inspector --timeout=120s
kubectl exec -n "$NAMESPACE" qwen-perf-inspector -- \
  find /perf-cache/aiperf/qwen36/tp2-2p2d -maxdepth 7 -type f
kubectl delete pod -n "$NAMESPACE" qwen-perf-inspector
```

The Job refuses to overwrite an existing raw `cN` directory. Choose a new
`ARTIFACT_ROOT` for reruns.

## Cleanup

```bash
kubectl delete job -n "$NAMESPACE" "$PERF_JOB_NAME" --ignore-not-found
kubectl delete dynamographdeployment.nvidia.com -n "$NAMESPACE" \
  "$DEPLOYMENT" --wait=false --ignore-not-found
kubectl delete pod -n "$NAMESPACE" qwen-smoke qwen-perf-inspector \
  --ignore-not-found
```

Cleanup intentionally preserves `model-cache` and `perf-cache` PVC contents.

## Troubleshooting

- `Pending` workers: check role labels, four free GPUs per role node,
  `rdma/ib`, Multus, and `qwen-roce`.
- Hybrid/HMA error: stop; verify the exact 1.3.0 image and vLLM 0.23.0. Do not
  disable HMA to force the experiment through.
- No KV events: verify prefill `--kv-events-config`, `PYTHONHASHSEED=0`, ZMQ
  logs, and `/metrics`; do not invent metric names.
- Shared prompts but zero router hits: inspect attention-group events and
  router selections; Qwen3.6 recurrent groups are not equivalent to standard
  attention KV blocks.
- AIPerf install failure: the client Pod needs package-index egress or a
  prebuilt internal image containing exactly AIPerf 0.10.0.
- Artifact collision: select a new `ARTIFACT_ROOT`; do not delete old results
  unless they were archived.
- Timeouts on 32K/256 or 2K/4096: increase `REQUEST_TIMEOUT` and
  `GRACE_PERIOD`; do not accept timeout-defined saturation.
