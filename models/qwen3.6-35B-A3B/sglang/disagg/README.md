# SGLang disaggregated TP comparison

These are the supported SGLang comparison recipes in this directory:

| Recipe | Prefill | Decode | Total GPUs |
|---|---:|---:|---:|
| [TP1 4P4D](tp1-4p4d/README.md) | 4 × 1 GPU | 4 × 1 GPU | 8 |
| [TP2 2P2D](tp2-2p2d/README.md) | 2 × 2 GPUs | 2 × 2 GPUs | 8 |
| [TP1-attention + EP2 4P4D](tp1-ep2-4p4d/README.md) | 4 × 2 GPUs | 4 × 2 GPUs | 16 |

The EP2 recipe uses `tp-size=dp-size=ep-size=2`, DP attention, and
`moe-dense-tp-size=1`. SGLang's worker world is two GPUs, while attention and
dense non-expert layers execute at effective TP=1 and experts are distributed
at EP=2. It also enables Dynamo KV-aware routing,
SGLang/Dynamo metrics, and NIXL Prometheus telemetry; see its
[metrics runbook](tp1-ep2-4p4d/metrics.md).

The old top-level `deploy.yaml` requests 16 GPUs and is retained only as
historical material. Do not apply it for this comparison.

SGLang/Dynamo supports NIXL-based prefill/decode disaggregation, but that does
not prove this Qwen3.6 hybrid model is compatible. Each recipe requires startup
logs and a real cross-node transfer before benchmarking.

## Shared recovery and prerequisites

```bash
export NAMESPACE=qwen32-bench
export RECIPE_ROOT=/ephemeral/shared/qwen3.6-35b-a3b
export MODEL_CACHE_DIR="$RECIPE_ROOT/model-cache"
export ROCE_NETWORK=qwen-roce

kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml |
  kubectl apply -f -
```

If `model-cache` or `perf-cache` is missing after namespace deletion, recover
the retained PVs:

```bash
for binding in \
  qwen32-model-cache-pv:model-cache \
  qwen32-vllm-perf-cache-pv:perf-cache; do
  pv="${binding%%:*}"
  pvc="${binding#*:}"
  phase="$(kubectl get pv "$pv" -o jsonpath='{.status.phase}')"
  reclaim="$(kubectl get pv "$pv" -o jsonpath='{.spec.persistentVolumeReclaimPolicy}')"
  [ "$reclaim" = Retain ] || exit 1

  case "$phase" in
    Released)
      kubectl patch pv "$pv" --type=merge -p '{"spec":{"claimRef":null}}'
      kubectl wait --for=jsonpath='{.status.phase}'=Available \
        "pv/$pv" --timeout=120s
      ;;
    Available)
      ;;
    Bound)
      claim="$(kubectl get pv "$pv" \
        -o jsonpath='{.spec.claimRef.namespace}/{.spec.claimRef.name}')"
      [ "$claim" = "$NAMESPACE/$pvc" ] || {
        echo "$pv is bound to $claim" >&2
        exit 1
      }
      ;;
    *)
      echo "$pv cannot be recovered while phase=$phase" >&2
      exit 1
      ;;
  esac
done

kubectl apply -n "$NAMESPACE" -f - <<'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
spec:
  accessModes: [ReadWriteMany]
  storageClassName: qwen-shared-manual
  volumeName: qwen32-model-cache-pv
  resources:
    requests:
      storage: 100Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: perf-cache
spec:
  accessModes: [ReadWriteMany]
  storageClassName: qwen-shared-manual
  volumeName: qwen32-vllm-perf-cache-pv
  resources:
    requests:
      storage: 50Gi
EOF

kubectl wait -n "$NAMESPACE" --for=jsonpath='{.status.phase}'=Bound \
  pvc/model-cache pvc/perf-cache --timeout=120s
```

Restore the namespace-local RoCE attachment if necessary and verify capacity:

```bash
if ! kubectl get network-attachment-definition "$ROCE_NETWORK" \
  -n "$NAMESPACE" >/dev/null 2>&1; then
  kubectl patch macvlannetwork "$ROCE_NETWORK" --type=merge \
    -p "{\"spec\":{\"networkNamespace\":\"$NAMESPACE\"}}"
fi

kubectl get pvc model-cache perf-cache -n "$NAMESPACE"
kubectl get network-attachment-definition "$ROCE_NETWORK" -n "$NAMESPACE"
kubectl get nodes -L qwen.nvidia.com/role \
  -o custom-columns='NODE:.metadata.name,ROLE:.metadata.labels.qwen\.nvidia\.com/role,GPU:.status.allocatable.nvidia\.com/gpu,RDMA:.status.allocatable.rdma/ib'
```

If the pinned snapshot is not already cached, run the existing download Job:

```bash
kubectl delete job -n "$NAMESPACE" qwen36-35b-a3b-fp8-download \
  --ignore-not-found
kubectl apply -n "$NAMESPACE" -f "$MODEL_CACHE_DIR/model-download.yaml"
kubectl logs -n "$NAMESPACE" -f job/qwen36-35b-a3b-fp8-download
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  job/qwen36-35b-a3b-fp8-download --timeout=3600s
```

Then choose exactly one recipe above. Never run recipes concurrently. The EP2
recipe consumes all 16 GPUs; each TP-only comparison recipe consumes eight.
