# NVIDIA Dynamo Experiments Deployment & Operation Guide

This reference provides exact export variables, launch commands, pod monitoring, and cleanup procedures for all vLLM experiments on the 16x H100 cluster (`gpu05` and `gpu06`).

---

## Summary of Available Experiments

| Category | Experiment Name | Model | Topology | GPU Allocation | Key Manifest Path |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Aggregated** | `qwen3-32b-fp8-vllm-agg-tp2` | Qwen3-32B-FP8 | 8 Workers x TP=2 | 16 GPUs (8x `gpu05`, 8x `gpu06`) | `models/qwen3-32B/experiments/vllm/agg-routing/deploy.yaml` |
| **Disaggregated baseline (non-KV-aware)** | `qwen3-32b-fp8-vllm-disagg` | Qwen3-32B-FP8 | 6 Prefill (TP=2) + 2 Decode (TP=2) | 16 GPUs (12x Prefill, 4x Decode) | `models/qwen3-32B/experiments/vllm/disagg-routing/deploy.yaml` |
| **Disaggregated (KV-Aware)** | `qwen3-32b-fp8-vllm-disagg-kv-aware` | Qwen3-32B-FP8 | 6 Prefill (TP=2) + 2 Decode (TP=2) | 16 GPUs (12x Prefill, 4x Decode) | `models/qwen3-32B/experiments/vllm/disagg-routing-kv-aware/deploy.yaml` |

---

## Shared prerequisites and capacity

Each experiment requests all 16 H100 GPUs, so run only one experiment at a
time. All current manifests attach worker Pods to the `qwen32-bench/qwen-roce`
secondary network. Before applying any manifest, verify the shared model cache,
RDMA resources, and pod-native RoCE objects:

```bash
export NAMESPACE=qwen32-bench
export NETOP_NAMESPACE=nvidia-network-operator
export ROCE_NETWORK=qwen-roce
export ROCE_POOL=qwen-roce-pool

kubectl get pvc model-cache -n "$NAMESPACE"
kubectl get nodes \
  -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,RDMA:.status.allocatable.rdma/ib'
kubectl get network-attachment-definition "$ROCE_NETWORK" \
  -n "$NAMESPACE"
kubectl get macvlannetwork "$ROCE_NETWORK"
kubectl get ippool "$ROCE_POOL" -n "$NETOP_NAMESPACE"
```

Do not deploy unless `model-cache` is `Bound`, all 16 GPUs and the required
`rdma/ib` resources are available, and the RoCE objects are ready. Use the
[pod-native RoCE runbook](models/qwen3-32B/experiments/vllm/disagg-routing-kv-aware/pod-native-roce.md)
to create or diagnose the secondary network.

---

## 1. Aggregated vLLM Deployment — 8 Workers x TP=2 (16 GPUs) (`qwen3-32b-fp8-vllm-agg-tp2`) - WORKING

### Export Variables

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3-32b/experiments/vllm/agg-routing
export DEPLOYMENT=qwen3-32b-fp8-vllm-agg-tp2
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
```

### Apply Manifest

```bash
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
```

### See Pods Running

```bash
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide -w
```

### Delete / Stop Experiment

```bash
kubectl delete dynamographdeployment.nvidia.com "$DEPLOYMENT" \
  -n "$NAMESPACE" --wait=false --ignore-not-found
kubectl delete pods -l "$GRAPH_LABEL" -n "$NAMESPACE" \
  --force --grace-period=0 --ignore-not-found
```

---

## 2. Disaggregated vLLM baseline — 6 Prefill (TP=2) + 2 Decode (TP=2) (16 GPUs) (`qwen3-32b-fp8-vllm-disagg`) - WORKING

### Export Variables

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3-32b/experiments/vllm/disagg-routing
export DEPLOYMENT=qwen3-32b-fp8-vllm-disagg
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
```

### Apply Manifest

```bash
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
```

This is the non-KV-aware baseline: the frontend uses default routing, workers
do not publish KV events, and prefix caching is disabled. Its frontend requests
32 CPUs and 64 GiB of memory on one schedulable node and has a 128 GiB memory
limit.

### See Pods Running

```bash
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -L nvidia.com/dynamo-component-type -o wide -w
```

### Delete / Stop Experiment

```bash
kubectl delete dynamographdeployment.nvidia.com "$DEPLOYMENT" \
  -n "$NAMESPACE" --wait=false --ignore-not-found
kubectl delete pods -l "$GRAPH_LABEL" -n "$NAMESPACE" \
  --force --grace-period=0 --ignore-not-found
```

---

## 3. Disaggregated KV-Aware Deployment — 6 Prefill (TP=2) + 2 Decode (TP=2) (16 GPUs) (`qwen3-32b-fp8-vllm-disagg-kv-aware`) - WORKING

### Export Variables

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3-32b/experiments/vllm/disagg-routing-kv-aware
export DEPLOYMENT=qwen3-32b-fp8-vllm-disagg-kv-aware
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
```

### Apply Manifest

```bash
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
```

### See Pods Running

```bash
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -L nvidia.com/dynamo-component-type -o wide -w
```

### Delete / Stop Experiment

```bash
kubectl delete dynamographdeployment.nvidia.com "$DEPLOYMENT" \
  -n "$NAMESPACE" --wait=false --ignore-not-found
kubectl delete pods -l "$GRAPH_LABEL" -n "$NAMESPACE" \
  --force --grace-period=0 --ignore-not-found
```

---

## 4. Complete Cluster VRAM & Process Destroy (Emergency Host Cleanup)

Run this whenever lingering GPU memory or orphan vLLM processes persist across runs on **both `gpu05` and `gpu06`**:

```bash
# 1. Delete all K8s deployments and force-delete pods
export NAMESPACE=qwen32-bench
kubectl delete dynamographdeployment.nvidia.com --all \
  -n "$NAMESPACE" --wait=false --ignore-not-found
kubectl delete pods --all -n "$NAMESPACE" \
  --force --grace-period=0 --ignore-not-found

# 2. Kill orphan host processes (run on both gpu05 and gpu06)
pkill -9 -f "dynamo\.vllm|dynamo\.frontend|vllm" || true
sudo fuser -v /dev/nvidia* 2>/dev/null | awk '{print $2}' | xargs -r sudo kill -9

# 3. Verify all GPU VRAM is zeroed out
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
```
