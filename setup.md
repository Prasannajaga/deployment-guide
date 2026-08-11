# NVIDIA Dynamo Experiments Deployment & Operation Guide

This reference provides exact export variables, launch commands, pod monitoring, and cleanup procedures for all vLLM experiments on the 16x H100 cluster (`gpu05` and `gpu06`).

---

## Summary of Available Experiments

| Category | Experiment Name | Model | Topology | GPU Allocation | Key Manifest Path |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Aggregated** | `qwen3-32b-fp8-vllm-agg-tp2` | Qwen3-32B-FP8 | 8 Workers x TP=2 | 16 GPUs (8x `gpu05`, 8x `gpu06`) | `models/qwen3-32B/experiments/vllm/agg-routing/deploy.yaml` |
| **Disaggregated** | `qwen3-32b-fp8-vllm-disagg` | Qwen3-32B-FP8 | 2 Prefill (TP=2) + 1 Decode (TP=4) | 8 GPUs (4x Prefill, 4x Decode) | `models/qwen3-32B/experiments/vllm/disagg-routing/deploy.yaml` |
| **Disaggregated (KV-Aware)** | `qwen3-32b-fp8-vllm-disagg-kv-aware` | Qwen3-32B-FP8 | 6 Prefill (TP=2) + 2 Decode (TP=2) | 16 GPUs (12x Prefill, 4x Decode) | `models/qwen3-32B/experiments/vllm/disagg-routing-kv-aware/README.md` |

---

## 1. Aggregated vLLM Deployment — 8 Workers x TP=2 (16 GPUs) (`qwen3-32b-fp8-vllm-agg-tp2`) - WORKING

### Export Variables

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3-32b/experiments/vllm/agg-routing
export DGD=qwen3-32b-fp8-vllm-agg-tp2
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=$DGD"
```

### Apply Manifest

```bash
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
```

### See Pods Running

```bash
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide -w
```

### Delete / Stop Experiment

```bash
kubectl delete dynamographdeployment "$DGD" -n "$NAMESPACE" --ignore-not-found
kubectl delete pods -l "$GRAPH_LABEL" -n "$NAMESPACE" --force --grace-period=0
```

---

## 2. Disaggregated vLLM Deployment — 2 Prefill (TP=2) + 1 Decode (TP=4) (8 GPUs) (`qwen3-32b-fp8-vllm-disagg`) - WORKING

### Export Variables

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3-32b/experiments/vllm/disagg-routing
export DGD=qwen3-32b-fp8-vllm-disagg
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=$DGD"
```

### Apply Manifest

```bash
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
```

### See Pods Running

```bash
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -L nvidia.com/dynamo-component-type -o wide -w
```

### Delete / Stop Experiment

```bash
kubectl delete dynamographdeployment "$DGD" -n "$NAMESPACE" --ignore-not-found
kubectl delete pods -l "$GRAPH_LABEL" -n "$NAMESPACE" --force --grace-period=0
```

---

## 3. Disaggregated KV-Aware Deployment — 6 Prefill (TP=2) + 2 Decode (TP=2) (16 GPUs) (`qwen3-32b-fp8-vllm-disagg-kv-aware`) - NOT WORKING

### Export Variables

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3-32b/experiments/vllm/disagg-routing-kv-aware
export DGD=qwen3-32b-fp8-vllm-disagg-kv-aware
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=$DGD"
```

### Apply Manifest

```bash
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
```

### See Pods Running

```bash
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -L nvidia.com/dynamo-component-type -o wide -w
```

### Delete / Stop Experiment

```bash
kubectl delete dynamographdeployment "$DGD" -n "$NAMESPACE" --ignore-not-found
kubectl delete pods -l "$GRAPH_LABEL" -n "$NAMESPACE" --force --grace-period=0
```

---

## 4. Complete Cluster VRAM & Process Destroy (Emergency Host Cleanup)

Run this whenever lingering GPU memory or orphan vLLM processes persist across runs on **both `gpu05` and `gpu06`**:

```bash
# 1. Delete all K8s deployments and force-delete pods
kubectl delete dynamographdeployments --all -n qwen32-bench --ignore-not-found
kubectl delete pods --all -n qwen32-bench --force --grace-period=0

# 2. Kill orphan host processes (run on both gpu05 and gpu06)
pkill -9 -f "dynamo\.vllm|dynamo\.frontend|vllm" || true
sudo fuser -v /dev/nvidia* 2>/dev/null | awk '{print $2}' | xargs -r sudo kill -9

# 3. Verify all GPU VRAM is zeroed out
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
```
