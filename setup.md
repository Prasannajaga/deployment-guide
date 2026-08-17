# NVIDIA Dynamo Experiments Deployment & Operation Guide

This reference provides the shared prerequisite checks and one example launch
workflow for the experiments in the `models/` directory on the 16 × H100
cluster (`gpu05` and `gpu06`). Run only one 16-GPU experiment at a time.

## Summary of Available Experiments

| Experiment Name | Model | Status | Topology | GPU Allocation | Key Manifest Path |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `deepseek-v4-flash-fp8-sglang-agg-tp4` | DeepSeek-V4-Flash-FP8 | Not Working | 4 aggregated workers × TP=4 | 16 GPUs | [`models/deepseek-v4-flash-fp8/sglang/agg/deploy.yaml`](models/deepseek-v4-flash-fp8/sglang/agg/deploy.yaml) |
| `glm52-fp8-vllm-agg-tp16` | GLM-5.2-FP8 | Working | One two-node vLLM replica, TP=16 | 16 GPUs | [`models/glm-5.2-fp8/vllm/agg/deploy.yaml`](models/glm-5.2-fp8/vllm/agg/deploy.yaml) |
| Llama-3.1-8B TP=16 | Llama-3.1-8B-Instruct | Working | Cross-node vLLM tensor parallel | 16 GPUs | [`models/llama-8B/setup.md`](models/llama-8B/setup.md) |
| `qwen3-235b-a22b-fp8-sglang-agg-tp4` | Qwen3-235B-A22B-FP8 | Working | 4 aggregated workers × TP=4 | 16 GPUs | [`models/qwen3-235B-A22B/sglang/agg/deploy.yaml`](models/qwen3-235B-A22B/sglang/agg/deploy.yaml) |
| `qwen3-32b-fp8-vllm-agg-tp2` | Qwen3-32B-FP8 | Working | 8 aggregated workers × TP=2 | 16 GPUs | [`models/qwen3-32B/experiments/vllm/agg-routing/deploy.yaml`](models/qwen3-32B/experiments/vllm/agg-routing/deploy.yaml) |
| `qwen3-32b-fp8-vllm-agg-kv-offload` | Qwen3-32B-FP8 | Working | 8 aggregated workers × TP=2 with host KV tier | 16 GPUs | [`models/qwen3-32B/experiments/vllm/agg-routing-kv-offloading/deploy.yaml`](models/qwen3-32B/experiments/vllm/agg-routing-kv-offloading/deploy.yaml) |
| `qwen3-32b-fp8-vllm-disagg` | Qwen3-32B-FP8 | Working | 6 prefill × TP=2 + 2 decode × TP=2 | 16 GPUs | [`models/qwen3-32B/experiments/vllm/disagg-routing/deploy.yaml`](models/qwen3-32B/experiments/vllm/disagg-routing/deploy.yaml) |
| `qwen3-32b-fp8-vllm-disagg-kv-aware` | Qwen3-32B-FP8 | Working | 6 prefill × TP=2 + 2 decode × TP=2 with KV routing | 16 GPUs | [`models/qwen3-32B/experiments/vllm/disagg-routing-kv-aware/deploy.yaml`](models/qwen3-32B/experiments/vllm/disagg-routing-kv-aware/deploy.yaml) |
| `qwen3-32b-fp8-sglang-agg-tp2` | Qwen3-32B-FP8 | Working | 8 aggregated workers × TP=2 | 16 GPUs | [`models/qwen3-32B/experiments/sglang/agg-routing/deploy.yaml`](models/qwen3-32B/experiments/sglang/agg-routing/deploy.yaml) |
| `qwen3-32b-fp8-sglang-agg-kv-offload` | Qwen3-32B-FP8 | Working | 8 aggregated workers × TP=2 with CPU HiCache | 16 GPUs | [`models/qwen3-32B/experiments/sglang/agg-routing-kv-offloading/deploy.yaml`](models/qwen3-32B/experiments/sglang/agg-routing-kv-offloading/deploy.yaml) |
| `qwen3-32b-fp8-sglang-disagg` | Qwen3-32B-FP8 | Working | 2 prefill × TP=2 + 1 decode × TP=4 | 8 GPUs | [`models/qwen3-32B/experiments/sglang/disagg-routing/deploy.yaml`](models/qwen3-32B/experiments/sglang/disagg-routing/deploy.yaml) |
| `qwen3-32b-fp8-sglang-disagg-kv-aware` | Qwen3-32B-FP8 | Working | 6 prefill × TP=2 + 2 decode × TP=2 with KV routing | 16 GPUs | [`models/qwen3-32B/experiments/sglang/disagg-routing-kv-aware/deploy.yaml`](models/qwen3-32B/experiments/sglang/disagg-routing-kv-aware/deploy.yaml) |
| `qwen36-35b-a3b-fp8-sglang-agg-tp2` | Qwen3.6-35B-A3B-FP8 | Working | Aggregated TP=2, KEDA scales 1–8 workers | 2–16 GPUs | [`models/qwen3.6-35B-A3B/sglang/agg-autoscaling/deploy.yaml`](models/qwen3.6-35B-A3B/sglang/agg-autoscaling/deploy.yaml) |
| `qwen36-35b-a3b-fp8-sglang-disagg-tp2` | Qwen3.6-35B-A3B-FP8 | Working | 4 prefill × TP=2 + 4 decode × TP=2 | 16 GPUs | [`models/qwen3.6-35B-A3B/sglang/disagg/deploy.yaml`](models/qwen3.6-35B-A3B/sglang/disagg/deploy.yaml) |
| `q36-sgl-pd-tp1-4p4d` | Qwen3.6-35B-A3B-FP8 | Working | 4 prefill × TP=1 + 4 decode × TP=1 | 8 GPUs | [`models/qwen3.6-35B-A3B/sglang/disagg/tp1-4p4d/deploy.yaml`](models/qwen3.6-35B-A3B/sglang/disagg/tp1-4p4d/deploy.yaml) |
| `q36-sgl-pd-tp2-2p2d` | Qwen3.6-35B-A3B-FP8 | Working | 2 prefill × TP=2 + 2 decode × TP=2 | 8 GPUs | [`models/qwen3.6-35B-A3B/sglang/disagg/tp2-2p2d/deploy.yaml`](models/qwen3.6-35B-A3B/sglang/disagg/tp2-2p2d/deploy.yaml) |
| `q36-sgl-pd-tp1ep2-4p4d` | Qwen3.6-35B-A3B-FP8 | Working | 4 prefill + 4 decode, two-GPU DP-attention/EP=2 workers | 16 GPUs | [`models/qwen3.6-35B-A3B/sglang/disagg/tp1-ep2-4p4d/deploy.yaml`](models/qwen3.6-35B-A3B/sglang/disagg/tp1-ep2-4p4d/deploy.yaml) |

The table lists experiments with manifests present in the workspace. Some
model READMEs mention additional recipes whose manifests are not present.

## Shared prerequisites and capacity

Set the common variables once. Use the selected experiment's README for its
exact model cache, deployment, and benchmark values.

```bash
export NAMESPACE=qwen32-bench
export SHARED_ROOT=/ephemeral/shared
export NETOP_NAMESPACE=nvidia-network-operator
export ROCE_NETWORK=qwen-roce
export ROCE_POOL=qwen-roce-pool

kubectl get crd dynamographdeployments.nvidia.com
kubectl get pvc -n "$NAMESPACE"
kubectl get nodes \
  -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,RDMA:.status.allocatable.rdma/ib'
kubectl get dynamographdeployments.nvidia.com -A
# Required for prefill/decode experiments or any manifest that requests RDMA.
kubectl get network-attachment-definition "$ROCE_NETWORK" \
  -n "$NAMESPACE"
kubectl get macvlannetwork "$ROCE_NETWORK"
kubectl get ippool "$ROCE_POOL" -n "$NETOP_NAMESPACE"
```

Do not deploy unless the required PVC is `Bound`, the selected GPUs and
`rdma/ib` resources are available, and the selected recipe's network
requirements are ready.

## One deployment example

This is one example for the Qwen3-32B aggregated vLLM experiment. Replace the
recipe variables with the values from the selected experiment's README.

```bash
# Setup
export RECIPE_ROOT="$SHARED_ROOT/qwen3-32b"
export EXP_DIR="$RECIPE_ROOT/experiments/vllm/agg-routing"
export DEPLOYMENT=qwen3-32b-fp8-vllm-agg-tp2
export MODEL=Qwen/Qwen3-32B-FP8
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
export FRONTEND_SERVICE="${DEPLOYMENT}-frontend"

# Deploy
test -f "$EXP_DIR/deploy.yaml"
kubectl apply --dry-run=server -n "$NAMESPACE" \
  -f "$EXP_DIR/deploy.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"

# Run
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide -w
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --all-containers --prefix --tail=500

# Delete
kubectl delete dynamographdeployment.nvidia.com "$DEPLOYMENT" \
  -n "$NAMESPACE" --wait=true --ignore-not-found
```
