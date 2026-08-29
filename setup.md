# NVIDIA Dynamo Experiments Deployment & Operation Guide

This reference provides the shared prerequisite checks and one example launch
workflow for the experiments in the `models/` directory on the 16 × H100
cluster (`inst-1onle-devrel-rdma-pool` and `inst-g9dwj-devrel-rdma-pool`). Run only one 16-GPU experiment at a time.

## Canonical shared configuration

The values below are the single source of truth for every active recipe in
this repository. A recipe may add model-, framework-, or topology-specific
variables, but it must not replace these shared values.

```bash
export NAMESPACE=dynamo-bench
export NETOP_NAMESPACE=nvidia-network-operator
export ONE_EXPERIMENT_AT_A_TIME=true

export PREFILL_NODE=inst-1onle-devrel-rdma-pool
export DECODE_NODE=inst-g9dwj-devrel-rdma-pool
export GPU_COUNT_PER_NODE=8
export TOTAL_GPU_COUNT=16

export SHARED_ROOT=/ephemeral/shared
export EXP_DIR=/ephemeral/shared/networking
export RECIPE_EXP_DIR_PATTERN='/ephemeral/shared/<model>/<framework>/<recipe>'

export ROCE_MASTER=rdma7
export ROCE_HCA=mlx5_8
export ROCE_NETWORK=roce
export ROCE_POOL=roce-pool

export MODEL_CACHE_PVC=model-cache
export MODEL_CACHE_PV=qwen32-model-cache-pv
export MODEL_CACHE_STORAGE_CLASS=qwen-shared-manual
export MODEL_CACHE_CAPACITY=100Gi
export MODEL_CACHE_ACCESS_MODE=ReadWriteMany
export MODEL_CACHE_RECLAIM_POLICY=Retain
export MODEL_CACHE_HOST_PATH=/ephemeral/shared/huggingface
export DELETE_DOWNLOAD_JOB_BEFORE_APPLY=true
export DOWNLOAD_MOUNT_PATH=/model-store
export WORKER_MOUNT_PATH=/opt/models
export HF_HOME_IN_DOWNLOAD_JOB=/model-store
export HF_HOME_IN_WORKER=/opt/models

export PERF_CACHE_PVC=perf-cache
export PERF_CACHE_PV=qwen32-vllm-perf-cache-pv
export PERF_CACHE_STORAGE_CLASS=qwen-shared-manual
export PERF_CACHE_CAPACITY=50Gi
export PERF_CACHE_MOUNT_PATH=/perf-cache

export COMPILATION_CACHE_POLICY=vllm-only
export COMPILATION_CACHE_PVC=compilation-cache
export COMPILATION_CACHE_MOUNT_PATH=/home/dynamo/.cache/vllm

export HF_TOKEN_SECRET=hf-token-secret
export IMAGE_PULL_SECRET=nvcrimagepullsecret
```

Each node has eight GPUs, for 16 GPUs total. Run one experiment at a time.
Each recipe defines the model revision it needs. Download Jobs use
model-specific names so cached models can coexist in `dynamo-bench`. Delete a
Job before recreating that same Job name with an updated specification.

The model cache is one shared Hugging Face cache. Workers select a model with
their recipe-specific `MODEL_PATH`; they do not create model-specific cache
PVCs. Benchmark results share the `perf-cache` PVC, while each recipe preserves
its existing `ARTIFACT_ROOT` and subdirectory layout.

The separate `compilation-cache` PVC is used only by vLLM workers. It stores
reusable vLLM compilation artifacts through `VLLM_CACHE_ROOT`; it is not a
model-weight cache, a runtime KV cache, or a benchmark-results directory.

The existing PV and storage-class object names are intentionally retained even
though some contain `qwen32`. Renaming a live storage object provides no data
layout benefit and would require a controlled PVC/PV rebinding.

## Summary of Available Experiments

| Experiment Name | Model | Status | Topology | GPU Allocation | Key Manifest Path |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `deepseek-v4-flash-fp8-sglang-agg-tp4` | DeepSeek-V4-Flash-FP8 | Not Working | 4 aggregated workers × TP=4 | 16 GPUs | [`models/deepseek-v4-flash-fp8/sglang/agg/deploy.yaml`](models/deepseek-v4-flash-fp8/sglang/agg/deploy.yaml) |
| `glm52-fp8-vllm-agg-tp16` | GLM-5.2-FP8 | Working | One two-node vLLM replica, TP=16 | 16 GPUs | [`models/glm-5.2-fp8/vllm/agg/deploy.yaml`](models/glm-5.2-fp8/vllm/agg/deploy.yaml) |
| Llama-3.1-8B TP=16 | Llama-3.1-8B-Instruct | Working | Cross-node vLLM tensor parallel | 16 GPUs | [`models/llama-8B/setup.md`](models/llama-8B/setup.md) |
| `qwen3-32b-fp8-sglang-agg-tp2` | Qwen3-32B-FP8 | Working | 8 aggregated workers × TP=2 | 16 GPUs | [`models/qwen3-32b-fp8/sglang/agg-routing/deploy.yaml`](models/qwen3-32b-fp8/sglang/agg-routing/deploy.yaml) |
| `qwen3-32b-fp8-sglang-agg-kv-offload` | Qwen3-32B-FP8 | Working | 8 aggregated workers × TP=2 with CPU HiCache | 16 GPUs | [`models/qwen3-32b-fp8/sglang/agg-routing-kv-offloading/deploy.yaml`](models/qwen3-32b-fp8/sglang/agg-routing-kv-offloading/deploy.yaml) |
| `qwen3-32b-fp8-sglang-disagg` | Qwen3-32B-FP8 | Working | 2 prefill × TP=2 + 1 decode × TP=4 | 8 GPUs | [`models/qwen3-32b-fp8/sglang/disagg-routing/deploy.yaml`](models/qwen3-32b-fp8/sglang/disagg-routing/deploy.yaml) |
| `qwen3-32b-fp8-sglang-disagg-kv-aware` | Qwen3-32B-FP8 | Working | 6 prefill × TP=2 + 2 decode × TP=2 with KV routing | 16 GPUs | [`models/qwen3-32b-fp8/sglang/disagg-routing-kv-aware/deploy.yaml`](models/qwen3-32b-fp8/sglang/disagg-routing-kv-aware/deploy.yaml) |
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
export NAMESPACE=dynamo-bench
export SHARED_ROOT=/ephemeral/shared
export NETOP_NAMESPACE=nvidia-network-operator
export ROCE_NETWORK=roce
export ROCE_POOL=roce-pool

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

Bootstrap the shared workload namespace and Pod RoCE resources once by
following [Network Setup](network-setup.md). That runbook materializes the
cluster manifests below `/ephemeral/shared/networking` and
creates the `roce` attachment in `dynamo-bench`. All model families use this
namespace and reference that attachment by the bare name `roce`; do not create
per-model workload namespaces or use namespace-qualified network references.

Do not deploy unless the required PVC is `Bound`, the selected GPUs and
`rdma/ib` resources are available, and the selected recipe's network
requirements are ready.

## One deployment example

This is one example for the Qwen3-32B aggregated vLLM experiment. Replace the
recipe variables with the values from the selected experiment's README.

```bash
# Setup
export RECIPE_ROOT="$SHARED_ROOT/qwen3-32b"
export EXP_DIR="$RECIPE_ROOT/vllm/agg-routing"
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
