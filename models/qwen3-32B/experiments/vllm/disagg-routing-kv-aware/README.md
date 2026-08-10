# Qwen3-32B-FP8 vLLM disaggregated deployment (6 Prefill + 2 Decode, TP=2)

This guide adapts the disaggregated inference setup for the 16 x H100 cluster with 6 prefill workers and 2 decode workers, all using Tensor Parallelism (TP=2).

Key parameters:

- `MODEL_PATH` uses the immutable snapshot on the existing `model-cache` PVC;
- Prefill topology: 6 replicas x TP=2 (12 H100 GPUs total);
- Decode topology: 2 replicas x TP=2 (4 H100 GPUs total);
- Total cluster GPU allocation: 16 H100 GPUs across 2 nodes (8 GPUs on `gpu05`, 8 GPUs on `gpu06`);
- `UCX_NET_DEVICES=mlx5_8:1` selects the active non-bonded RoCE HCA found on both nodes;
- `UCX_IB_ADDR_TYPE=eth` and `UCX_IB_GID_INDEX=3` select the validated RoCE address (`10.224.7.x`);
- `hostNetwork: true` makes the host's `rdma7` RoCE netdevice visible in each worker network namespace, and `ClusterFirstWithHostNet` retains cluster DNS;
- RDMA resource requests: 2 per TP=2 worker;
- `IPC_LOCK`, `SYS_RESOURCE`, and 40 GiB shared memory are retained.

## 1. Create the deployment file

**Run only on the Kubernetes control-plane node (`gpu05`).**

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3-32b/experiments/vllm/disagg-routing-kv-aware

mkdir -p "$EXP_DIR"

tee "$EXP_DIR/deploy.yaml" >/dev/null <<'EOF_DEPLOY_YAML'
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-32b-fp8-vllm-disagg-kv-aware
spec:
  backendFramework: vllm
  pvcs:
    - name: model-cache
      create: false
  services:
    Frontend:
      componentType: frontend
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          workingDir: /workspace/examples/backends/vllm
          command:
            - python3
            - -m
            - dynamo.frontend
          args:
            - --router-mode
            - kv
      envs:
        - name: HF_HOME
          value: /opt/models
      replicas: 1
    VllmPrefillWorker:
      componentType: worker
      subComponentType: prefill
      envFromSecret: hf-token-secret
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 40Gi
      extraPodSpec: 
        mainContainer:
          ports: []
          env:
            - name: SERVED_MODEL_NAME
              value: "Qwen/Qwen3-32B-FP8"
            - name: MODEL_PATH
              value: "/opt/models/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
            - name: HF_HOME
              value: /opt/models
            - name: VLLM_NIXL_SIDE_CHANNEL_HOST
              valueFrom:
                fieldRef:
                  fieldPath: status.podIP
            - name: UCX_TLS
              value: "rc_x,rc,cuda_copy,cuda_ipc"
            - name: UCX_NET_DEVICES
              value: "mlx5_8:1"
            - name: UCX_IB_ADDR_TYPE
              value: "eth"
            - name: UCX_IB_GID_INDEX
              value: "3"
            - name: UCX_RNDV_SCHEME
              value: "get_zcopy"
            - name: UCX_RNDV_THRESH
              value: "0"
            - name: UCX_IB_REG_METHODS
              value: "odp,rcache"
            - name: UCX_RCACHE_MAX_UNRELEASED
              value: "1024"
            - name: UCX_RC_TIMEOUT
              value: "600s"
            - name: UCX_KEEPALIVE_INTERVAL
              value: "300s"
            - name: UCX_LOG_LEVEL
              value: "info"
            - name: NIXL_LOG_LEVEL
              value: "INFO"
          args:
          - |
            ulimit -l unlimited && python3 -m dynamo.vllm \
              --model $MODEL_PATH \
              --served-model-name $SERVED_MODEL_NAME \
              --tensor-parallel-size 2 \
              --data-parallel-size 1 \
              --disaggregation-mode prefill \
              --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
              --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:0","enable_kv_cache_events":true}' \
              --gpu-memory-utilization 0.90 \
              --max-model-len 32768 \
              --no-enable-prefix-caching \
              --block-size 128
          command:
          - /bin/sh
          - -c
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          workingDir: /workspace/examples/backends/vllm
          securityContext:
            runAsUser: 0
            capabilities:
              add:
                - IPC_LOCK
                - SYS_RESOURCE
      replicas: 6
      resources:
        limits:
          gpu: "2"
          custom:
            rdma/ib: "2"
        requests:
          gpu: "2"
          custom:
            rdma/ib: "2"
    VllmDecodeWorker:
      componentType: worker
      subComponentType: decode
      envFromSecret: hf-token-secret
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 40Gi
      extraPodSpec: 
        mainContainer:
          ports: []
          env:
            - name: SERVED_MODEL_NAME
              value: "Qwen/Qwen3-32B-FP8"
            - name: MODEL_PATH
              value: "/opt/models/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
            - name: HF_HOME
              value: /opt/models
            - name: VLLM_NIXL_SIDE_CHANNEL_HOST
              valueFrom:
                fieldRef:
                  fieldPath: status.podIP
            - name: UCX_TLS
              value: "rc_x,rc,cuda_copy,cuda_ipc"
            - name: UCX_NET_DEVICES
              value: "mlx5_8:1"
            - name: UCX_IB_ADDR_TYPE
              value: "eth"
            - name: UCX_IB_GID_INDEX
              value: "3"
            - name: UCX_RNDV_SCHEME
              value: "get_zcopy"
            - name: UCX_RNDV_THRESH
              value: "0"
            - name: UCX_IB_REG_METHODS
              value: "odp,rcache"
            - name: UCX_RCACHE_MAX_UNRELEASED
              value: "1024"
            - name: UCX_RC_TIMEOUT
              value: "600s"
            - name: UCX_KEEPALIVE_INTERVAL
              value: "300s"
            - name: UCX_LOG_LEVEL
              value: "info"
            - name: NIXL_LOG_LEVEL
              value: "INFO"
          args:
          - |
            ulimit -l unlimited && python3 -m dynamo.vllm \
              --model $MODEL_PATH \
              --served-model-name $SERVED_MODEL_NAME \
              --tensor-parallel-size 2 \
              --data-parallel-size 1 \
              --disaggregation-mode decode \
              --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' \
              --gpu-memory-utilization 0.90 \
              --max-model-len 32768 \
              --no-enable-prefix-caching \
              --block-size 128
          command:
          - /bin/sh
          - -c
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          workingDir: /workspace/examples/backends/vllm
          securityContext:
            runAsUser: 0
            capabilities:
              add:
                - IPC_LOCK
                - SYS_RESOURCE
      replicas: 2
      resources:
        limits:
          gpu: "2"
          custom:
            rdma/ib: "2"
        requests:
          gpu: "2"
          custom:
            rdma/ib: "2"
EOF_DEPLOY_YAML
```

## 2. Stop old workers before preflight

**Run only on `gpu05`.**

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3-32b/experiments/vllm/disagg-routing-kv-aware
export DGD=qwen3-32b-fp8-vllm-disagg-kv-aware
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=$DGD"

kubectl delete dynamographdeployment qwen3-32b-fp8-vllm-disagg \
  -n "$NAMESPACE" --ignore-not-found
kubectl delete dynamographdeployment "$DGD" \
  -n "$NAMESPACE" --ignore-not-found

kubectl wait --for=delete pod \
  -l nvidia.com/dynamo-graph-deployment-name=qwen3-32b-fp8-vllm-disagg \
  -n "$NAMESPACE" --timeout=5m || true
kubectl wait --for=delete pod -l "$GRAPH_LABEL" \
  -n "$NAMESPACE" --timeout=5m || true

kubectl get pods -n "$NAMESPACE" -o wide
```

### 3.2 Kubernetes gate

**Run only on `gpu05`.**

```bash
kubectl get nodes -L qwen.nvidia.com/role -o wide
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\tGPU="}{.status.allocatable.nvidia\.com/gpu}{"\tRDMA="}{.status.allocatable.rdma/ib}{"\n"}{end}'
```

## 4. Deploy

**Run only on `gpu05`.**

```bash
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"

kubectl get pods -n "$NAMESPACE" \
  -l "$GRAPH_LABEL" \
  -L nvidia.com/dynamo-component-type \
  -o wide -w
```

Expected topology: 1 frontend, 6 prefill workers (TP=2), 2 decode workers (TP=2). Total GPUs = 16 H100.

## 5. Observe startup and diagnose failures

```bash
kubectl logs -n "$NAMESPACE" \
  -l "$GRAPH_LABEL" \
  --all-containers=true \
  --prefix \
  --timestamps \
  --tail=200 \
  --max-log-requests=10 \
  --ignore-errors=true \
  --follow
```

## 6. Readiness and inference smoke test

```bash
kubectl wait --for=condition=Ready pod \
  -l "$GRAPH_LABEL" \
  -n "$NAMESPACE" --timeout=45m

kubectl port-forward -n "$NAMESPACE" \
  service/qwen3-32b-fp8-vllm-disagg-kv-aware-frontend 8000:8000
```

In another terminal:

```bash
curl -fsS http://127.0.0.1:8000/v1/models | jq

curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-32B-FP8",
    "messages": [{"role": "user", "content": "Reply with only: VLLM_KV_AWARE_OK"}],
    "temperature": 0,
    "max_tokens": 32
  }' | jq
```

## 7. Shutdown & Complete GPU VRAM Cleanup

### 7.1 Delete Kubernetes Resources (`gpu05`)

```bash
export NAMESPACE=qwen32-bench

# Delete all Dynamo Graph Deployments in the namespace
kubectl delete dynamographdeployments --all -n "$NAMESPACE" --ignore-not-found

# Force-delete all lingering pods to immediately release GPU claims
kubectl delete pods --all -n "$NAMESPACE" --force --grace-period=0

# Confirm all pods are deleted
kubectl wait --for=delete pod --all -n "$NAMESPACE" --timeout=1m || true
```

### 7.2 Host GPU VRAM Cleanup (`gpu05` & `gpu06`)

Because `hostNetwork: true` and IPC locks are used, orphaned PyTorch / CUDA worker processes can occasionally persist on the host OS after Pod deletion. Run this on **both `gpu05` and `gpu06`**:

```bash
# 1. Kill any lingering vLLM / Dynamo Python processes on the host
pkill -9 -f "dynamo\.vllm|dynamo\.frontend|vllm" || true

# 2. Kill any processes holding /dev/nvidia* device handles
sudo fuser -v /dev/nvidia* 2>/dev/null | awk '{print $2}' | xargs -r sudo kill -9

# 3. Verify all GPU VRAM is completely freed (0 MiB used)
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
```
