# vLLM disaggregated routing with KV awareness

This experiment uses six TP=2 prefill workers and two TP=2 decode workers. The
frontend consumes worker KV events and routes repeated prefixes toward cached
prefill workers. Total allocation is 16 H100 GPUs.

## Variables

Set these once before running any command in this recipe:

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3-32b/experiments/vllm/disagg-routing-kv-aware
export DEPLOYMENT=qwen3-32b-fp8-vllm-disagg-kv-aware
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
export FRONTEND_SERVICE="${DEPLOYMENT}-frontend"
export LOCAL_PORT=8000
```

## Files

- `deploy.yaml` — deployment and NIXL/UCX settings

## Create deploy.yaml

The quoted `EOF` delimiter preserves shell variables inside the manifest.
Running this block writes only the local configuration file.

```bash
mkdir -p "$EXP_DIR"
tee "$EXP_DIR/deploy.yaml" >/dev/null <<'EOF'
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
            - name: PYTHONHASHSEED
              value: "0"
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
              --max-model-len 40960 \
              --enable-prefix-caching \
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
            - name: PYTHONHASHSEED
              value: "0"
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
              --max-model-len 40960 \
              --enable-prefix-caching \
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
EOF
```

## Deploy

```bash
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl get pods -n "$NAMESPACE" \
  -l "$GRAPH_LABEL" \
  -o wide -w
```

## Check logs

Tail or stream logs for all containers in the deployment:

```bash
# Tail recent 500 lines across all containers
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" --all-containers --prefix --tail=500

# Stream live logs in real time
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" --all-containers --prefix -f

# Filter for NIXL, UCX, KV events, and errors
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" --all-containers --prefix --tail=500 | grep -Ei 'nixl|ucx|kv.event|error|traceback'

# Check prefill worker logs specifically
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL,nvidia.com/dynamo-sub-component-type=prefill" --all-containers --prefix --tail=500

# Check decode worker logs specifically
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL,nvidia.com/dynamo-sub-component-type=decode" --all-containers --prefix --tail=500

# Inspect logs from a previous crashed container instance
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" --all-containers --prefix --previous --tail=500
```

## Acceptance checks

```bash
kubectl port-forward -n "$NAMESPACE" \
  service/"$FRONTEND_SERVICE" "$LOCAL_PORT":8000
```

Send the same long-prefix request twice. Accept the run only if requests
succeed, NIXL initializes, and KV-event counters increase without worker
restarts.

## Clean up

```bash
kubectl delete dynamographdeployment.nvidia.com "$DEPLOYMENT" \
  -n "$NAMESPACE" --wait=false --ignore-not-found
kubectl delete pods -l "$GRAPH_LABEL" -n "$NAMESPACE" \
  --force --grace-period=0 --ignore-not-found
```

The first command prevents the controller from keeping the experiment alive;
the second immediately force-deletes its pods. The local `deploy.yaml` remains
unchanged.
