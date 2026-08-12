# SGLang prefill/decode disaggregation

This mirrors `vllm/disagg-routing`: two TP=2 prefill workers, one TP=4 decode
worker, NIXL over the cluster's UCX/RDMA path, and eight H100 GPUs total.

## Variables

Set these once before running any command in this recipe:

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3-32b/experiments/sglang/disagg-routing
export DEPLOYMENT=qwen3-32b-fp8-sglang-disagg
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
export FRONTEND_SERVICE="${DEPLOYMENT}-frontend"
export LOCAL_PORT=8000
```

## Create deploy.yaml

The quoted `EOF` delimiter preserves shell variables inside the manifest.
Running this block writes only the local configuration file.

```bash
mkdir -p "$EXP_DIR"
tee "$EXP_DIR/deploy.yaml" >/dev/null <<'EOF'
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-32b-fp8-sglang-disagg
spec:
  backendFramework: sglang
  pvcs:
    - name: model-cache
      create: false
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0
          workingDir: /workspace/examples/backends/sglang
    SglangPrefillWorker:
      componentType: worker
      subComponentType: prefill
      replicas: 2
      envFromSecret: hf-token-secret
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 40Gi
      extraPodMetadata:
        annotations:
          k8s.v1.cni.cncf.io/networks: qwen32-bench/qwen-roce
      extraPodSpec:
        hostNetwork: false
        dnsPolicy: ClusterFirst
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0
          workingDir: /workspace/examples/backends/sglang
          command:
            - /bin/sh
            - -c
          args:
            - |
              ulimit -l unlimited && python3 -m dynamo.sglang \
                --model-path "$MODEL_PATH" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --tp-size 2 \
                --context-length 40960 \
                --page-size 64 \
                --mem-fraction-static 0.90 \
                --trust-remote-code \
                --skip-tokenizer-init \
                --disaggregation-mode prefill \
                --disaggregation-transfer-backend nixl \
                --disaggregation-bootstrap-port 30001 \
                --host 0.0.0.0
          env:
            - name: SERVED_MODEL_NAME
              value: Qwen/Qwen3-32B-FP8
            - name: MODEL_PATH
              value: /opt/models/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df
            - name: HF_HOME
              value: /opt/models
            - name: DYN_SYSTEM_PORT
              value: "9081"
            - name: SGLANG_DISAGGREGATION_NIXL_BACKEND
              value: UCX
            - name: UCX_TLS
              value: rc_x,rc,cuda_copy,cuda_ipc
            - name: UCX_NET_DEVICES
              value: mlx5_8:1
            - name: UCX_IB_ADDR_TYPE
              value: eth
            - name: UCX_IB_GID_INDEX
              value: "3"
          ports:
            - name: system
              containerPort: 9081
              hostPort: 9081
            - name: bootstrap
              containerPort: 30001
              hostPort: 30001
          securityContext:
            runAsUser: 0
            capabilities:
              add:
                - IPC_LOCK
                - SYS_RESOURCE
      resources:
        limits:
          gpu: "2"
          custom:
            rdma/ib: "2"
        requests:
          gpu: "2"
          custom:
            rdma/ib: "2"
    SglangDecodeWorker:
      componentType: worker
      subComponentType: decode
      replicas: 1
      envFromSecret: hf-token-secret
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 40Gi
      extraPodSpec:
        hostNetwork: false
        dnsPolicy: ClusterFirst
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0
          workingDir: /workspace/examples/backends/sglang
          command:
            - /bin/sh
            - -c
          args:
            - |
              ulimit -l unlimited && python3 -m dynamo.sglang \
                --model-path "$MODEL_PATH" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --tp-size 4 \
                --context-length 40960 \
                --page-size 64 \
                --mem-fraction-static 0.90 \
                --trust-remote-code \
                --skip-tokenizer-init \
                --disaggregation-mode decode \
                --disaggregation-transfer-backend nixl \
                --disaggregation-bootstrap-port 30002 \
                --host 0.0.0.0
          env:
            - name: SERVED_MODEL_NAME
              value: Qwen/Qwen3-32B-FP8
            - name: MODEL_PATH
              value: /opt/models/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df
            - name: HF_HOME
              value: /opt/models
            - name: DYN_SYSTEM_PORT
              value: "9082"
            - name: SGLANG_DISAGGREGATION_NIXL_BACKEND
              value: UCX
            - name: UCX_TLS
              value: rc_x,rc,cuda_copy,cuda_ipc
            - name: UCX_NET_DEVICES
              value: mlx5_8:1
            - name: UCX_IB_ADDR_TYPE
              value: eth
            - name: UCX_IB_GID_INDEX
              value: "3"
          ports:
            - name: system
              containerPort: 9082
              hostPort: 9082
            - name: bootstrap
              containerPort: 30002
              hostPort: 30002
          securityContext:
            runAsUser: 0
            capabilities:
              add:
                - IPC_LOCK
                - SYS_RESOURCE
      resources:
        limits:
          gpu: "4"
          custom:
            rdma/ib: "4"
        requests:
          gpu: "4"
          custom:
            rdma/ib: "4"

EOF
```

## Deploy

```bash
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
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

# Filter for NIXL, UCX, bootstrap, and errors
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" --all-containers --prefix --tail=500 | grep -Ei 'nixl|ucx|bootstrap|error|traceback'

# Check prefill worker logs specifically
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL,nvidia.com/dynamo-sub-component-type=prefill" --all-containers --prefix --tail=500

# Check decode worker logs specifically
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL,nvidia.com/dynamo-sub-component-type=decode" --all-containers --prefix --tail=500

# Inspect logs from a previous crashed container instance
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" --all-containers --prefix --previous --tail=500
```

## Verify

```bash
kubectl port-forward -n "$NAMESPACE" \
  service/"$FRONTEND_SERVICE" "$LOCAL_PORT":8000
```

Accept only after all workers are ready, the NIXL UCX backend initializes, and
the standard chat-completions smoke test succeeds.

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

## Reference

- [Dynamo SGLang disaggregation](https://docs.nvidia.com/dynamo/latest/backends/sg-lang/disaggregation)
