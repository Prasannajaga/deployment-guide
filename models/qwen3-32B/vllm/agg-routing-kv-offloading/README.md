# vLLM aggregated routing with KV-cache offloading

Eight TP=2 aggregated workers use vLLM's native `OffloadingConnector`. Required
worker anti-affinity places one worker on each of eight nodes. Each engine
receives a 32 GiB pinned host-memory KV tier shared by its two TP ranks. The
frontend uses event-driven KV routing and credits CPU-resident prefixes.

This is KV-cache offloading, not model-weight `--cpu-offload-gb`.

## Variables

Set these once before running any command in this recipe:

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3-32b/vllm/agg-routing-kv-offloading
export DEPLOYMENT=qwen3-32b-fp8-vllm-agg-kv-offload
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
export FRONTEND_SERVICE="${DEPLOYMENT}-frontend"
export LOCAL_PORT=8000
```

## Capacity gate

Each node runs one worker. The pod requests 64 GiB and limits memory at 96 GiB;
reserve at least 64 GiB of allocatable RAM per node for its 32 GiB KV tier plus
runtime overhead:

```bash
free -h
kubectl top nodes
```

Reduce `cpu_bytes_to_use` and the pod memory request together if the nodes do
not have that headroom.

## Create deploy.yaml

The quoted `EOF` delimiter preserves shell variables inside the manifest.
Running this block writes only the local configuration file.

```bash
mkdir -p "$EXP_DIR"
tee "$EXP_DIR/deploy.yaml" >/dev/null <<'EOF'
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-32b-fp8-vllm-agg-kv-offload
spec:
  backendFramework: vllm
  pvcs:
    - name: model-cache
      create: false
  services:
    Frontend:
      componentType: frontend
      replicas: 1
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
            - --router-host-cache-hit-weight
            - "0.75"
    VllmWorker:
      componentType: worker
      replicas: 8
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
        affinity:
          podAntiAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
              - labelSelector:
                  matchLabels:
                    nvidia.com/dynamo-graph-deployment-name: qwen3-32b-fp8-vllm-agg-kv-offload
                    nvidia.com/dynamo-component-type: worker
                topologyKey: kubernetes.io/hostname
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          workingDir: /workspace/examples/backends/vllm
          command:
            - /bin/sh
            - -c
          args:
            - |
              python3 -m dynamo.vllm \
                --model "$MODEL_PATH" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --tensor-parallel-size 2 \
                --data-parallel-size 1 \
                --gpu-memory-utilization 0.90 \
                --max-model-len 40960 \
                --enable-prefix-caching \
                --block-size 16 \
                --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":34359738368,"block_size":256,"self_describing_kv_events":true}}' \
                --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
          env:
            - name: SERVED_MODEL_NAME
              value: Qwen/Qwen3-32B-FP8
            - name: MODEL_PATH
              value: /opt/models/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df
            - name: HF_HOME
              value: /opt/models
            - name: PYTHONHASHSEED
              value: "0"
          securityContext:
            runAsUser: 0
            capabilities:
              add:
                - IPC_LOCK
                - SYS_RESOURCE
      resources:
        limits:
          gpu: "2"
          cpu: "16"
          memory: 96Gi
        requests:
          gpu: "2"
          cpu: "8"
          memory: 64Gi
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

# Filter for offloading, CPU KV tier, KV events, and errors
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" --all-containers --prefix --tail=500 | grep -Ei 'offload|cpu|kv.event|error|traceback'

# Check worker logs specifically
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL,nvidia.com/dynamo-component-type=worker" --all-containers --prefix --tail=500

# Inspect logs from a previous crashed container instance
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" --all-containers --prefix --previous --tail=500
```

## Verify offloading

```bash
kubectl port-forward -n "$NAMESPACE" \
  service/"$FRONTEND_SERVICE" "$LOCAL_PORT":8000
```

Send repeated requests with the same long prefix. Confirm successful requests,
increasing stored KV-event counters, CPU-tier activity, and stable pod memory.

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

## References

- [Native vLLM KV offloading](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/backends/v-llm/native-kv-offloading)
- [Dynamo KV-cache offloading](https://docs.nvidia.com/dynamo/backends/v-llm/kv-cache-offloading)
