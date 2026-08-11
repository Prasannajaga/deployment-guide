# SGLang aggregated routing with KV-cache offloading

Eight TP=2 aggregated workers use SGLang HiCache. Required worker anti-affinity
places one worker on each of eight nodes. GPU HBM is L1 and pinned host RAM is
L2; `--hicache-ratio 2` makes each rank's host KV pool twice its device KV pool.
The frontend consumes KV events and credits host-resident prefixes.

This mirrors `vllm/agg-routing-kv-offloading`, using the native offloader for
each framework.

## Variables

Set these once before running any command in this recipe:

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3-32b/experiments/sglang/agg-routing-kv-offloading
export DEPLOYMENT=qwen3-32b-fp8-sglang-agg-kv-offload
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
export FRONTEND_SERVICE="${DEPLOYMENT}-frontend"
export LOCAL_PORT=8000
```

## Capacity gate

HiCache allocates the host tier per TP rank. Each node runs one worker whose pod
requests 224 GiB and has a 256 GiB memory limit. Verify that capacity after
system reservations, watch memory during startup, and stop if a node approaches
swap or OOM pressure:

```bash
free -h
kubectl top nodes
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
  name: qwen3-32b-fp8-sglang-agg-kv-offload
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
          command:
            - python3
            - -m
            - dynamo.frontend
          args:
            - --router-mode
            - kv
            - --router-host-cache-hit-weight
            - "0.75"
    SglangWorker:
      componentType: worker
      replicas: 8
      envFromSecret: hf-token-secret
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 40Gi
      extraPodSpec:
        affinity:
          podAntiAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
              - labelSelector:
                  matchLabels:
                    nvidia.com/dynamo-graph-deployment-name: qwen3-32b-fp8-sglang-agg-kv-offload
                    nvidia.com/dynamo-component-type: worker
                topologyKey: kubernetes.io/hostname
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0
          workingDir: /workspace/examples/backends/sglang
          command:
            - /bin/sh
            - -c
          args:
            - |
              python3 -m dynamo.sglang \
                --model-path "$MODEL_PATH" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --tp-size 2 \
                --context-length 40960 \
                --page-size 64 \
                --mem-fraction-static 0.82 \
                --trust-remote-code \
                --skip-tokenizer-init \
                --host 0.0.0.0 \
                --enable-hierarchical-cache \
                --hicache-ratio 2 \
                --hicache-write-policy write_through \
                --hicache-io-backend kernel \
                --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}'
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
          memory: 256Gi
        requests:
          gpu: "2"
          cpu: "8"
          memory: 224Gi
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

# Filter for HiCache, host memory, KV events, and errors
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" --all-containers --prefix --tail=500 | grep -Ei 'hicache|host memory|kv.event|error|traceback'

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

Send a repeated long-prefix workload and confirm HiCache allocation, CPU-tier KV
events, stable host RAM, and lower repeat-prefix TTFT.

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

- [Dynamo SGLang HiCache](https://docs.nvidia.com/dynamo/dev/integrations/kv-cache-integrations/hi-cache)
- [SGLang HiCache design](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/hicache_design.mdx)
