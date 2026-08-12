# SGLang aggregated routing

Eight aggregated SGLang workers run at TP=2. Required worker anti-affinity places
one worker on each of eight nodes, allocating 16 H100 GPUs. Each worker performs
prefill and decode. This mirrors `vllm/agg-routing` without a host KV-cache tier.

## Variables

Set these once before running any command in this recipe:

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3-32b/experiments/sglang/agg-routing
export DEPLOYMENT=qwen3-32b-fp8-sglang-agg-tp2
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
export FRONTEND_SERVICE="${DEPLOYMENT}-frontend"
export LOCAL_PORT=8000
export BENCH_URL="http://127.0.0.1:${LOCAL_PORT}"
export MODEL=Qwen/Qwen3-32B-FP8
```

## Create deploy.yaml

The quoted `EOF` delimiter preserves shell variables inside the manifest.
Running this block writes only the local configuration file.

```bash
mkdir -p "$EXP_DIR"
sudo tee "$EXP_DIR/deploy.yaml" >/dev/null <<'EOF'
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-32b-fp8-sglang-agg-tp2
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
    SglangWorker:
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
            preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm
               labelSelector:
                  matchLabels:
                    nvidia.com/dynamo-graph-deployment-name: qwen3-32b-fp8-sglang-agg-tp2
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
                --mem-fraction-static 0.90 \
                --trust-remote-code \
                --skip-tokenizer-init \
                --host 0.0.0.0 \
                --enable-metrics
          env:
            - name: SERVED_MODEL_NAME
              value: Qwen/Qwen3-32B-FP8
            - name: MODEL_PATH
              value: /opt/models/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df
            - name: HF_HOME
              value: /opt/models
      resources:
        limits:
          gpu: "2"
        requests:
          gpu: "2"
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

# Filter for errors and tracebacks
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" --all-containers --prefix --tail=500 | grep -Ei 'error|traceback'

# Check worker logs specifically
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL,nvidia.com/dynamo-component-type=worker" --all-containers --prefix --tail=500

# Inspect logs from a previous crashed container instance
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" --all-containers --prefix --previous --tail=500
```


## Test

```bash
kubectl port-forward -n "$NAMESPACE" \
  service/"$FRONTEND_SERVICE" "$LOCAL_PORT":8000
```

```bash
curl -fsS "$BENCH_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<EOF
  {
    "model":"$MODEL",
    "messages":[{"role":"user","content":"Reply with: ready"}],
    "temperature":0,
    "max_tokens":16
  }
EOF
```

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
