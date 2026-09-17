# DeepSeek-V4-Flash-FP8 SGLang aggregated TP=4 baseline

Four independent SGLang workers run at TP=4. Each worker performs both prefill
and decode, so the deployment uses all 16 H100 GPUs without expert
parallelism, disaggregated serving, KV-aware routing, KV offloading, or NIXL.

```text
4 replicas x TP=4 = 16 GPUs
```

This is an experimental H100 capacity probe. The checkpoint is approximately
294 GB, so four replicas place approximately 1.176 TB of weights into the
cluster's 1.28 TB of aggregate VRAM. The official SGLang Hopper TP=4 recipe is
validated on H200, not H100. Read the [capacity warning](../../README.md) before
deploying, and run no other GPU workload at the same time.

## Variables

Set these variables in the shell used to create and operate the deployment:

```bash
export NAMESPACE=qwen32-bench
export RECIPE_ROOT=/ephemeral/shared/deepseek-v4-flash-fp8
export EXP_DIR="${RECIPE_ROOT}/sglang/agg"
export MODEL_CACHE_DIR="${RECIPE_ROOT}/model-cache"
export DEPLOYMENT=deepseek-v4-flash-fp8-sglang-agg-tp4
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
export FRONTEND_SERVICE="${DEPLOYMENT}-frontend"
export LOCAL_PORT=8000
export BENCH_URL="http://127.0.0.1:${LOCAL_PORT}"
export MODEL=sgl-project/DeepSeek-V4-Flash-FP8
export MODEL_REVISION=ae01d80c06cdfe30581edfd0e1c5449dc7ed7f17
```

`EXP_DIR` is the cluster's shared checkout location. Change `RECIPE_ROOT` if
the repository was copied elsewhere. Change `NAMESPACE` only if the same
namespace already contains the shared `model-cache` PVC and the Dynamo
operator can watch it.

## Files

| File | Purpose |
|---|---|
| `deploy.yaml` | Aggregated TP=4 `DynamoGraphDeployment` manifest |
| `../../model-cache/model-download.yaml` | Pinned model-cache population Job |

## Prerequisites and capacity checks

Run these checks from a Kubernetes administrator shell:

```bash
kubectl get crd dynamographdeployments.nvidia.com
kubectl get pvc model-cache -n "$NAMESPACE"
kubectl get nodes \
  -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
kubectl get dynamographdeployments.nvidia.com -n "$NAMESPACE"
kubectl get pods -A \
  -o custom-columns='NAMESPACE:.metadata.namespace,POD:.metadata.name,NODE:.spec.nodeName,GPU:.spec.containers[*].resources.requests.nvidia\.com/gpu' \
  | grep -v '<none>' || true
```

Continue only when:

- the Dynamo CRD exists;
- `model-cache` is `Bound`, shared by both GPU nodes, and has at least 350 GiB
  available for the pinned snapshot and cache metadata;
- the two nodes expose 16 H100 80 GB GPUs in total;
- no other deployment is consuming those GPUs.

Every worker requests four GPUs. Kubernetes therefore keeps each TP group
inside one 8-GPU node and should place two workers on each node. This recipe
does not request RDMA resources.

## Create and populate the model cache

If the pinned snapshot is not already present, create the download Job. The
quoted `EOF` keeps the container-side `$MODEL_NAME` and `$MODEL_REVISION`
variables intact.

```bash
mkdir -p "$MODEL_CACHE_DIR"
tee "$MODEL_CACHE_DIR/model-download.yaml" >/dev/null <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: deepseek-v4-flash-fp8-download
spec:
  backoffLimit: 3
  template:
    metadata:
      labels:
        app: deepseek-v4-flash-fp8-download
    spec:
      restartPolicy: Never
      containers:
        - name: model-download
          image: python:3.10-slim
          command:
            - /bin/sh
            - -c
          args:
            - |
              set -eu
              pip install --no-cache-dir huggingface_hub==1.16.4
              hf download "$MODEL_NAME" --revision "$MODEL_REVISION"
          env:
            - name: MODEL_NAME
              value: sgl-project/DeepSeek-V4-Flash-FP8
            - name: MODEL_REVISION
              value: ae01d80c06cdfe30581edfd0e1c5449dc7ed7f17
            - name: HF_HOME
              value: /model-store
            - name: HF_XET_HIGH_PERFORMANCE
              value: "1"
          envFrom:
            - secretRef:
                name: hf-token-secret
                optional: true
          resources:
            requests:
              cpu: "4"
              memory: 64Gi
            limits:
              cpu: "8"
              memory: 64Gi
          volumeMounts:
            - name: model-cache
              mountPath: /model-store
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache
EOF
```

Validate the Job against the cluster API, apply it, and wait for the exact
revision to finish downloading:

```bash
kubectl apply --dry-run=server -n "$NAMESPACE" \
  -f "$MODEL_CACHE_DIR/model-download.yaml"
kubectl apply -n "$NAMESPACE" -f "$MODEL_CACHE_DIR/model-download.yaml"
kubectl wait -n "$NAMESPACE" \
  --for=condition=Complete job/deepseek-v4-flash-fp8-download \
  --timeout=7200s
kubectl logs -n "$NAMESPACE" job/deepseek-v4-flash-fp8-download --tail=100
```

Confirm that the exact pinned snapshot is present and inspect remaining cache
capacity:

```bash
export DOWNLOAD_POD="$(kubectl get pod -n "$NAMESPACE" \
  -l app=deepseek-v4-flash-fp8-download \
  -o jsonpath='{.items[0].metadata.name}')"
kubectl exec -n "$NAMESPACE" "$DOWNLOAD_POD" -- \
  test -f "/model-store/hub/models--sgl-project--DeepSeek-V4-Flash-FP8/snapshots/${MODEL_REVISION}/config.json"
kubectl exec -n "$NAMESPACE" "$DOWNLOAD_POD" -- df -h /model-store
```

## Create deploy.yaml

The quoted `EOF` delimiter preserves `$MODEL_PATH` and `$SERVED_MODEL_NAME` for
expansion inside each worker container. It creates or replaces only this
recipe's local `deploy.yaml`.

```bash
mkdir -p "$EXP_DIR"
tee "$EXP_DIR/deploy.yaml" >/dev/null <<'EOF'
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: deepseek-v4-flash-fp8-sglang-agg-tp4
spec:
  backendFramework: sglang
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
          image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0
          workingDir: /workspace/examples/backends/sglang
      envs:
        - name: HF_HOME
          value: /opt/models
      replicas: 1
      resources:
        requests:
          cpu: "8"
          memory: 32Gi
        limits:
          memory: 64Gi
    SglangWorker:
      componentType: worker
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 64Gi
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0
          workingDir: /workspace/examples/backends/sglang
          command:
            - /bin/sh
            - -c
          args:
            - |
              exec python3 -m dynamo.sglang \
                --model-path "$MODEL_PATH" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --tp-size 8 \
                --context-length 8192 \
                --chunked-prefill-size 4096 \
                --page-size 64 \
                --max-running-requests 8 \
                --mem-fraction-static 0.90 \
                --disable-cuda-graph \
                --trust-remote-code \
                --reasoning-parser deepseek-v4 \
                --dyn-reasoning-parser deepseek_v4 \
                --dyn-tool-call-parser deepseek_v4 \
                --host 0.0.0.0 \
                --enable-metrics
          env:
            - name: SERVED_MODEL_NAME
              value: sgl-project/DeepSeek-V4-Flash-FP8
            - name: MODEL_PATH
              value: /opt/models/hub/models--sgl-project--DeepSeek-V4-Flash-FP8/snapshots/ae01d80c06cdfe30581edfd0e1c5449dc7ed7f17
            - name: HF_HOME
              value: /opt/models
            - name: PYTHONHASHSEED
              value: "0"
            - name: SGLANG_DSV4_FP4_EXPERTS
              value: "0"
            - name: SGLANG_JIT_DEEPGEMM_PRECOMPILE
              value: "1"
            - name: SGLANG_JIT_DEEPGEMM_FAST_WARMUP
              value: "1"
      replicas: 2
      resources:
        limits:
          gpu: "8"
        requests:
          gpu: "8"
EOF
```

## Validate and deploy

Validate the local manifest against the installed Dynamo CRD before changing
cluster state:

```bash
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
```

Apply it and watch scheduling and startup:

```bash
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl get pods -n "$NAMESPACE" \
  -l "$GRAPH_LABEL" \
  -L nvidia.com/dynamo-component-type \
  -o wide -w
```

Expected result: one frontend and four worker Pods. Every worker requests four
GPUs, so a fully ready deployment consumes all 16 GPUs and normally places two
workers on each 8-GPU node.

In a separate terminal, wait for all graph Pods to become ready. Model loading
can take a long time for four independent copies:

```bash
kubectl wait -n "$NAMESPACE" \
  --for=condition=Ready pod \
  -l "$GRAPH_LABEL" \
  --timeout=2400s
```

## Check frontend and SGLang worker logs

Show recent logs from every graph container:

```bash
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --all-containers --prefix --tail=500
```

Stream only the SGLang workers:

```bash
kubectl logs -n "$NAMESPACE" \
  -l "$GRAPH_LABEL,nvidia.com/dynamo-component-type=worker" \
  --all-containers --prefix -f --max-log-requests=4
```

Stream only the frontend:

```bash
kubectl logs -n "$NAMESPACE" \
  -l "$GRAPH_LABEL,nvidia.com/dynamo-component-type=frontend" \
  --all-containers --prefix -f
```

Inspect one worker when interleaved logs are too noisy:

```bash
export WORKER_POD="$(kubectl get pods -n "$NAMESPACE" \
  -l "$GRAPH_LABEL,nvidia.com/dynamo-component-type=worker" \
  -o jsonpath='{.items[0].metadata.name}')"
kubectl logs -n "$NAMESPACE" "$WORKER_POD" \
  --all-containers --prefix -f
```

Search for load failures, CUDA OOMs, unsupported architecture/parser errors,
and inspect the previous container attempt after a restart:

```bash
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --all-containers --prefix --tail=2000 \
  | grep -Ei 'error|exception|traceback|out of memory|oom|unsupported|failed' || true

kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --all-containers --prefix --previous --tail=1000
```

Confirm that there are four workers, each received four GPUs, placement is
balanced, and no container is restarting:

```bash
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  -o custom-columns='POD:.metadata.name,NODE:.spec.nodeName,READY:.status.containerStatuses[*].ready,RESTARTS:.status.containerStatuses[*].restartCount,GPU:.spec.containers[*].resources.requests.nvidia\.com/gpu'
```

If any worker dies with CUDA OOM during weight loading or memory profiling,
this TP=4 H100 layout does not fit. Repeated restarts will not fix it; remove
the deployment using the cleanup command below. The first practical fallback
is two replicas at TP=8.

## Smoke test

Forward the private frontend Service from an administrator terminal:

```bash
kubectl port-forward -n "$NAMESPACE" \
  service/"$FRONTEND_SERVICE" "$LOCAL_PORT":8000
```

Keep the port-forward running. In another terminal, export the request
variables and send a deterministic request with thinking disabled:

```bash
export BENCH_URL=http://127.0.0.1:8000
export MODEL=sgl-project/DeepSeek-V4-Flash-FP8

curl -fsS "$BENCH_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON
{
  "model": "$MODEL",
  "messages": [
    {"role": "user", "content": "Reply with exactly: ready"}
  ],
  "chat_template_kwargs": {"thinking": false},
  "temperature": 0,
  "max_tokens": 16
}
JSON
```

Pass criteria:

- the request succeeds without a 4xx/5xx response;
- the response model is `sgl-project/DeepSeek-V4-Flash-FP8`;
- the content is `ready` apart from harmless whitespace or punctuation;
- all four workers remain ready with zero restarts.

## Cleanup

Stop the port-forward with `Ctrl-C`, then remove the graph deployment:

```bash
kubectl delete dynamographdeployment.nvidia.com "$DEPLOYMENT" \
  -n "$NAMESPACE" --wait=false --ignore-not-found
```

Watch until the controller-owned Pods disappear:

```bash
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -w
```

If terminated Pods remain stuck after the graph has been deleted, force-delete
only Pods carrying this deployment's exact graph label:

```bash
kubectl delete pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --force --grace-period=0 --ignore-not-found
```

The cleanup does not delete the shared `model-cache` PVC, the pinned snapshot,
the namespace, or either local YAML file. Delete the completed download Job
only when its Pod is no longer needed for cache inspection:

```bash
kubectl delete job deepseek-v4-flash-fp8-download \
  -n "$NAMESPACE" --ignore-not-found
```
