# SGLang aggregated KEDA autoscaling

The deployment starts with one aggregated SGLang worker at TP=2. KEDA can
scale the worker service from one to eight replicas using Dynamo frontend
active-request count, while the frontend remains at one replica. Each worker
performs both prefill and decode and consumes two H100 GPUs. The topology does
not use data parallelism, expert parallelism, KV-aware routing, KV offloading,
or NIXL.

The frontend retains enough CPU and memory to remain stable during load. Do
not deliberately throttle it to manufacture queue depth: that can reset
clients and measures frontend starvation rather than worker demand. The KEDA
trigger uses `dynamo_frontend_active_requests`, Dynamo 1.3's gauge for requests
from frontend entry through completion.

This is the aggregated comparison point for the matching
[disaggregated recipe](../disagg/README.md). The complete KEDA, Prometheus,
Grafana, load-test, and live Pod workflow is in
[autoscaling.md](autoscaling.md). Do not run another GPU experiment while
allowing this recipe to scale to its eight-worker maximum.

## Variables

Set these variables in the shell used to create and operate the deployment:

```bash
export NAMESPACE=dynamo-bench
export RECIPE_ROOT=/ephemeral/shared/qwen3.6-35b-a3b
export EXP_DIR="${RECIPE_ROOT}/sglang/agg-autoscaling"
export MODEL_CACHE_DIR="${RECIPE_ROOT}/model-cache"
export MODEL_DOWNLOAD_JOB=qwen36-35b-a3b-fp8-download
export DEPLOYMENT=qwen36-35b-a3b-fp8-sglang-agg-tp2
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
export FRONTEND_SERVICE="${DEPLOYMENT}-frontend"
export LOCAL_PORT=8000
export BENCH_URL="http://127.0.0.1:${LOCAL_PORT}"
export MODEL=Qwen/Qwen3.6-35B-A3B-FP8
export MODEL_REVISION=95a723d08a9490559dae23d0cff1d9466213d989
```

`EXP_DIR` is the cluster's shared checkout location. Change `RECIPE_ROOT` if
the repository was copied elsewhere, but keep every other value unchanged so
the commands and manifest labels continue to match.

## Files

| File | Purpose |
|---|---|
| `preflight.yaml` | Isolated one-worker TP=2 compatibility canary |
| `deploy.yaml` | One-worker `DynamoGraphDeployment` with a worker scaling adapter |
| `scaledobject.yaml` | KEDA active-request scaler for the worker DGDSA |
| `autoscaling.md` | KEDA, Prometheus, Grafana, load-test, and live scaling runbook |
| `../../model-cache/model-download.yaml` | Optional pinned model-cache population Job |

## Prerequisites

Run these checks from a Kubernetes administrator shell before creating the
manifest:

```bash
kubectl get crd dynamographdeployments.nvidia.com
kubectl get crd dynamographdeploymentscalingadapters.nvidia.com
kubectl get pvc model-cache -n "$NAMESPACE"
kubectl get nodes \
  -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
kubectl get dynamographdeployments.nvidia.com -n "$NAMESPACE"
```

Continue only when:

- the Dynamo deployment and scaling-adapter CRDs exist;
- `model-cache` is `Bound` and is accessible from both GPU nodes;
- at least two H100 GPUs are available for the initial worker;
- up to 16 H100 GPUs are available if KEDA may reach eight workers;
- no other experiment competes for the GPU capacity KEDA is allowed to use.

This aggregated topology does not request RDMA resources and does not require
the `roce` NetworkAttachmentDefinition. Complete this README first, then
continue with [autoscaling.md](autoscaling.md) to install and configure KEDA.

## Create model-download.yaml

If the pinned snapshot has not been downloaded, create the cache-population
Job. The quoted `EOF` keeps the container-side `$MODEL_NAME` and
`$MODEL_REVISION` variables intact.

```bash
mkdir -p "$MODEL_CACHE_DIR"
tee "$MODEL_CACHE_DIR/model-download.yaml" >/dev/null <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: qwen36-35b-a3b-fp8-download
spec:
  backoffLimit: 3
  template:
    metadata:
      labels:
        app: qwen36-35b-a3b-fp8-download
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
              value: Qwen/Qwen3.6-35B-A3B-FP8
            - name: MODEL_REVISION
              value: 95a723d08a9490559dae23d0cff1d9466213d989
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
              cpu: "2"
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
kubectl delete job "$MODEL_DOWNLOAD_JOB" -n "$NAMESPACE" \
  --ignore-not-found
kubectl apply --dry-run=server -n "$NAMESPACE" \
  -f "$MODEL_CACHE_DIR/model-download.yaml"
kubectl apply -n "$NAMESPACE" -f "$MODEL_CACHE_DIR/model-download.yaml"
kubectl wait -n "$NAMESPACE" \
  --for=condition=Complete "job/$MODEL_DOWNLOAD_JOB" \
  --timeout=3600s
kubectl logs -n "$NAMESPACE" "job/$MODEL_DOWNLOAD_JOB" --tail=100
```

## Create deploy.yaml

The quoted `EOF` delimiter is intentional: it preserves `$MODEL_PATH` and
`$SERVED_MODEL_NAME` for expansion inside each worker container instead of in
the administrator's shell. This command creates or replaces only the recipe's
local `deploy.yaml`.

```bash
mkdir -p "$EXP_DIR"
tee "$EXP_DIR/deploy.yaml" >/dev/null <<'EOF'
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: qwen36-35b-a3b-fp8-sglang-agg-tp2
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
      scalingAdapter:
        enabled: true
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 40Gi
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
                --tp-size 2 \
                --context-length 131072 \
                --page-size 64 \
                --mem-fraction-static 0.85 \
                --reasoning-parser qwen3 \
                --dyn-reasoning-parser qwen3 \
                --dyn-tool-call-parser qwen3_coder \
                --host 0.0.0.0 \
                --enable-metrics
          env:
            - name: SERVED_MODEL_NAME
              value: Qwen/Qwen3.6-35B-A3B-FP8
            - name: MODEL_PATH
              value: /opt/models/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989
            - name: HF_HOME
              value: /opt/models
            - name: PYTHONHASHSEED
              value: "0"
      replicas: 1
      resources:
        limits:
          gpu: "2"
        requests:
          gpu: "2"
EOF
```

## Validate and deploy

Validate against the installed Dynamo CRD before changing cluster state:

```bash
kubectl apply --dry-run=server -n "$NAMESPACE" \
  -f "$EXP_DIR/deploy.yaml"
```

Apply the manifest and watch placement:

```bash
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl get pods -n "$NAMESPACE" \
  -l "$GRAPH_LABEL" \
  -L nvidia.com/dynamo-component-type \
  -o wide -w
```

Expected initial result: one frontend and one worker Pod. The worker requests
two GPUs. `replicas: 1` is both the deployment seed and the safe minimum;
[autoscaling.md](autoscaling.md) configures KEDA to retain that minimum while
scaling as high as eight workers.

The frontend should report an eight-CPU request with no CPU limit. Confirm the
applied resources before generating load:

```bash
kubectl get pods -n "$NAMESPACE" \
  -l "$GRAPH_LABEL,nvidia.com/dynamo-component-type=frontend" \
  -o custom-columns='POD:.metadata.name,CPU_REQUEST:.spec.containers[*].resources.requests.cpu,CPU_LIMIT:.spec.containers[*].resources.limits.cpu,MEMORY_REQUEST:.spec.containers[*].resources.requests.memory,MEMORY_LIMIT:.spec.containers[*].resources.limits.memory'
```

In a separate terminal, wait for every deployment Pod to become ready:

```bash
kubectl wait -n "$NAMESPACE" \
  --for=condition=Ready pod \
  -l "$GRAPH_LABEL" \
  --timeout=1800s
```

## Check logs and GPU allocation

Tail recent logs across the complete graph:

```bash
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --all-containers --prefix --tail=500
```

Stream worker logs:

```bash
kubectl logs -n "$NAMESPACE" \
  -l "$GRAPH_LABEL,nvidia.com/dynamo-component-type=worker" \
  --all-containers --prefix -f
```

Check for startup failures and inspect a previously crashed container when
needed:

```bash
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --all-containers --prefix --tail=1000 \
  | grep -Ei 'error|exception|traceback|oom|failed' || true

kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --all-containers --prefix --previous --tail=500
```

Confirm that the initial worker received two GPUs and that no Pod is
restarting:

```bash
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  -o custom-columns='POD:.metadata.name,NODE:.spec.nodeName,READY:.status.containerStatuses[*].ready,RESTARTS:.status.containerStatuses[*].restartCount,GPU:.spec.containers[*].resources.requests.nvidia\.com/gpu'
```

## Smoke test

Forward the private frontend Service from an administrator terminal:

```bash
kubectl port-forward -n "$NAMESPACE" \
  service/"$FRONTEND_SERVICE" "$LOCAL_PORT":8000
```

Keep the port-forward running. In another terminal, export the same request
variables and send a deterministic non-thinking request:

```bash
export BENCH_URL=http://127.0.0.1:8000
export MODEL=Qwen/Qwen3.6-35B-A3B-FP8

curl -fsS "$BENCH_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON
{
  "model": "$MODEL",
  "messages": [
    {"role": "user", "content": "Reply with exactly: ready"}
  ],
  "chat_template_kwargs": {"enable_thinking": false},
  "temperature": 0,
  "max_tokens": 16
}
JSON
```

Pass criteria:

- HTTP request succeeds without a 4xx/5xx response;
- response model is `Qwen/Qwen3.6-35B-A3B-FP8`;
- response content is `ready` apart from harmless whitespace or punctuation;
- the frontend and initial worker remain ready with zero restarts.

After the smoke test passes, follow [autoscaling.md](autoscaling.md) to apply
the KEDA `ScaledObject`, create the Grafana panels, generate load, and watch
worker Pods scale live.

## Cleanup

Stop the port-forward with `Ctrl-C`. If KEDA was configured, remove its
`ScaledObject` before removing the graph deployment:

```bash
kubectl delete scaledobject qwen36-35b-a3b-sglang-worker \
  -n "$NAMESPACE" --ignore-not-found
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

The cleanup does not delete the shared `model-cache` PVC, the pinned model
snapshot, the namespace, or the local `deploy.yaml`.
