# GLM-5.2-FP8 aggregated 16-GPU recipe

This recipe serves the pinned public checkpoint `zai-org/GLM-5.2-FP8` as one
aggregated SGLang worker on the repository's two-node, 16 x H100 80 GB
cluster. It uses NVIDIA Dynamo 1.3.0 and the existing pod-native RoCE network.
The Dynamo 1.3.0 runtime contains SGLang 0.5.14, which satisfies GLM-5.2's
upstream minimum of SGLang 0.5.13.post1.

```text
1 model copy x (2 nodes x 8 GPUs) = 16 GPUs
TP=16 for attention and dense layers
EP=16 for 256 routed experts = 16 experts per rank
```

The checkpoint is 756 GB. TP=8 would place about 94.5 GB of checkpoint data on
each 80 GB H100 and cannot fit. TP=16 places about 47.25 GB per GPU before
runtime allocations. A second model copy or prefill/decode disaggregation
would require about 1.51 TB of weights and cannot fit in the cluster's 1.28 TB
of aggregate HBM.

This is a safe bring-up baseline, not a claimed benchmark result. It starts at
a 131,072-token context, 0.80 static-memory fraction, 8,192-token prefill
chunks, and 16 running requests. Hopper uses SGLang's automatic BF16 DSA KV
cache selection. MTP speculative decoding, KV offload, and disaggregation are
deliberately disabled until the baseline passes.

## Files

```text
glm-5.2-fp8/
├── README.md
├── model-cache/
│   └── model-download.yaml
└── sglang/agg/
    └── deploy.yaml
```

## 1. Set variables

Run all commands from a Kubernetes administrator shell:

```bash
export NAMESPACE=qwen32-bench
export RECIPE_ROOT=/ephemeral/shared/glm-5.2-fp8
export MODEL_CACHE_DIR="${RECIPE_ROOT}/model-cache"
export EXP_DIR="${RECIPE_ROOT}/sglang/agg"
export DEPLOYMENT=glm52-fp8-sglang-agg-tp16
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
export FRONTEND_SERVICE="${DEPLOYMENT}-frontend"
export MODEL=zai-org/GLM-5.2-FP8
export MODEL_REVISION=ba978f7d347eaf65d22f1a86833408afdb953541
export LOCAL_PORT=8000
```

## 2. Preflight

The multinode worker requires Dynamo's Grove orchestration, two nodes with
eight free H100 GPUs each, and the existing RoCE attachment and RDMA resource.
The shared PVC needs at least 850 GiB free for the 756 GB snapshot plus cache
metadata and download headroom.

```bash
kubectl get crd \
  dynamographdeployments.nvidia.com \
  podcliquesets.grove.io \
  network-attachment-definitions.k8s.cni.cncf.io

kubectl get pvc model-cache -n "$NAMESPACE"
kubectl get network-attachment-definition qwen-roce -n "$NAMESPACE"
kubectl get nodes \
  -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,RDMA:.status.allocatable.rdma/ib'
kubectl get pods -A \
  -o custom-columns='NAMESPACE:.metadata.namespace,POD:.metadata.name,NODE:.spec.nodeName,GPU:.spec.containers[*].resources.requests.nvidia\.com/gpu' \
  | grep -v '<none>' || true
```

Continue only when both nodes are ready, all 16 GPUs and at least eight
`rdma/ib` shares per node are free, `model-cache` is `Bound` and shared by both
nodes, `qwen-roce` exists, and no other 16-GPU graph is running.

## 3. Populate the model cache

Create the pinned download Job. The checkpoint is public; the Job uses
`hf-token-secret` only when that secret already exists.

```bash
mkdir -p "$MODEL_CACHE_DIR"
tee "$MODEL_CACHE_DIR/model-download.yaml" >/dev/null <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: glm52-fp8-download
spec:
  backoffLimit: 2
  activeDeadlineSeconds: 43200
  template:
    metadata:
      labels:
        app: glm52-fp8-download
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

              SNAPSHOT_DIR="$HF_HOME/hub/models--zai-org--GLM-5.2-FP8/snapshots/$MODEL_REVISION"
              test -s "$SNAPSHOT_DIR/config.json"
              test -s "$SNAPSHOT_DIR/model.safetensors.index.json"
              test -s "$SNAPSHOT_DIR/model-00141-of-00141.safetensors"
              du -sh "$SNAPSHOT_DIR"
          env:
            - name: MODEL_NAME
              value: zai-org/GLM-5.2-FP8
            - name: MODEL_REVISION
              value: ba978f7d347eaf65d22f1a86833408afdb953541
            - name: HF_HOME
              value: /model-store
            - name: HF_XET_HIGH_PERFORMANCE
              value: "1"
            - name: HF_HUB_DOWNLOAD_TIMEOUT
              value: "120"
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

Validate, apply, and wait. The Job checks the config, index, and final weight
shard before it can complete.

```bash
kubectl apply --dry-run=server -n "$NAMESPACE" \
  -f "$MODEL_CACHE_DIR/model-download.yaml"
kubectl apply -n "$NAMESPACE" -f "$MODEL_CACHE_DIR/model-download.yaml"
kubectl wait -n "$NAMESPACE" \
  --for=condition=Complete job/glm52-fp8-download \
  --timeout=43200s
kubectl logs -n "$NAMESPACE" job/glm52-fp8-download --tail=100
```

If a failed Job must be retried, delete only the Job and apply it again. The
partial Hugging Face cache remains on the PVC and the next run resumes it.

```bash
kubectl delete job glm52-fp8-download \
  -n "$NAMESPACE" --ignore-not-found
kubectl apply -n "$NAMESPACE" -f "$MODEL_CACHE_DIR/model-download.yaml"
```

## 4. Create the deployment manifest

The Dynamo operator adds the SGLang `--nnodes`, `--node-rank`, and
`--dist-init-addr` arguments for the two Pods in this worker group. Do not add
them manually.

```bash
mkdir -p "$EXP_DIR"
tee "$EXP_DIR/deploy.yaml" >/dev/null <<'EOF'
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: glm52-fp8-sglang-agg-tp16
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
          cpu: "16"
          memory: 64Gi
    SglangWorker:
      componentType: worker
      multinode:
        nodeCount: 2
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 128Gi
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
              set -eu
              ulimit -l unlimited
              exec python3 -m dynamo.sglang \
                --model-path "$MODEL_PATH" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --tp-size 16 \
                --ep-size 16 \
                --moe-a2a-backend deepep \
                --deepep-mode auto \
                --moe-runner-backend deep_gemm \
                --context-length 131072 \
                --chunked-prefill-size 8192 \
                --page-size 64 \
                --max-running-requests 16 \
                --mem-fraction-static 0.80 \
                --watchdog-timeout 1200 \
                --dsa-topk-backend sgl-kernel \
                --reasoning-parser glm45 \
                --dyn-reasoning-parser glm45 \
                --dyn-tool-call-parser glm47 \
                --host 0.0.0.0 \
                --enable-metrics
          env:
            - name: SERVED_MODEL_NAME
              value: zai-org/GLM-5.2-FP8
            - name: MODEL_PATH
              value: /opt/models/hub/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541
            - name: HF_HOME
              value: /opt/models
            - name: PYTHONHASHSEED
              value: "0"
            - name: GLOO_SOCKET_IFNAME
              value: eth0
            - name: NCCL_SOCKET_IFNAME
              value: eth0
            - name: NCCL_IB_DISABLE
              value: "0"
            - name: NCCL_IB_HCA
              value: mlx5_8:1
            - name: NCCL_DEBUG
              value: INFO
            - name: NVSHMEM_HCA_LIST
              value: mlx5_8:1
            - name: NVSHMEM_ENABLE_NIC_PE_MAPPING
              value: "1"
            - name: SGLANG_ENABLE_JIT_DEEPGEMM
              value: "1"
          securityContext:
            runAsUser: 0
            capabilities:
              add:
                - IPC_LOCK
                - SYS_RESOURCE
      replicas: 1
      resources:
        requests:
          gpu: "8"
          cpu: "32"
          memory: 256Gi
          custom:
            rdma/ib: "8"
        limits:
          gpu: "8"
          cpu: "64"
          memory: 512Gi
          custom:
            rdma/ib: "8"
EOF
```

The RoCE HCA is pinned to `mlx5_8:1`, but no GID index is hardcoded. In this
pod-native MacVLAN layout, a host GID index is not guaranteed to identify the
Pod's `net1` address.

## 5. Validate and deploy

```bash
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"

kubectl get pods -n "$NAMESPACE" \
  -l "$GRAPH_LABEL" \
  -L nvidia.com/dynamo-component-type \
  -o wide -w
```

Expected topology: one frontend Pod and one two-Pod SGLang worker group, with
one eight-GPU worker Pod on each node. Weight loading and kernel compilation
can take well over 30 minutes.

```bash
kubectl wait -n "$NAMESPACE" \
  --for=condition=Ready pod \
  -l "$GRAPH_LABEL" \
  --timeout=7200s

kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --all-containers --prefix --tail=500
```

Confirm that both worker Pods request eight GPUs and have zero restarts:

```bash
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  -o custom-columns='POD:.metadata.name,NODE:.spec.nodeName,READY:.status.containerStatuses[*].ready,RESTARTS:.status.containerStatuses[*].restartCount,GPU:.spec.containers[*].resources.requests.nvidia\.com/gpu,RDMA:.spec.containers[*].resources.requests.rdma/ib'

kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --all-containers --prefix --tail=2000 \
  | grep -Ei 'error|exception|traceback|out of memory|oom|watchdog|failed' || true
```

## 6. Smoke test

Forward the frontend in one terminal:

```bash
kubectl port-forward -n "$NAMESPACE" \
  service/"$FRONTEND_SERVICE" "$LOCAL_PORT":8000
```

In another terminal, check model discovery and send a short non-thinking
request:

```bash
curl -fsS --max-time 30 \
  "http://127.0.0.1:${LOCAL_PORT}/v1/models"

curl -fsS --max-time 600 \
  "http://127.0.0.1:${LOCAL_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON
{
  "model": "$MODEL",
  "messages": [
    {"role": "user", "content": "Reply with exactly: ready"}
  ],
  "chat_template_kwargs": {"enable_thinking": false},
  "temperature": 0,
  "max_tokens": 16,
  "stream": false
}
JSON
```

Pass only when the model appears in `/v1/models`, the completion returns
`ready` without a 4xx/5xx response, both worker Pods remain ready, and their
restart counts stay at zero.

## 7. Cleanup

Stop the port-forward with `Ctrl-C`, then remove only this deployment:

```bash
kubectl delete dynamographdeployment.nvidia.com "$DEPLOYMENT" \
  -n "$NAMESPACE" --wait=false --ignore-not-found
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -w
```

If the graph is gone but its Pods remain stuck, force-delete only Pods with
this graph label:

```bash
kubectl delete pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --force --grace-period=0 --ignore-not-found
```

Remove the completed download Job after inspecting its logs:

```bash
kubectl delete job glm52-fp8-download \
  -n "$NAMESPACE" --ignore-not-found
```

Cleanup intentionally retains the shared `model-cache` PVC and the pinned
756 GB snapshot so later deployments do not download the model again.

## References

- [GLM-5.2-FP8 checkpoint](https://huggingface.co/zai-org/GLM-5.2-FP8)
- [SGLang GLM-5.2 cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2)
- [Dynamo multinode deployments](https://docs.nvidia.com/dynamo/dev/knowledge-base/kubernetes/multinode/multinode-deployments)
- [Dynamo SGLang backend](https://docs.nvidia.com/dynamo/latest/knowledge-base/modular-components/backends/sg-lang/overview)
- [Dynamo parser configuration](https://docs.nvidia.com/dynamo/latest/user-guides/parsing/parser-configuration)
