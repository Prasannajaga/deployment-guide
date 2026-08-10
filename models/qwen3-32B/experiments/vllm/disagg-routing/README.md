# Qwen3-32B-FP8 vLLM disaggregation + KV routing on H100

This is an upstream-first trial for the 16 x H100 cluster. It preserves the
provided NVIDIA vLLM deployment's tuning arguments and environment variables.
These functional values differ from the H200 source:

- checkpoint: `Qwen/Qwen3-32B` -> `Qwen/Qwen3-32B-FP8`;
- model loading: `--model` points at the pinned snapshot on the model-cache
  PVC and `--served-model-name Qwen/Qwen3-32B-FP8` preserves the API name;
- worker permissions: prefill and decode containers run as UID 0 with
  `IPC_LOCK` and `SYS_RESOURCE` so the shared vLLM compilation cache is
  writable;
- prefill replicas: `6 -> 4`;
- decode replicas: `2 -> 4`.

The upstream metadata name `disagg-router-6p-2d` is intentionally retained so
this first trial changes no unrelated deployment fields, even though the
actual topology is 4P/4D. Each worker remains TP=2, consuming eight H100s for
prefill and eight H100s for decode.

The local snapshot path prevents Dynamo's startup `fetch_model()` from trying
to update Hugging Face cache metadata owned by the download Job. UID 0 fixes
the same ownership mismatch at
`/home/dynamo/.cache/vllm/torch_compile_cache` without enabling privileged
container mode. No node selectors, host networking, UCX variables, port
overrides, or context reduction are added. The upstream 131,072-token YaRN
configuration remains in place. Kubernetes decides worker placement.

Sources:

- [NVIDIA Qwen3-32B recipe](https://docs.nvidia.com/dynamo/dev/recipes/qwen3-32b)
- [upstream deployment](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b/vllm/disagg-kv-router/deploy.yaml)
- [upstream model download](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b/model-cache/model-download.yaml)
- [upstream cache PVCs](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b/model-cache/cache.yaml)

## 1. Create every YAML directly on `gpu05`

Log in to `gpu05` using the cluster's approved access method. Run this entire
block there; no repository checkout or file transfer is required.

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3-32b/experiments/vllm/disagg-routing

mkdir -p "$EXP_DIR" \
  /ephemeral/shared/huggingface \
  /ephemeral/shared/qwen3-32b/vllm-compilation-cache \
  /ephemeral/shared/qwen3-32b/perf-cache

tee "$EXP_DIR/cache.yaml" >/dev/null <<'EOF_CACHE_YAML'
# Static cache volumes for the gpu05/gpu06 lab only.
# /ephemeral/shared must already be the same shared filesystem on both nodes.
apiVersion: v1
kind: PersistentVolume
metadata:
  name: qwen32-model-cache-pv
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: qwen-shared-manual
  volumeMode: Filesystem
  hostPath:
    path: /ephemeral/shared/huggingface
    type: Directory
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
  namespace: qwen32-bench
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: qwen-shared-manual
  volumeName: qwen32-model-cache-pv
---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: qwen32-vllm-compilation-cache-pv
spec:
  capacity:
    storage: 50Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: qwen-shared-manual
  volumeMode: Filesystem
  hostPath:
    path: /ephemeral/shared/qwen3-32b/vllm-compilation-cache
    type: Directory
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: compilation-cache
  namespace: qwen32-bench
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: qwen-shared-manual
  volumeName: qwen32-vllm-compilation-cache-pv
---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: qwen32-vllm-perf-cache-pv
spec:
  capacity:
    storage: 50Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: qwen-shared-manual
  volumeMode: Filesystem
  hostPath:
    path: /ephemeral/shared/qwen3-32b/perf-cache
    type: Directory
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: perf-cache
  namespace: qwen32-bench
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: qwen-shared-manual
  volumeName: qwen32-vllm-perf-cache-pv
EOF_CACHE_YAML

tee "$EXP_DIR/model-download.yaml" >/dev/null <<'EOF_MODEL_DOWNLOAD_YAML'
apiVersion: batch/v1
kind: Job
metadata:
  name: model-download
  namespace: qwen32-bench
spec:
  backoffLimit: 3
  completions: 1
  parallelism: 1
  template:
    metadata:
      labels:
        app: model-download
    spec:
      restartPolicy: Never
      nodeSelector:
        qwen.nvidia.com/role: prefill
      tolerations:
        - key: node-role.kubernetes.io/control-plane
          operator: Exists
          effect: NoSchedule
        - key: nvidia.com/gpu
          operator: Equal
          value: "true"
          effect: NoSchedule
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
              value: Qwen/Qwen3-32B-FP8
            - name: MODEL_REVISION
              value: aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df
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
EOF_MODEL_DOWNLOAD_YAML

tee "$EXP_DIR/deploy.yaml" >/dev/null <<'EOF_DEPLOY_YAML'
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: disagg-router-6p-2d
spec:
  pvcs:
    - create: false
      name: model-cache
    - create: false
      name: compilation-cache
  services:
    Frontend:
      componentType: frontend
      envs:
        - name: HF_HOME
          value: /home/dynamo/.cache/huggingface
      extraPodSpec:
        mainContainer:
          args:
            - --router-mode
            - kv
          command:
            - python
            - -m
            - dynamo.frontend
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          workingDir: /workspace
      replicas: 1
      resources:
        requests:
          cpu: "8"
        limits:
          cpu: "8"
      subComponentType: null
    VllmDecodeWorker:
      componentType: worker
      envFromSecret: hf-token-secret
      extraPodSpec:
        mainContainer:
          args:
            - --model
            - /home/dynamo/.cache/huggingface/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df
            - --served-model-name
            - Qwen/Qwen3-32B-FP8
            - --disaggregation-mode
            - decode
            - --kv-transfer-config
            - '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
            - --tensor-parallel-size
            - "2"
            - --no-enable-log-requests
            - --gpu-memory-utilization
            - "0.90"
            - --no-enable-prefix-caching
            - --async-scheduling
            - --block-size
            - "64"
            - --hf-overrides
            - '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768},"max_position_embeddings":131072}'
            - --max-model-len
            - "131072"
          command:
            - python3
            - -m
            - dynamo.vllm
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          workingDir: /workspace
          env:
            - name: DYN_HEALTH_CHECK_ENABLED
              value: "false"
            - name: HF_HOME
              value: /home/dynamo/.cache/huggingface
          securityContext:
            runAsUser: 0
            capabilities:
              add:
                - IPC_LOCK
                - SYS_RESOURCE
      replicas: 4
      resources:
        limits:
          gpu: "2"
          custom:
            rdma/ib: "2"
        requests:
          gpu: "2"
      subComponentType: decode
      volumeMounts:
        - name: model-cache
          mountPoint: /home/dynamo/.cache/huggingface
        - name: compilation-cache
          mountPoint: /home/dynamo/.cache/vllm
          useAsCompilationCache: true
    VllmPrefillWorker:
      componentType: worker
      envFromSecret: hf-token-secret
      extraPodMetadata:
        annotations:
          prometheus.io/scrape: "true"
          prometheus.io/port: "9400"
          prometheus.io/path: /metrics
      extraPodSpec:
        mainContainer:
          args:
            - --model
            - /home/dynamo/.cache/huggingface/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df
            - --served-model-name
            - Qwen/Qwen3-32B-FP8
            - --disaggregation-mode
            - prefill
            - --kv-transfer-config
            - '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
            - --tensor-parallel-size
            - "2"
            - --no-enable-log-requests
            - --gpu-memory-utilization
            - "0.90"
            - --async-scheduling
            - --block-size
            - "64"
            - --hf-overrides
            - '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768},"max_position_embeddings":131072}'
            - --max-model-len
            - "131072"
            - --kv-events-config
            - '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
          command:
            - python3
            - -m
            - dynamo.vllm
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          env:
            - name: DYN_HEALTH_CHECK_ENABLED
              value: "false"
            - name: HF_HOME
              value: /home/dynamo/.cache/huggingface
          workingDir: /workspace
          securityContext:
            runAsUser: 0
            capabilities:
              add:
                - IPC_LOCK
                - SYS_RESOURCE
      replicas: 4
      resources:
        limits:
          gpu: "2"
          custom:
            rdma/ib: "2"
        requests:
          gpu: "2"
      subComponentType: prefill
      volumeMounts:
        - name: model-cache
          mountPoint: /home/dynamo/.cache/huggingface
        - name: compilation-cache
          mountPoint: /home/dynamo/.cache/vllm
          useAsCompilationCache: true
EOF_DEPLOY_YAML
```

Verify all three files before applying them:

```bash
sha256sum "$EXP_DIR/cache.yaml" \
  "$EXP_DIR/model-download.yaml" \
  "$EXP_DIR/deploy.yaml"
```

Expected hashes:

```text
a0e3b38b1d514bfbea2c86ccf6c79b1905108ab1de759a85a7ee579b20bcb944  cache.yaml
f3dfd55a4afb4eebd2372ae25d124668e581d5e495bdd7d48264d2b284fb1044  model-download.yaml
b5c2ff5fe308cb2aa536b073fadac3fec534aee141e618a8ce873216cf88b9da  deploy.yaml
```

Stop if any hash differs.

## 2. Preflight and validation

```bash
kubectl get nodes -L qwen.nvidia.com/role
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\tGPU="}{.status.allocatable.nvidia\.com/gpu}{"\tRDMA="}{.status.allocatable.rdma/ib}{"\n"}{end}'
kubectl get dgd,pods,pvc -n "$NAMESPACE" -o wide
```

Require 16 free GPUs and sufficient `rdma/ib` resources. Stop any existing
16-GPU experiment first.

Validate the replica counts, FP8 checkpoint, and installed DGD schema:

```bash
python3 - "$EXP_DIR/deploy.yaml" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as stream:
    doc = yaml.safe_load(stream)

services = doc["spec"]["services"]
prefill = services["VllmPrefillWorker"]
decode = services["VllmDecodeWorker"]
snapshot = "/home/dynamo/.cache/huggingface/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
assert doc["metadata"]["name"] == "disagg-router-6p-2d"
assert prefill["replicas"] == 4
assert decode["replicas"] == 4
assert prefill["resources"]["limits"]["gpu"] == "2"
assert decode["resources"]["limits"]["gpu"] == "2"
for worker in (prefill, decode):
    container = worker["extraPodSpec"]["mainContainer"]
    args = container["args"]
    assert args[args.index("--model") + 1] == snapshot
    assert args[args.index("--served-model-name") + 1] == "Qwen/Qwen3-32B-FP8"
    assert container["securityContext"]["runAsUser"] == 0
    assert set(container["securityContext"]["capabilities"]["add"]) == {"IPC_LOCK", "SYS_RESOURCE"}
print("validated: Qwen3-32B-FP8, 4 prefill + 4 decode, TP=2")
PY

kubectl apply --dry-run=server -f "$EXP_DIR/deploy.yaml" -n "$NAMESPACE"
```

Do not continue if either validation fails.

## 3. Create caches and the Hugging Face Secret

```bash
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f "$EXP_DIR/cache.yaml"
kubectl wait --for=jsonpath='{.status.phase}'=Bound \
  pvc/model-cache pvc/compilation-cache pvc/perf-cache \
  -n "$NAMESPACE" --timeout=2m
```

The deployment requires `hf-token-secret`. Create it interactively without
saving the token in a file or shell history:

```bash
read -rsp 'Hugging Face read token: ' HF_TOKEN_INPUT
echo
printf '%s' "$HF_TOKEN_INPUT" | kubectl create secret generic hf-token-secret \
  -n "$NAMESPACE" \
  --from-file=HF_TOKEN=/dev/stdin \
  --dry-run=client -o yaml | kubectl apply -f -
unset HF_TOKEN_INPUT
```

Do not print or export the Secret during diagnostics.

## 4. Download the pinned FP8 checkpoint

```bash
kubectl delete job model-download -n "$NAMESPACE" --ignore-not-found
kubectl apply -f "$EXP_DIR/model-download.yaml"
kubectl logs -f job/model-download -n "$NAMESPACE"
kubectl wait --for=condition=Complete job/model-download \
  -n "$NAMESPACE" --timeout=60m
```

The job downloads `Qwen/Qwen3-32B-FP8` revision
`aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`.

## 5. Deploy and observe

```bash
kubectl apply -f "$EXP_DIR/deploy.yaml" -n "$NAMESPACE"

kubectl get pods -n "$NAMESPACE" \
  -l nvidia.com/dynamo-graph-deployment-name=disagg-router-6p-2d \
  -L nvidia.com/dynamo-component-type -o wide -w
```

Expected graph: one frontend, four TP=2 prefill pods, and four TP=2 decode
pods. Inspect actual node placement because the source manifest has no node
selectors.

To check logs for any pod (works for both running pods and crashed/restarted pods):

```bash
POD=disagg-router-6p-2d-0-vllmprefillworker-wrl52

# Running pod logs
kubectl logs "$POD" -n "$NAMESPACE" --all-containers --timestamps --tail=500

# Crashed/restarted pod logs (previous container instance)
kubectl logs "$POD" -n "$NAMESPACE" --all-containers --previous --timestamps --tail=500 2>/dev/null || true
```


## 6. Smoke test

```bash
kubectl get svc -n "$NAMESPACE" | grep disagg-router-6p-2d
kubectl port-forward -n "$NAMESPACE" \
  svc/disagg-router-6p-2d-frontend 8000:8000
```

In another terminal:

```bash
curl -fsS http://127.0.0.1:8000/v1/models | jq

curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-32B-FP8",
    "messages": [{"role": "user", "content": "Reply with only: VLLM_PD_KV_OK"}],
    "max_tokens": 24,
    "temperature": 0
  }' | jq
```

Acceptance requires all nine pods to remain Ready, the request to succeed,
prefill KV events to reach the KV router, and NIXL transfer to initialize.

## 7. Shutdown

```bash
kubectl delete dgd disagg-router-6p-2d \
  -n "$NAMESPACE" --ignore-not-found
kubectl get pods -n "$NAMESPACE" -w
```

Keep the retained model and compilation caches for the next run.
