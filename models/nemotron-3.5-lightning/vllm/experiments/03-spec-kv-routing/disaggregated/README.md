# Experiment 3: speculative decoding × KV routing

This recipe uses exactly four H100s at TP=1. The default is one prefill and
three decode workers for decode stress; a two-prefill/two-decode alternative
is included below. Each worker Pod requests one GPU and one `rdma/ib`
allocation. Two CPU-only frontends serve either layout.

All worker Pods attach to `qwen-roce` and use `UCX_NET_DEVICES=mlx5_8:1`,
`UCX_TLS=rc_x,rc,cuda_copy,cuda_ipc`, and the UCX tuning from the
[Qwen3.6 recipe](../../../../../qwen3.6-35B-A3B/sglang/disagg/tp2-2p2d/deploy.yaml).
The vLLM NIXL side channel advertises the primary Pod IP on port 5600;
allow primary Pod-network connectivity as well as RoCE data traffic.
Existing H100 affinity applies, without Qwen-specific role labels.

Keep the same load, timing, and A–D cells when comparing with aggregated TP=1.
The default dedicates three GPUs to decode; 2P2D dedicates two, while aggregated
workers can use all four for decode. This compares resource allocation and
transfer behavior at fixed TP and GPU count. With only one prefill worker,
1P3D cannot demonstrate cache-aware selection among prefill replicas; use
2P2D for that comparison. Measure MTP with A versus B and C versus D.

Both roles must use `VLLM_SSM_CONV_STATE_LAYOUT=DS` in the observed runtime.
Its NIXL worker rejects SD at startup with `3-read Mamba conv transfer requires
DS conv state layout`, even with matching TP=1. The earlier SD workaround was
invalid for this build. Aggregated serving does not initialize this NIXL
transfer path, so its successful MTP run does not establish compatibility here.

The reported vLLM 0.26.0 run completed C but crashed in D during Mamba
align-state copying. This recipe now requires vLLM 0.27.1, which follows
[v0.27.0 with fix #49291](https://github.com/vllm-project/vllm/releases/tag/v0.27.0).
Keep DS and seven-token MTP. The derived image below retains Dynamo 1.4.1
and NIXL 1.3.2; this combination is an experimental integration, not a
published or cluster-validated NVIDIA runtime. Build dependency checks and the
startup/transfer gates must pass before benchmarking. Rerun C and D on the
same built image; preserve the old 0.26.0 results separately.

Both roles receive the same seven-token MTP configuration in B and D to keep
hybrid cache layouts aligned; measure speculative acceptance on the decode
role. Prefix caching and Mamba align mode stay enabled in all four cells.
The pinned runtime must support their combination with NIXL and MTP: successful
startup alone is insufficient. Before a sweep, repeat the same prompt through
the frontend in every cell and verify valid output, successful transfers, and
no cache-layout or SSM assertions in either role. If that compatibility gate
fails, stop rather than silently disabling prefix caching or changing the image.
Earlier runtimes have reported this [Mamba/NIXL prefix-cache limitation](https://github.com/ai-dynamo/dynamo/issues/12197).
The connector configuration follows the [Dynamo vLLM reference](https://docs.nvidia.com/dynamo/reference/backends/v-llm-configuration).

| Cell | Routing Strategy | Speculative Decoding | Description |
| :--- | :--- | :--- | :--- |
| **A** | Round-Robin | Disabled (`off`) | Baseline control without speculative decoding or KV routing |
| **B** | Round-Robin | Multi-Token Prediction (`mtp`, 7 tokens) | Speculative decoding isolated under round-robin routing |
| **C** | KV-Aware Routing | Disabled (`off`) | KV cache-aware routing isolated without speculative decoding |
| **D** | KV-Aware Routing | Multi-Token Prediction (`mtp`, 7 tokens) | Combined speculative decoding and KV cache-aware routing |

The first stage uses built-in MTP with seven speculative tokens. This runbook
downloads the pinned base, DFlash, and DSpark checkpoints before creating the
graph. DFlash and DSpark should be introduced only after the MTP 2×2 has
produced stable results.

## 1. Variables and preflight

Run these commands from a cluster-administration host.

```bash
export NAMESPACE=qwen32-bench
export RECIPE_ROOT=/ephemeral/shared/nemotron-3.5-lightning
export MODEL_CACHE_DIR="$RECIPE_ROOT/model-cache"
export EXP_DIR=/ephemeral/shared/nemotron-3.5-lightning/vllm/experiments/03-spec-kv-routing/disaggregated
export DOWNLOAD_JOB=nemotron35-model-download
export TOPOLOGY=disaggregated
export PREFILL_WORKERS=1 DECODE_WORKERS=3
export PD_LAYOUT=tp1-1p3d
export ARTIFACT_ROOT="/perf-cache/specrouting/disaggregated/$PD_LAYOUT"
export DEPLOYMENT=nemotron35-vllm-e3
export PERF_JOB=nemotron35-vllm-e3-perf
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=$DEPLOYMENT"
# Set this to your writable registry/repository with a new, unique tag.
export RUNTIME_IMAGE="${RUNTIME_IMAGE:?Set RUNTIME_IMAGE to your custom vLLM 0.27.1 image reference}"
export CAPABILITY_POD=nemotron35-vllm-capability
mkdir -p "$MODEL_CACHE_DIR" "$EXP_DIR"

kubectl get crd dynamographdeployments.nvidia.com
kubectl get pvc model-cache perf-cache -n "$NAMESPACE"
kubectl get network-attachment-definition qwen-roce -n "$NAMESPACE"
kubectl get nodes \
  -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,RDMA:.status.allocatable.rdma/ib'
```

### Build the required runtime image

Run the following on a Docker build host with registry access after setting
`EXP_DIR` and `RUNTIME_IMAGE` above. The base NVIDIA runtime ships vLLM 0.26.0;
changing only the version assertion or the NVIDIA image tag does not install
the fix. Do not install `ai-dynamo[vllm]==1.4.1` again in this image, because
that extra pins vLLM back to 0.26.0. Keep the existing Dynamo installation and
let pip resolve the new vLLM dependencies. A dependency conflict is a failed
build, not a reason to bypass dependency checks.

```bash
mkdir -p "$EXP_DIR/runtime-build"
tee "$EXP_DIR/runtime-build/Dockerfile" >/dev/null <<'RUNTIME_EOF'
# Experimental Dynamo 1.4.1 + vLLM 0.27.1 integration for the DS/MTP fix.
# Build and validate before deployment; not a published NVIDIA image.
FROM nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1
USER root
RUN python3 -m pip install --no-cache-dir --upgrade \
      "vllm[flashinfer,runai,otel]==0.27.1" "nixl[cu13]==1.3.2" \
    && python3 -m pip check \
    && python3 -c 'import importlib.metadata as m; assert m.version("vllm") == "0.27.1"; assert m.version("ai-dynamo") == "1.4.1"; import dynamo.vllm, nixl'
RUNTIME_EOF
docker build --pull -t "$RUNTIME_IMAGE" "$EXP_DIR/runtime-build"
docker run --rm --entrypoint python3 "$RUNTIME_IMAGE" \
  -m dynamo.vllm --help > "$EXP_DIR/runtime-build/dynamo-vllm-help.txt"
docker push "$RUNTIME_IMAGE"
docker image inspect "$RUNTIME_IMAGE" \
  > "$EXP_DIR/runtime-build/image-inspect.json"
```

The cluster must be able to pull this image. Use an immutable tag or registry
digest for both C and D. Import and CLI checks do not prove GPU correctness;
complete the capability check below and the existing streaming/transfer checks.

The graph needs four free H100s and four free `rdma/ib` allocations in total,
with one of each available for every worker Pod. The node
listing shows allocatable capacity, so also account for existing Pod requests.
Require `qwen-roce` in this namespace and `mlx5_8` port 1 on every eligible
worker node. The attachment name and HCA are cluster-specific values taken
from the Qwen recipe; both must match the actual fabric. Verify that the
runtime contains vLLM 0.27.1 before scheduling it.

```bash
kubectl delete pod "$CAPABILITY_POD" -n "$NAMESPACE" --ignore-not-found
kubectl run "$CAPABILITY_POD" -n "$NAMESPACE" \
  --image="$RUNTIME_IMAGE" --restart=Never --command -- \
  python3 -c 'import importlib.metadata as m; version=m.version("vllm"); print("vLLM", version); assert version == "0.27.1", version'
kubectl wait -n "$NAMESPACE" --for=jsonpath='{.status.phase}'=Succeeded \
  "pod/$CAPABILITY_POD" --timeout=300s
kubectl logs -n "$NAMESPACE" "$CAPABILITY_POD"
kubectl delete pod "$CAPABILITY_POD" -n "$NAMESPACE" --ignore-not-found
```

## 2. Download and verify the pinned models

The deployment uses `HF_HUB_OFFLINE=1` and exact snapshot paths, so model
download is a mandatory gate. The Job below runs before any
`DynamoGraphDeployment` is created. Re-running it is safe: the Job verifies the
exact snapshot's configuration and weight files first and skips the Hugging
Face download call when they are complete. Exact commit SHAs prevent a moving
`main` branch from changing the experiment.

```bash
tee "$MODEL_CACHE_DIR/model-download.yaml" >/dev/null <<'MODEL_EOF'
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
apiVersion: batch/v1
kind: Job
metadata:
  name: nemotron35-model-download
spec:
  backoffLimit: 3
  activeDeadlineSeconds: 43200
  template:
    metadata:
      labels:
        app: nemotron35-model-download
    spec:
      restartPolicy: Never
      containers:
        - name: model-download
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1
          imagePullPolicy: IfNotPresent
          command: [python3, -c]
          args:
            - |
              import json
              import os
              from pathlib import Path
              from huggingface_hub import snapshot_download

              cache_root = Path("/model-cache/hub")

              def snapshot_path(repo_id, revision):
                  return (
                      cache_root
                      / f"models--{repo_id.replace('/', '--')}"
                      / "snapshots"
                      / revision
                  )

              def complete(snapshot):
                  config = snapshot / "config.json"
                  if not config.is_file() or config.stat().st_size == 0:
                      return False
                  for index_name in (
                      "model.safetensors.index.json",
                      "pytorch_model.bin.index.json",
                  ):
                      index = snapshot / index_name
                      if not index.is_file() or index.stat().st_size == 0:
                          continue
                      try:
                          weight_map = json.loads(
                              index.read_text(encoding="utf-8")
                          )["weight_map"]
                      except (KeyError, json.JSONDecodeError):
                          return False
                      weights = set(weight_map.values())
                      return bool(weights) and all(
                          (snapshot / weight).is_file()
                          and (snapshot / weight).stat().st_size > 0
                          for weight in weights
                      )
                  direct_weights = tuple(snapshot.glob("*.safetensors")) + tuple(
                      snapshot.glob("pytorch_model*.bin")
                  )
                  return any(
                      weight.is_file() and weight.stat().st_size > 0
                      for weight in direct_weights
                  )

              models = (
                  ("BASE_MODEL", "BASE_REVISION"),
                  ("DFLASH_MODEL", "DFLASH_REVISION"),
                  ("DSPARK_MODEL", "DSPARK_REVISION"),
              )
              for model_key, revision_key in models:
                  repo_id = os.environ[model_key]
                  revision = os.environ[revision_key]
                  expected = snapshot_path(repo_id, revision)
                  if complete(expected):
                      snapshot = expected
                      print(
                          f"Cache hit for {repo_id}@{revision}; skipping download",
                          flush=True,
                      )
                  else:
                      print(f"Downloading {repo_id}@{revision}", flush=True)
                      snapshot = Path(
                          snapshot_download(repo_id=repo_id, revision=revision)
                      )
                  if not complete(snapshot):
                      raise RuntimeError(
                          f"incomplete model snapshot after download: {snapshot}"
                      )
                  print(f"Verified {repo_id}@{revision}: {snapshot}", flush=True)

              for path in sorted(cache_root.glob("*/snapshots/*")):
                  print(path)
          env:
            - name: BASE_MODEL
              value: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
            - name: BASE_REVISION
              value: cc84af2fe71647d87f4486c064f320e1e7535243
            - name: DFLASH_MODEL
              value: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash
            - name: DFLASH_REVISION
              value: 7fc1f1ff4b82b917efbd0710df0872c2bb89caa5
            - name: DSPARK_MODEL
              value: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark
            - name: DSPARK_REVISION
              value: d10c6ff40d6e69d1f92e407e027de3eafdb77645
            - name: HF_HOME
              value: /model-cache
            - name: HF_XET_HIGH_PERFORMANCE
              value: "1"
          resources:
            requests:
              cpu: "4"
              memory: 16Gi
            limits:
              cpu: "8"
              memory: 32Gi
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: [ALL]
          volumeMounts:
            - name: model-cache
              mountPath: /model-cache
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache
MODEL_EOF

kubectl delete job "$DOWNLOAD_JOB" -n "$NAMESPACE" --ignore-not-found
kubectl apply --dry-run=server -n "$NAMESPACE" \
  -f "$MODEL_CACHE_DIR/model-download.yaml"
kubectl apply -n "$NAMESPACE" -f "$MODEL_CACHE_DIR/model-download.yaml"
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  "job/$DOWNLOAD_JOB" --timeout=43200s
kubectl logs -n "$NAMESPACE" "job/$DOWNLOAD_JOB" --tail=200
```

Do not continue to deployment unless the Job is `Complete` and its log contains
all three `Verified ...@<commit>` lines. A failed or timed-out Job leaves the
deployment blocked; inspect it before retrying:

```bash
kubectl get job "$DOWNLOAD_JOB" -n "$NAMESPACE" -o wide
kubectl get pods -n "$NAMESPACE" -l "job-name=$DOWNLOAD_JOB" -o wide
kubectl describe job "$DOWNLOAD_JOB" -n "$NAMESPACE"
```

## 3. Create the cell-aware deployment template

```bash
tee "$EXP_DIR/deploy.template.yaml" >/dev/null <<'DEPLOY_EOF'
# Replace __RUNTIME_IMAGE__ with the built vLLM 0.27.1 image before applying.
# Set experiment-cell to A, B, C, or D before applying. See README.md.
apiVersion: v1
kind: ConfigMap
metadata:
  name: nemotron35-vllm-e3-settings
  labels:
    research.nvidia.com/experiment: spec-kv-routing
data:
  experiment-cell: A
  speculative-config: |-
    {"method":"mtp","num_speculative_tokens":7}
---
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: nemotron35-vllm-e3
  labels:
    app.kubernetes.io/name: nemotron35-vllm-e3
    app.kubernetes.io/part-of: nemotron35-research
    research.nvidia.com/experiment: spec-kv-routing
spec:
  backendFramework: vllm
  components:
    - name: Frontend
      type: frontend
      replicas: 2
      podTemplate:
        metadata:
          labels:
            app.kubernetes.io/name: nemotron35-vllm-e3
            app.kubernetes.io/component: frontend
            research.nvidia.com/experiment: spec-kv-routing
        spec:
          containers:
            - name: main
              image: &runtime_image __RUNTIME_IMAGE__
              imagePullPolicy: IfNotPresent
              command: [/bin/sh, -c]
              args:
                - |
                  set -eu
                  set -- python3 -m dynamo.frontend --trust-remote-code
                  case "$EXPERIMENT_CELL" in
                    A|B)
                      set -- "$@" --router-mode round-robin
                      ;;
                    C|D)
                      set -- "$@" \
                        --router-mode kv \
                        --router-kv-events \
                        --kv-cache-block-size 64
                      ;;
                    *)
                      echo "EXPERIMENT_CELL must be A, B, C, or D" >&2
                      exit 2
                      ;;
                  esac
                  exec "$@"
              env:
                - name: EXPERIMENT_CELL
                  valueFrom:
                    configMapKeyRef:
                      name: nemotron35-vllm-e3-settings
                      key: experiment-cell
                - name: DYN_HTTP_BODY_LIMIT_MB
                  value: "200"
                - name: HF_HOME
                  value: /opt/models
                - name: HF_HUB_OFFLINE
                  value: "1"
                - name: HF_MODULES_CACHE
                  value: /tmp/hf_modules
              resources:
                requests:
                  cpu: "8"
                  memory: 8Gi
                limits:
                  cpu: "16"
                  memory: 16Gi
              startupProbe:
                httpGet:
                  path: /health
                  port: http
                periodSeconds: 10
                timeoutSeconds: 5
                failureThreshold: 90
              readinessProbe:
                httpGet:
                  path: /health
                  port: http
                initialDelaySeconds: 10
                periodSeconds: 10
                timeoutSeconds: 3
                failureThreshold: 3
              livenessProbe:
                httpGet:
                  path: /health
                  port: http
                initialDelaySeconds: 15
                periodSeconds: 10
                timeoutSeconds: 3
                failureThreshold: 3
              ports:
                - name: http
                  containerPort: 8000
              volumeMounts:
                - name: model-cache
                  mountPath: /opt/models
                  readOnly: true
          volumes:
            - name: model-cache
              persistentVolumeClaim:
                claimName: model-cache

    - name: VllmPrefillWorker
      type: prefill
      replicas: 1
      sharedMemorySize: 40Gi
      podTemplate:
        metadata:
          labels:
            app.kubernetes.io/name: nemotron35-vllm-e3
            app.kubernetes.io/component: worker
            research.nvidia.com/worker-role: prefill
            research.nvidia.com/experiment: spec-kv-routing
          annotations:
            k8s.v1.cni.cncf.io/networks: qwen-roce
        spec:
          hostNetwork: false
          dnsPolicy: ClusterFirst
          affinity:
            nodeAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
                nodeSelectorTerms:
                  - matchExpressions:
                      - key: nvidia.com/gpu.present
                        operator: In
                        values: ["true"]
                      - key: nvidia.com/gpu.product
                        operator: In
                        values: [NVIDIA-H100-80GB-HBM3]
          tolerations:
            - key: nvidia.com/gpu
              operator: Equal
              value: "true"
              effect: NoSchedule
          volumes:
            - name: model-cache
              persistentVolumeClaim:
                claimName: model-cache
            - name: dshm
              emptyDir:
                medium: Memory
                sizeLimit: 40Gi
          containers:
            - name: main
              image: *runtime_image
              imagePullPolicy: IfNotPresent
              workingDir: /workspace
              command: [/bin/sh, -c]
              args:
                - |
                  set -eu
                  ulimit -l unlimited
                  set -- python3 -m dynamo.vllm \
                    --model "$MODEL_PATH" \
                    --served-model-name "$SERVED_MODEL_NAME" \
                    --trust-remote-code \
                    --tensor-parallel-size 1 \
                    --max-model-len 65536 \
                    --max-num-seqs 512 \
                    --max-num-batched-tokens 32768 \
                    --gpu-memory-utilization 0.90 \
                    --async-scheduling \
                    --enable-prefix-caching \
                    --block-size 64 \
                    --mamba-cache-mode align \
                    --mamba-backend flashinfer \
                    --mamba-ssm-cache-dtype float16 \
                    --enable-mamba-cache-stochastic-rounding \
                    --mamba-cache-philox-rounds 5 \
                    --dyn-tool-call-parser nemotron_nano \
                    --dyn-reasoning-parser nemotron_nano \
                    --reasoning-parser nemotron_v3 \
                    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
                    --disaggregation-mode prefill

                  case "$EXPERIMENT_CELL" in
                    A)
                      ;;
                    B)
                      set -- "$@" --speculative-config "$SPECULATIVE_CONFIG"
                      ;;
                    C)
                      set -- "$@" --kv-events-config "$KV_EVENTS_CONFIG"
                      ;;
                    D)
                      set -- "$@" \
                        --speculative-config "$SPECULATIVE_CONFIG" \
                        --kv-events-config "$KV_EVENTS_CONFIG"
                      ;;
                    *)
                      echo "EXPERIMENT_CELL must be A, B, C, or D" >&2
                      exit 2
                      ;;
                  esac
                  exec "$@"
              env:
                - name: EXPERIMENT_CELL
                  valueFrom:
                    configMapKeyRef:
                      name: nemotron35-vllm-e3-settings
                      key: experiment-cell
                - name: SPECULATIVE_CONFIG
                  valueFrom:
                    configMapKeyRef:
                      name: nemotron35-vllm-e3-settings
                      key: speculative-config
                - name: KV_EVENTS_CONFIG
                  value: '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
                - name: SERVED_MODEL_NAME
                  value: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
                - name: MODEL_PATH
                  value: /opt/models/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/snapshots/cc84af2fe71647d87f4486c064f320e1e7535243
                - name: HF_HOME
                  value: /opt/models
                - name: HF_HUB_OFFLINE
                  value: "1"
                - name: HF_MODULES_CACHE
                  value: /tmp/hf_modules
                - name: TRITON_CACHE_DIR
                  value: /tmp/.triton-cache
                - name: VLLM_CONFIG_ROOT
                  value: /tmp/vllm-config
                - name: VLLM_CACHE_ROOT
                  value: /tmp/vllm-cache
                - name: VLLM_NIXL_SIDE_CHANNEL_HOST
                  valueFrom:
                    fieldRef: {fieldPath: status.podIP}
                - {name: VLLM_NIXL_SIDE_CHANNEL_PORT, value: '5600'}
                - {name: VLLM_SSM_CONV_STATE_LAYOUT, value: DS}
                - name: UCX_TLS
                  value: 'rc_x,rc,cuda_copy,cuda_ipc'
                - name: UCX_NET_DEVICES
                  value: 'mlx5_8:1'
                - name: UCX_IB_ADDR_TYPE
                  value: 'eth'
                - name: UCX_RNDV_SCHEME
                  value: 'get_zcopy'
                - name: UCX_RNDV_THRESH
                  value: '0'
                - name: UCX_IB_REG_METHODS
                  value: 'odp,rcache'
                - name: UCX_RCACHE_MAX_UNRELEASED
                  value: '1024'
                - name: UCX_RC_TIMEOUT
                  value: '600s'
                - name: UCX_KEEPALIVE_INTERVAL
                  value: '300s'
                - name: UCX_LOG_LEVEL
                  value: 'info'
                - name: NIXL_LOG_LEVEL
                  value: 'INFO'
                - name: NCCL_IB_DISABLE
                  value: "1"
                - name: PYTHONHASHSEED
                  value: "0"
              resources:
                requests:
                  cpu: "16"
                  memory: 96Gi
                  nvidia.com/gpu: "1"
                  rdma/ib: "1"
                limits:
                  cpu: "32"
                  memory: 128Gi
                  nvidia.com/gpu: "1"
                  rdma/ib: "1"
              securityContext:
                runAsUser: 0
                capabilities:
                  add: [IPC_LOCK, SYS_RESOURCE]
              startupProbe:
                httpGet:
                  path: /live
                  port: 9090
                periodSeconds: 30
                timeoutSeconds: 20
                failureThreshold: 120
              ports:
                - name: system
                  containerPort: 9090
              volumeMounts:
                - name: model-cache
                  mountPath: /opt/models
                  readOnly: true
                - name: dshm
                  mountPath: /dev/shm

    - name: VllmDecodeWorker
      type: decode
      replicas: 3
      sharedMemorySize: 40Gi
      podTemplate:
        metadata:
          labels:
            app.kubernetes.io/name: nemotron35-vllm-e3
            app.kubernetes.io/component: worker
            research.nvidia.com/worker-role: decode
            research.nvidia.com/experiment: spec-kv-routing
          annotations:
            k8s.v1.cni.cncf.io/networks: qwen-roce
        spec:
          hostNetwork: false
          dnsPolicy: ClusterFirst
          affinity:
            nodeAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
                nodeSelectorTerms:
                  - matchExpressions:
                      - key: nvidia.com/gpu.present
                        operator: In
                        values: ["true"]
                      - key: nvidia.com/gpu.product
                        operator: In
                        values: [NVIDIA-H100-80GB-HBM3]
          tolerations:
            - key: nvidia.com/gpu
              operator: Equal
              value: "true"
              effect: NoSchedule
          volumes:
            - name: model-cache
              persistentVolumeClaim:
                claimName: model-cache
            - name: dshm
              emptyDir:
                medium: Memory
                sizeLimit: 40Gi
          containers:
            - name: main
              image: *runtime_image
              imagePullPolicy: IfNotPresent
              workingDir: /workspace
              command: [/bin/sh, -c]
              args:
                - |
                  set -eu
                  ulimit -l unlimited
                  set -- python3 -m dynamo.vllm \
                    --model "$MODEL_PATH" \
                    --served-model-name "$SERVED_MODEL_NAME" \
                    --trust-remote-code \
                    --tensor-parallel-size 1 \
                    --max-model-len 65536 \
                    --max-num-seqs 512 \
                    --max-num-batched-tokens 32768 \
                    --gpu-memory-utilization 0.90 \
                    --async-scheduling \
                    --enable-prefix-caching \
                    --block-size 64 \
                    --mamba-cache-mode align \
                    --mamba-backend flashinfer \
                    --mamba-ssm-cache-dtype float16 \
                    --enable-mamba-cache-stochastic-rounding \
                    --mamba-cache-philox-rounds 5 \
                    --dyn-tool-call-parser nemotron_nano \
                    --dyn-reasoning-parser nemotron_nano \
                    --reasoning-parser nemotron_v3 \
                    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
                    --disaggregation-mode decode

                  case "$EXPERIMENT_CELL" in
                    A)
                      ;;
                    B)
                      set -- "$@" --speculative-config "$SPECULATIVE_CONFIG"
                      ;;
                    C)
                      set -- "$@" --kv-events-config "$KV_EVENTS_CONFIG"
                      ;;
                    D)
                      set -- "$@" \
                        --speculative-config "$SPECULATIVE_CONFIG" \
                        --kv-events-config "$KV_EVENTS_CONFIG"
                      ;;
                    *)
                      echo "EXPERIMENT_CELL must be A, B, C, or D" >&2
                      exit 2
                      ;;
                  esac
                  exec "$@"
              env:
                - name: EXPERIMENT_CELL
                  valueFrom:
                    configMapKeyRef:
                      name: nemotron35-vllm-e3-settings
                      key: experiment-cell
                - name: SPECULATIVE_CONFIG
                  valueFrom:
                    configMapKeyRef:
                      name: nemotron35-vllm-e3-settings
                      key: speculative-config
                - name: KV_EVENTS_CONFIG
                  value: '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
                - name: SERVED_MODEL_NAME
                  value: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
                - name: MODEL_PATH
                  value: /opt/models/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/snapshots/cc84af2fe71647d87f4486c064f320e1e7535243
                - name: HF_HOME
                  value: /opt/models
                - name: HF_HUB_OFFLINE
                  value: "1"
                - name: HF_MODULES_CACHE
                  value: /tmp/hf_modules
                - name: TRITON_CACHE_DIR
                  value: /tmp/.triton-cache
                - name: VLLM_CONFIG_ROOT
                  value: /tmp/vllm-config
                - name: VLLM_CACHE_ROOT
                  value: /tmp/vllm-cache
                - name: VLLM_NIXL_SIDE_CHANNEL_HOST
                  valueFrom:
                    fieldRef: {fieldPath: status.podIP}
                - {name: VLLM_NIXL_SIDE_CHANNEL_PORT, value: '5600'}
                - {name: VLLM_SSM_CONV_STATE_LAYOUT, value: DS}
                - name: UCX_TLS
                  value: 'rc_x,rc,cuda_copy,cuda_ipc'
                - name: UCX_NET_DEVICES
                  value: 'mlx5_8:1'
                - name: UCX_IB_ADDR_TYPE
                  value: 'eth'
                - name: UCX_RNDV_SCHEME
                  value: 'get_zcopy'
                - name: UCX_RNDV_THRESH
                  value: '0'
                - name: UCX_IB_REG_METHODS
                  value: 'odp,rcache'
                - name: UCX_RCACHE_MAX_UNRELEASED
                  value: '1024'
                - name: UCX_RC_TIMEOUT
                  value: '600s'
                - name: UCX_KEEPALIVE_INTERVAL
                  value: '300s'
                - name: UCX_LOG_LEVEL
                  value: 'info'
                - name: NIXL_LOG_LEVEL
                  value: 'INFO'
                - name: NCCL_IB_DISABLE
                  value: "1"
                - name: PYTHONHASHSEED
                  value: "0"
              resources:
                requests:
                  cpu: "16"
                  memory: 96Gi
                  nvidia.com/gpu: "1"
                  rdma/ib: "1"
                limits:
                  cpu: "32"
                  memory: 128Gi
                  nvidia.com/gpu: "1"
                  rdma/ib: "1"
              securityContext:
                runAsUser: 0
                capabilities:
                  add: [IPC_LOCK, SYS_RESOURCE]
              startupProbe:
                httpGet:
                  path: /live
                  port: 9090
                periodSeconds: 30
                timeoutSeconds: 20
                failureThreshold: 120
              ports:
                - name: system
                  containerPort: 9090
              volumeMounts:
                - name: model-cache
                  mountPath: /opt/models
                  readOnly: true
                - name: dshm
                  mountPath: /dev/shm
DEPLOY_EOF
python3 - "$EXP_DIR/deploy.template.yaml" "$RUNTIME_IMAGE" <<'IMAGE_EOF'
from pathlib import Path
import sys
path = Path(sys.argv[1])
image = sys.argv[2]
assert image and not any(c.isspace() for c in image) and "__" not in image
path.write_text(path.read_text().replace("__RUNTIME_IMAGE__", image))
IMAGE_EOF
```

### Optional 2P2D layout

The template above defaults to 1P3D. For 2P2D, replace it with this complete
manifest before rendering a cell. Keep the layout fixed across A–D and use a
separate result directory. Rerun the 1P3D heredoc above to switch back.

```bash
tee "$EXP_DIR/deploy.template.yaml" >/dev/null <<'DEPLOY_2P2D_EOF'
# Replace __RUNTIME_IMAGE__ with the built vLLM 0.27.1 image before applying.
# Set experiment-cell to A, B, C, or D before applying. See README.md.
apiVersion: v1
kind: ConfigMap
metadata:
  name: nemotron35-vllm-e3-settings
  labels:
    research.nvidia.com/experiment: spec-kv-routing
data:
  experiment-cell: A
  speculative-config: |-
    {"method":"mtp","num_speculative_tokens":7}
---
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: nemotron35-vllm-e3
  labels:
    app.kubernetes.io/name: nemotron35-vllm-e3
    app.kubernetes.io/part-of: nemotron35-research
    research.nvidia.com/experiment: spec-kv-routing
spec:
  backendFramework: vllm
  components:
    - name: Frontend
      type: frontend
      replicas: 2
      podTemplate:
        metadata:
          labels:
            app.kubernetes.io/name: nemotron35-vllm-e3
            app.kubernetes.io/component: frontend
            research.nvidia.com/experiment: spec-kv-routing
        spec:
          containers:
            - name: main
              image: &runtime_image __RUNTIME_IMAGE__
              imagePullPolicy: IfNotPresent
              command: [/bin/sh, -c]
              args:
                - |
                  set -eu
                  set -- python3 -m dynamo.frontend --trust-remote-code
                  case "$EXPERIMENT_CELL" in
                    A|B)
                      set -- "$@" --router-mode round-robin
                      ;;
                    C|D)
                      set -- "$@" \
                        --router-mode kv \
                        --router-kv-events \
                        --kv-cache-block-size 64
                      ;;
                    *)
                      echo "EXPERIMENT_CELL must be A, B, C, or D" >&2
                      exit 2
                      ;;
                  esac
                  exec "$@"
              env:
                - name: EXPERIMENT_CELL
                  valueFrom:
                    configMapKeyRef:
                      name: nemotron35-vllm-e3-settings
                      key: experiment-cell
                - name: DYN_HTTP_BODY_LIMIT_MB
                  value: "200"
                - name: HF_HOME
                  value: /opt/models
                - name: HF_HUB_OFFLINE
                  value: "1"
                - name: HF_MODULES_CACHE
                  value: /tmp/hf_modules
              resources:
                requests:
                  cpu: "8"
                  memory: 8Gi
                limits:
                  cpu: "16"
                  memory: 16Gi
              startupProbe:
                httpGet:
                  path: /health
                  port: http
                periodSeconds: 10
                timeoutSeconds: 5
                failureThreshold: 90
              readinessProbe:
                httpGet:
                  path: /health
                  port: http
                initialDelaySeconds: 10
                periodSeconds: 10
                timeoutSeconds: 3
                failureThreshold: 3
              livenessProbe:
                httpGet:
                  path: /health
                  port: http
                initialDelaySeconds: 15
                periodSeconds: 10
                timeoutSeconds: 3
                failureThreshold: 3
              ports:
                - name: http
                  containerPort: 8000
              volumeMounts:
                - name: model-cache
                  mountPath: /opt/models
                  readOnly: true
          volumes:
            - name: model-cache
              persistentVolumeClaim:
                claimName: model-cache

    - name: VllmPrefillWorker
      type: prefill
      replicas: 2
      sharedMemorySize: 40Gi
      podTemplate:
        metadata:
          labels:
            app.kubernetes.io/name: nemotron35-vllm-e3
            app.kubernetes.io/component: worker
            research.nvidia.com/worker-role: prefill
            research.nvidia.com/experiment: spec-kv-routing
          annotations:
            k8s.v1.cni.cncf.io/networks: qwen-roce
        spec:
          hostNetwork: false
          dnsPolicy: ClusterFirst
          affinity:
            nodeAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
                nodeSelectorTerms:
                  - matchExpressions:
                      - key: nvidia.com/gpu.present
                        operator: In
                        values: ["true"]
                      - key: nvidia.com/gpu.product
                        operator: In
                        values: [NVIDIA-H100-80GB-HBM3]
          tolerations:
            - key: nvidia.com/gpu
              operator: Equal
              value: "true"
              effect: NoSchedule
          volumes:
            - name: model-cache
              persistentVolumeClaim:
                claimName: model-cache
            - name: dshm
              emptyDir:
                medium: Memory
                sizeLimit: 40Gi
          containers:
            - name: main
              image: *runtime_image
              imagePullPolicy: IfNotPresent
              workingDir: /workspace
              command: [/bin/sh, -c]
              args:
                - |
                  set -eu
                  ulimit -l unlimited
                  set -- python3 -m dynamo.vllm \
                    --model "$MODEL_PATH" \
                    --served-model-name "$SERVED_MODEL_NAME" \
                    --trust-remote-code \
                    --tensor-parallel-size 1 \
                    --max-model-len 65536 \
                    --max-num-seqs 512 \
                    --max-num-batched-tokens 32768 \
                    --gpu-memory-utilization 0.90 \
                    --async-scheduling \
                    --enable-prefix-caching \
                    --block-size 64 \
                    --mamba-cache-mode align \
                    --mamba-backend flashinfer \
                    --mamba-ssm-cache-dtype float16 \
                    --enable-mamba-cache-stochastic-rounding \
                    --mamba-cache-philox-rounds 5 \
                    --dyn-tool-call-parser nemotron_nano \
                    --dyn-reasoning-parser nemotron_nano \
                    --reasoning-parser nemotron_v3 \
                    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
                    --disaggregation-mode prefill

                  case "$EXPERIMENT_CELL" in
                    A)
                      ;;
                    B)
                      set -- "$@" --speculative-config "$SPECULATIVE_CONFIG"
                      ;;
                    C)
                      set -- "$@" --kv-events-config "$KV_EVENTS_CONFIG"
                      ;;
                    D)
                      set -- "$@" \
                        --speculative-config "$SPECULATIVE_CONFIG" \
                        --kv-events-config "$KV_EVENTS_CONFIG"
                      ;;
                    *)
                      echo "EXPERIMENT_CELL must be A, B, C, or D" >&2
                      exit 2
                      ;;
                  esac
                  exec "$@"
              env:
                - name: EXPERIMENT_CELL
                  valueFrom:
                    configMapKeyRef:
                      name: nemotron35-vllm-e3-settings
                      key: experiment-cell
                - name: SPECULATIVE_CONFIG
                  valueFrom:
                    configMapKeyRef:
                      name: nemotron35-vllm-e3-settings
                      key: speculative-config
                - name: KV_EVENTS_CONFIG
                  value: '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
                - name: SERVED_MODEL_NAME
                  value: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
                - name: MODEL_PATH
                  value: /opt/models/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/snapshots/cc84af2fe71647d87f4486c064f320e1e7535243
                - name: HF_HOME
                  value: /opt/models
                - name: HF_HUB_OFFLINE
                  value: "1"
                - name: HF_MODULES_CACHE
                  value: /tmp/hf_modules
                - name: TRITON_CACHE_DIR
                  value: /tmp/.triton-cache
                - name: VLLM_CONFIG_ROOT
                  value: /tmp/vllm-config
                - name: VLLM_CACHE_ROOT
                  value: /tmp/vllm-cache
                - name: VLLM_NIXL_SIDE_CHANNEL_HOST
                  valueFrom:
                    fieldRef: {fieldPath: status.podIP}
                - {name: VLLM_NIXL_SIDE_CHANNEL_PORT, value: '5600'}
                - {name: VLLM_SSM_CONV_STATE_LAYOUT, value: DS}
                - name: UCX_TLS
                  value: 'rc_x,rc,cuda_copy,cuda_ipc'
                - name: UCX_NET_DEVICES
                  value: 'mlx5_8:1'
                - name: UCX_IB_ADDR_TYPE
                  value: 'eth'
                - name: UCX_RNDV_SCHEME
                  value: 'get_zcopy'
                - name: UCX_RNDV_THRESH
                  value: '0'
                - name: UCX_IB_REG_METHODS
                  value: 'odp,rcache'
                - name: UCX_RCACHE_MAX_UNRELEASED
                  value: '1024'
                - name: UCX_RC_TIMEOUT
                  value: '600s'
                - name: UCX_KEEPALIVE_INTERVAL
                  value: '300s'
                - name: UCX_LOG_LEVEL
                  value: 'info'
                - name: NIXL_LOG_LEVEL
                  value: 'INFO'
                - name: NCCL_IB_DISABLE
                  value: "1"
                - name: PYTHONHASHSEED
                  value: "0"
              resources:
                requests:
                  cpu: "16"
                  memory: 96Gi
                  nvidia.com/gpu: "1"
                  rdma/ib: "1"
                limits:
                  cpu: "32"
                  memory: 128Gi
                  nvidia.com/gpu: "1"
                  rdma/ib: "1"
              securityContext:
                runAsUser: 0
                capabilities:
                  add: [IPC_LOCK, SYS_RESOURCE]
              startupProbe:
                httpGet:
                  path: /live
                  port: 9090
                periodSeconds: 30
                timeoutSeconds: 20
                failureThreshold: 120
              ports:
                - name: system
                  containerPort: 9090
              volumeMounts:
                - name: model-cache
                  mountPath: /opt/models
                  readOnly: true
                - name: dshm
                  mountPath: /dev/shm

    - name: VllmDecodeWorker
      type: decode
      replicas: 2
      sharedMemorySize: 40Gi
      podTemplate:
        metadata:
          labels:
            app.kubernetes.io/name: nemotron35-vllm-e3
            app.kubernetes.io/component: worker
            research.nvidia.com/worker-role: decode
            research.nvidia.com/experiment: spec-kv-routing
          annotations:
            k8s.v1.cni.cncf.io/networks: qwen-roce
        spec:
          hostNetwork: false
          dnsPolicy: ClusterFirst
          affinity:
            nodeAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
                nodeSelectorTerms:
                  - matchExpressions:
                      - key: nvidia.com/gpu.present
                        operator: In
                        values: ["true"]
                      - key: nvidia.com/gpu.product
                        operator: In
                        values: [NVIDIA-H100-80GB-HBM3]
          tolerations:
            - key: nvidia.com/gpu
              operator: Equal
              value: "true"
              effect: NoSchedule
          volumes:
            - name: model-cache
              persistentVolumeClaim:
                claimName: model-cache
            - name: dshm
              emptyDir:
                medium: Memory
                sizeLimit: 40Gi
          containers:
            - name: main
              image: *runtime_image
              imagePullPolicy: IfNotPresent
              workingDir: /workspace
              command: [/bin/sh, -c]
              args:
                - |
                  set -eu
                  ulimit -l unlimited
                  set -- python3 -m dynamo.vllm \
                    --model "$MODEL_PATH" \
                    --served-model-name "$SERVED_MODEL_NAME" \
                    --trust-remote-code \
                    --tensor-parallel-size 1 \
                    --max-model-len 65536 \
                    --max-num-seqs 512 \
                    --max-num-batched-tokens 32768 \
                    --gpu-memory-utilization 0.90 \
                    --async-scheduling \
                    --enable-prefix-caching \
                    --block-size 64 \
                    --mamba-cache-mode align \
                    --mamba-backend flashinfer \
                    --mamba-ssm-cache-dtype float16 \
                    --enable-mamba-cache-stochastic-rounding \
                    --mamba-cache-philox-rounds 5 \
                    --dyn-tool-call-parser nemotron_nano \
                    --dyn-reasoning-parser nemotron_nano \
                    --reasoning-parser nemotron_v3 \
                    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
                    --disaggregation-mode decode

                  case "$EXPERIMENT_CELL" in
                    A)
                      ;;
                    B)
                      set -- "$@" --speculative-config "$SPECULATIVE_CONFIG"
                      ;;
                    C)
                      set -- "$@" --kv-events-config "$KV_EVENTS_CONFIG"
                      ;;
                    D)
                      set -- "$@" \
                        --speculative-config "$SPECULATIVE_CONFIG" \
                        --kv-events-config "$KV_EVENTS_CONFIG"
                      ;;
                    *)
                      echo "EXPERIMENT_CELL must be A, B, C, or D" >&2
                      exit 2
                      ;;
                  esac
                  exec "$@"
              env:
                - name: EXPERIMENT_CELL
                  valueFrom:
                    configMapKeyRef:
                      name: nemotron35-vllm-e3-settings
                      key: experiment-cell
                - name: SPECULATIVE_CONFIG
                  valueFrom:
                    configMapKeyRef:
                      name: nemotron35-vllm-e3-settings
                      key: speculative-config
                - name: KV_EVENTS_CONFIG
                  value: '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
                - name: SERVED_MODEL_NAME
                  value: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
                - name: MODEL_PATH
                  value: /opt/models/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/snapshots/cc84af2fe71647d87f4486c064f320e1e7535243
                - name: HF_HOME
                  value: /opt/models
                - name: HF_HUB_OFFLINE
                  value: "1"
                - name: HF_MODULES_CACHE
                  value: /tmp/hf_modules
                - name: TRITON_CACHE_DIR
                  value: /tmp/.triton-cache
                - name: VLLM_CONFIG_ROOT
                  value: /tmp/vllm-config
                - name: VLLM_CACHE_ROOT
                  value: /tmp/vllm-cache
                - name: VLLM_NIXL_SIDE_CHANNEL_HOST
                  valueFrom:
                    fieldRef: {fieldPath: status.podIP}
                - {name: VLLM_NIXL_SIDE_CHANNEL_PORT, value: '5600'}
                - {name: VLLM_SSM_CONV_STATE_LAYOUT, value: DS}
                - name: UCX_TLS
                  value: 'rc_x,rc,cuda_copy,cuda_ipc'
                - name: UCX_NET_DEVICES
                  value: 'mlx5_8:1'
                - name: UCX_IB_ADDR_TYPE
                  value: 'eth'
                - name: UCX_RNDV_SCHEME
                  value: 'get_zcopy'
                - name: UCX_RNDV_THRESH
                  value: '0'
                - name: UCX_IB_REG_METHODS
                  value: 'odp,rcache'
                - name: UCX_RCACHE_MAX_UNRELEASED
                  value: '1024'
                - name: UCX_RC_TIMEOUT
                  value: '600s'
                - name: UCX_KEEPALIVE_INTERVAL
                  value: '300s'
                - name: UCX_LOG_LEVEL
                  value: 'info'
                - name: NIXL_LOG_LEVEL
                  value: 'INFO'
                - name: NCCL_IB_DISABLE
                  value: "1"
                - name: PYTHONHASHSEED
                  value: "0"
              resources:
                requests:
                  cpu: "16"
                  memory: 96Gi
                  nvidia.com/gpu: "1"
                  rdma/ib: "1"
                limits:
                  cpu: "32"
                  memory: 128Gi
                  nvidia.com/gpu: "1"
                  rdma/ib: "1"
              securityContext:
                runAsUser: 0
                capabilities:
                  add: [IPC_LOCK, SYS_RESOURCE]
              startupProbe:
                httpGet:
                  path: /live
                  port: 9090
                periodSeconds: 30
                timeoutSeconds: 20
                failureThreshold: 120
              ports:
                - name: system
                  containerPort: 9090
              volumeMounts:
                - name: model-cache
                  mountPath: /opt/models
                  readOnly: true
                - name: dshm
                  mountPath: /dev/shm
DEPLOY_2P2D_EOF
python3 - "$EXP_DIR/deploy.template.yaml" "$RUNTIME_IMAGE" <<'IMAGE_EOF'
from pathlib import Path
import sys
path = Path(sys.argv[1])
image = sys.argv[2]
assert image and not any(c.isspace() for c in image) and "__" not in image
path.write_text(path.read_text().replace("__RUNTIME_IMAGE__", image))
IMAGE_EOF
export PREFILL_WORKERS=2 DECODE_WORKERS=2
export PD_LAYOUT=tp1-2p2d
export ARTIFACT_ROOT="/perf-cache/specrouting/disaggregated/$PD_LAYOUT"
```

The template defaults to A. Render exactly one cell at a time and delete the
old graph before switching, because a ConfigMap-only edit does not guarantee a
worker restart.

```bash
export CELL=A
case "$CELL" in A|B|C|D) ;; *) echo 'CELL must be A, B, C, or D' >&2; exit 2 ;; esac
sed "s/experiment-cell: A/experiment-cell: $CELL/" \
  "$EXP_DIR/deploy.template.yaml" > "$EXP_DIR/deploy-$CELL.yaml"

kubectl delete job "$PERF_JOB" -n "$NAMESPACE" --ignore-not-found
kubectl delete dynamographdeployment "$DEPLOYMENT" -n "$NAMESPACE" --ignore-not-found
kubectl delete configmap nemotron35-vllm-e3-settings -n "$NAMESPACE" --ignore-not-found
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/deploy-$CELL.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy-$CELL.yaml"
kubectl wait -n "$NAMESPACE" --for=jsonpath='{.status.state}'=successful \
  "dynamographdeployment/$DEPLOYMENT" --timeout=60m
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide -w
kubectl get endpoints "${DEPLOYMENT}-frontend" -n "$NAMESPACE"
```

Before accepting a run, inspect all worker logs and frontend metrics. B and D must
show MTP initialization; C and D must expose KV-router and applied-event metric
families. A and B must not publish KV events.

### Verify the RoCE attachment

After all four worker Pods are running, inspect their Multus network status and
RDMA device visibility before sending the frontend test request.

```bash
for pod in $(kubectl get pods -n "$NAMESPACE" \
  -l "app.kubernetes.io/name=$DEPLOYMENT,app.kubernetes.io/component=worker" \
  -o name); do
  kubectl get -n "$NAMESPACE" "$pod" \
    -o go-template='{{index .metadata.annotations "k8s.v1.cni.cncf.io/network-status"}}{{"\n"}}'
  kubectl exec -n "$NAMESPACE" "$pod" -c main -- sh -ec '
    test -d /sys/class/infiniband/mlx5_8/ports/1
    test -c /dev/infiniband/rdma_cm
    cat /sys/class/infiniband/mlx5_8/ports/1/state
    cat /sys/class/infiniband/mlx5_8/ports/1/link_layer
    env | sort | grep -E "^(UCX_|NIXL_|VLLM_NIXL_)"
  '
done
```

Require `qwen-roce` in each network-status annotation, an active Ethernet RDMA
port, and the configured UCX values. A ready Pod alone does not validate NIXL:
after the streaming test, inspect all worker logs for UCX/NIXL transfer errors
and require a valid response before benchmarking.

### Port-forward and test the frontend

Keep this command running in a dedicated terminal. The Kubernetes frontend
listens on port 8000; `LOCAL_PORT` is the port exposed only on the administration
host. Once the deployment endpoints are active, port-forward using the frontend
Service:

```bash
export LOCAL_PORT=8000
kubectl port-forward -n "$NAMESPACE" \
  "svc/${DEPLOYMENT}-frontend" "${LOCAL_PORT}:8000"
```

If you need to test a specific frontend pod directly without routing through the
Service object:

```bash
export LOCAL_PORT=8000
kubectl port-forward -n "$NAMESPACE" \
  "$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=$DEPLOYMENT,app.kubernetes.io/component=frontend" -o name | head -n 1)" \
  "${LOCAL_PORT}:8000"
```

From a second terminal, verify health and confirm that the exact served model
is registered before sending a streaming chat request:

```bash
export LOCAL_PORT=8000

curl -fsS "http://127.0.0.1:${LOCAL_PORT}/health"
curl -fsS "http://127.0.0.1:${LOCAL_PORT}/v1/models" | jq .

curl -fsS -N "http://127.0.0.1:${LOCAL_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    "messages": [
      {"role": "user", "content": "Reply with exactly: frontend ready"}
    ],
    "max_tokens": 64,
    "temperature": 0,
    "stream": true
  }'
```

Do not start the benchmark unless `/health` succeeds, `/v1/models` contains
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, and the chat request
returns a valid streaming response. Stop the port-forward with `Ctrl-C` after
testing; the in-cluster benchmark uses the frontend Service directly.

## 4. Run the saturation and breaker benchmark

The benchmark Job, high-concurrency sweep, stopping rules, and matched MTP
on/off comparison procedure are in [`perf.md`](../perf.md). The default Job now
starts at concurrency 64. The decode scheduling limit is 1536 sequences
for 1P3D or 1024 for 2P2D (512 per decode worker). Find saturation independently
for this topology before attempting the larger breaker points.

Configure the Prometheus and DCGM capture in [`metrics.md`](../metrics.md) before
the first load point. MTP acceptance, engine queue depth, GPU activity, KV-cache
pressure, errors, and pod restarts must be recorded over the same UTC interval
as every AIPerf result.

## 5. Cleanup

```bash
kubectl delete job "$PERF_JOB" -n "$NAMESPACE" --ignore-not-found
kubectl delete dynamographdeployment "$DEPLOYMENT" -n "$NAMESPACE" --ignore-not-found
kubectl delete configmap nemotron35-vllm-e3-settings -n "$NAMESPACE" --ignore-not-found
kubectl delete job "$DOWNLOAD_JOB" -n "$NAMESPACE" --ignore-not-found
```

Deleting the download Job does not remove the checkpoints stored on the
`model-cache` PVC.
