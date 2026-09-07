# Nemotron 3.5 Lightning research recipes

This model area standardizes the selected experiments on Dynamo 1.4.1,
vLLM 0.26.0, NIXL 1.3.2, and
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`. The deployment shapes
keep tensor parallelism at one so that each H100 is an independently
fail-able worker. Only experiment 1 and experiment 3 are represented here.

```text
nemotron-3.5-lightning/
├── model-cache/
│   └── model-download.yaml
└── vllm/experiments/
    ├── 01-migration-avalanche/
    │   ├── README.md
    │   ├── deploy.yaml
    │   ├── fault-inject.sh
    │   ├── matrix.yaml
    │   └── perf.yaml
    └── 03-spec-kv-routing/
        ├── README.md
        ├── analyze_interaction.py
        ├── deploy.yaml
        ├── matrix.yaml
        ├── perf.yaml
        └── run-experiment.sh
```

The model revisions are immutable Hugging Face commit SHAs. The runtime image
is pinned to `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1`; the preflight in
each experiment checks the bundled vLLM version before a graph is created.
The recipes use the existing `model-cache` and `perf-cache` PVCs and the
`qwen-roce` Multus network in the `qwen32-bench` namespace. The model and
runtime images used here are public and do not require registry or Hugging Face
secrets.

## Populate the model cache

Run these commands from a cluster-administration host. They generate the
runtime manifest under `/ephemeral/shared`; the repository path is not used by
the cluster. The Job verifies each exact snapshot before downloading, so a
complete cached model is reported as a cache hit and is not downloaded again.

```bash
export NAMESPACE=qwen32-bench
export RECIPE_ROOT=/ephemeral/shared/nemotron-3.5-lightning
export MODEL_CACHE_DIR="$RECIPE_ROOT/model-cache"
export DOWNLOAD_JOB=nemotron35-model-download
mkdir -p "$MODEL_CACHE_DIR"

kubectl get pvc model-cache perf-cache -n "$NAMESPACE"

tee "$MODEL_CACHE_DIR/model-download.yaml" >/dev/null <<'EOF'
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
EOF

kubectl delete job "$DOWNLOAD_JOB" -n "$NAMESPACE" --ignore-not-found
kubectl apply --dry-run=server -n "$NAMESPACE" \
  -f "$MODEL_CACHE_DIR/model-download.yaml"
kubectl apply -n "$NAMESPACE" -f "$MODEL_CACHE_DIR/model-download.yaml"
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  "job/$DOWNLOAD_JOB" --timeout=43200s
kubectl logs -n "$NAMESPACE" "job/$DOWNLOAD_JOB" --tail=100
```

MTP uses the base checkpoint's built-in prediction heads. The DFlash and
DSpark revisions are cached now so that experiment 3 can advance to its second
and third stages without changing model state mid-project.

## Sources

The model flags and H100 topology are adapted from NVIDIA's Nemotron 3.5
Lightning Dynamo recipes. Migration limits are frontend settings in Dynamo
1.4.1, and all experiment manifests keep the router and vLLM KV block size at
64 tokens.

- <https://docs.nvidia.com/dynamo/v1.4.0/recipes/nemotron-3-5-lightning>
- <https://docs.nvidia.com/dynamo/dev/kubernetes/fault-tolerance/request-migration>
- <https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4>
