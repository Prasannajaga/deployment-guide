# Qwen3-235B-A22B-FP8 model cache

Run this once from a Kubernetes admin host when the shared cache does not already contain the pinned model snapshot.

```bash
export NAMESPACE=qwen235-bench
export EXP_DIR=/ephemeral/shared/qwen3-235b-a22b-fp8/model-cache
mkdir -p "$EXP_DIR"

tee "$EXP_DIR/model-download.yaml" >/dev/null <<'EOF'
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
apiVersion: batch/v1
kind: Job
metadata:
  name: model-download
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
              value: Qwen/Qwen3-235B-A22B-FP8
            - name: MODEL_REVISION
              value: 39eb2b067ea6b8e3e1dd97d3cd0c7ffeaf3e1a35
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

kubectl get pvc model-cache -n "$NAMESPACE"
kubectl get secret hf-token-secret -n "$NAMESPACE"
kubectl delete job model-download -n "$NAMESPACE" --ignore-not-found
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/model-download.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/model-download.yaml"
kubectl wait -n "$NAMESPACE" --for=condition=Complete job/model-download --timeout=2h
kubectl logs -n "$NAMESPACE" job/model-download --tail=100
```
