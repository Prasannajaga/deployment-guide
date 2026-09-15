# Model Caching for Dynamo on Kubernetes

This document explains how to download a pinned Hugging Face model revision once into a shared `ReadWriteMany` (RWX) PersistentVolumeClaim (PVC). After that, Dynamo components(e.g. Frontend, VllmWorker etc.) can mount the same cache, avoiding repeated downloads during deployment, restart, and scale-out.

In this document, we will use `Qwen/Qwen3-32B-FP8` as an example. When adapting it for another model, change `MODEL_NAME` and `MODEL_REVISION`, then update the matching snapshot path in the DGD.

| Setting | Value |
| --- | --- |
| Namespace | `dynamo-bench` |
| PVC | `model-cache` |
| Model | `Qwen/Qwen3-32B-FP8` |
| Revision | `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df` |
| Download Job cache root | `/model-store` |
| Worker cache root | `/opt/models` |

## 1. Set variables and check prerequisites

Run the commands in this guide from a Kubernetes administrator host. Store all generated manifests under a shared cluster-side experiment directory:

```bash
export NAMESPACE=dynamo-bench
export EXP_DIR=/ephemeral/shared/model-cache/qwen3-32b-fp8
export MODEL_DOWNLOAD_JOB=qwen3-32b-fp8-download

mkdir -p "$EXP_DIR"

kubectl get namespace "$NAMESPACE"
kubectl get crd dynamographdeployments.nvidia.com
kubectl auth can-i create persistentvolumeclaims -n "$NAMESPACE"
kubectl auth can-i create jobs.batch -n "$NAMESPACE"
```

The cluster must have an RWX-capable storage class, such as NFS, CephFS, or a cloud-provider shared filesystem. The example requests `100Gi`. Increase the capacity when the model and its revisions require more space.

List the available storage classes before creating the PVC:

```bash
kubectl get storageclass
```

## 2. Create or verify the shared PVC

First check whether the namespace already has a suitable `model-cache` PVC:

```bash
kubectl get pvc model-cache -n "$NAMESPACE"
```

If it exists, verify that it is `Bound` (meaning that the PVC is successfully bound to a PV(PersistentVolume)), uses `ReadWriteMany`, and has enough capacity.

If it does not exist, create the manifest below.

```bash
tee "$EXP_DIR/model-cache-pvc.yaml" >/dev/null <<'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
EOF

kubectl apply --dry-run=server -n "$NAMESPACE" \
  -f "$EXP_DIR/model-cache-pvc.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/model-cache-pvc.yaml"
kubectl wait -n "$NAMESPACE" \
  --for=jsonpath='{.status.phase}'=Bound pvc/model-cache \
  --timeout=5m
kubectl get pvc model-cache -n "$NAMESPACE"
```

Do not continue until the PVC is `Bound`.

## 3. Optionally add Hugging Face authentication

Public models can usually be downloaded without a token. However, gated or private models require a Hugging Face read token. Using a token can also help avoid anonymous download limits. You can create one in the [Hugging Face access-token settings](https://huggingface.co/settings/tokens).

If a token is required, enter it without placing it in shell history. This command writes the token directly to a Kubernetes Secret:

```bash
read -rsp 'Hugging Face read token: ' HF_TOKEN_INPUT
echo
printf '%s' "$HF_TOKEN_INPUT" \
  | kubectl create secret generic hf-token-secret \
      --namespace "$NAMESPACE" \
      --from-file=HF_TOKEN=/dev/stdin \
      --dry-run=client -o yaml \
  | kubectl apply -f -
unset HF_TOKEN_INPUT
```

Avoid revealing the token value; verify only that the Secret exists:

```bash
kubectl get secret hf-token-secret -n "$NAMESPACE"
```

## 4. Create the one-time download Job

The Job mounts the shared PVC at `/model-store`, sets `HF_HOME` to the same location, and downloads an immutable model revision. Workers mount the same PVC at `/opt/models`, so both paths expose the same cache contents. The Secret reference is optional, so the same Job works for public models without a Secret.

```bash
tee "$EXP_DIR/model-download.yaml" >/dev/null <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: qwen3-32b-fp8-download
  namespace: dynamo-bench
spec:
  template:
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
          volumeMounts:
            - name: model-cache
              mountPath: /model-store
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache
EOF
```

## 5. Download and verify the pinned snapshot

Validate the Job against the live cluster API, apply it, and wait for it to complete. Delete only an old Job with the same name before rerunning. `--ignore-not-found` suppresses an error when no matching Job exists. Deleting the download Job does not affect files already stored on the PVC.

```bash
kubectl get pvc model-cache -n "$NAMESPACE"
kubectl delete job "$MODEL_DOWNLOAD_JOB" \
  -n "$NAMESPACE" --ignore-not-found
kubectl apply --dry-run=server -n "$NAMESPACE" \
  -f "$EXP_DIR/model-download.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/model-download.yaml"
kubectl wait -n "$NAMESPACE" \
  --for=condition=Complete "job/$MODEL_DOWNLOAD_JOB" \
  --timeout=2h
kubectl logs -n "$NAMESPACE" \
  "job/$MODEL_DOWNLOAD_JOB" --tail=100
```

Hugging Face stores the downloaded model in this cache layout:

```text
hub/models--<org>--<model>/snapshots/<commit-hash>/
```

For this example, the complete path in the download Job is:

```text
/model-store/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df
```

Mount the PVC in a temporary Pod and verify the model directory actually exists:

```bash
kubectl run model-cache-inspect -n "$NAMESPACE" \
  --rm -i --restart=Never --image=busybox:1.36 \
  --overrides='{
    "spec": {
      "volumes": [
        {
          "name": "model-cache",
          "persistentVolumeClaim": {"claimName": "model-cache"}
        }
      ],
      "containers": [
        {
          "name": "model-cache-inspect",
          "image": "busybox:1.36",
          "volumeMounts": [
            {"name": "model-cache", "mountPath": "/opt/models"}
          ]
        }
      ]
    }
  }' \
  -- sh -c 'test -d /opt/models/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df && echo "Pinned snapshot: present"'
```

## 6. Mount the cache in a DynamoGraphDeployment

In a DynamoGraphDeployment (DGD), `create: false` tells the Dynamo operator to use the existing PVC (`model-cache`). Mount the cache into every component that reads model files. Workers require the model weights, and the frontend can reuse tokenizer and configuration files from the same cache.

Add the following fields to the complete deployment manifest for the recipe. DGD services use `mountPoint`.

```yaml
spec:
  pvcs:
    - name: model-cache
      create: false
  services:
    Frontend:
      componentType: frontend
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      envs:
        - name: HF_HOME
          value: /opt/models
    VllmWorker:
      componentType: worker
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      extraPodSpec:
        mainContainer:
          env:
            - name: SERVED_MODEL_NAME
              value: Qwen/Qwen3-32B-FP8
            - name: MODEL_PATH
              value: /opt/models/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df
            - name: HF_HOME
              value: /opt/models
```

After updating the complete recipe manifest, save it as `$EXP_DIR/deploy.yaml`, then validate it before deployment:

```bash
kubectl apply --dry-run=server -n "$NAMESPACE" \
  -f "$EXP_DIR/deploy.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
```

## 7. Troubleshooting

If the download or deployment fails, check in this order:

1. PVC state, access mode, capacity, and storage class

```bash
kubectl get pvc model-cache -n "$NAMESPACE" -o wide
kubectl describe pvc model-cache -n "$NAMESPACE"
```

2. Download Job state and events

```bash
kubectl get job "$MODEL_DOWNLOAD_JOB" -n "$NAMESPACE"
kubectl describe job "$MODEL_DOWNLOAD_JOB" -n "$NAMESPACE"
```

3. Downloader logs

```bash
kubectl logs -n "$NAMESPACE" \
  "job/$MODEL_DOWNLOAD_JOB" --tail=200
```

4. DGD and generated Pod state

```bash
kubectl get dynamographdeployments -n "$NAMESPACE"
kubectl get pods -n "$NAMESPACE" -o wide
kubectl get events -n "$NAMESPACE" \
  --sort-by='.lastTimestamp' | tail -n 50
```


## References

- [NVIDIA Dynamo Model Caching](https://docs.nvidia.com/dynamo/kubernetes/model-deployment/model-loading/model-caching)
