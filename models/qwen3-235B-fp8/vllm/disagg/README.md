# Qwen3-235B-A22B-FP8 disaggregated recipe

This recipe runs two TP4 prefill workers and two TP4 decode workers on 16 H100 GPUs. We've reused the the existing RoCE network configured in dynamo-bench, which is used for KV Cache Transfer.

## 1. Variables and preflight

```bash
export NAMESPACE=dynamo-bench
export EXP_DIR=/ephemeral/shared/qwen3-235b-a22b-fp8/vllm/disagg
export DEPLOYMENT=qwen3-235b-a22b-fp8-vllm-disagg
export PERF_JOB=qwen3-235b-a22b-fp8-vllm-disagg-bench
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=$DEPLOYMENT"
mkdir -p "$EXP_DIR"

kubectl get pvc model-cache perf-cache -n "$NAMESPACE"
kubectl get secret hf-token-secret nvcrimagepullsecret -n "$NAMESPACE"
kubectl get network-attachment-definition roce -n "$NAMESPACE"
kubectl get nodes -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,RDMA:.status.allocatable.rdma/ib'
```

The four workers require 16 GPUs and eight `rdma/ib` resources in total.

Download the shared model cache first if needed. See the [pinned model-cache docs](../../model-cache/README.md).

## 2. Create the deployment manifest

```bash
tee "$EXP_DIR/deploy.yaml" >/dev/null <<'EOF'
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-235b-a22b-fp8-vllm-disagg
spec:
  backendFramework: vllm
  pvcs:
    - name: model-cache
      create: false
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      envs:
        - name: HF_HOME
          value: /opt/models
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          workingDir: /workspace/examples/backends/vllm
          command:
            - /bin/sh
            - -c
          args:
            - python3 -m dynamo.frontend --router-mode round-robin --http-port 8000

    VllmPrefillWorker:
      componentType: worker
      subComponentType: prefill
      replicas: 2
      envFromSecret: hf-token-secret
      sharedMemory:
        size: 80Gi
      extraPodMetadata:
        annotations:
          k8s.v1.cni.cncf.io/networks: roce
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      resources:
        limits:
          gpu: "4"
          custom:
            rdma/ib: "2"
        requests:
          gpu: "4"
          custom:
            rdma/ib: "2"
      extraPodSpec:
        hostNetwork: false
        dnsPolicy: ClusterFirst
        affinity:
          nodeAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
              nodeSelectorTerms:
                - matchExpressions:
                    - key: nvidia.com/gpu.present
                      operator: In
                      values:
                        - "true"
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          workingDir: /workspace/examples/backends/vllm
          command:
            - /bin/sh
            - -c
          args:
            - |
              ulimit -l unlimited
              exec python3 -m dynamo.vllm \
                --model "$MODEL_PATH" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --tensor-parallel-size 4 \
                --data-parallel-size 1 \
                --enable-expert-parallel \
                --disaggregation-mode prefill \
                --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
                --gpu-memory-utilization 0.90 \
                --max-model-len 8192 \
                --enable-chunked-prefill \
                --max-num-seqs 128 \
                --max-num-batched-tokens 8192 \
                --enable-prefix-caching \
                --block-size 128 \
                --no-enable-log-requests
          env:
            - name: SERVED_MODEL_NAME
              value: Qwen/Qwen3-235B-A22B-FP8
            - name: MODEL_PATH
              value: Qwen/Qwen3-235B-A22B-FP8
            - name: HF_HOME
              value: /opt/models
            - name: PYTHONHASHSEED
              value: "0"
            - name: VLLM_NIXL_SIDE_CHANNEL_HOST
              valueFrom:
                fieldRef:
                  fieldPath: status.podIP
            - name: VLLM_NIXL_SIDE_CHANNEL_PORT
              value: "5600"
            - name: NIXL_TELEMETRY_ENABLE
              value: "y"
            - name: NIXL_TELEMETRY_EXPORTER
              value: prometheus
            - name: NIXL_TELEMETRY_MULTIPROC_DIR
              value: /dev/shm/nixl-telemetry
            - name: NIXL_TELEMETRY_PROMETHEUS_PORT
              value: "19090"
            - name: UCX_TLS
              value: rc_x,rc,cuda_copy,cuda_ipc
            - name: UCX_IB_ADDR_TYPE
              value: eth
          securityContext:
            runAsUser: 0
            capabilities:
              add:
                - IPC_LOCK
                - SYS_RESOURCE
          ports:
            - name: system
              containerPort: 9090
            - name: nixl-side
              containerPort: 5600
            - name: nixl-metrics
              containerPort: 19090

    VllmDecodeWorker:
      componentType: worker
      subComponentType: decode
      replicas: 2
      envFromSecret: hf-token-secret
      sharedMemory:
        size: 80Gi
      extraPodMetadata:
        annotations:
          k8s.v1.cni.cncf.io/networks: roce
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      resources:
        limits:
          gpu: "4"
          custom:
            rdma/ib: "2"
        requests:
          gpu: "4"
          custom:
            rdma/ib: "2"
      extraPodSpec:
        hostNetwork: false
        dnsPolicy: ClusterFirst
        affinity:
          nodeAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
              nodeSelectorTerms:
                - matchExpressions:
                    - key: nvidia.com/gpu.present
                      operator: In
                      values:
                        - "true"
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          workingDir: /workspace/examples/backends/vllm
          command:
            - /bin/sh
            - -c
          args:
            - |
              ulimit -l unlimited
              exec python3 -m dynamo.vllm \
                --model "$MODEL_PATH" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --tensor-parallel-size 4 \
                --data-parallel-size 1 \
                --enable-expert-parallel \
                --disaggregation-mode decode \
                --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
                --gpu-memory-utilization 0.90 \
                --max-model-len 8192 \
                --enable-chunked-prefill \
                --max-num-seqs 128 \
                --max-num-batched-tokens 8192 \
                --enable-prefix-caching \
                --block-size 128 \
                --no-enable-log-requests
          env:
            - name: SERVED_MODEL_NAME
              value: Qwen/Qwen3-235B-A22B-FP8
            - name: MODEL_PATH
              value: Qwen/Qwen3-235B-A22B-FP8
            - name: HF_HOME
              value: /opt/models
            - name: PYTHONHASHSEED
              value: "0"
            - name: VLLM_NIXL_SIDE_CHANNEL_HOST
              valueFrom:
                fieldRef:
                  fieldPath: status.podIP
            - name: VLLM_NIXL_SIDE_CHANNEL_PORT
              value: "5600"
            - name: NIXL_TELEMETRY_ENABLE
              value: "y"
            - name: NIXL_TELEMETRY_EXPORTER
              value: prometheus
            - name: NIXL_TELEMETRY_MULTIPROC_DIR
              value: /dev/shm/nixl-telemetry
            - name: NIXL_TELEMETRY_PROMETHEUS_PORT
              value: "19090"
            - name: UCX_TLS
              value: rc_x,rc,cuda_copy,cuda_ipc
            - name: UCX_IB_ADDR_TYPE
              value: eth
          securityContext:
            runAsUser: 0
            capabilities:
              add:
                - IPC_LOCK
                - SYS_RESOURCE
          ports:
            - name: system
              containerPort: 9090
            - name: nixl-side
              containerPort: 5600
            - name: nixl-metrics
              containerPort: 19090
EOF
```

## 3. Validate and deploy

```bash
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl wait -n "$NAMESPACE" --for=jsonpath='{.status.state}'=successful "dynamographdeployment/$DEPLOYMENT" --timeout=45m
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide
```

Verify the actual RoCE attachment on a prefill worker:

```bash
export PREFILL_POD=$(kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o name | rg 'vllmprefillworker' | head -1 | sed 's#pod/##')
kubectl get pod -n "$NAMESPACE" "$PREFILL_POD" -o json | jq '{node:.spec.nodeName,networks:.metadata.annotations["k8s.v1.cni.cncf.io/networks"],networkStatus:.metadata.annotations["k8s.v1.cni.cncf.io/network-status"]}'
kubectl exec -n "$NAMESPACE" "$PREFILL_POD" -- sh -lc 'test -d /dev/infiniband && ls /dev/infiniband'
kubectl exec -n "$NAMESPACE" "$PREFILL_POD" -- printenv UCX_TLS UCX_IB_ADDR_TYPE
kubectl exec -n "$NAMESPACE" "$PREFILL_POD" -- ibv_devices
```

## 4. Create and run the benchmark

```bash
tee "$EXP_DIR/perf.yaml" >/dev/null <<'EOF'
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
apiVersion: batch/v1
kind: Job
metadata:
  name: qwen3-235b-a22b-fp8-vllm-disagg-bench
spec:
  backoffLimit: 1
  completions: 1
  parallelism: 1
  template:
    metadata:
      labels:
        app: qwen3-235b-a22b-fp8-vllm-disagg-bench
    spec:
      containers:
      - command:
        - /bin/sh
        - -c
        - |
          apt-get update && apt-get install -y curl jq procps git build-essential && apt-get clean
          pip install "aiperf==0.10.0";
          echo "aiperf installation completed";
          sysctl -w net.ipv4.ip_local_port_range="1024 65000"
          cat /proc/sys/net/ipv4/ip_local_port_range
          export COLUMNS=200
          EPOCH=$(date +%s)
          wait_for_model_ready() {
            echo "Waiting for model '$TARGET_MODEL' at $ENDPOINT/v1/models (checking every 5s)..."
            while ! curl -s "http://$ENDPOINT/v1/models" | jq -e --arg model "$TARGET_MODEL" '.data[]? | select(.id == $model)' >/dev/null 2>&1; do
                echo "[$(date '+%H:%M:%S')] Model not ready yet, sleeping 5s before checking again http://$ENDPOINT/v1/models"
                sleep 5
            done
            echo "Model '$TARGET_MODEL' is now available!"
            curl -s "http://$ENDPOINT/v1/models" | jq .
          }
          run_perf() {
            local concurrency=$1
            local isl=$2
            local osl=$3
            key=concurrency_${concurrency}
            export ARTIFACT_DIR="${ROOT_ARTIFACT_DIR}/${EPOCH}_${JOB_NAME}/${key}"
            mkdir -p "$ARTIFACT_DIR"
            echo "ARTIFACT_DIR: $ARTIFACT_DIR"
            aiperf profile --artifact-dir $ARTIFACT_DIR \
                --model $TARGET_MODEL \
                --tokenizer $TARGET_MODEL \
                --endpoint-type chat \
                --endpoint /v1/chat/completions \
                --streaming \
                --use-server-token-count \
                --url http://$ENDPOINT \
                --synthetic-input-tokens-mean $isl \
                --synthetic-input-tokens-stddev 0 \
                --output-tokens-mean $osl \
                --output-tokens-stddev 0 \
                --extra-inputs "max_tokens:$osl" \
                --extra-inputs "min_tokens:$osl" \
                --extra-inputs "ignore_eos:true" \
                --extra-inputs "repetition_penalty:1.0" \
                --extra-inputs "temperature: 0.0" \
                --concurrency $concurrency \
                --request-count $((10*concurrency)) \
                --warmup-request-count $concurrency \
                --num-dataset-entries 12800 \
                --random-seed 100 \
                --workers-max 252 \
                -H 'Authorization: Bearer NOT USED' \
                -H 'Accept: text/event-stream'\
                --record-processors 32 \
                --ui simple
            echo "ARTIFACT_DIR: $ARTIFACT_DIR"
            ls -la $ARTIFACT_DIR
          }
          wait_for_model_ready
          mkdir -p "${ROOT_ARTIFACT_DIR}/${EPOCH}_${JOB_NAME}"
          # Calculate total concurrency based on per-GPU concurrency and GPU count
          TOTAL_CONCURRENCY=$((CONCURRENCY_PER_GPU * DEPLOYMENT_GPU_COUNT))
          echo "Calculated total concurrency: $TOTAL_CONCURRENCY (${CONCURRENCY_PER_GPU} per GPU x ${DEPLOYMENT_GPU_COUNT} GPUs)"
          # Write input_config.json
          cat > "${ROOT_ARTIFACT_DIR}/${EPOCH}_${JOB_NAME}/input_config.json" <<EOF
          {
            "gpu_count": $DEPLOYMENT_GPU_COUNT,
            "concurrency_per_gpu": $CONCURRENCY_PER_GPU,
            "total_concurrency": $TOTAL_CONCURRENCY,
            "mode": "$DEPLOYMENT_MODE",
            "isl": $ISL,
            "osl": $OSL,
            "endpoint": "$ENDPOINT",
            "model endpoint": "$TARGET_MODEL"
          }
          EOF

          # Run perf with calculated total concurrency
          run_perf $TOTAL_CONCURRENCY $ISL $OSL
          echo "done with concurrency $TOTAL_CONCURRENCY"
        env:
        - name: TARGET_MODEL
          value: Qwen/Qwen3-235B-A22B-FP8
        - name: ENDPOINT
          value: qwen3-235b-a22b-fp8-vllm-disagg-frontend:8000
        - name: CONCURRENCY_PER_GPU
          value: "2"
        - name: DEPLOYMENT_GPU_COUNT
          value: "16"
        - name: ISL
          value: "4000"
        - name: OSL
          value: "200"
        - name: DEPLOYMENT_MODE
          value: disagg
        - name: AIPERF_HTTP_CONNECTION_LIMIT
          value: "200"
        - name: JOB_NAME
          valueFrom:
            fieldRef:
              apiVersion: v1
              fieldPath: metadata.labels['job-name']
        - name: ROOT_ARTIFACT_DIR
          value: /perf-cache
        - name: HF_HOME
          value: /model-cache
        - name: PYTHONUNBUFFERED
          value: "1"
        image: python:3.12-slim
        imagePullPolicy: IfNotPresent
        name: perf
        securityContext:
          privileged: true
        volumeMounts:
        - name: model-cache
          mountPath: /model-cache
        - name: perf-cache
          mountPath: /perf-cache
        workingDir: /workspace
      imagePullSecrets:
      - name: nvcrimagepullsecret
      restartPolicy: Never
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache
      - name: perf-cache
        persistentVolumeClaim:
          claimName: perf-cache
EOF

kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/perf.yaml"
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/perf.yaml"
kubectl logs -n "$NAMESPACE" -f "job/$PERF_JOB"
kubectl wait -n "$NAMESPACE" --for=condition=Complete "job/$PERF_JOB" --timeout=2h
```

The benchmark uses fixed synthetic workload:

- 4,000 ISL, 200 OSL
- 32 concurrent requests
- 32 warmup requests and 320 measured requests

## 5. Cleanup

Copy the benchmark artifacts before deleting the graph.

```bash
kubectl delete job "$PERF_JOB" -n "$NAMESPACE" --ignore-not-found
kubectl delete dynamographdeployment "$DEPLOYMENT" -n "$NAMESPACE" --ignore-not-found --wait=false
kubectl wait -n "$NAMESPACE" --for=delete pod -l "$GRAPH_LABEL" --timeout=15m
```
