# vLLM prefill/decode disaggregation

This is the validated vLLM P/D baseline: two TP=2 prefill workers, one TP=4
decode worker, NIXL over UCX/RDMA, and one frontend. It allocates eight H100
GPUs. The manifest keeps the cluster-specific RoCE device and unique host ports.

## Variables

Set these once before running any command in this recipe:

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3-32b/experiments/vllm/disagg-routing
export DEPLOYMENT=qwen3-32b-fp8-vllm-disagg
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
export FRONTEND_SERVICE="${DEPLOYMENT}-frontend"
export PERF_JOB=qwen3-32b-fp8-vllm-disagg-perf
export LOCAL_PORT=8000
export BENCH_URL="http://127.0.0.1:${LOCAL_PORT}"
export MODEL=Qwen/Qwen3-32B-FP8
```

## Files

| File | Purpose |
| --- | --- |
| `deploy.yaml` | Runtime deployment |
| `perf.yaml` | In-cluster 35-cell AIPerf matrix |
| `plot-config.yaml` | TTFT/TPOT plotting defaults |
| `fetch-metrics.md` | Prometheus/DCGM export commands |
| `model-download.yaml` | Optional model-cache population Job |
| `cache.yaml` | Optional cache PVC helper |

## Create deploy.yaml

The quoted `EOF` delimiter preserves shell variables inside the manifest.
Running this block writes only the local configuration file.

```bash
mkdir -p "$EXP_DIR"
tee "$EXP_DIR/deploy.yaml" >/dev/null <<'EOF'
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-32b-fp8-vllm-disagg
spec:
  backendFramework: vllm
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
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          workingDir: /workspace/examples/backends/vllm
      envs:
        - name: HF_HOME
          value: /opt/models
      replicas: 1
    VllmPrefillWorker:
      componentType: worker
      subComponentType: prefill
      envFromSecret: hf-token-secret
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 40Gi
      extraPodSpec:
        hostNetwork: true
        dnsPolicy: ClusterFirstWithHostNet
        affinity:
          podAffinity:
            preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                  - key: nvidia.com/dynamo-component-type
                    operator: In
                    values:
                    - worker
                topologyKey: kubernetes.io/hostname
        mainContainer:
          env:
            - name: SERVED_MODEL_NAME
              value: "Qwen/Qwen3-32B-FP8"
            - name: MODEL_PATH
              value: "/opt/models/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
            - name: HF_HOME
              value: /opt/models
            - name: DYN_SYSTEM_PORT
              value: "9090"
            - name: NIXL_TELEMETRY_ENABLE
              value: "y"
            - name: NIXL_TELEMETRY_EXPORTER
              value: "prometheus"
            - name: NIXL_TELEMETRY_MULTIPROC_DIR
              value: /dev/shm/nixl-telemetry
            - name: NIXL_TELEMETRY_PROMETHEUS_PORT
              value: "19090"
            - name: DYN_FORWARDPASS_METRIC_PORT
              value: "20380"
            - name: VLLM_NIXL_SIDE_CHANNEL_PORT
              value: "5600"
            - name: UCX_TLS
              value: "rc_x,rc,cuda_copy,cuda_ipc"
            - name: UCX_NET_DEVICES
              value: "mlx5_8:1"
            - name: UCX_IB_ADDR_TYPE
              value: "eth"
            - name: UCX_IB_GID_INDEX
              value: "3"
            - name: UCX_RNDV_SCHEME
              value: "get_zcopy"
            - name: UCX_RNDV_THRESH
              value: "0"
            - name: UCX_IB_REG_METHODS
              value: "odp,rcache"
            - name: UCX_RCACHE_MAX_UNRELEASED
              value: "1024"
            - name: UCX_RC_TIMEOUT
              value: "600s"
            - name: UCX_KEEPALIVE_INTERVAL
              value: "300s"
            - name: UCX_LOG_LEVEL
              value: "info"
            - name: NIXL_LOG_LEVEL
              value: "INFO"
          args:
          - |
            ulimit -l unlimited && python3 -m dynamo.vllm \
              --model $MODEL_PATH \
              --served-model-name $SERVED_MODEL_NAME \
              --tensor-parallel-size 2 \
              --data-parallel-size 1 \
              --disaggregation-mode prefill \
              --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
              --gpu-memory-utilization 0.90 \
              --max-model-len 40960 \
              --no-enable-prefix-caching \
              --block-size 128
          command:
          - /bin/sh
          - -c
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          workingDir: /workspace/examples/backends/vllm
          ports:
            - name: system
              containerPort: 9090
              hostPort: 9090
            - name: nixl-metrics
              containerPort: 19090
              hostPort: 19090
            - name: fpm
              containerPort: 20380
              hostPort: 20380
            - name: nixl-side
              containerPort: 5600
              hostPort: 5600
          securityContext:
            runAsUser: 0
            capabilities:
              add:
                - IPC_LOCK
                - SYS_RESOURCE
      replicas: 2
      resources:
        limits:
          gpu: "2"
          custom:
            rdma/ib: "2"
        requests:
          gpu: "2"
          custom:
            rdma/ib: "2"
    VllmDecodeWorker:
      componentType: worker
      subComponentType: decode
      envFromSecret: hf-token-secret
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 40Gi
      extraPodSpec:
        hostNetwork: true
        dnsPolicy: ClusterFirstWithHostNet
        affinity:
          podAffinity:
            preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                  - key: nvidia.com/dynamo-component-type
                    operator: In
                    values:
                    - worker
                topologyKey: kubernetes.io/hostname
        mainContainer:
          env:
            - name: SERVED_MODEL_NAME
              value: "Qwen/Qwen3-32B-FP8"
            - name: MODEL_PATH
              value: "/opt/models/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
            - name: HF_HOME
              value: /opt/models
            - name: DYN_SYSTEM_PORT
              value: "9091"
            - name: NIXL_TELEMETRY_ENABLE
              value: "y"
            - name: NIXL_TELEMETRY_EXPORTER
              value: "prometheus"
            - name: NIXL_TELEMETRY_MULTIPROC_DIR
              value: /dev/shm/nixl-telemetry
            - name: NIXL_TELEMETRY_PROMETHEUS_PORT
              value: "19091"
            - name: DYN_FORWARDPASS_METRIC_PORT
              value: "20381"
            - name: VLLM_NIXL_SIDE_CHANNEL_PORT
              value: "5601"
            - name: UCX_TLS
              value: "rc_x,rc,cuda_copy,cuda_ipc"
            - name: UCX_NET_DEVICES
              value: "mlx5_8:1"
            - name: UCX_IB_ADDR_TYPE
              value: "eth"
            - name: UCX_IB_GID_INDEX
              value: "3"
            - name: UCX_RNDV_SCHEME
              value: "get_zcopy"
            - name: UCX_RNDV_THRESH
              value: "0"
            - name: UCX_IB_REG_METHODS
              value: "odp,rcache"
            - name: UCX_RCACHE_MAX_UNRELEASED
              value: "1024"
            - name: UCX_RC_TIMEOUT
              value: "600s"
            - name: UCX_KEEPALIVE_INTERVAL
              value: "300s"
            - name: UCX_LOG_LEVEL
              value: "info"
            - name: NIXL_LOG_LEVEL
              value: "INFO"
          args:
          - |
            ulimit -l unlimited && python3 -m dynamo.vllm \
              --model $MODEL_PATH \
              --served-model-name $SERVED_MODEL_NAME \
              --tensor-parallel-size 2 \
              --data-parallel-size 1 \
              --disaggregation-mode decode \
              --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' \
              --gpu-memory-utilization 0.90 \
              --max-model-len 40960 \
              --no-enable-prefix-caching \
              --block-size 128
          command:
          - /bin/sh
          - -c
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          workingDir: /workspace/examples/backends/vllm
          ports:
            - name: system
              containerPort: 9091
              hostPort: 9091
            - name: nixl-metrics
              containerPort: 19091
              hostPort: 19091
            - name: fpm
              containerPort: 20381
              hostPort: 20381
            - name: nixl-side
              containerPort: 5601
              hostPort: 5601
          securityContext:
            runAsUser: 0
            capabilities:
              add:
                - IPC_LOCK
                - SYS_RESOURCE
      replicas: 2
      resources:
        limits:
          gpu: "2"
          custom:
            rdma/ib: "2"
        requests:
          gpu: "2"
          custom:
            rdma/ib: "2"
EOF
```

## 1. Preflight

Run on the Kubernetes control-plane host:

```bash
kubectl get pvc -n "$NAMESPACE" model-cache
kubectl get nodes -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,RDMA:.status.allocatable.rdma/ib'
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
```

The two GPU nodes must expose `mlx5_8:1`, RoCE GID index `3`, the `rdma/ib`
resource, and the cached model snapshot referenced by `MODEL_PATH`.

## 2. Deploy

```bash
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl get pods -n "$NAMESPACE" \
  -l "$GRAPH_LABEL" \
  -o wide -w
```

Expected: one frontend, two prefill workers using two GPUs each, and one decode
worker using four GPUs.

## 3. Verify NIXL, readiness, and logs

Tail or stream logs for all containers in the deployment:

```bash
# Tail recent 500 lines across all containers
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" --all-containers --prefix --tail=500

# Stream live logs in real time
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" --all-containers --prefix -f

# Filter for NIXL, UCX, readiness, and errors
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" --all-containers --prefix --tail=500 | grep -Ei 'nixl|ucx|ready|error|traceback'

# Check prefill worker logs specifically
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL,nvidia.com/dynamo-sub-component-type=prefill" --all-containers --prefix --tail=500

# Check decode worker logs specifically
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL,nvidia.com/dynamo-sub-component-type=decode" --all-containers --prefix --tail=500

# Inspect logs from a previous crashed container instance
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" --all-containers --prefix --previous --tail=500
```

```bash
kubectl get pods -n "$NAMESPACE" \
  -l "$GRAPH_LABEL"
```

Do not benchmark until every pod is `Running`, containers are ready, and logs
show NIXL/UCX initialization without transport errors.

## 4. Smoke test

```bash
kubectl port-forward -n "$NAMESPACE" \
  service/"$FRONTEND_SERVICE" "$LOCAL_PORT":8000
```

In another terminal:

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

## 5. Benchmark

The benchmark Job runs inside Kubernetes, so a dropped SSH session or local
port-forward does not terminate the test.

```bash
kubectl delete job -n "$NAMESPACE" "$PERF_JOB" \
  --ignore-not-found
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/perf.yaml"
kubectl logs -n "$NAMESPACE" \
  job/"$PERF_JOB" -f
```

The matrix is fixed at:

- ISL: `1024 4096 8192 16384 32768`
- OSL: `256`
- concurrency: `1 8 16 32 64 128 256`
- TTFT goodput SLO: `2000 ms`
- TPOT goodput SLO: `50 ms/token`

Artifacts are written to the `perf-cache` PVC. Use `fetch-metrics.md` for the
matching eight-hour DCGM export. Install `plot-config.yaml` as
`~/.aiperf/plot_config.yaml` before running `aiperf plot` locally.

## 6. Clean up

```bash
kubectl delete job -n "$NAMESPACE" "$PERF_JOB" \
  --ignore-not-found
kubectl delete dynamographdeployment.nvidia.com "$DEPLOYMENT" \
  -n "$NAMESPACE" --wait=false --ignore-not-found
kubectl delete pods -l "$GRAPH_LABEL" -n "$NAMESPACE" \
  --force --grace-period=0 --ignore-not-found
```

The DGD and benchmark Job are removed first; the label-scoped pod command then
force-deletes the experiment pods immediately. The local `deploy.yaml` and
benchmark artifacts remain unchanged.

## References

- [Dynamo disaggregated communication](https://docs.nvidia.com/dynamo/dev/knowledge-base/kubernetes/kubernetes-operator/disagg-communication)
- [Dynamo disaggregated serving](https://docs.nvidia.com/dynamo/dev/kubernetes/disaggregated-serving/overview)
- [AIPerf plotting](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/plot.md)
