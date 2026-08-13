# vLLM prefill/decode disaggregation

This non-KV-aware vLLM P/D baseline uses six TP=2 prefill workers, two TP=2
decode workers, NIXL over UCX/RDMA, and one frontend. It allocates 16 H100 GPUs.
Worker Pods use a dedicated Multus/MacVLAN RoCE interface without host
networking or host ports.

## Variables

Set these once before running any command in this recipe:

```bash
export NAMESPACE=qwen32-bench
export NETOP_NAMESPACE=nvidia-network-operator
export EXP_DIR=/ephemeral/shared/qwen3-32b/experiments/vllm/deprecated/disagg-routing
export DEPLOYMENT=qwen3-32b-fp8-vllm-disagg
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
export FRONTEND_SERVICE="${DEPLOYMENT}-frontend"
export PERF_JOB=qwen3-32b-fp8-vllm-disagg-perf
export ROCE_NETWORK=qwen-roce
export ROCE_POOL=qwen-roce-pool
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

The `qwen32-bench/qwen-roce` secondary network must already exist. Follow the
[pod-native RoCE runbook](../disagg-routing-kv-aware/pod-native-roce.md) before
deploying if Multus, NV-IPAM, the IP pool, and the MacVLAN network have not been
configured and validated on every eligible GPU node.

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
      resources:
        requests:
          cpu: "32"
          memory: "64Gi"
        limits:
          memory: "128Gi"
    VllmPrefillWorker:
      componentType: worker
      subComponentType: prefill
      envFromSecret: hf-token-secret
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 40Gi
      extraPodMetadata:
        annotations:
          k8s.v1.cni.cncf.io/networks: qwen32-bench/qwen-roce
      extraPodSpec:
        hostNetwork: false
        dnsPolicy: ClusterFirst
        mainContainer:
          ports: []
          env:
            - name: SERVED_MODEL_NAME
              value: "Qwen/Qwen3-32B-FP8"
            - name: MODEL_PATH
              value: "/opt/models/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
            - name: HF_HOME
              value: /opt/models
            - name: PYTHONHASHSEED
              value: "0"
            - name: VLLM_NIXL_SIDE_CHANNEL_HOST
              valueFrom:
                fieldRef:
                  fieldPath: status.podIP
            - name: UCX_TLS
              value: "rc_x,rc,cuda_copy,cuda_ipc"
            - name: UCX_NET_DEVICES
              value: "mlx5_8:1"
            - name: UCX_IB_ADDR_TYPE
              value: "eth"
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
                --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' \
                --gpu-memory-utilization 0.90 \
                --max-model-len 40960 \
                --no-enable-prefix-caching \
                --block-size 128
          command:
            - /bin/sh
            - -c
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          workingDir: /workspace/examples/backends/vllm
          securityContext:
            runAsUser: 0
            capabilities:
              add:
                - IPC_LOCK
                - SYS_RESOURCE
      replicas: 6
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
      extraPodMetadata:
        annotations:
          k8s.v1.cni.cncf.io/networks: qwen32-bench/qwen-roce
      extraPodSpec:
        hostNetwork: false
        dnsPolicy: ClusterFirst
        mainContainer:
          ports: []
          env:
            - name: SERVED_MODEL_NAME
              value: "Qwen/Qwen3-32B-FP8"
            - name: MODEL_PATH
              value: "/opt/models/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
            - name: HF_HOME
              value: /opt/models
            - name: PYTHONHASHSEED
              value: "0"
            - name: VLLM_NIXL_SIDE_CHANNEL_HOST
              valueFrom:
                fieldRef:
                  fieldPath: status.podIP
            - name: UCX_TLS
              value: "rc_x,rc,cuda_copy,cuda_ipc"
            - name: UCX_NET_DEVICES
              value: "mlx5_8:1"
            - name: UCX_IB_ADDR_TYPE
              value: "eth"
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
                --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}' \
                --gpu-memory-utilization 0.90 \
                --max-model-len 40960 \
                --no-enable-prefix-caching \
                --block-size 128
          command:
            - /bin/sh
            - -c
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          workingDir: /workspace/examples/backends/vllm
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
kubectl get network-attachment-definition "$ROCE_NETWORK" \
  -n "$NAMESPACE" -o yaml
kubectl get macvlannetwork "$ROCE_NETWORK" -o yaml
kubectl get ippool "$ROCE_POOL" -n "$NETOP_NAMESPACE" -o yaml
kubectl apply --dry-run=server -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
```

PASS only if every eligible GPU node exposes `mlx5_8:1` and the `rdma/ib`
resource, the model snapshot referenced by `MODEL_PATH` is cached, and the
`qwen-roce` NetworkAttachmentDefinition uses the ready `qwen-roce` MacVLAN
network and `qwen-roce-pool` IP pool. The physical host may expose its RoCE GID
at index 3, but `UCX_IB_GID_INDEX` must remain unset in the manifest so UCX can
select the Pod-specific GID created for `net1`.

## 2. Deploy

```bash
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl get pods -n "$NAMESPACE" \
  -l "$GRAPH_LABEL" \
  -o wide -w
```

Expected: one frontend, six prefill workers using two GPUs each, and two decode
workers using two GPUs each.

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

Verify that all eight worker Pods received a unique secondary-network address:

```bash
kubectl get pods -n "$NAMESPACE" \
  -l "$GRAPH_LABEL,nvidia.com/dynamo-component-type=worker" \
  -o name |
while read -r pod; do
  echo "===== $pod ====="
  kubectl get -n "$NAMESPACE" "$pod" -o json |
    jq -r '.metadata.annotations["k8s.v1.cni.cncf.io/network-status"] | fromjson | .[] | select(.interface == "net1") | [.interface, (.ips | join(","))] | @tsv'
  kubectl exec -n "$NAMESPACE" "$pod" -- env |
    grep -E '^UCX_(TLS|NET_DEVICES|IB_ADDR_TYPE|IB_GID_INDEX)='
done
```

Do not benchmark until every pod is `Running`, containers are ready, and logs
show NIXL/UCX initialization without transport errors. Each worker must show a
unique `net1` address, `UCX_NET_DEVICES=mlx5_8:1`, and no
`UCX_IB_GID_INDEX` entry.

## 4. Smoke test

```bash
kubectl port-forward -n "$NAMESPACE" \
  service/"$FRONTEND_SERVICE" "$LOCAL_PORT":8000
```

In another terminal, send a smoke-test request:

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

Immediately check for successful KV transfers and transport failures:

```bash
kubectl get pods -n "$NAMESPACE" \
  -l "$GRAPH_LABEL,nvidia.com/dynamo-component-type=worker" \
  -o name |
while read -r pod; do
  kubectl logs -n "$NAMESPACE" "$pod" \
    --all-containers --prefix --since=10m
done |
grep -Ei \
  'KV Transfer metrics|successful transfers|nixl|rc_mlx5|failed transfers|NIXL_ERR|Address not valid|No such device' |
tail -n 500
```

The request must succeed, successful-transfer evidence must appear, and no
failed-transfer increase, TCP fallback, `NIXL_ERR`, `Address not valid`, or
`No such device` error may appear.

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
