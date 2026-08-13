# vLLM disaggregated routing with KV awareness

This experiment uses six TP=2 prefill workers and two TP=2 decode workers. The
frontend consumes worker KV events and routes repeated prefixes toward cached
prefill workers. Total allocation is 16 H100 GPUs.

## Variables

Set these once before running any command in this recipe:

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3-32b/experiments/vllm/deprecated/disagg-routing-kv-aware
export DEPLOYMENT=qwen3-32b-fp8-vllm-disagg-kv-aware
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
export FRONTEND_SERVICE="${DEPLOYMENT}-frontend"
export LOCAL_PORT=8000
```

## Files

- `deploy.yaml` — deployment and NIXL/UCX settings
- `prefill-nixl-preflight.yaml` — temporary prefill NIXL/UCX initialization check
- `pod-native-roce.md` — Multus/MacVLAN runbook for RoCE without host networking

## Create deploy.yaml

The quoted `EOF` delimiter preserves shell variables inside the manifest.
Running this block writes only the local configuration file.

```bash
mkdir -p "$EXP_DIR"
tee "$EXP_DIR/deploy-pod-roce.yaml" >/dev/null <<'EOF'
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-32b-fp8-vllm-disagg-kv-aware
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
          command:
            - python3
            - -m
            - dynamo.frontend
          args:
            - --router-mode
            - kv
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
                --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:0","enable_kv_cache_events":true}' \
                --gpu-memory-utilization 0.90 \
                --max-model-len 40960 \
                --enable-prefix-caching \
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
                --enable-prefix-caching \
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
EOFtee "$EXP_DIR/prefill-nixl-preflight.yaml" >/dev/null <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: qwen3-32b-prefill-nixl-preflight
  labels:
    app: qwen3-32b-prefill-nixl-preflight
spec:
  restartPolicy: Never
  activeDeadlineSeconds: 600
  nodeName: inst-1onle-devrel-rdma-pool
  hostNetwork: false
  dnsPolicy: ClusterFirst
  tolerations:
    - key: node-role.kubernetes.io/control-plane
      operator: Exists
      effect: NoSchedule
    - key: nvidia.com/gpu
      operator: Equal
      value: "true"
      effect: NoSchedule
  containers:
    - name: prefill-nixl-check
      image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
      imagePullPolicy: IfNotPresent
      workingDir: /workspace/examples/backends/vllm
      command:
        - /bin/bash
        - -lc
      args:
        - |
          set -euo pipefail
          echo "POD=$POD_NAME NODE=$NODE_NAME POD_IP=$POD_IP"
          exec sleep infinity
      env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
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
      securityContext:
        runAsUser: 0
        capabilities:
          add:
            - IPC_LOCK
            - SYS_RESOURCE
      resources:
        requests:
          nvidia.com/gpu: "2"
          rdma/ib: "2"
        limits:
          nvidia.com/gpu: "2"
          rdma/ib: "2"
      volumeMounts:
        - name: model-cache
          mountPath: /opt/models
        - name: shared-memory
          mountPath: /dev/shm
  volumes:
    - name: model-cache
      persistentVolumeClaim:
        claimName: model-cache
    - name: shared-memory
      emptyDir:
        medium: Memory
        sizeLimit: 40Gi
EOF

kubectl delete pod qwen3-32b-prefill-nixl-preflight \
  -n "$NAMESPACE" --ignore-not-found --wait=true
kubectl apply -n "$NAMESPACE" \
  -f "$EXP_DIR/prefill-nixl-preflight.yaml"

kubectl wait \
  --for=condition=Ready \
  pod/qwen3-32b-prefill-nixl-preflight \
  -n "$NAMESPACE" --timeout=5m

kubectl get pod qwen3-32b-prefill-nixl-preflight \
  -n "$NAMESPACE" -o wide
```

First inspect what the isolated prefill Pod actually sees. This is the direct
check for whether `mlx5_8`, port `1`, and GID index `3` are valid inside the
Pod rather than only on the host:

```bash
kubectl exec -n "$NAMESPACE" \
  pod/qwen3-32b-prefill-nixl-preflight -- bash -lc '
    set -euo pipefail

    echo "Configured UCX values:"
    env | grep -E "^UCX_(NET_DEVICES|IB_ADDR_TYPE|IB_GID_INDEX)="

    echo "Verbs character devices:"
    ls -la /dev/infiniband

    echo "RDMA devices visible in this Pod:"
    find /sys/class/infiniband -mindepth 1 -maxdepth 1 \
      -printf "%f\n" | sort

    echo "Pod network interfaces:"
    ip -brief address || true

    echo "RDMA-to-netdev mapping:"
    ibdev2netdev || true
    rdma link || true

    test -d /sys/class/infiniband/mlx5_8/ports/1
    echo "mlx5_8 port 1 exists"

    printf "state="
    cat /sys/class/infiniband/mlx5_8/ports/1/state
    printf "link_layer="
    cat /sys/class/infiniband/mlx5_8/ports/1/link_layer
    printf "gid[3]="
    cat /sys/class/infiniband/mlx5_8/ports/1/gids/3

    echo "Netdevices associated with mlx5_8:"
    find /sys/class/infiniband/mlx5_8/device/net \
      -mindepth 1 -maxdepth 1 -printf "%f\n" 2>/dev/null || true

    echo "Nonzero GIDs visible across all RDMA devices:"
    for gid_file in /sys/class/infiniband/*/ports/*/gids/*; do
      gid=$(cat "$gid_file")
      case "$gid" in
        0000:0000:0000:0000:0000:0000:0000:0000) continue ;;
      esac
      printf "%s=%s\n" "$gid_file" "$gid"
    done
  '
```

Stop here if `mlx5_8` does not exist. If it exists, require port `1` to be
`ACTIVE`, the link layer to be `Ethernet`, and GID index `3` to be nonzero.
Also check whether the associated RoCE netdevice is visible in the Pod's
network namespace.

Next initialize NIXL with the three values currently pinned in the production
prefill manifest:

```bash
kubectl exec -n "$NAMESPACE" \
  pod/qwen3-32b-prefill-nixl-preflight -- \
  python3 -c "import os, nixl; agent = nixl.nixl_agent(os.environ['POD_NAME']); print('NIXL_PINNED_UCX_BACKEND_OK')"
```

The pinned test passes only when it exits successfully and shows UCX backend
creation on `mlx5_8:1`, without TCP fallback. Expected success evidence is:

```text
Created backend: UCX
Backend UCX was instantiated
Initialized NIXL agent
NIXL_PINNED_UCX_BACKEND_OK
```

If the pinned test fails, compare it with the vLLM guide's literal
`UCX_NET_DEVICES=all` setting. Remove the pinned address type and GID index so
UCX can choose values appropriate to whichever device it selects:

```bash
kubectl exec -n "$NAMESPACE" \
  pod/qwen3-32b-prefill-nixl-preflight -- \
  env -u UCX_IB_ADDR_TYPE -u UCX_IB_GID_INDEX UCX_NET_DEVICES=all \
  python3 -c "import os, nixl; agent = nixl.nixl_agent(os.environ['POD_NAME'] + '-all'); print('NIXL_ALL_DEVICES_UCX_BACKEND_OK')"
```

This intentionally retains the production RDMA-only `UCX_TLS` value instead
of changing it to `all`. The test must not pass by silently selecting TCP.
Inspect the UCX worker configuration in the output and require at least one
`rc_mlx5/<device>:<port>` transport.

Interpret the comparison as follows:

- `mlx5_8` absent: `UCX_NET_DEVICES=mlx5_8:1` is invalid inside the Pod.
- Pinned initialization fails but the all-devices test succeeds over
  `rc_mlx5`: at least one of the three pinned UCX values is wrong for the Pod.
- The all-devices test succeeds without `rc_mlx5`: this is not an RDMA pass.
- Both initialization attempts fail: investigate RDMA device injection and
  the Pod network namespace before changing the DGD.

Only if the all-devices test prints `NIXL_ALL_DEVICES_UCX_BACKEND_OK` and
selects `rc_mlx5` should the production prefill and decode environments be
changed to:

```yaml
- name: UCX_NET_DEVICES
  value: "all"
```

In that case, remove `UCX_IB_ADDR_TYPE` and `UCX_IB_GID_INDEX` from both roles
rather than combining auto device selection with the currently invalid pinned
GID. Re-run this preflight on both GPU nodes before applying the DGD.

Do not deploy if either output contains `NIXL_ERR_BACKEND`, `Address not
valid`, `No such device`, or an unexpected TCP transport. The Pod remains
running after either test, so it can be inspected further with:

```bash
kubectl describe pod qwen3-32b-prefill-nixl-preflight -n "$NAMESPACE"
kubectl exec -it -n "$NAMESPACE" \
  pod/qwen3-32b-prefill-nixl-preflight -- bash
```

After a successful check, release its GPU and RDMA allocations before the
main deployment:

```bash
kubectl delete pod qwen3-32b-prefill-nixl-preflight \
  -n "$NAMESPACE" --wait=true
```

This comparison tests the suggested hypothesis without enabling host
networking. It verifies device visibility and UCX backend creation, but a real
prefill-to-decode request after deployment is still required to prove an
end-to-end RDMA KV transfer.

## Deploy

```bash
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"
kubectl get pods -n "$NAMESPACE" \
  -l "$GRAPH_LABEL" \
  -o wide -w
```

## Check logs

Tail or stream logs for all containers in the deployment:

```bash
# Tail recent 500 lines across all containers
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" --all-containers --prefix --tail=500

# Stream live logs in real time
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" --all-containers --prefix -f

# Filter for NIXL, UCX, KV events, and errors
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" --all-containers --prefix --tail=500 | grep -Ei 'nixl|ucx|kv.event|error|traceback'

# Check prefill worker logs specifically
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL,nvidia.com/dynamo-sub-component-type=prefill" --all-containers --prefix --tail=500

# Check decode worker logs specifically
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL,nvidia.com/dynamo-sub-component-type=decode" --all-containers --prefix --tail=500

# Inspect logs from a previous crashed container instance
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" --all-containers --prefix --previous --tail=500
```

## Acceptance checks

```bash
kubectl port-forward -n "$NAMESPACE" \
  service/"$FRONTEND_SERVICE" "$LOCAL_PORT":8000
```

Send the same long-prefix request twice. Accept the run only if requests
succeed, NIXL initializes, and KV-event counters increase without worker
restarts.

## Clean up

```bash
kubectl delete dynamographdeployment.nvidia.com "$DEPLOYMENT" \
  -n "$NAMESPACE" --wait=false --ignore-not-found
kubectl delete pods -l "$GRAPH_LABEL" -n "$NAMESPACE" \
  --force --grace-period=0 --ignore-not-found
```

The first command prevents the controller from keeping the experiment alive;
the second immediately force-deletes its pods. The local `deploy.yaml` remains
unchanged.
