# Pod-native RoCE for the Qwen3-32B vLLM deployment

This runbook gives the Qwen prefill and decode Pods a dedicated RoCE interface
without using `hostNetwork: true`.

The resulting network layout is:

```text
eth0  Calico network  Kubernetes Services and the vLLM NIXL handshake
net1  MacVLAN/rdma7   UCX, RoCE, and NIXL KV transfers
```

The existing RDMA shared-device plugin exposes the verbs devices. Multus and
MacVLAN add the missing network interface, IP address, route, and RoCE GID to
each Pod network namespace.

Do not deploy the model until every `PASS` gate below succeeds.

## 1. Set the fixed cluster values

Run on the Kubernetes control-plane host:

```bash
export NAMESPACE=qwen32-bench
export NETOP_NAMESPACE=nvidia-network-operator
export EXP_DIR=/ephemeral/shared/qwen3-32b/vllm/disagg-routing-kv-aware
export DEPLOYMENT=qwen3-32b-fp8-vllm-disagg-kv-aware

export PREFILL_NODE=inst-1onle-devrel-rdma-pool
export DECODE_NODE=inst-g9dwj-devrel-rdma-pool

export ROCE_MASTER=rdma7
export ROCE_HCA=mlx5_8
export ROCE_NETWORK=qwen-roce
export ROCE_POOL=qwen-roce-pool

mkdir -p "$EXP_DIR"
```

Check that the namespace, nodes, PVC, RDMA resource, and Network Operator
exist:

```bash
kubectl get namespace "$NAMESPACE"
kubectl get node "$PREFILL_NODE" "$DECODE_NODE" -o wide
kubectl get pvc model-cache -n "$NAMESPACE"
kubectl get nicclusterpolicy nic-cluster-policy

kubectl get nodes \
  -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,RDMA:.status.allocatable.rdma/ib'
```

PASS only if:

- both nodes are `Ready`;
- `model-cache` is `Bound`;
- both nodes advertise GPUs and a nonzero `rdma/ib` value;
- no existing model workload is consuming the GPUs needed by the tests.

List existing GPU workloads before continuing:

```bash
kubectl get pods -A \
  -o custom-columns='NAMESPACE:.metadata.namespace,POD:.metadata.name,NODE:.spec.nodeName,PHASE:.status.phase,GPU:.spec.containers[*].resources.limits.nvidia\.com/gpu'
```

## 2. Check the physical RoCE interface on both nodes

Run the following block directly on both GPU nodes:

```bash
hostname
ip -o -4 address show dev rdma7
ip link show dev rdma7
ip route show dev rdma7

cat /sys/class/infiniband/mlx5_8/ports/1/state
cat /sys/class/infiniband/mlx5_8/ports/1/link_layer
cat /sys/class/infiniband/mlx5_8/ports/1/gids/3

rdma system show
```

PASS only if both nodes show:

- interface `rdma7`;
- HCA `mlx5_8`, port `1`;
- state `4: ACTIVE`;
- link layer `Ethernet`;
- shared RDMA namespace mode;
- working connectivity on the same physical RoCE fabric.

Stop if the interface or HCA names differ between nodes. Substitute the actual
common interface/HCA in every later section rather than guessing.

## 3. Reserve a Pod IP subnet

Obtain a dedicated, unused IP subnet on the same L2 RoCE fabric as `rdma7`.
The allocation must be approved by whoever manages the fabric.

Do not give NV-IPAM the existing `10.224.7.0/24` subnet unless that entire
subnet has been delegated to NV-IPAM. The hosts already use addresses from
that subnet, and NV-IPAM must not allocate a duplicate address.

Set the reserved values:

```bash
export ROCE_SUBNET='REPLACE_WITH_RESERVED_SUBNET/CIDR'
export ROCE_GATEWAY='REPLACE_WITH_GATEWAY'
```

Protect against accidentally applying the placeholders:

```bash
case "$ROCE_SUBNET $ROCE_GATEWAY" in
  *REPLACE*)
    echo "STOP: set the reserved ROCE_SUBNET and ROCE_GATEWAY first" >&2
    false
    ;;
esac
```

PASS only if:

- the subnet is unused by hosts, Pods, DHCP, and other IPAM systems;
- it is present on the same L2 fabric on both nodes;
- it contains enough addresses for eight workers plus future test Pods;
- the switch permits MacVLAN child MAC addresses on the server-facing ports.

If the switch or provider blocks multiple MAC addresses, stop here. Use
SR-IOV VFs with Multus instead of MacVLAN; UCX environment changes cannot
work around a fabric anti-spoofing policy.

## 4. Enable Multus, CNI plugins, and NV-IPAM

This merge patch preserves the existing RDMA shared-device plugin section of
the `NicClusterPolicy` and adds only the secondary-network components.

```bash
tee "$EXP_DIR/nicclusterpolicy-secondary-network-patch.yaml" >/dev/null <<'EOF'
spec:
  nvIpam:
    image: nvidia-k8s-ipam
    repository: nvcr.io/nvidia/mellanox
    version: network-operator-v26.4.0
    imagePullSecrets: []
    enableWebhook: false
  secondaryNetwork:
    cniPlugins:
      image: plugins
      repository: nvcr.io/nvidia/mellanox
      version: network-operator-v26.4.0
      imagePullSecrets: []
    multus:
      image: multus-cni
      repository: nvcr.io/nvidia/mellanox
      version: network-operator-v26.4.0
      imagePullSecrets: []
EOF

kubectl patch nicclusterpolicy nic-cluster-policy \
  --type=merge \
  --patch-file="$EXP_DIR/nicclusterpolicy-secondary-network-patch.yaml"
```

Watch reconciliation:

```bash
kubectl get nicclusterpolicy nic-cluster-policy -w
```

In another terminal, inspect the components:

```bash
kubectl get pods -n "$NETOP_NAMESPACE" -o wide
kubectl get daemonsets -n "$NETOP_NAMESPACE"
kubectl get deployments -n "$NETOP_NAMESPACE"

kubectl get crd | grep -E \
  'network-attachment-definitions|macvlannetworks|ippools'
```

PASS only if:

- `nic-cluster-policy` returns to `ready`;
- Multus is ready on both nodes;
- the CNI plugin DaemonSet is ready on both nodes;
- the NV-IPAM controller and node components are ready;
- the `NetworkAttachmentDefinition`, `MacvlanNetwork`, and `IPPool` CRDs
  exist;
- the existing `rdma-shared-dp-ds` remains ready on both nodes.

## 5. Create the NV-IPAM pool

This EOF expands only the two reserved network variables set in section 3.

```bash
tee "$EXP_DIR/qwen-roce-pool.yaml" >/dev/null <<EOF
apiVersion: nv-ipam.nvidia.com/v1alpha1
kind: IPPool
metadata:
  name: ${ROCE_POOL}
  namespace: ${NETOP_NAMESPACE}
spec:
  subnet: "${ROCE_SUBNET}"
  gateway: "${ROCE_GATEWAY}"
  perNodeBlockSize: 16
EOF

kubectl apply -f "$EXP_DIR/qwen-roce-pool.yaml"
kubectl get ippool "$ROCE_POOL" \
  -n "$NETOP_NAMESPACE" -o yaml
```

PASS only if the stored `subnet` and `gateway` exactly match the reserved
allocation. Stop if the IPPool reports an error or cannot allocate a unique
block to both nodes.

## 6. Create the MacVLAN secondary network

`mtu: 0` inherits the MTU of `rdma7`.

```bash
tee "$EXP_DIR/qwen-roce-network.yaml" >/dev/null <<EOF
apiVersion: mellanox.com/v1alpha1
kind: MacvlanNetwork
metadata:
  name: ${ROCE_NETWORK}
spec:
  networkNamespace: "${NAMESPACE}"
  master: "${ROCE_MASTER}"
  mode: bridge
  mtu: 0
  ipam: |
    {
      "type": "nv-ipam",
      "poolName": "${ROCE_POOL}"
    }
EOF

kubectl apply -f "$EXP_DIR/qwen-roce-network.yaml"

kubectl get macvlannetwork "$ROCE_NETWORK" -o yaml
kubectl get network-attachment-definition "$ROCE_NETWORK" \
  -n "$NAMESPACE" -o yaml
```

PASS only if:

- the `MacvlanNetwork` state is ready;
- the generated `NetworkAttachmentDefinition` exists in `qwen32-bench`;
- its master is `rdma7` and its IPAM pool is `qwen-roce-pool`.

## 7. Validate cross-node RDMA with `rping`

Create one small RDMA Pod on each node. These Pods do not request GPUs.

```bash
tee "$EXP_DIR/roce-rping-test.yaml" >/dev/null <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: qwen-roce-rping-server
  namespace: ${NAMESPACE}
  annotations:
    k8s.v1.cni.cncf.io/networks: ${NAMESPACE}/${ROCE_NETWORK}
spec:
  restartPolicy: Never
  nodeName: ${PREFILL_NODE}
  tolerations:
    - key: node-role.kubernetes.io/control-plane
      operator: Exists
      effect: NoSchedule
    - key: nvidia.com/gpu
      operator: Equal
      value: "true"
      effect: NoSchedule
  containers:
    - name: rping
      image: mellanox/rping-test
      command: ["/bin/sh", "-c"]
      args: ["exec sleep 3600"]
      securityContext:
        capabilities:
          add: ["IPC_LOCK"]
      resources:
        requests:
          rdma/ib: "1"
        limits:
          rdma/ib: "1"
---
apiVersion: v1
kind: Pod
metadata:
  name: qwen-roce-rping-client
  namespace: ${NAMESPACE}
  annotations:
    k8s.v1.cni.cncf.io/networks: ${NAMESPACE}/${ROCE_NETWORK}
spec:
  restartPolicy: Never
  nodeName: ${DECODE_NODE}
  tolerations:
    - key: node-role.kubernetes.io/control-plane
      operator: Exists
      effect: NoSchedule
    - key: nvidia.com/gpu
      operator: Equal
      value: "true"
      effect: NoSchedule
  containers:
    - name: rping
      image: mellanox/rping-test
      command: ["/bin/sh", "-c"]
      args: ["exec sleep 3600"]
      securityContext:
        capabilities:
          add: ["IPC_LOCK"]
      resources:
        requests:
          rdma/ib: "1"
        limits:
          rdma/ib: "1"
EOF

kubectl delete -f "$EXP_DIR/roce-rping-test.yaml" \
  --ignore-not-found --wait=true
kubectl apply -f "$EXP_DIR/roce-rping-test.yaml"

kubectl wait --for=condition=Ready \
  pod/qwen-roce-rping-server pod/qwen-roce-rping-client \
  -n "$NAMESPACE" --timeout=5m
```

Inspect Multus status and capture the secondary addresses:

```bash
for pod in qwen-roce-rping-server qwen-roce-rping-client; do
  echo "===== $pod ====="
  kubectl get pod "$pod" -n "$NAMESPACE" -o json |
    jq -r '.metadata.annotations["k8s.v1.cni.cncf.io/network-status"] | fromjson'
done

export RPING_SERVER_IP="$({
  kubectl get pod qwen-roce-rping-server -n "$NAMESPACE" -o json |
    jq -r '.metadata.annotations["k8s.v1.cni.cncf.io/network-status"]
      | fromjson
      | map(select(.interface == "net1"))[0].ips[0]'
} | cut -d/ -f1)"

export RPING_CLIENT_IP="$({
  kubectl get pod qwen-roce-rping-client -n "$NAMESPACE" -o json |
    jq -r '.metadata.annotations["k8s.v1.cni.cncf.io/network-status"]
      | fromjson
      | map(select(.interface == "net1"))[0].ips[0]'
} | cut -d/ -f1)"

test -n "$RPING_SERVER_IP"
test "$RPING_SERVER_IP" != null
test -n "$RPING_CLIENT_IP"
test "$RPING_CLIENT_IP" != null
echo "RPING_SERVER_IP=$RPING_SERVER_IP"
echo "RPING_CLIENT_IP=$RPING_CLIENT_IP"
```

Run the RDMA connection test:

```bash
kubectl exec -n "$NAMESPACE" qwen-roce-rping-server -- \
  rping -s -a "$RPING_SERVER_IP" -v &
export RPING_SERVER_PROCESS=$!

sleep 3

kubectl exec -n "$NAMESPACE" qwen-roce-rping-client -- \
  rping -c -I "$RPING_CLIENT_IP" -a "$RPING_SERVER_IP" -v -C 10

wait "$RPING_SERVER_PROCESS"
```

PASS only if:

- both Pods have a distinct `net1` IP from the reserved subnet;
- neither Pod reports a Multus or IPAM error;
- the client completes all RDMA iterations;
- the server exits cleanly after the client finishes.

If `rping` fails, do not continue to NIXL. Check switch MAC policy, VLAN/L2
reachability, MTU consistency, PFC/ECN configuration, and the IP pool.

Clean up the `rping` Pods after a pass:

```bash
kubectl delete -f "$EXP_DIR/roce-rping-test.yaml" --wait=true
```

## 8. Run the NIXL preflight on both nodes

The following function writes one Pod YAML per node. Each Pod uses the same
runtime, network attachment, UCX transport selection, security capabilities,
shared memory, PVC, and `2 GPU + 2 RDMA` allocation as a TP=2 worker.

GID index is intentionally not pinned. UCX must select the GID created for the
Pod's `net1` address.

```bash
write_nixl_preflight() {
  local pod_name="$1"
  local node_name="$2"

  tee "$EXP_DIR/${pod_name}.yaml" >/dev/null <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${pod_name}
  namespace: ${NAMESPACE}
  annotations:
    k8s.v1.cni.cncf.io/networks: ${NAMESPACE}/${ROCE_NETWORK}
spec:
  restartPolicy: Never
  activeDeadlineSeconds: 900
  nodeName: ${node_name}
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
    - name: nixl-check
      image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
      imagePullPolicy: IfNotPresent
      workingDir: /workspace/examples/backends/vllm
      command: ["/bin/bash", "-lc"]
      args:
        - |
          set -euo pipefail
          echo "POD=\$POD_NAME NODE=\$NODE_NAME POD_IP=\$POD_IP"
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
      securityContext:
        runAsUser: 0
        capabilities:
          add: ["IPC_LOCK", "SYS_RESOURCE"]
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
}

write_nixl_preflight qwen-nixl-prefill-check "$PREFILL_NODE"
write_nixl_preflight qwen-nixl-decode-check "$DECODE_NODE"

kubectl delete pod qwen-nixl-prefill-check qwen-nixl-decode-check \
  -n "$NAMESPACE" --ignore-not-found --wait=true

kubectl apply -f "$EXP_DIR/qwen-nixl-prefill-check.yaml"
kubectl apply -f "$EXP_DIR/qwen-nixl-decode-check.yaml"

kubectl wait --for=condition=Ready \
  pod/qwen-nixl-prefill-check pod/qwen-nixl-decode-check \
  -n "$NAMESPACE" --timeout=10m
```

Verify that both Pods have `net1` addresses:

```bash
for pod in qwen-nixl-prefill-check qwen-nixl-decode-check; do
  echo "===== $pod ====="
  kubectl get pod "$pod" -n "$NAMESPACE" -o json |
    jq -r '.metadata.annotations["k8s.v1.cni.cncf.io/network-status"] | fromjson'
done
```

Verify nonzero GIDs and run NIXL initialization in both Pods:

```bash
for pod in qwen-nixl-prefill-check qwen-nixl-decode-check; do
  echo "===== $pod: device and GID check ====="

  kubectl exec -n "$NAMESPACE" "$pod" -- bash -lc '
    set -euo pipefail
    test -d /sys/class/infiniband/mlx5_8/ports/1
    cat /sys/class/infiniband/mlx5_8/ports/1/state
    cat /sys/class/infiniband/mlx5_8/ports/1/link_layer

    found=0
    for gid_file in /sys/class/infiniband/mlx5_8/ports/1/gids/*; do
      gid=$(cat "$gid_file")
      case "$gid" in
        0000:0000:0000:0000:0000:0000:0000:0000) continue ;;
      esac
      echo "$gid_file=$gid"
      found=1
    done
    test "$found" -eq 1
  '

  echo "===== $pod: NIXL check ====="

  kubectl exec -n "$NAMESPACE" "$pod" -- \
    python3 -c "import os,nixl; a=nixl.nixl_agent(os.environ['POD_NAME'] + '-macvlan'); print('NIXL_MACVLAN_RDMA_OK')"
done
```

PASS only if both Pods show:

```text
4: ACTIVE
Ethernet
NIXL_MACVLAN_RDMA_OK
```

The UCX output must select `rc_mlx5/mlx5_8:1`. Stop if it shows
`NIXL_ERR_BACKEND`, `Address not valid`, `No such device`, or TCP.

Delete both preflight Pods after they pass so the model can use their GPUs:

```bash
kubectl delete pod qwen-nixl-prefill-check qwen-nixl-decode-check \
  -n "$NAMESPACE" --wait=true
```

## 9. Create the pod-native DGD

This manifest keeps `hostNetwork` disabled, attaches the secondary network to
both worker roles, pins the known HCA, and lets UCX select the Pod-specific
RoCE GID. The Calico Pod IP remains the vLLM NIXL handshake address.

```bash
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
EOF
```

Validate the manifest before deploying:

```bash
grep -nE \
  'hostNetwork|k8s.v1.cni.cncf.io/networks|UCX_NET_DEVICES|UCX_IB_ADDR_TYPE|UCX_IB_GID_INDEX|rdma/ib' \
  "$EXP_DIR/deploy-pod-roce.yaml"

kubectl apply --dry-run=server \
  -n "$NAMESPACE" \
  -f "$EXP_DIR/deploy-pod-roce.yaml"
```

PASS only if:

- both worker roles have the `qwen-roce` Multus annotation;
- both explicitly use `hostNetwork: false`;
- both use `UCX_NET_DEVICES=mlx5_8:1`;
- neither contains `UCX_IB_GID_INDEX`;
- the server-side dry run succeeds.

## 10. Deploy and verify every generated worker Pod

Apply only after sections 1 through 9 pass:

```bash
kubectl apply -n "$NAMESPACE" \
  -f "$EXP_DIR/deploy-pod-roce.yaml"

kubectl get pods -n "$NAMESPACE" -o wide -w
```

Wait for one frontend, six prefill workers, and two decode workers. Every
container must become ready with zero restarts.

Check that every worker received the secondary network:

```bash
kubectl get pods -n "$NAMESPACE" -o name |
grep -E 'vllm(prefill|decode)worker' |
while read -r pod; do
  echo "===== $pod ====="
  kubectl get -n "$NAMESPACE" "$pod" -o json |
    jq -r '.metadata.annotations["k8s.v1.cni.cncf.io/network-status"] | fromjson'
done
```

PASS only if all eight workers have unique `net1` addresses from the reserved
RoCE subnet.

Check the actual worker environments:

```bash
kubectl get pods -n "$NAMESPACE" -o name |
grep -E 'vllm(prefill|decode)worker' |
while read -r pod; do
  echo "===== $pod ====="
  kubectl exec -n "$NAMESPACE" "$pod" -- env |
    grep -E '^UCX_(TLS|NET_DEVICES|IB_ADDR_TYPE|IB_GID_INDEX)='
done
```

PASS only if `UCX_IB_GID_INDEX` is absent and the other values match the
manifest.

Check NIXL and UCX logs without relying on a DGD label selector:

```bash
kubectl get pods -n "$NAMESPACE" -o name |
grep -E 'vllm(prefill|decode)worker' |
while read -r pod; do
  kubectl logs -n "$NAMESPACE" "$pod" \
    --all-containers --prefix --tail=500
done |
grep -Ei \
  'Created backend|Backend UCX|Initialized NIXL|rc_mlx5|KV Transfer metrics|NIXL_ERR|Address not valid|No such device|fallback|traceback|error' |
tail -n 500
```

PASS only if:

- every worker initializes NIXL and UCX;
- UCX selects `rc_mlx5/mlx5_8:1`;
- there is no `NIXL_ERR_BACKEND`;
- there is no `Address not valid` or `No such device`;
- there is no TCP fallback;
- no worker restarts.

## 11. Prove an end-to-end NIXL KV transfer

Find the frontend Service:

```bash
kubectl get service -n "$NAMESPACE" |
grep 'qwen3-32b-fp8-vllm-disagg-kv-aware.*frontend'
```

Open a local tunnel and keep it running:

```bash
kubectl port-forward -n "$NAMESPACE" \
  service/qwen3-32b-fp8-vllm-disagg-kv-aware-frontend \
  8000:8000
```

In another terminal, send the same prefix twice:

```bash
export MODEL='Qwen/Qwen3-32B-FP8'

for attempt in 1 2; do
  curl -fsS http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    --data-binary @- <<EOF
{
  "model": "${MODEL}",
  "messages": [
    {
      "role": "user",
      "content": "This is a repeated-prefix NIXL validation request. Summarize why a dedicated secondary RoCE interface is required for an isolated Kubernetes Pod. Attempt ${attempt}."
    }
  ],
  "temperature": 0,
  "max_tokens": 64
}
EOF
done
```

Immediately inspect transfer evidence:

```bash
kubectl get pods -n "$NAMESPACE" -o name |
grep -E 'vllm(prefill|decode)worker' |
while read -r pod; do
  kubectl logs -n "$NAMESPACE" "$pod" \
    --all-containers --prefix --since=10m
done |
grep -Ei \
  'KV Transfer metrics|successful transfers|nixl|rc_mlx5|failed transfers|NIXL_ERR|Address not valid|No such device' |
tail -n 500
```

Final acceptance requires all of the following:

- both HTTP requests succeed;
- logs report successful NIXL KV transfers;
- UCX uses `rc_mlx5/mlx5_8:1`;
- failed-transfer counters do not increase;
- no TCP fallback appears;
- every worker remains ready with zero restarts.

Pod readiness and successful NIXL initialization alone are not proof of a KV
transfer. The repeated request and transfer evidence are mandatory.

## 12. Test-resource cleanup

The `rping` and NIXL preflight Pods should already be deleted. Confirm:

```bash
kubectl get pod -n "$NAMESPACE" |
grep -E 'qwen-roce-rping|qwen-nixl-(prefill|decode)-check' || true
```

Do not delete the `MacvlanNetwork`, `IPPool`, Multus, NV-IPAM, or CNI plugins
while the DGD is using them.

## References

- [NVIDIA: Deploy MacVLAN Network with RDMA Shared Device](https://docs.nvidia.com/networking/display/kubernetes25100/quick-start/macvlan-rdma-shared.html)
- [NVIDIA Network Operator deployment guide](https://docs.nvidia.com/networking/display/kubernetes2640/deployment-guide-kubernetes.html)
- [NVIDIA Dynamo Kubernetes API reference](https://docs.nvidia.com/dynamo/dev/reference/api/kubernetes/full-api-reference)
- [vLLM NixlConnector usage guide](https://docs.vllm.ai/en/stable/features/nixl_connector_usage/)
- [RDMA Core `rping` manual](https://man7.org/linux/man-pages/man1/rping.1.html)
