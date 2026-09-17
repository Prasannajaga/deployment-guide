# Pod-native RoCE for Dynamo workers

Use this after the base setup in [`cluster.md`](cluster.md). It resolves the networking problem that blocked replicated disaggregated workers.

## 1. Problem and solution

We encountered two mutually exclusive failure modes when scaling disaggregated workers across nodes:

| Worker networking | Result |
|---|---|
| `hostNetwork: true` | UCX could use the host RoCE interface, but workers shared the node's TCP-port namespace. After the first workers reserved the generated ports, KAI rejected the remaining replicas with `didn't have free ports for the requested pod ports`. |
| `hostNetwork: false` with only Calico | Workers had isolated ports and could be scheduled, but their network namespaces had no usable RoCE interface and Pod-specific GID. UCX failed with `Address not valid`, followed by `NIXL_ERR_BACKEND`. |

The fix keeps `hostNetwork: false` and gives each worker two separate interfaces:

```text
eth0  Calico             Kubernetes Services and the NIXL handshake
net1  Multus/MacVLAN     UCX/RoCE and NIXL KV-cache transfers
```

Setting `hostNetwork: false` with Calico provides every Pod an isolated IP and TCP port namespace, eliminating host port collisions during scaling. Multus attaches `net1` using MacVLAN to connect directly to the physical `rdma7` HCA, while NV-IPAM assigns a unique RoCE IP address that establishes a Pod-specific RoCE GID. In parallel, the RDMA Shared Device Plugin exposes `/dev/infiniband` verbs devices and advertises schedulable `rdma/ib` resources to Kubernetes. Finally, UCX is pinned to `mlx5_8:1` and uses the Pod's native `net1` GID for high-speed data transfers.

MacVLAN restores the actual RoCE GID and network path inside the container namespace, while the device plugin exposes the verbs hardware devices. Both components are required on this cluster to achieve Pod-native RDMA serving.

## 2. Set the cluster values

Run all `kubectl` commands from the Kubernetes administrator terminal on
`gpu05`. Blocks marked **both nodes** must be run directly on each host.

```bash
export NAMESPACE=qwen32-bench
export NETOP_NAMESPACE=nvidia-network-operator
export EXP_DIR=/ephemeral/shared/qwen3-32b/pod-native-roce
export PREFILL_NODE=inst-1onle-devrel-rdma-pool
export DECODE_NODE=inst-g9dwj-devrel-rdma-pool
export ROCE_MASTER=rdma7
export ROCE_HCA=mlx5_8
export ROCE_NETWORK=qwen-roce
export ROCE_POOL=qwen-roce-pool

mkdir -p "$EXP_DIR"
```

## 3. Validate the physical RoCE path

**Both nodes:**

```bash
hostname
ip -o -4 address show dev rdma7
ip link show dev rdma7
ibdev2netdev

cat /sys/class/infiniband/mlx5_8/ports/1/state
cat /sys/class/infiniband/mlx5_8/ports/1/link_layer
rdma system show
lsmod | grep -E 'nvidia_peermem|nv_peer_mem'
```

Require `rdma7`, HCA `mlx5_8` port 1, state `4: ACTIVE`, link layer
`Ethernet`, and `nvidia_peermem`. Stop if the interface or HCA differs between
nodes; substitute the actual common device everywhere below.

### Enable shared RDMA namespace mode when needed

If `rdma system show` already reports `netns shared`, skip this subsection.
Otherwise configure one node at a time, rebooting and validating `gpu06`
before changing `gpu05`.

**Both nodes, one at a time:**

```bash
sudo tee /etc/modprobe.d/99-rdma-shared-netns.conf >/dev/null <<'EOF'
options ib_core netns_mode=1
EOF

sudo tee /etc/default/grub.d/99-rdma-shared-netns.cfg >/dev/null <<'EOF'
GRUB_CMDLINE_LINUX_DEFAULT="${GRUB_CMDLINE_LINUX_DEFAULT} ib_core.netns_mode=1"
EOF

sudo update-initramfs -u
sudo update-grub
sudo reboot
```

After each reboot:

```bash
cat /sys/module/ib_core/parameters/netns_mode
rdma system show
```

Require `Y` and `netns shared`. Changing this live can fail with `Device or
resource busy`; the boot configuration is the persistent fix.

## 4. Install the secondary-network components

This complete policy creates the Kubernetes resource `rdma/ib` and installs
Multus, CNI plugins, and NV-IPAM. It does not replace the host OFED driver.

```bash
tee "$EXP_DIR/nic-cluster-policy.yaml" >/dev/null <<'EOF'
apiVersion: mellanox.com/v1alpha1
kind: NicClusterPolicy
metadata:
  name: nic-cluster-policy
spec:
  rdmaSharedDevicePlugin:
    image: k8s-rdma-shared-dev-plugin
    repository: nvcr.io/nvidia/mellanox
    version: network-operator-v26.4.0
    config: |
      {
        "configList": [
          {
            "resourcePrefix": "rdma",
            "resourceName": "ib",
            "rdmaHcaMax": 64,
            "selectors": {"vendors": ["15b3"]}
          }
        ]
      }
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

kubectl apply -f "$EXP_DIR/nic-cluster-policy.yaml"

kubectl wait nicclusterpolicy/nic-cluster-policy \
  --for=jsonpath='{.status.state}'=ready --timeout=15m

kubectl get pods -n "$NETOP_NAMESPACE" -o wide
kubectl get nodes \
  -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,RDMA:.status.allocatable.rdma/ib'
```

Continue only when Network Operator components are healthy and both nodes
advertise a positive `rdma/ib` value.

## 5. Create the Pod RoCE network

Obtain a dedicated unused subnet on the same L2 fabric as `rdma7`. It must not
overlap host addresses, DHCP, or another IPAM system. The switch must permit
multiple MacVLAN child MAC addresses on each server port.

Do not use the hosts' existing `10.224.7.0/24` subnet unless the network
administrator explicitly delegates a non-overlapping range to NV-IPAM.

```bash
export ROCE_SUBNET='REPLACE_WITH_RESERVED_SUBNET/CIDR'
export ROCE_GATEWAY='REPLACE_WITH_GATEWAY'

case "$ROCE_SUBNET $ROCE_GATEWAY" in
  *REPLACE*)
    echo 'STOP: set the approved RoCE subnet and gateway' >&2
    false
    ;;
esac
```

Create the pool and network:

```bash
tee "$EXP_DIR/qwen-roce.yaml" >/dev/null <<EOF
apiVersion: nv-ipam.nvidia.com/v1alpha1
kind: IPPool
metadata:
  name: ${ROCE_POOL}
  namespace: ${NETOP_NAMESPACE}
spec:
  subnet: "${ROCE_SUBNET}"
  gateway: "${ROCE_GATEWAY}"
  perNodeBlockSize: 16
---
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

kubectl apply -f "$EXP_DIR/qwen-roce.yaml"

kubectl get ippool "$ROCE_POOL" -n "$NETOP_NAMESPACE"
kubectl get macvlannetwork "$ROCE_NETWORK"
kubectl get network-attachment-definition "$ROCE_NETWORK" \
  -n "$NAMESPACE"
```

If the fabric blocks child MAC addresses, stop and use SR-IOV VFs. UCX
environment variables cannot bypass an L2 anti-spoofing policy.

## 6. Validate cross-node RDMA

Create one lightweight `rping` Pod on each node:

```bash
tee "$EXP_DIR/rping.yaml" >/dev/null <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: roce-rping-server
  namespace: ${NAMESPACE}
  annotations:
    k8s.v1.cni.cncf.io/networks: ${NAMESPACE}/${ROCE_NETWORK}
spec:
  restartPolicy: Never
  nodeName: ${PREFILL_NODE}
  tolerations:
    - {key: node-role.kubernetes.io/control-plane, operator: Exists, effect: NoSchedule}
    - {key: nvidia.com/gpu, operator: Equal, value: "true", effect: NoSchedule}
  containers:
    - name: rping
      image: mellanox/rping-test
      command: ["/bin/sh", "-c"]
      args: ["exec sleep 3600"]
      resources:
        requests: {rdma/ib: "1"}
        limits: {rdma/ib: "1"}
---
apiVersion: v1
kind: Pod
metadata:
  name: roce-rping-client
  namespace: ${NAMESPACE}
  annotations:
    k8s.v1.cni.cncf.io/networks: ${NAMESPACE}/${ROCE_NETWORK}
spec:
  restartPolicy: Never
  nodeName: ${DECODE_NODE}
  tolerations:
    - {key: node-role.kubernetes.io/control-plane, operator: Exists, effect: NoSchedule}
    - {key: nvidia.com/gpu, operator: Equal, value: "true", effect: NoSchedule}
  containers:
    - name: rping
      image: mellanox/rping-test
      command: ["/bin/sh", "-c"]
      args: ["exec sleep 3600"]
      resources:
        requests: {rdma/ib: "1"}
        limits: {rdma/ib: "1"}
EOF

kubectl delete -f "$EXP_DIR/rping.yaml" --ignore-not-found --wait=true
kubectl apply -f "$EXP_DIR/rping.yaml"

kubectl wait --for=condition=Ready \
  pod/roce-rping-server pod/roce-rping-client \
  -n "$NAMESPACE" --timeout=5m
```

Capture the two `net1` addresses:

```bash
net1_ip() {
  kubectl get pod "$1" -n "$NAMESPACE" -o json |
    jq -r '.metadata.annotations["k8s.v1.cni.cncf.io/network-status"]
      | fromjson | map(select(.interface == "net1"))[0].ips[0]' |
    cut -d/ -f1
}

export RPING_SERVER_IP="$(net1_ip roce-rping-server)"
export RPING_CLIENT_IP="$(net1_ip roce-rping-client)"

test -n "$RPING_SERVER_IP" && test "$RPING_SERVER_IP" != null
test -n "$RPING_CLIENT_IP" && test "$RPING_CLIENT_IP" != null
printf 'server=%s client=%s\n' "$RPING_SERVER_IP" "$RPING_CLIENT_IP"
```

Run the transfer test:

```bash
kubectl exec -n "$NAMESPACE" roce-rping-server -- \
  rping -s -a "$RPING_SERVER_IP" -v &
export RPING_SERVER_PROCESS=$!

sleep 3

kubectl exec -n "$NAMESPACE" roce-rping-client -- \
  rping -c -I "$RPING_CLIENT_IP" \
    -a "$RPING_SERVER_IP" -v -C 10

wait "$RPING_SERVER_PROCESS"
```

Pass only when all ten RDMA iterations complete. Then release the resources:

```bash
kubectl delete -f "$EXP_DIR/rping.yaml" --wait=true
```

## 7. Apply the worker settings

Add this pattern to every prefill and decode worker. Keep the full model DGD
in its experiment guide rather than copying it here.

```yaml
extraPodMetadata:
  annotations:
    k8s.v1.cni.cncf.io/networks: qwen32-bench/qwen-roce
extraPodSpec:
  hostNetwork: false
  dnsPolicy: ClusterFirst
  mainContainer:
    ports: []
    env:
      - name: VLLM_NIXL_SIDE_CHANNEL_HOST
        valueFrom:
          fieldRef:
            fieldPath: status.podIP
      - {name: UCX_TLS, value: "rc_x,rc,cuda_copy,cuda_ipc"}
      - {name: UCX_NET_DEVICES, value: "mlx5_8:1"}
      - {name: UCX_IB_ADDR_TYPE, value: "eth"}
      - {name: UCX_RNDV_SCHEME, value: "get_zcopy"}
      - {name: UCX_RNDV_THRESH, value: "0"}
      - {name: UCX_RCACHE_MAX_UNRELEASED, value: "1024"}
    securityContext:
      runAsUser: 0
      capabilities:
        add: [IPC_LOCK, SYS_RESOURCE]
resources:
  requests:
    gpu: "2"
    custom:
      rdma/ib: "2"
  limits:
    gpu: "2"
    custom:
      rdma/ib: "2"
```

Important rules:

- `status.podIP` is the Calico `eth0` address used for the NIXL handshake.
- UCX uses `mlx5_8:1` and the Pod-specific GID created for `net1`.
- Do not set `UCX_IB_GID_INDEX=3`; index 3 belongs to the host address.
- Do not use `hostNetwork: true` or fixed `hostPort` values on replicas.
- Request one `rdma/ib` allocation per GPU; TP=2 requests two of each.
- For SGLang, omit `VLLM_NIXL_SIDE_CHANNEL_HOST`; keep the network and UCX
  settings.

After model deployment, verify every worker:

```bash
kubectl get pods -n "$NAMESPACE" -o wide

for pod in $(kubectl get pods -n "$NAMESPACE" \
  -l nvidia.com/dynamo-component-type \
  -o name); do
  echo "===== $pod ====="
  kubectl get -n "$NAMESPACE" "$pod" -o json |
    jq -r '.metadata.annotations["k8s.v1.cni.cncf.io/network-status"] // "NO NETWORK STATUS"'
done
```

Worker logs must show UCX/NIXL initialization without TCP fallback:

```bash
kubectl logs -n "$NAMESPACE" \
  -l nvidia.com/dynamo-component-type \
  --all-containers=true --prefix --timestamps \
  --tail=500 --max-log-requests=20 |
grep -Ei 'NIXL|UCX|mlx5|Address not valid|No such device|TCP'
```

The fix is complete only when every replica is scheduled, every worker has a
unique `net1` address, NIXL instantiates UCX without TCP fallback, and a real
prefill-to-decode inference request succeeds.

If a Pod still fails:

```bash
kubectl describe pod POD_NAME -n "$NAMESPACE" | sed -n '/Events:/,$p'
kubectl logs POD_NAME -n "$NAMESPACE" --previous --timestamps --tail=300
kubectl get pods -n "$NETOP_NAMESPACE" -o wide
```

## References

- [Dynamo RDMA setup](https://docs.nvidia.com/dynamo/dev/kubernetes/installation/rdma-setup/overview)
- [Dynamo disaggregated communication](https://docs.nvidia.com/dynamo/dev/knowledge-base/kubernetes/kubernetes-operator/disagg-communication)
- [Network Operator MacVLAN with RDMA Shared Device](https://docs.nvidia.com/networking/display/kubernetes2610/quick-start/macvlan-rdma-shared.html)
