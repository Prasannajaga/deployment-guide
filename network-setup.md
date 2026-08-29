# Network Setup

This is the runbook for building the cluster's Pod-native RoCE network. This repository's experiments use the `roce` secondary network and the `rdma/ib` resource in the `dynamo-bench` workload namespace.

This contains only the commands and their pass criteria. To understand the network architecture behind it, read [Network Architecture](network-architecture.md) alongside. For the definition of an individual term, see the [Glossary](GLOSSARY.md).

## Before you start

- Complete the base cluster bring-up in [Cluster Setup](cluster.md). Kubernetes, Calico, Helm, and the NVIDIA operators must already be installed.

The steps run in this order:

1. Set the shared cluster values.
2. Validate the physical RoCE path on both hosts.
3. Install the secondary-network components and the `rdma/ib` resource.
4. Create the Pod RoCE network (`IPPool` + `MacvlanNetwork`).
5. Validate cross-node RDMA with a disposable `rping` pair.

## 1. Set the cluster values

```bash
export NAMESPACE=dynamo-bench
export NETOP_NAMESPACE=nvidia-network-operator
export EXP_DIR=/ephemeral/shared/networking

export PREFILL_NODE=inst-1onle-devrel-rdma-pool
export DECODE_NODE=inst-g9dwj-devrel-rdma-pool

export ROCE_MASTER=rdma7
export ROCE_HCA=mlx5_8
export ROCE_NETWORK=roce
export ROCE_POOL=roce-pool

mkdir -p "$EXP_DIR"
```


## 2. Validate the physical RoCE path

Start by listing the RDMA devices and the Linux network interfaces they map to.

**Both the `$PREFILL_NODE` and `$DECODE_NODE` hosts:**

```bash
hostname
ibdev2netdev
rdma link show

for device in /sys/class/infiniband/*; do
  hca=${device##*/}
  for port in "$device"/ports/*; do
    port_number=${port##*/}
    printf '%s port %s: state=%s link_layer=%s\n' \
      "$hca" "$port_number" \
      "$(cat "$port/state")" \
      "$(cat "$port/link_layer")"
  done
done
```

Example output on `inst-1onle-devrel-rdma-pool`:

```bash
inst-1onle-devrel-rdma-pool
mlx5_0 port 1 ==> rdma0 (Up)
# ...
mlx5_8 port 1 ==> rdma7 (Up)
mlx5_9 port 1 ==> rdma8 (Up)
link mlx5_2/1 state ACTIVE physical_state LINK_UP netdev eth0
# ...
link mlx5_8/1 state ACTIVE physical_state LINK_UP netdev rdma7
link mlx5_17/1 state ACTIVE physical_state LINK_UP netdev rdma15
mlx5_0 port 1: state=4: ACTIVE link_layer=Ethernet
# ...
mlx5_8 port 1: state=4: ACTIVE link_layer=Ethernet
mlx5_9 port 1: state=4: ACTIVE link_layer=Ethernet
```

The values in step 1 select one active Ethernet RDMA port for the single-rail
path. `$ROCE_MASTER` is its Linux interface name and must be identical on every
node this `MacvlanNetwork` targets. `$ROCE_HCA` is the paired verbs device.

Validate the path on both hosts:

```bash
ip -o -4 address show dev "$ROCE_MASTER"
ip link show dev "$ROCE_MASTER"

cat "/sys/class/infiniband/$ROCE_HCA/ports/1/state"
cat "/sys/class/infiniband/$ROCE_HCA/ports/1/link_layer"
rdma system show
lsmod | grep -E 'nvidia_peermem|nv_peer_mem'
```

Example output:

```bash
58: rdma7    inet 10.224.7.143/12 brd 10.239.255.255 scope global rdma7\       valid_lft forever preferred_lft forever
58: rdma7: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 4220 qdisc mq state UP mode DEFAULT group default qlen 20000
    link/ether a0:88:c2:2d:e3:61 brd ff:ff:ff:ff:ff:ff
4: ACTIVE
Ethernet
netns shared copy-on-fork on
nvidia_peermem         16384  0
ib_core               557056  9 rdma_cm,ib_ipoib,nvidia_peermem,iw_cm,ib_umad,rdma_ucm,ib_uverbs,mlx5_ib,ib_cm
nvidia              14393344  790 nvidia_uvm,nvidia_peermem,nvidia_modeset
```

**Pass when** `$ROCE_MASTER` and `$ROCE_HCA` line up as expected, both selected nodes report state `4: ACTIVE` with link layer `Ethernet`, and `nvidia_peermem` is loaded.

### 2a. Enable shared RDMA namespace mode if needed

Skip this part if `rdma system show` already reports `netns shared`. If it reports `exclusive` instead, the selected `$ROCE_HCA` can stay in the host network namespace and never show up in the Pod network namespace.

Configure one selected node at a time: reboot and validate `$PREFILL_NODE` before changing `$DECODE_NODE`.

**Both `$PREFILL_NODE` and `$DECODE_NODE`, one at a time:**

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

Example output:

```bash
Y
netns shared copy-on-fork on
```

**Pass when** the first command prints `Y` and the second reports `netns shared`.

## 3. Install the secondary-network components

The policy below creates the `rdma/ib` Kubernetes resource and installs Multus, the CNI plugins, and NV-IPAM.

```bash
tee "$EXP_DIR/nic-cluster-policy.yaml" >/dev/null <<EOF
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
            "selectors": {
              "vendors": ["15b3"],
              "ifNames": ["${ROCE_MASTER}"]
            }
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

sleep 45
kubectl wait nicclusterpolicy/nic-cluster-policy \
  --for=jsonpath='{.status.state}'=ready --timeout=15m

kubectl get pods -n "$NETOP_NAMESPACE" -o wide
kubectl get nodes \
  -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,RDMA:.status.allocatable.rdma/ib'
```

Example output:

```bash
nicclusterpolicy.mellanox.com/nic-cluster-policy condition met

NAME                      READY   STATUS    NODE
rdma-shared-dp-ds-lfx2z   1/1     Running   inst-1onle-devrel-rdma-pool
rdma-shared-dp-ds-nvh79   1/1     Running   inst-g9dwj-devrel-rdma-pool
# cni-plugins, Multus, and NV-IPAM Pods are also Running on both nodes

NODE                          GPU   RDMA
inst-1onle-devrel-rdma-pool   8     64
inst-g9dwj-devrel-rdma-pool   8     64
```

**Pass when** the Network Operator components are healthy and both selected nodes show a positive `rdma/ib` value.

> [!NOTE]
> The RDMA selector combines the Mellanox vendor filter with `"ifNames": ["${ROCE_MASTER}"]`, which limits `rdma/ib` to the single HCA paired with `net1` (the same host interface used by the MacVLAN network), rather than any other HCA in the host.

## 4. Create the Pod RoCE network

Use the Pod RoCE address range configured directly in the `IPPool` below.
NV-IPAM divides subnet into node-local blocks:

```bash
tee "$EXP_DIR/namespace.yaml" >/dev/null <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: ${NAMESPACE}
EOF

tee "$EXP_DIR/${ROCE_POOL}.yaml" >/dev/null <<EOF
apiVersion: nv-ipam.nvidia.com/v1alpha1
kind: IPPool
metadata:
  name: ${ROCE_POOL}
  namespace: ${NETOP_NAMESPACE}
spec:
  subnet: "10.224.240.0/24"
  perNodeBlockSize: 16
EOF

tee "$EXP_DIR/${ROCE_NETWORK}.yaml" >/dev/null <<EOF
apiVersion: mellanox.com/v1alpha1
kind: MacvlanNetwork
metadata:
  name: ${ROCE_NETWORK}
spec:
  networkNamespace: "${NAMESPACE}"
  master: "${ROCE_MASTER}"
  mode: bridge
  ipam: |
    {
      "type": "nv-ipam",
      "poolName": "${ROCE_POOL}"
    }
EOF

kubectl apply -f "$EXP_DIR/namespace.yaml"
kubectl apply -f "$EXP_DIR/${ROCE_POOL}.yaml"
kubectl apply -f "$EXP_DIR/${ROCE_NETWORK}.yaml"

kubectl get ippools.nv-ipam.nvidia.com "$ROCE_POOL" \
  -n "$NETOP_NAMESPACE" \
  -o custom-columns='NAME:.metadata.name,SUBNET:.spec.subnet,BLOCK_SIZE:.spec.perNodeBlockSize'
kubectl get macvlannetwork "$ROCE_NETWORK" \
  -o custom-columns='NAME:.metadata.name,STATUS:.status.state'
kubectl get network-attachment-definition "$ROCE_NETWORK" \
  -n "$NAMESPACE" \
  -o custom-columns='NAME:.metadata.name'
```

Example output:

```bash
NAME        SUBNET            BLOCK_SIZE
roce-pool   10.224.240.0/24   16

NAME   STATUS
roce   ready

NAME
roce
```

**Pass when** all three network resources exist, including the NAD generated in `$NAMESPACE`. `$ROCE_POOL` remains in `$NETOP_NAMESPACE` because NV-IPAM reads pools from the operator namespace.

The previous `rdma/roce` NAD was removed when the operator moved the generated NAD to `dynamo-bench`.

If your fabric blocks child MAC addresses, stop here and use [SR-IOV VFs](https://docs.nvidia.com/networking/display/kubernetes2570/quick-start/sriov-ib-rdma.html) instead.

## 5. Validate cross-node RDMA

Create one lightweight `rping` server Pod on `$PREFILL_NODE` and one client Pod on `$DECODE_NODE` to validate the RDMA connection:

```bash
tee "$EXP_DIR/rping.yaml" >/dev/null <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${ROCE_NETWORK}-rping-server
  namespace: ${NAMESPACE}
  annotations:
    k8s.v1.cni.cncf.io/networks: ${ROCE_NETWORK}
spec:
  restartPolicy: Never
  nodeName: ${PREFILL_NODE}
  tolerations:
    - {key: node-role.kubernetes.io/control-plane, operator: Exists, effect: NoSchedule}
    - {key: nvidia.com/gpu, operator: Equal, value: "true", effect: NoSchedule}
  containers:
    - name: rping
      image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0
      imagePullPolicy: IfNotPresent
      command: ["/bin/sh", "-c"]
      args: ["exec sleep 3600"]
      securityContext:
        runAsUser: 0
        capabilities:
          add: [IPC_LOCK, SYS_RESOURCE]
      resources:
        requests: {rdma/ib: "1"}
        limits: {rdma/ib: "1"}
---
apiVersion: v1
kind: Pod
metadata:
  name: ${ROCE_NETWORK}-rping-client
  namespace: ${NAMESPACE}
  annotations:
    k8s.v1.cni.cncf.io/networks: ${ROCE_NETWORK}
spec:
  restartPolicy: Never
  nodeName: ${DECODE_NODE}
  tolerations:
    - {key: node-role.kubernetes.io/control-plane, operator: Exists, effect: NoSchedule}
    - {key: nvidia.com/gpu, operator: Equal, value: "true", effect: NoSchedule}
  containers:
    - name: rping
      image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0
      imagePullPolicy: IfNotPresent
      command: ["/bin/sh", "-c"]
      args: ["exec sleep 3600"]
      securityContext:
        runAsUser: 0
        capabilities:
          add: [IPC_LOCK, SYS_RESOURCE]
      resources:
        requests: {rdma/ib: "1"}
        limits: {rdma/ib: "1"}
EOF

kubectl delete -f "$EXP_DIR/rping.yaml" --ignore-not-found --wait=true
kubectl apply -f "$EXP_DIR/rping.yaml"

sleep 45
kubectl wait --for=condition=Ready \
  "pod/${ROCE_NETWORK}-rping-server" "pod/${ROCE_NETWORK}-rping-client" \
  -n "$NAMESPACE" --timeout=5m
```

Example output:

```bash
pod/roce-rping-server condition met
pod/roce-rping-client condition met
```

Install dependencies for `rping` in the two temporary test Pods:

```bash
for pod in "${ROCE_NETWORK}-rping-server" "${ROCE_NETWORK}-rping-client"; do
  kubectl exec -n "$NAMESPACE" "$pod" -- bash -lc '
    export DEBIAN_FRONTEND=noninteractive
    mkdir -p /var/lib/apt/lists/partial
    apt-get update
    apt-get install -y --no-install-recommends \
      rdma-core ibverbs-utils rdmacm-utils perftest iputils-ping iproute2
  '
  printf '%s: dependencies installed\n' "$pod"
done
```

Example output:

```bash
roce-rping-server: dependencies installed
roce-rping-client: dependencies installed
```

Verify that the resource allocation exposes the HCA selected by `ifNames` and no other Mellanox HCA:

```bash
for pod in "${ROCE_NETWORK}-rping-server" "${ROCE_NETWORK}-rping-client"; do
  observed_hcas="$(
    kubectl exec -n "$NAMESPACE" "$pod" -- ibv_devices |
      awk '$1 ~ /^mlx5_/ {print $1}' |
      sort -u
  )"
  printf '%s: %s\n' "$pod" "$observed_hcas"
  test "$observed_hcas" = "$ROCE_HCA"
done
```

**Pass when** both Pods print only `mlx5_8`. If another HCA appears, stop and fix the RDMA device-plugin allocation before deploying workers; the selector has not produced the device set assumed by this guide.

Example output:

```bash
roce-rping-server: mlx5_8
roce-rping-client: mlx5_8
```

List the selected device files as a second allocation check:

```bash
for pod in "${ROCE_NETWORK}-rping-server" "${ROCE_NETWORK}-rping-client"; do
  printf '%s:\n' "$pod"
  kubectl exec -n "$NAMESPACE" "$pod" -- \
    find /dev/infiniband -maxdepth 1 -type c -printf '%f\n' | sort
done
```

Example output:

```bash
roce-rping-server:
issm8
rdma_cm
umad8
uverbs8
roce-rping-client:
issm8
rdma_cm
umad8
uverbs8
```

Capture the `net1` address of each Pod:

```bash
net1_ip() {
  kubectl get pod "$1" -n "$NAMESPACE" -o json |
    jq -r '.metadata.annotations["k8s.v1.cni.cncf.io/network-status"]
      | fromjson | map(select(.interface == "net1"))[0].ips[0]' |
    cut -d/ -f1
}

export RPING_SERVER_IP="$(net1_ip "${ROCE_NETWORK}-rping-server")"
export RPING_CLIENT_IP="$(net1_ip "${ROCE_NETWORK}-rping-client")"

test -n "$RPING_SERVER_IP" && test "$RPING_SERVER_IP" != null
test -n "$RPING_CLIENT_IP" && test "$RPING_CLIENT_IP" != null
printf 'server=%s client=%s\n' "$RPING_SERVER_IP" "$RPING_CLIENT_IP"
```

Example output:

```bash
server=10.224.240.5 client=10.224.240.20
```

Run the transfer test:

```bash
timeout 90 kubectl exec -n "$NAMESPACE" "${ROCE_NETWORK}-rping-server" -- \
  rping -s -a "$RPING_SERVER_IP" -V -v \
  >"$EXP_DIR/rping-server.log" 2>&1 &
RPING_SERVER_PID=$!

sleep 2

set +e
timeout 90 kubectl exec -n "$NAMESPACE" "${ROCE_NETWORK}-rping-client" -- \
  rping -c -I "$RPING_CLIENT_IP" \
    -a "$RPING_SERVER_IP" -V -v -C 10 2>&1 | \
  tee "$EXP_DIR/rping-client.log"
RPING_CLIENT_RC=${PIPESTATUS[0]}

wait "$RPING_SERVER_PID"
RPING_SERVER_RC=$?
set -e

printf 'client_rc=%s server_rc=%s\n' "$RPING_CLIENT_RC" "$RPING_SERVER_RC"
test "$RPING_CLIENT_RC" -eq 0
test "$RPING_SERVER_RC" -eq 0
```

Example output:

```bash
ping data: rdma-ping-0: ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqr
# ...
ping data: rdma-ping-9: JKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyzA
client_rc=0 server_rc=0
```

**Pass when** all RDMA iterations complete.


Then clean up the test Pods:

```bash
kubectl delete -f "$EXP_DIR/rping.yaml" --wait=true
```

Example output:

```bash
pod "roce-rping-server" deleted from dynamo-bench namespace
pod "roce-rping-client" deleted from dynamo-bench namespace
```

The Pod RoCE network is now ready.

Continue with [Deployment Setup](setup.md) to validate the shared prerequisites and launch a model recipe.

## Troubleshooting

If Pod fails, check the logs:

```bash
kubectl describe pod POD_NAME -n "$NAMESPACE" | sed -n '/Events:/,$p'
kubectl logs POD_NAME -n "$NAMESPACE" --previous --timestamps --tail=300
kubectl get pods -n "$NETOP_NAMESPACE" -o wide
```

## References

- [Dynamo RDMA setup](https://docs.nvidia.com/dynamo/dev/kubernetes/installation/rdma-setup/overview)
- [Dynamo disaggregated communication](https://docs.nvidia.com/dynamo/dev/knowledge-base/kubernetes/kubernetes-operator/disagg-communication)
- [Network Operator MacVLAN with RDMA Shared Device](https://docs.nvidia.com/networking/display/kubernetes2610/quick-start/macvlan-rdma-shared.html)
