# Network Architecture

This document explains the cluster's network configuration.

See [Network Setup](network-setup.md) for the runbook. For the definition of an individual term, see [Glossary](GLOSSARY.md).

## Starting from a bare-metal cluster

Let's look at our two-node bare-metal cluster:

```text
gpu05                                       gpu06
┌──────────────────────────────┐             ┌──────────────────────────────┐
│ Ubuntu Linux                 │             │ Ubuntu Linux                 │
│ 8 × H100 GPUs                │             │ 8 × H100 GPUs                │
│ ConnectX-7 NICs              │◀─ Ethernet ─▶│ ConnectX-7 NICs             │
└──────────────────────────────┘             └──────────────────────────────┘
```

In this example, `gpu05` runs the Prefill workers and `gpu06` runs the Decode workers. The Prefill workers create the KV cache and transfer it to the Decode workers.

The conventional network path sends data through the operating system's socket stack, which adds CPU work and data-path overhead. However this is not ideal for sending KV cache in disaggregated serving because the KV cache grows [very large](https://developer.nvidia.com/blog/how-to-reduce-kv-cache-bottlenecks-with-nvidia-dynamo/#why_is_kv_cache_a_bottleneck_for_llm_inference) with the context length and must be transferred frequently, at each prefill-to-decode handoff. [RDMA (Remote Direct Memory Access)](GLOSSARY.md#rdma-and-gpudirect-rdma) provides a faster, more efficient path for these transfers.

This is the minimum background needed to understand the configuration.

The Kubernetes and networking details below serve one goal: give KV-cache
transfers a faster RDMA path.

## Problem we faced when applying Kubernetes

When a [Pod](GLOSSARY.md#kubernetes-pod) starts, the container runtime invokes the cluster's configured [CNI](GLOSSARY.md#cni-and-cni-plugin) plugin. In our cluster, [Calico](GLOSSARY.md#calico) provides the default Pod network and creates `eth0`. Applications can use TCP over the network.

However for the transfer sizes used by disaggregated inference, TCP fallback can potentioally become the system bottleneck. In fact the [Dynamo documentation](https://docs.nvidia.com/dynamo/dev/kubernetes/installation/install-dynamo) reports roughly **200–500×** degradation in speed when TCP fallback is used instead of faster RDMA.

Applying RDMA to KV cache transfer starts with checking whether every
participating node has an active [RDMA-capable Ethernet](GLOSSARY.md#roce)
port. Our nodes did, so a RoCE path was viable.

After validating the network cards, the first approach we tried was setting [`hostNetwork: true`](GLOSSARY.md#hostnetwork), which exposes the host's network namespace to the Pod.

All Pods then share the host's network namespace, including its interfaces,
addresses, routes, and TCP/UDP port space.

This approach did **not** work for multiple replicas (DP) on the same node. For example, with TP=2 on an 8-GPU node, Dynamo creates four replicas. Each replica tries to bind the same ports, such as `9000`, so four Pods compete for the same port and fail.

With separate [network namespaces](GLOSSARY.md#linux-network-namespace), each Pod could bind those ports independently. However in this case the Pod could not use RoCE interface for RDMA because it wasn't visible to Pod.

Concretely, we hit two mutually exclusive failure modes when scaling disaggregated workers across nodes:

| Worker networking | Result |
| --- | --- |
| `hostNetwork: true` | RDMA worked, but workers on the same node shared its TCP ports, causing conflicts between replicas. |
| `hostNetwork: false` with only Calico | Workers had isolated ports, but no usable RoCE interface inside the Pod, so [UCX](GLOSSARY.md#ucx)/[NIXL](GLOSSARY.md#nixl) could not use RDMA. |

So `hostNetwork: true` was not viable in our case.

## Solution we derived: Use MacVLAN

To use RDMA for sending KV cache while avoiding port conflicts, we need to add a second [network interface](GLOSSARY.md#network-interface) beyond the default `eth0`.

There are multiple ways to do this.

One option is [SR-IOV (Single Root I/O Virtualization)](GLOSSARY.md#sr-iov-physical-function-and-virtual-function), which configures a [NIC](GLOSSARY.md#nic-hca-and-connectx)'s physical function to expose virtual functions at the PCIe level. SR-IOV can provide RDMA access, but MacVLAN was more straightforward (less clunky to set up) for this cluster.

We added [MacVLAN](GLOSSARY.md#macvlan) alongside Calico for the RoCE network. MacVLAN derives a virtual interface from the host's `rdma7` interface and places it in the Pod as `net1`. Each Pod receives its own `net1`, MAC address, IP address, and network namespace. The separate namespace and IP allows replicas bind the same port number without colliding.

We also changed the CNI topology. A Pod normally receives the cluster's default CNI network. However since we want two different network connection, we use Multus. [Multus](GLOSSARY.md#multus) acts as a meta-plugin (or orchestrator) that retains that default network while attaching additional networks through other CNI plugins, including MacVLAN.

The default topology was:

```
kubelet
  ↓ (asks the container runtime to create the Pod sandbox)
container runtime
  ↓ (invokes the configured CNI)
Calico (manages normal network, creates `eth0`)
```

and we changed it to:

```
kubelet
  ↓ (asks the container runtime to create the Pod sandbox)
container runtime
  ↓ (invokes the configured CNI)
Multus  ←── meta CNI plugin (orchestrator) that can call multiple CNI Plugins
  ↓ (Multus internally calls multiple CNI plugins)
  ├─→ Calico   (handles the "default network" → creates eth0)
  └─→ MacVLAN  (handles the "additional network" → creates net1, only when annotation present)
```

So now each worker Pod has two interfaces. It keeps `eth0` for normal Kubernetes communication, and MacVLAN adds `net1` for the RoCE path:

```text
gpu05                                       gpu06
┌──────────────────────────┐                ┌──────────────────────────┐
│ eth0: comm / control     │ ─── Calico ──▶ │ eth0: comm / control     │
│ net1: transfer KV cache  │ ══ RoCE/RDMA ═▶│ net1: transfer KV cache  │
└──────────────────────────┘                └──────────────────────────┘
```

This looks like the full solution, but there is still one missing piece. While `net1` gives the Pod an address on the RoCE network, it does not by itself let the process issue RDMA operations to the NIC.

## Why `net1` alone is not enough

With normal TCP over `eth0`, the application writes to a socket and the Linux kernel handles the rest of the network path. RDMA works differently. After connection setup, the application submits work through the verbs API instead of sending the data through the normal kernel socket path.

That leaves us with two separate things to solve:

1. Give the Pod its own path and identity on the [RoCE fabric](GLOSSARY.md#roce-fabric).
2. Give the process access to the NIC's RDMA API.

**1. A path and an identity on the fabric**

The first half is the part MacVLAN and NV-IPAM solve. A [container](GLOSSARY.md#container) runs in the Pod's network namespace, so it does not automatically inherit the host's `rdma7` interface. MacVLAN creates `net1` inside that namespace using `rdma7` as the parent.

`net1` needs two addresses:

1. **L2 (MAC address).** MacVLAN gives each `net1` its own MAC. It allows multiple Pods to share one physical NIC and still be told apart at the switch.
2. **L3 (IP address).** Each Pod's `net1` also needs an IP address so other Pods can reach it over RoCEv2. [NV-IPAM](GLOSSARY.md#ipam-and-nv-ipam) allocates an address from the configured RoCE network pool.

**2. Direct access to the NIC's RDMA API**

To register memory, create [queue pairs](GLOSSARY.md#queue-pair), and issue RDMA READ/WRITE operations, the process must open device files such as `/dev/infiniband/uverbsX`.

The **[RDMA Shared Device Plugin](GLOSSARY.md#kubernetes-device-plugin)** gives the container access to those files. So `net1` and `/dev/infiniband` do different jobs, even though both lead back to the same physical ConnectX port.

This is how the pieces line up:

| What's required | Layer | Who provides it |
| --- | --- | --- |
| An RDMA-capable NIC the kernel recognizes | hardware / driver | the host, independent of Kubernetes |
| An interface inside the Pod's network namespace | L2 | MacVLAN |
| A unique MAC address for that interface | L2 | MacVLAN |
| A valid IP on the RoCE fabric | L3 | NV-IPAM |
| Access to `/dev/infiniband/*` | verbs / device | [RDMA shared device plugin](GLOSSARY.md#rdma-shared-device-plugin-and-rdmaib) |

When applying this in the system:
- `NicClusterPolicy` installs the RDMA Shared Device Plugin, NV-IPAM, Multus, and the CNI plugins. It also defines which host interface backs `rdma/ib`.
- `IPPool` contains the RoCE addresses.
- `MacvlanNetwork` tells the Network Operator how to generate the [NAD (`NetworkAttachmentDefinition`)](GLOSSARY.md#nad).
- Finally when deploying worker pod, pod annotation tells Multus to attach the predefined NAD.

## Installing the pieces with `NicClusterPolicy`

`NicClusterPolicy` is the cluster-level installation and configuration object:

```yaml
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
              "ifNames": ["rdma7"]
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
```

`rdmaSharedDevicePlugin` installs the device plugin and creates the Kubernetes resource name `rdma/ib`.

`nvIpam` installs the IPAM component that allocates an address to each Pod's secondary `net1` interface.

`secondaryNetwork` installs the CNI binaries and Multus, which invokes the requested secondary network during Pod creation.

Below this section, `MacvlanNetwork` describes the network and causes the Network Operator to generate the NAD.

Then worker manifest(`deploy.yaml`) finally asks Multus to use it.

## "NIC names" - one NIC has two names

Linux networking uses a name such as [`rdma7`](GLOSSARY.md#rdma-and-mlx), while the [RDMA verbs](GLOSSARY.md#rdma-verbs) side uses a name such as [`mlx5_8`](GLOSSARY.md#rdma-and-mlx).

By using the `ibdev2netdev` command, we can inspect the mapping:

```
ibdev2netdev
mlx5_0 port 1 ==> rdma0 (Up)
mlx5_1 port 1 ==> rdma1 (Up)
mlx5_10 port 1 ==> rdma9 (Up)
mlx5_11 port 1 ==> eth1 (Up)
mlx5_12 port 1 ==> rdma10 (Up)
mlx5_13 port 1 ==> rdma11 (Up)
mlx5_14 port 1 ==> rdma12 (Up)
mlx5_15 port 1 ==> rdma13 (Up)
mlx5_16 port 1 ==> rdma14 (Up)
mlx5_17 port 1 ==> rdma15 (Up)
mlx5_2 port 1 ==> eth0 (Up)
mlx5_3 port 1 ==> rdma2 (Up)
mlx5_4 port 1 ==> rdma3 (Up)
mlx5_5 port 1 ==> rdma4 (Up)
mlx5_6 port 1 ==> rdma5 (Up)
mlx5_7 port 1 ==> rdma6 (Up)
mlx5_8 port 1 ==> rdma7 (Up)
mlx5_9 port 1 ==> rdma8 (Up)
```

MacVLAN CNI works with Linux network interfaces, so the NAD uses `rdmaN`. UCX works with the RDMA device and port so it sees the same physical port as `mlx5_N:1`.

When using the specific NIC/HCA, two names must align to the same physical port.

## Generating NAD with MacvlanNetwork

We materialize it as `roce.yaml` under `/ephemeral/shared/networking`:

```yaml
apiVersion: mellanox.com/v1alpha1
kind: MacvlanNetwork
metadata:
  name: roce
spec:
  # Workload namespace where the `roce` NAD is created
  networkNamespace: dynamo-bench
  # Host Linux interface used as the parent when creating the Pod's `net1`
  master: rdma7
  # Lets MacVLAN interfaces on the same parent communicate at L2
  mode: bridge
  ipam: |
    { "type": "nv-ipam", "poolName": "roce-pool" }
```

The [Network Operator](GLOSSARY.md#nvidia-network-operator) reads this resource and generates the `roce` NAD in the workload [namespace](GLOSSARY.md#namespace).

The Pod now has its own MAC and IP on `net1` and the packets leave through the physical link behind `rdma7`.

## Attaching `roce` in `deploy.yaml`

Creating the NAD does not attach it to every Pod automatically. The final step is to ask for RDMA connection in each worker that needs the RoCE path.

Here is the relevant part of a prefill worker. The decode worker uses the same settings:

```yaml
# ...
    VllmPrefillWorker:
      componentType: worker
      subComponentType: prefill
      replicas: 4
      extraPodMetadata:
        annotations:
          k8s.v1.cni.cncf.io/networks: roce
      extraPodSpec:
        hostNetwork: false
        mainContainer:
          env:
            - name: UCX_TLS
              value: rc_x,rc,cuda_copy,cuda_ipc
            - name: UCX_IB_ADDR_TYPE
              value: eth
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

### `k8s.v1.cni.cncf.io/networks: roce`

When a Pod is deployed through `deploy.yaml`, Multus checks for the `k8s.v1.cni.cncf.io/networks` annotation. If the annotation is absent, the Pod receives only the default Calico network and `eth0`. If it is present, Multus also attaches the secondary network.

In this case, Multus looks for a NAD named `roce` and attaches the network described by that NAD.

The worker and generated NAD both live in `dynamo-bench`, so Multus can resolve the bare name `roce` without a namespace qualifier.

### UCX setting

UCX uses both pieces of this path: `net1` supplies the network identity, and
`/dev/infiniband` supplies the RDMA API for memory registration, queue-pair
creation, and READ/WRITE operations.

`UCX_TLS` restricts which transports UCX may use. The list is an allowlist, not a left-to-right preference order; UCX selects among the enabled transports according to endpoint capabilities and performance characteristics.

- `rc_x`: a high-performance variant of `rc` that takes fuller advantage of recent `mlx5` hardware features.
- `rc`: Reliable Connected, one of RDMA's standard transport types (alongside others like UD and DC). As the name suggests, it's a reliable connection: it guarantees packet order and resends any packets that are lost.
- `cuda_copy`: for copying between GPU memory and CPU memory.
- `cuda_ipc`: IPC = Inter-Process Communication, for GPU-to-GPU transfers within the same node.

The full list available can be found [here](https://docs.nvidia.com/dynamo/dev/knowledge-base/kubernetes/kubernetes-operator/disagg-communication#core-transport-selection).

> [!NOTE]
> We intentionally omit `tcp`. Allowing it could let a transfer succeed over a much slower TCP path and mask an RDMA configuration problem. Restricting the transport list makes failures in the intended RDMA path visible instead.

`UCX_IB_ADDR_TYPE` controls the address type UCX uses for IB devices. We set it to `eth` so UCX uses Ethernet-style addressing for our [RoCE](GLOSSARY.md#roce) fabric rather than native [InfiniBand](GLOSSARY.md#infiniband) addressing.

The RDMA Shared Device Plugin prevents UCX from choosing another HCA by
combining the Mellanox vendor selector with `ifNames: ["rdma7"]`. This narrows
the `rdma/ib` device set to the [HCA](GLOSSARY.md#nic-hca-and-connectx) paired
with the MacVLAN parent. The Pod receives access to that selected HCA's device
files, so UCX can discover it without setting
`UCX_NET_DEVICES=mlx5_8:1` in every worker manifest.

### `rdma/ib`

A request such as `rdma/ib: "2"` uses the Kubernetes extended resource created
by the RDMA Shared Device Plugin. A Pod must request this resource before
Kubernetes gives the container access to the selected RDMA device files.

Our plugin config uses `rdmaHcaMax: 64`, so each node reports 64 logical allocation units when the selected HCA is present. These units are only (virtual) Kubernetes accounting, not a physical split.

A Pod receives the same selected-HCA device-file set whether it requests `rdma/ib: "1"` or `rdma/ib: "2"`. Usually we use one unit per GPU as convention, so a TP=2 worker requests `rdma/ib: "2"`.

## Summary

We've gone through a lot, so let's quickly review what we needed to do to enable RDMA across multiple nodes and Pods.

We used MacVLAN to attach each Pod to the RDMA network and NV-IPAM to assign its IP address. Then configured the RDMA Shared Device Plugin to select the same host interface, making the corresponding RDMA device-file set available to Pods that request `rdma/ib`.

This gives each Pod both parts of the path: a network interface for RDMA traffic and access to the RDMA device behind it.

See the actual setup in [network setup](network-setup.md).

## References

- [NVIDIA Network Operator](https://docs.nvidia.com/networking/display/kubernetes2641/index.html)
- [MacVLAN Network with RDMA Shared Device](https://docs.nvidia.com/networking/display/kubernetes2641/quick-start/macvlan-rdma-shared.html)
- [RDMA Shared Device Plugin](https://github.com/Mellanox/k8s-rdma-shared-dev-plugin)
- [NVIDIA RDMA over Converged Ethernet](https://networking-docs.nvidia.com/doca/archive/3-4-0/rdma-over-converged-ethernet)
- [UCX in NVIDIA HPC-X](https://networking-docs.nvidia.com/hpcxum/2.50/unified-communication-x-framework-library)
- [NIXL](https://github.com/ai-dynamo/nixl/blob/main/README.md)
- [Dynamo RDMA setup](https://docs.nvidia.com/dynamo/dev/kubernetes/installation/rdma-setup/overview)
- [Dynamo disaggregated communication](https://docs.nvidia.com/dynamo/dev/knowledge-base/kubernetes/kubernetes-operator/disagg-communication)
- [RoCE v2](https://www.ibm.com/docs/en/aix/7.3.0?topic=access-rdma-over-converged-ethernet-roce-version-2)
