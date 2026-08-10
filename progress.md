# Cluster Progress & Model Benchmarks

## Llama 8B (`meta-llama/Llama-3.1-8B-Instruct`)

I started off with the baseline deployment of Llama 8B using **TP=16** stretched across both nodes (`gpu05` and `gpu06`). While it worked to validate our multi-node setup and network connectivity.

it was pretty messy each node ended up occupying roughly 95% of its GPU VRAM (~76GB/80GB), and inter-node network communication became a bottleneck for every single layer!

![Llama 8B TP=16 NVIDIA-SMI Memory Usage](assets/llama8B-TP16.png)

### **TP=16 (Baseline):** Runs 1 single model instance across 16 GPUs

**Weight Distribution:** Llama 8B in FP16 precision takes ~16GB of total model weights. At TP=16, these weights are sharded across 16 GPUs, so **each individual GPU holds only ~1GB of model weights!** The rest of the 76GB+ VRAM allocated per GPU (as shown in the `nvidia-smi` output above) is pre-allocated by vLLM for KV cache and execution buffers.

  Because each GPU does tiny 1GB math operations, GPU computation finishes in microseconds while inter-node network communication between `gpu05` and `gpu06` stalls every transformer layer.

For a smaller model like 8B, scaling down to lower TP sizes (such as TP=8 for 2 single-node replicas on local NVLink, or TP=4 / TP=2 / TP=1 for multi-replica serving) would eliminate inter-node network communication and yield significantly higher request throughput. However, since we aren't experimenting further or benchmarking Llama 8B, we won't be deploying these lower TP configurations.

I'm not trying to play around with this model with higher throughput. I've done this only to test and verify that multi-node TP deployment is working properly with our current setup here, so we focus on the bigger models now!

## GLM-5.2 (FP8)

Evaluating feasibility for GLM-5.2 FP8 (~744–750 GB model weights) on our 16× H100 80GB cluster (1,280 GB total VRAM):

**Aggregated TP16 (1 Model Copy across 16 GPUs) — Feasible**  
Running a single model copy across all 16 GPUs works because the ~744 GB model weights split to roughly 46.5 GB per GPU (~744 GB total). This leaves about 530 GB of gross VRAM across the cluster for KV cache and runtime execution buffers.

**Disaggregated Prefill + Decode (8 H100 Prefill + 8 H100 Decode) — Not Feasible**  
Disaggregated P/D requires two complete model copies (one for prefill and one for decode), totaling ~1,488 GB in weights alone—which exceeds our cluster's 1,280 GB total VRAM. Additionally, an 8× H100 node only has 640 GB VRAM, but one copy requires ~744 GB (~93 GB/GPU), so an 8-GPU group cannot hold the model. This is why NVIDIA's reference P/D recipes target H200 nodes (141GB VRAM per GPU).



## Qwen3-32B-FP8: RDMA/NIXL backend incident

The Qwen disaggregated vLLM deployment encountered two separate RDMA
problems. The first prevented Kubernetes from advertising an RDMA resource.
The second allowed the Pod to open the RDMA device, but prevented UCX from
constructing a usable RoCE network path. Both had to be fixed before NIXL
could initialize.

### 1. Kubernetes initially advertised no `rdma/ib` resource

The following command showed eight GPUs but an empty RDMA allocation on the
nodes:

```bash
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\tGPU="}{.status.allocatable.nvidia\.com/gpu}{"\tRDMA="}{.status.allocatable.rdma/ib}{"\n"}{end}'
```

The RDMA Shared Device Plugin requires the kernel RDMA subsystem to operate in
shared network-namespace mode on this cluster. `ib_core` was not persistently
configured with `netns_mode=1`, so the plugin could not publish usable
`rdma/ib` resources even though the physical HCAs were active.

The persistent host configuration was:

```bash
sudo tee /etc/modprobe.d/99-rdma-shared-netns.conf >/dev/null <<'EOF'
options ib_core netns_mode=1
EOF

sudo tee /etc/default/grub.d/99-rdma-shared-netns.cfg >/dev/null <<'EOF'
GRUB_CMDLINE_LINUX_DEFAULT="${GRUB_CMDLINE_LINUX_DEFAULT} ib_core.netns_mode=1"
EOF

sudo update-initramfs -u
sudo update-grub

grep -R --line-number \
  'ib_core\.netns_mode\|netns_mode=1' \
  /etc/modprobe.d \
  /etc/default/grub.d

sudo reboot
```

After rebooting both nodes, the device-plugin Pods became healthy and the
nodes advertised a nonzero `rdma/ib` allocation. This fixed Kubernetes RDMA
resource discovery, but it did not by itself make RoCE usable from an
isolated application Pod.

### 2. NIXL still failed inside ordinary Pod networking

Early worker manifests also inherited the upstream example
`UCX_NET_DEVICES=mlx5_0:1`, and a later revision selected `mlx5_8:1` without
pinning the cluster's known-good GID. Those configuration defects were fixed
first by setting `mlx5_8:1`, GID index `3`, and Ethernet address mode on both
roles. NIXL still failed after those values were correct, which exposed the
separate network-namespace problem described below.

Host inspection established the intended RoCE path:

```text
mlx5_8 port 1 -> rdma7
GID index 3   -> RoCE v2
prefill GID   -> 10.224.7.143
decode GID    -> 10.224.7.236
```

`ibv_devinfo` reported the port as active with `link_layer: Ethernet`. That
means the cluster uses RoCE rather than native InfiniBand. The zero LIDs in
`ibv_devinfo` were therefore not the error. The production UCX selection was
correct:

```yaml
- name: UCX_NET_DEVICES
  value: mlx5_8:1
- name: UCX_IB_GID_INDEX
  value: "3"
- name: UCX_IB_ADDR_TYPE
  value: eth
```

The failed preflight Pod requested `rdma/ib`, saw `/dev/infiniband`, read the
correct `mlx5_8` GID, and progressed far enough for UCX to create queues. It
then failed while constructing the RoCE address handle:

```text
ibv_create_ah(... dgid=::ffff:10.224.7.236 sgid_index=3 ...)
for UD mlx5 connect on mlx5_8 failed: No such device
UCX error: Address not valid
createBackend: backend 'UCX' ... NIXL_ERR_BACKEND
```

This error chain was the decisive evidence:

1. NIXL asked UCX to create its backend.
2. UCX opened `mlx5_8` and selected GID index 3 successfully.
3. UCX uses a UD address handle during connection setup, including when its
   main data transport is reliable-connected `rc_mlx5`.
4. Creating a RoCE address handle requires the Ethernet netdevice associated
   with the GID so the kernel can resolve the destination MAC and route.
5. The ordinary Calico Pod network namespace contained the overlay `eth0`, but
   not the host's physical `rdma7` netdevice.
6. The kernel therefore returned `ENODEV` (`No such device`), UCX translated
   it to `Address not valid`, and NIXL reported `NIXL_ERR_BACKEND`.

The later warnings about `mlx5dv_devx_obj_destroy(SRQ/CQ)`, pending
operations, objects not returned to the memory pool, and a nonempty async
event hash were cleanup after endpoint creation failed. They were not the
original cause. An inactive HCA, an incorrect GID, incompatible firmware, and
missing `/dev/infiniband` were ruled out by the host and Pod evidence.

### Why `hostNetwork` fixed it

A simple way to think about the failure is:

- `/dev/infiniband` gave the Pod the controls for the RDMA engine;
- the isolated Pod network namespace did not contain the `rdma7` road;
- UCX could start the engine, but it could not build a path to the destination.

The working Pod configuration is:

```yaml
spec:
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet
```

`hostNetwork: true` gives the worker the host's network namespace, including
`rdma7`, its `10.224.7.x` address, its GID-to-netdevice association, route, and
neighbor-resolution context. `ClusterFirstWithHostNet` preserves Kubernetes
Service DNS while using that host network.

After adding these settings independently to the prefill and decode
preflight Pods, both produced all of the required success evidence:

```text
Created backend: UCX
Backend UCX was instantiated
Initialized NIXL agent
NIXL_UCX_BACKEND_OK
ucp_context ... rma(rc_mlx5/mlx5_8:1) ... am(rc_mlx5/mlx5_8:1)
```

There was no TCP fallback. The `invalid gid[3] on mlx5_11` diagnostic was
irrelevant because UCX was pinned to `mlx5_8:1`. The `no active primary CUDA
context` diagnostic was expected because the initialization-only preflight
did not allocate or transfer a GPU buffer.

### Production settings and port isolation

The worker configuration must retain both:

```yaml
hostNetwork: true
dnsPolicy: ClusterFirstWithHostNet
```

Host networking removes the normal per-Pod TCP port namespace. Workers that
can share a physical node must therefore use distinct supported ports. The
current deployment assigns:

```text
Prefill: DYN_SYSTEM_PORT=9090, NIXL telemetry=19090
Decode:  DYN_SYSTEM_PORT=9091, NIXL telemetry=19091
```

The associated `hostPort` declarations also prevent the two identical
prefill replicas from being scheduled on the same node. Do not add
`--metrics-endpoint-port`: the `dynamo.vllm` argument parser in the pinned
`nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0` image rejects that option as an
unknown argument.

### Root cause summary

The final `NIXL_ERR_BACKEND` was not caused by an inactive HCA or an incorrect
GID. Kubernetes successfully exposed the RDMA verbs devices, but the ordinary
Calico Pod network namespace did not expose the physical RoCE netdevice
associated with the selected GID. UCX failed at `ibv_create_ah()` with
`ENODEV` because it could not construct the Ethernet/RoCE address path. Using
the host network restored the missing netdevice and allowed the UCX/NIXL
backend to initialize.

The successful initialization preflight proves device selection and backend
creation. Full acceptance still requires a real prefill-to-decode inference
request, no worker restarts, no TCP fallback, and successful NIXL KV-transfer
evidence in the production logs.
