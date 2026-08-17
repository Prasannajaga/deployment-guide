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
  dnsPolicy: ClusterFirst
```

`hostNetwork: true` gives the worker the host's network namespace, including
`rdma7`, its `10.224.7.x` address, its GID-to-netdevice association, route, and
neighbor-resolution context. `ClusterFirst` preserves Kubernetes
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
dnsPolicy: ClusterFirst
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

## Port collisions after enabling host networking

The RDMA fix introduced a separate networking consequence. Before
`hostNetwork: true`, every Pod had its own network namespace and therefore
its own private TCP port space. Two Pods on the same node could both listen on
`*:20380` or `*:5600` because those sockets existed in different namespaces.

With `hostNetwork: true`, the worker processes bind directly to the node's IP
and share the node's single TCP port space. A listening address is identified
by the combination of protocol, local IP, and port. Only one process can own a
particular combination at a time. For example, on node
`10.18.96.143`:

```text
Prefill binds tcp://*:20380       -> succeeds
Decode tries tcp://*:20380       -> EADDRINUSE / Address already in use

Prefill binds tcp://10.18.96.143:5600 -> succeeds
Decode tries the same address          -> EADDRINUSE / Address already in use
```

This is similar to two apartments becoming one shared house. With isolated
Pod networking, each apartment can have its own room number 5600. With host
networking, there is only one room 5600 in the house, so the second occupant
must use another number.

### Collision 1: forward-pass metrics ZMQ port `20380`

The first observed failure was:

```text
zmq.error.ZMQError: Address already in use (addr='tcp://*:20380')
```

`20380` is used by Dynamo's instrumented scheduler for its forward-pass
metrics ZMQ PUB socket. Both roles inherited the same default. When a prefill
worker and the decode worker were placed on the same host-networked node, the
first worker claimed `20380` and the second worker could not start its engine
core.

The supported environment variable was used to separate the roles:

```text
Prefill: DYN_FORWARDPASS_METRIC_PORT=20380
Decode:  DYN_FORWARDPASS_METRIC_PORT=20381
```

An attempted `--metrics-endpoint-port` workaround was removed because it is
not a valid command-line argument in the pinned
`nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0` image. The runtime terminated
with an unknown-argument error before startup. `DYN_FORWARDPASS_METRIC_PORT`
is the setting that controls the port involved in this collision.

### Collision 2: NIXL handshake side-channel port `5600`

After resolving `20380`, the decode worker reached NIXL initialization but
its handshake-listener thread failed with:

```text
zmq.error.ZMQError: Address already in use
(addr='tcp://10.18.96.143:5600')
```

Port `5600` is the default vLLM NIXL side channel. It carries NIXL connection
metadata and handshake messages; it is separate from the high-volume RDMA KV
data path. The prefill worker on `10.18.96.143` already owned port `5600`, so
the decode listener could not bind it.

The roles now use different supported side-channel ports:

```text
Prefill: VLLM_NIXL_SIDE_CHANNEL_PORT=5600
Decode:  VLLM_NIXL_SIDE_CHANNEL_PORT=5601
```

The decode process remained in Kubernetes `Running` state after this error
because only the background `nixl_handshake_listener` thread had exited.
However, the Pod stayed `0/1` and could not perform a valid NIXL handshake.
Changing the environment variable alone could not revive that dead thread;
the corrected manifest had to be applied and the decode Pod recreated.

### Complete role-specific port allocation

The deployment gives every potentially co-located role a distinct port for
each listener:

| Listener | Prefill | Decode | Reason |
| --- | ---: | ---: | --- |
| Dynamo system | `9090` | `9091` | Avoid a possible host-level system listener conflict |
| NIXL Prometheus telemetry | `19090` | `19091` | Avoid a possible metrics listener conflict |
| Forward-pass metrics ZMQ | `20380` | `20381` | Fix the observed `20380` collision |
| NIXL handshake side channel | `5600` | `5601` | Fix the observed `5600` collision |

The `20380` and `5600` failures were directly observed. The system and NIXL
telemetry pairs were separated proactively because those listeners also share
the host port namespace. These port errors were not RDMA hardware, GID, UCX,
or NIXL backend failures; they appeared only after the RDMA fix allowed worker
initialization to progress further under host networking.

Matching `containerPort` and `hostPort` entries document and reserve these
ports for Kubernetes scheduling. Because both prefill replicas request the
same prefill `hostPort` values, Kubernetes cannot schedule them on the same
node. A prefill and decode worker can share a node because all four of their
host ports differ.

Useful checks are:

```bash
# Show the listening processes and ports on a node.
sudo ss -lntp | grep -E ':(9090|9091|5600|5601|19090|19091|20380|20381)\b'

# Confirm the values injected into a worker Pod.
kubectl exec -n "$NAMESPACE" "$WORKER_POD" -- env \
  | grep -E 'DYN_SYSTEM_PORT|NIXL_TELEMETRY_PROMETHEUS_PORT|DYN_FORWARDPASS_METRIC_PORT|VLLM_NIXL_SIDE_CHANNEL_PORT'
```

The general rule for this deployment is: every process that may run on the
same physical node with `hostNetwork: true` must have a unique port for every
socket it binds. A port may be reused only when scheduling guarantees that
the processes will be on different nodes.

## Why 2P+1D works but 6P+2D does not

The 2-prefill + 1-decode disaggregated deployment works on our 2-node cluster
because the port math fits. The 6-prefill + 2-decode KV-aware routing
experiment does not, and the reason is entirely in how `hostPort` interacts
with replica counts under `hostNetwork: true`.

### The scheduling rule

When a Pod declares `hostPort: 9090`, the Kubernetes scheduler treats that
port as an exclusive resource on the physical node. Only one Pod can own
`hostPort: 9090` on a given node at a time. If a second Pod requests the same
`hostPort` on the same node, the scheduler leaves it `Pending` with the event:

```text
no nodes with enough resources were found:
2 node(s) didn't have free ports for the requested pod ports.
```

This is not a bug. It is the scheduler enforcing that two processes on the
same host network cannot bind the same port.

### Why 2P+1D fits

The working deployment explicitly assigns different port sets to prefill and
decode:

```text
Prefill:  system=9090  nixl-metrics=19090  fpm=20380  nixl-side=5600
Decode:   system=9091  nixl-metrics=19091  fpm=20381  nixl-side=5601
```

Both prefill replicas declare `hostPort: 9090`. The scheduler therefore
forces them onto separate nodes — one prefill per node. With 2 nodes, that
is exactly 2 prefill replicas. The single decode replica declares
`hostPort: 9091`, which does not collide with `9090`, so it can share either
node with a prefill worker.

The layout ends up being:

```text
gpu05:  1 prefill (ports 9090/19090/20380/5600)
        1 decode  (ports 9091/19091/20381/5601)
gpu06:  1 prefill (ports 9090/19090/20380/5600)
```

Three pods, two nodes, no port conflict, 8 GPUs fully used. The `hostPort`
constraint actually helps here — it acts as an implicit anti-affinity that
guarantees the two identical prefill replicas land on different nodes.

### Why 6P+2D does not fit

The 6P+2D experiment uses the same Dynamo Operator with `hostNetwork: true`.
The operator injects two `hostPort` declarations into every worker Pod
regardless of what we put in the manifest:

```text
Every prefill pod:  system=9090  nixl=19090
Every decode pod:   system=9090  nixl=19090
```

I tried `ports: []` in `mainContainer` to suppress these, but the operator
ignores that override — it always writes `hostPort: 9090` and
`hostPort: 19090` into the generated Pod spec. I confirmed this by inspecting
the actual Pod JSON:

```bash
kubectl get pod "$POD" -n "$NAMESPACE" -o json | jq '.spec.containers[].ports'
```

Every single worker pod had `hostPort: 9090` and `hostPort: 19090` regardless
of my `ports: []` override.

With those fixed ports, the scheduler can only place **one prefill per node**
and **one decode per node**:

```text
gpu05:  1 prefill (ports 9090/19090)  +  1 decode (ports 9090/19090) ← COLLISION
gpu06:  1 prefill (ports 9090/19090)  +  1 decode (ports 9090/19090) ← COLLISION
```

Wait, it is even worse. Prefill and decode use the **same** ports (9090 and
19090), so a prefill and a decode cannot even share a node. The scheduler can
place at most one worker of any type per node. With 2 nodes, that means
maximum 2 workers total — but we need 8 (6 prefill + 2 decode). The remaining
6 pods sit in `Pending` forever.

In the 2P+1D deployment, I could manually assign different ports to prefill
and decode because there are only two roles. For 6P+2D, I would need 8
unique port sets (one per replica), but the Dynamo Operator uses a single
component definition with `replicas: N` — every replica gets identical ports.

### The `VLLM_NIXL_SIDE_CHANNEL_HOST` confusion

I also added `VLLM_NIXL_SIDE_CHANNEL_HOST` set to `status.podIP` via the
Kubernetes downward API. This fixes a different problem entirely: without it,
vLLM defaults the NIXL side-channel address to `localhost`, so a decode worker
on `gpu06` cannot reach a prefill worker on `gpu05` for the initial NIXL
handshake. Setting it to the pod IP fixes cross-node reachability.

But `VLLM_NIXL_SIDE_CHANNEL_HOST` only controls the **IP address** the
side-channel socket advertises. It does not control the **port number**, and
it does not prevent the Dynamo Operator from injecting `hostPort: 19090` into
the Pod spec. The scheduling collision is a port problem, not an address
problem. These are two independent issues at two different layers:

- `VLLM_NIXL_SIDE_CHANNEL_HOST` → application layer, IP advertisement
- `hostPort` collision → Kubernetes scheduler layer, port reservation

### Without `hostNetwork` it also fails — differently

I tried removing `hostNetwork: true` to eliminate the port collision. The
pods scheduled fine — every pod got its own Calico overlay IP and its own
port namespace, so there was no port conflict at all. But then RDMA failed:

```text
ibv_create_ah(dgid=::ffff:10.224.7.143 sgid_index=3)
for UD mlx5 connect on mlx5_8 failed: No such device
NIXL_ERR_BACKEND
```

The Calico pod network namespace does not contain the host's `rdma7`
netdevice. The `dgid` in the error was a pod overlay IP that does not exist
in any GID table of `mlx5_8:1`. UCX could open the HCA but could not
construct the RoCE address path. This is the exact same failure we fixed
earlier by adding `hostNetwork: true` in the first place.

So we are stuck between two constraints:

- **With `hostNetwork: true`:** RDMA works, but `hostPort` limits us to 1
  worker per node.
- **Without `hostNetwork: true`:** No port collisions, but RDMA fails because
  the pod namespace lacks the physical RoCE netdevice.

## Final port-collision fix: isolated Pods with a secondary RoCE adapter

The Kubernetes scheduling problem was fixed without allocating a unique set
of host ports to every worker. Each prefill and decode Pod now keeps its
ordinary isolated network namespace and receives a second Kubernetes network
adapter dedicated to the RoCE fabric.

The resulting Pod network layout is:

```text
eth0  Calico Pod network   Kubernetes Services, control traffic, and the
                           vLLM NIXL handshake address
net1  Multus MacVLAN      rdma7-backed RoCE path for UCX/NIXL KV transfers
```

This design uses four cluster-level components:

1. **Multus** attaches more than one network interface to a Pod.
2. **MacVLAN CNI** creates the Pod's `net1` interface on the physical
   `rdma7` parent interface.
3. **NV-IPAM** gives each `net1` adapter a unique address from the reserved
   `qwen-roce-pool` subnet.
4. **RDMA Shared Device Plugin** exposes the `mlx5_8` verbs device through
   the `rdma/ib` extended resource.

The persistent cluster objects are:

```text
NV-IPAM IPPool:                 qwen-roce-pool
MacvlanNetwork:                 qwen-roce
NetworkAttachmentDefinition:   qwen32-bench/qwen-roce
Physical parent interface:     rdma7
RDMA HCA/port:                  mlx5_8:1
```

These objects are installed once per cluster and are not recreated for each
model deployment. New deployments reference the existing Network Attachment
Definition and request the existing RDMA resource.

### Worker manifest changes

Both prefill and decode components attach the secondary network:

```yaml
extraPodMetadata:
  annotations:
    k8s.v1.cni.cncf.io/networks: qwen32-bench/qwen-roce
```

They explicitly remain outside the host network:

```yaml
extraPodSpec:
  hostNetwork: false
  dnsPolicy: ClusterFirst
  mainContainer:
    ports: []
```

The workers retain access to the shared HCA by requesting `rdma/ib`:

```yaml
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

The NIXL handshake uses the primary Calico Pod IP from the Kubernetes
downward API. Because every Pod has a unique `eth0` address, replicas can all
listen on the same side-channel port without colliding:

```yaml
- name: VLLM_NIXL_SIDE_CHANNEL_HOST
  valueFrom:
    fieldRef:
      fieldPath: status.podIP
```

UCX remains restricted to the known RoCE HCA:

```yaml
- name: UCX_TLS
  value: "rc_x,rc,cuda_copy,cuda_ipc"
- name: UCX_NET_DEVICES
  value: "mlx5_8:1"
- name: UCX_IB_ADDR_TYPE
  value: "eth"
```

`UCX_IB_GID_INDEX` is deliberately not hardcoded in the pod-native manifests.
A GID index representing the host's `rdma7` address is not necessarily the
index associated with a Pod's MacVLAN `net1` address.

All `hostPort` declarations were removed. `ports: []` suppresses the default
worker port declarations produced by the DGD template in this configuration.

### Why this fixes Kubernetes scheduling

With `hostNetwork: true`, all workers placed on a node compete for the same
node-wide socket namespace. Kubernetes reserves each requested `hostPort` as
an exclusive per-node resource, which caused Kai Scheduler to report:

```text
2 node(s) didn't have free ports for the requested pod ports
```

With `hostNetwork: false`, each Pod owns a separate network namespace and
separate IP addresses. These socket tuples are distinct even when the port is
the same:

```text
192.168.12.145:5600  decode Pod A
192.168.12.186:5600  decode Pod B
192.168.63.147:5600  prefill Pod A
192.168.63.190:5600  prefill Pod B
```

Consequently, all replicas may use the same application ports. Kubernetes no
longer reserves `9090`, `19090`, `20380`, or `5600` on the physical nodes,
and Kai Scheduler can place multiple TP=2 workers on each eight-GPU node.

The separation of responsibilities is:

```text
Kubernetes scheduling fix: hostNetwork=false and no hostPort
RoCE device access:        rdma/ib resource exposes mlx5_8
RoCE network path:         Multus/MacVLAN adds net1 over rdma7
NIXL handshake:            unique Calico status.podIP on eth0
NIXL KV data:              UCX uses mlx5_8:1 through the RoCE path
```

### Verifying the port-collision fix

The live generated Pods, rather than only the DGD YAML, must be checked:

```bash
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o json |
jq -r '
  .items[] |
  [
    .metadata.name,
    (.spec.nodeName // "PENDING"),
    (.spec.hostNetwork // false),
    ([.spec.containers[].ports[]?.hostPort |
      select(. != null)] | join(","))
  ] | @tsv
'
```

PASS for the scheduling fix requires:

- every worker shows `hostNetwork=false`;
- the host-port column is empty;
- multiple workers can be placed on the same node;
- no Pod event contains `didn't have free ports`;
- Pods are no longer `Pending` because of port availability.

Verify that each worker also received the two network adapters:

```bash
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o name |
grep -E 'vllm(prefill|decode)worker' |
while read -r pod; do
  echo "===== $pod ====="
  kubectl get -n "$NAMESPACE" "$pod" -o json |
    jq -r '.metadata.annotations[
      "k8s.v1.cni.cncf.io/network-status"
    ] | fromjson'
done
```

PASS requires both `eth0` and a unique `net1` address on every worker.

## Qwen3-32B-FP8: benchmark completion and DCGM export

The disaggregated AIPerf benchmark matrix completed successfully and retained
its artifacts in:

```text
/ephemeral/shared/qwen3-32b/perf-cache/aiperf/disagg/1786442681_qwen3-32b-fp8-vllm-disagg-perf/
```

The cluster's client-facing Prometheus Service was confirmed as:

```text
namespace: monitoring
service:   monitoring-kube-prometheus-prometheus
port:      9090
```

Prometheus can therefore be opened locally with
`9095:9090` port-forwarding. The DCGM export is restricted to Kubernetes
workload label `exported_namespace="qwen32-bench"`, `exported_pod` names
matching the disaggregated prefill/decode workers, metric
`DCGM_FI_DEV_GPU_UTIL`, and the eight hours immediately before the query. The
plain `namespace` and `pod` labels identify the exporter in `gpu-operator`,
not the inference worker. Raw samples, a sample-level CSV, the exact query
window, and a per-worker/per-GPU utilization summary are saved under
`/ephemeral/shared/dynamo/aiperf-results/dcgm-last-8h/`.

The complete executable procedure is documented in
[`fetch-metrics.md`](models/qwen3-32B/experiments/vllm/disagg-routing/fetch-metrics.md).

NIXL transfer telemetry was not enabled for this completed run. NIXL latency
collection is deferred to the next benchmark round, when the NIXL exporter
will be enabled before traffic starts and scraped for the entire run.

## Progress in the last 24 hours — August 13–14, 2026

The last day expanded the cluster work from the earlier Qwen3-32B deployment
into three new MoE model families and several controlled serving topologies.
The most important completed result is that Qwen3.6-35B-A3B-FP8 was deployed
successfully with SGLang in two different disaggregated layouts and both
layouts completed the same seven-point AIPerf matrix. DeepSeek-V4-Flash-FP8
and Qwen3-235B-A22B-FP8 received pinned model-cache jobs, full SGLang
deployment manifests, and operational runbooks. We also prepared vLLM
Qwen3.6 P/D recipes and a new SGLang TP1-attention + EP2 experiment with
KV-aware routing and complete transfer telemetry.

The evidence level is intentionally separated below. A checked-in manifest is
not described as a successful deployment unless retained smoke-test or
benchmark artifacts prove that the model served requests.

| Model / topology | GPU layout | Status after this work |
|---|---:|---|
| Qwen3.6 SGLang P/D TP1 4P4D | 4 prefill × 1 GPU + 4 decode × 1 GPU = 8 | Deployed and benchmarked; all seven concurrency points passed |
| Qwen3.6 SGLang P/D TP2 2P2D | 2 prefill × 2 GPUs + 2 decode × 2 GPUs = 8 | Deployed and benchmarked; all seven concurrency points passed |
| DeepSeek-V4-Flash-FP8 SGLang aggregate | 4 replicas × TP=4 = 16 | H100 capacity probe prepared and attempted; no retained acceptance artifact, so not marked validated |
| Qwen3-235B-A22B-FP8 SGLang aggregate | 4 replicas × TP=4 = 16 | Complete pinned recipe and runbook prepared; cluster acceptance result not retained in this repository |
| Qwen3-235B-A22B-FP8 SGLang P/D | 2 prefill × TP=4 + 2 decode × TP=4 = 16 | Complete NIXL/RoCE recipe prepared; cluster acceptance result not retained |
| Qwen3.6 vLLM KV-aware P/D TP1 4P4D | 4 prefill + 4 decode = 8 | Deployment, hybrid-cache compatibility gate, and benchmark job prepared; not yet supported by retained pass evidence |
| Qwen3.6 vLLM KV-aware P/D TP2 2P2D | 2 prefill × TP=2 + 2 decode × TP=2 = 8 | Deployment, hybrid-cache compatibility gate, and benchmark job prepared; not yet supported by retained pass evidence |
| Qwen3.6 SGLang TP1-attention + EP2 4P4D | 4 prefill × 2 GPUs + 4 decode × 2 GPUs = 16 | KV-aware CPU-offload deployment passed a prefill-heavy sweep at concurrency 1–32; matching no-offload A/B run is still pending |

### DeepSeek-V4-Flash-FP8 H100 capacity probe

We added a pinned cache and SGLang aggregate recipe for
`sgl-project/DeepSeek-V4-Flash-FP8`, revision
`ae01d80c06cdfe30581edfd0e1c5449dc7ed7f17`. The checkpoint is approximately
294 GB. The intended experiment uses four independent TP=4 replicas, placing
approximately 1.176 TB of weights into the cluster's 1.28 TB of aggregate HBM
before KV cache, CUDA workspaces, and runtime allocations are counted.

This made it a deliberately aggressive H100 capacity probe. The official
SGLang Hopper TP=4 configuration is an H200-oriented layout; four copies leave
too little H100 headroom to assume success. The recipe therefore uses a short
8,192-token context, chunked prefill, eight maximum running requests,
`--disable-cuda-graph`, and explicit DeepSeek-V4 reasoning/tool parsers. It
also contains a hard cleanup path when a worker fails during loading or memory
profiling. The repository does not contain a completed smoke-test or benchmark
artifact for this model, so the probe must not be presented as a validated
serving result.

The executable recipe is in
[`models/deepseek-v4-flash-fp8/sglang/agg/`](models/deepseek-v4-flash-fp8/sglang/agg/README.md).

### Qwen3-235B-A22B-FP8 TP=4 baselines

We added a second large-model family using the pinned public checkpoint
`Qwen/Qwen3-235B-A22B-FP8`, revision
`2180ded38a22f6ab0ea405cbb02af2f7a6090379`. This is a roughly 239 GB sparse
MoE checkpoint with 235B total parameters, 22B active parameters per token,
128 experts, and eight selected experts per token.

TP=4 was chosen deliberately:

- the model has four KV heads, which divide exactly across four ranks;
- its FP8 MoE scale layout is compatible with TP=4 but not the proposed TP=8
  baseline;
- each rank holds roughly 59.75 GB of checkpoint weights, leaving materially
  more H100 runtime headroom than the DeepSeek-V4 four-copy probe.

Two matching SGLang recipes now exist:

1. **Aggregated:** four independent TP=4 workers, two per node, consuming all
   16 GPUs.
2. **Disaggregated:** two TP=4 prefill workers on `gpu05` and two TP=4 decode
   workers on `gpu06`, also consuming all 16 GPUs. Every P-to-D transfer
   crosses the physical RoCE link through NIXL/UCX.

Both layouts deliberately keep expert parallelism, speculative decoding,
KV-aware routing, and KV offloading disabled so that the initial comparison
changes only the serving topology. The disaggregated manifest reuses the
pod-native `qwen-roce` attachment, requests `rdma/ib`, pins UCX to
`mlx5_8:1`, and avoids `hostNetwork` and host-port collisions. Complete cache,
deploy, log, smoke-test, and cleanup procedures are under
[`models/qwen3-235B-A22B/`](models/qwen3-235B-A22B/README.md). No retained
request or benchmark artifact is present yet, so these remain ready-to-run
baselines rather than published performance results.

### Qwen3.6-35B-A3B-FP8 model bring-up

The main successful model deployment of the day was
`Qwen/Qwen3.6-35B-A3B-FP8`, pinned to revision
`95a723d08a9490559dae23d0cff1d9466213d989`. It is a 37.5 GB multimodal sparse
MoE with 35B total parameters, approximately 3B active parameters per token,
256 experts, eight selected experts, two KV heads, and a 262,144-token native
context. Three of every four language layers use Gated DeltaNet recurrent
linear attention, which makes disaggregated transfer more demanding than a
standard transformer KV cache.

The serving baseline was held at 131,072 tokens, 85% static GPU-memory
fraction, and a 64-token SGLang page size. Both layouts used Dynamo 1.3.0,
SGLang 0.5.14, namespace-isolated Pods, the existing Multus/MacVLAN
`qwen-roce` network, and NIXL/UCX over `mlx5_8:1`.

#### Successful SGLang TP1 4P4D deployment

The first controlled layout ran four single-GPU prefill workers and four
single-GPU decode workers:

```text
gpu05: 4 × prefill, TP=1
gpu06: 4 × decode,  TP=1
total: 8 H100 GPUs
```

The deployment served the model successfully and the benchmark completed at
concurrencies `1, 4, 8, 16, 32, 64, 128`. Every point is recorded as `PASS`
with zero entries in AIPerf's error summary.

#### Successful SGLang TP2 2P2D deployment

The matching TP2 layout ran two two-GPU prefill workers and two two-GPU decode
workers:

```text
gpu05: 2 × prefill, TP=2
gpu06: 2 × decode,  TP=2
total: 8 H100 GPUs
```

It used the same model revision, context, memory fraction, workload, random
seed, benchmark duration, and concurrency points. All seven points also
finished with `PASS` and no recorded AIPerf errors.

### Qwen3.6 TP1-versus-TP2 quick mixed benchmark

The comparable retained workload was a 60-second mixed-sequence run at each
concurrency after 16 warmup requests. Thinking was disabled and the random
seed was fixed at 42. The sequence distribution was:

```text
ISL 1,024 / OSL 256:   35%
ISL 4,096 / OSL 512:   30%
ISL 8,192 / OSL 1,024: 20%
ISL 16,384 / OSL 512:  10%
ISL 32,768 / OSL 256:   5%
```

Headline retained results are:

| Concurrency | TP1 output tok/s | TP2 output tok/s | TP1 median TTFT | TP2 median TTFT | TP1 median ITL | TP2 median ITL |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 214.60 | 247.77 | 138.55 ms | 146.35 ms | 4.05 ms | 3.51 ms |
| 4 | 819.60 | 931.42 | 134.55 ms | 143.76 ms | 4.21 ms | 3.73 ms |
| 8 | 1,529.92 | 1,634.64 | 134.40 ms | 144.98 ms | 4.56 ms | 4.30 ms |
| 16 | 2,573.50 | 2,709.46 | 137.11 ms | 238.59 ms | 5.43 ms | 4.95 ms |
| 32 | 4,179.99 | 4,302.14 | 224.84 ms | 304.62 ms | 6.66 ms | 6.13 ms |
| 64 | 6,470.49 | 6,226.02 | 238.95 ms | 445.88 ms | 8.51 ms | 7.80 ms |
| 128 | 9,435.67 | 7,201.80 | 474.86 ms | 1,073.26 ms | 11.44 ms | 8.85 ms |

TP2 had the expected small-concurrency advantage: at concurrency 1 its output
throughput was about 15.5% higher than TP1. The two layouts were close around
concurrency 32, where TP2 was about 2.9% higher. The crossover occurred before
concurrency 64. At concurrency 128 the replicated TP1 layout delivered about
31.0% more output-token throughput and substantially lower median TTFT, which
shows the benefit of having four independent horizontal workers instead of
two TP groups when the workload becomes highly concurrent. TP2 retained lower
median inter-token latency at the highest point, illustrating the latency
versus aggregate-capacity tradeoff rather than a universal winner.

These are short controlled comparison runs, not final production numbers.
They establish that both topologies function, exercise long mixed prompts,
and remain error-free through concurrency 128. The retained configurations,
matrix status files, frontend metric snapshots, raw profiles, CSV/JSON
exports, inputs, logs, and plots are stored under:

- [`tp1-4p4d/benchmarks`](models/qwen3.6-35B-A3B/sglang/disagg/tp1-4p4d/benchmarks/)
- [`tp2-2p2d/benchmarks`](models/qwen3.6-35B-A3B/sglang/disagg/tp2-2p2d/benchmarks/)

### Qwen3.6 vLLM KV-aware recipes

We also created controlled vLLM 0.23.0 P/D recipes for the same TP1 4P4D and
TP2 2P2D comparison. Both use Dynamo's KV-aware frontend, prefix caching,
128-token blocks, NIXL producer/consumer roles, pod-native RoCE, and explicit
hybrid/Mamba cache settings. Each topology has an AIPerf job using the same
workload controls.

Because Qwen3.6 contains Gated DeltaNet recurrent state, these recipes include
a strict compatibility gate rather than assuming ordinary transformer KV
transfer is enough. Acceptance requires successful model readiness, a real
P-to-D request, NIXL/UCX initialization, and logs proving that the pinned vLLM
connector supports the model's hybrid memory architecture. There are no
retained passing vLLM artifacts in the repository yet, so SGLang remains the
validated backend for this comparison.

### New SGLang TP1-attention + EP2 KV-aware recipe

The final addition was a full 16-GPU expert-parallel experiment for Qwen3.6.
Each worker has a two-GPU SGLang world configured as:

```text
--tp-size 2
--dp-size 2
--ep-size 2
--enable-dp-attention
--enable-dp-lm-head
--moe-dense-tp-size 1
```

With SGLang's DP-attention layout this means effective attention/dense TP=1
and expert parallelism EP=2; it is not a four-GPU `TP × EP` product. The full
deployment contains four prefill workers and four decode workers:

```text
4 prefill × 2 GPUs + 4 decode × 2 GPUs = 16 GPUs
```

The frontend uses `--router-mode kv`, and prefill workers publish KV events
through ZMQ for KV-aware routing. The transfer and observability stack now
contains three separate Prometheus paths:

- frontend and KV-router metrics on port `8000`;
- Dynamo plus SGLang engine metrics on worker port `9090`;
- NIXL transfer telemetry on port `19090` using
  `NIXL_TELEMETRY_ENABLE=y` and the Prometheus exporter.

The preflight does more than wait for Ready Pods. It sends a real disaggregated
request, checks SGLang/Dynamo metrics, requires a positive NIXL transferred-byte
counter, and rejects any nonzero failed-transfer counter. A separate metrics
runbook covers direct endpoint checks, Prometheus target discovery,
before/after snapshots, and range-query export. YAML parsing, embedded-manifest
equality, Bash syntax, topology accounting, and observability configuration
were validated locally. The CPU-offload deployment has now completed the
prefill-heavy sweep; the matching no-offload run and final A/B plots are still
the next acceptance steps.

The recipe is under
[`models/qwen3.6-35B-A3B/sglang/disagg/tp1-ep2-4p4d/`](models/qwen3.6-35B-A3B/sglang/disagg/tp1-ep2-4p4d/README.md),
with the workload and telemetry procedure in
[`benchmark.md`](models/qwen3.6-35B-A3B/sglang/disagg/tp1-ep2-4p4d/benchmark.md).

### Immediate next steps

1. Preserve the raw before/after NIXL and HiCache snapshots from the completed
   CPU-offload prefill-heavy run under the experiment directory.
2. Run the same prefill-heavy matrix without CPU offload, changing only the
   deployment variant, then generate the A/B plots.
3. Run and retain explicit acceptance artifacts for Qwen3-235B aggregate and
   disaggregated TP=4 layouts.
4. Treat the DeepSeek-V4 four-copy TP=4 topology as a capacity experiment; if
   H100 headroom is insufficient, change the topology rather than repeatedly
   restarting OOM workers.
5. Keep the vLLM Qwen3.6 results unpublished until the hybrid GDN/NIXL gate
   proves recurrent-state transfer end to end.


### Qwen3.6 TP1-attention + EP2 prefill-heavy CPU-offload test

I ran this preset against the CPU-offload deployment:

```bash
export PRESET=prefill-heavy
export WORKLOAD_NAME="${VARIANT}-${PRESET}"
capture_worker_metrics "${PRESET}-before"

kubectl delete job -n "$NAMESPACE" "$PERF_JOB_NAME" --ignore-not-found
kubectl set env --local -f "$EXP_DIR/perf.yaml" -o yaml \
  EXPERIMENT_VARIANT="$VARIANT" \
  WORKLOAD_MODE=fixed WORKLOAD_NAME="$WORKLOAD_NAME" \
  ISL=32768 OSL=256 PREFIX_MODE=shared \
  PREFIX_GROUPS=64 PREFIX_REUSE_PERCENT=75 \
  CONCURRENCIES='1 4 8 16 32' BENCHMARK_DURATION=300 \
  WARMUP_REQUESTS=16 RANDOM_SEED=42 NUM_DATASET_ENTRIES=4096 \
  ARTIFACT_ROOT="${RESULT_ROOT}/${VARIANT}" |
  kubectl apply -n "$NAMESPACE" -f -
kubectl logs -n "$NAMESPACE" -f "job/$PERF_JOB_NAME"
kubectl wait -n "$NAMESPACE" --for=condition=Complete \
  "job/$PERF_JOB_NAME" --timeout=14400s

capture_worker_metrics "${PRESET}-after"
```

Ran the prefill-heavy experiment at concurrency `1, 4, 8, 16, 32`, with
300 seconds at every point and 16 warmup requests before every measurement.
So this was 1,500 seconds of measured load across the full sweep, not one
single 300-second run.

Each request takes a 32K context window with OSL 256 to keep the decode
workload small and make this properly prefill heavy. With 75% prefix reuse,
the 32K input is split like this:

```text
shared prefix: 32,768 × 75% = 24,576 tokens
unique part:   32,768 - 24,576 = 8,192 tokens
output:        256 tokens
prefix groups: 64
```

This gives us a large working set: many requests can reuse a long 24K prefix,
but the 64 different prefix groups can still push older cache pages out of GPU
memory and into the CPU HiCache tier.

since this is prefill heavy, Dynamo spread different requests across the four
prefill replicas using KV locality and load. One important correction here:
it did not split one request into four pieces across all four replicas. Each
full request was routed to one prefill worker, and that selected worker used
its own two GPU ranks with `DP=2` and `EP=2`. Across all traffic, the four
prefill workers fed the four decode workers.

From the live NIXL counters, each prefill worker appeared to handle roughly
1 TB of cumulative transfer and the graph showed around 1.5 GB/s at sampled
points. I need to keep these as two different measurements:

- cumulative bytes tell how much moved during the chosen window;
- GB/s tells how quickly it moved at a point or averaged over a window;
- the correct average is `(after bytes - before bytes) / elapsed seconds`;
- 1 TB in one 300-second point is about 3.3 GB/s, while 1 TB across the full
  five-point 1,500-second sweep is about 0.67 GB/s. So 1.5 GB/s was likely a
  sampled rate, not the average that produces the 1 TB counter.

even with this sweep, the live graph kept the KV transfer under 60 ms, which
is nice considering the model is deployed with two GPU ranks per worker and
expert parallelism across them. This is NIXL P-to-D transfer time only; it is
not the full request TTFT. The final A/B result still needs the AIPerf TTFT,
TPOT, request-throughput, and output-token-throughput plots.

#### How the HiCache ratio math works here

We configured `--hicache-ratio 1.2`. In easy words, SGLang first works out the
size of the GPU KV pool for one rank, then creates a CPU host KV pool that is
1.2 times that size:

```text
CPU host KV capacity per rank = GPU KV capacity per rank × 1.2
```

For example, if one rank can hold 100,000 KV tokens on GPU, ratio 1.2 gives
that rank space for about 120,000 KV tokens in CPU memory. This ratio is based
on the GPU KV pool, not the full 80 GB GPU and not the full host RAM.

We have four prefill workers with two ranks each, so there are eight separate
prefill host pools using this math. Decode workers do not have CPU HiCache in
this recipe. Because the policy is `write_through`, new reusable pages are
copied into the CPU tier while a GPU copy may still exist. That means the CPU
tier is a backup/cache level, not automatically 1.2 times more unique KV on top
of everything still resident on GPU.

The 95% number also needs the right wording. If it came from:

```text
sglang:hicache_host_used_tokens / sglang:hicache_host_total_tokens
```

then it means the allocated CPU HiCache pool was 95% full. It does not mean
95% of every request's KV, or 95% of all KV traffic, was offloaded to CPU.
This is still very good evidence that the long-prefix workload actually filled
and exercised the host cache instead of only enabling the flag.

#### Why the CPU went 99% hot

The CPU becoming hot makes sense with this workload. We used `write_through`,
so new cache pages are copied from GPU memory into CPU memory as they are
created. If an older prefix is needed again after leaving GPU cache, HiCache
also has to find its pages and copy them back from CPU to GPU. Long 32K inputs,
64 prefix groups, and concurrency up to 32 keep that page-copy pipeline busy.

The `page_first_direct` memory layout and `direct` I/O backend make these local
GPU-to-CPU and CPU-to-GPU copies possible for this hybrid attention/Mamba
model. The CPU is coordinating page tables, cache bookkeeping, request
scheduling, network work, and host-memory copies. The model's MoE math is
still running on the GPUs; 99% CPU does not mean inference moved to CPU.

One more detail: if the 99% came from `top` for one process, it normally means
roughly one logical CPU core was fully busy, not 99% of the entire node. If it
came from a node-wide Grafana panel, then it means almost all node CPU capacity
was busy. I need to record which metric produced it before calling this a
node-wide CPU bottleneck.

There are also two different copy paths in this test:

- HiCache `direct` moves cache pages locally between a prefill GPU and that
  worker's CPU memory;
- NIXL/UCX moves the finished KV and recurrent state from the selected prefill
  worker to the selected decode worker over RDMA.

Interesting that the NIXL connection was very smooth without errors. But it
was not one KV object copied across all 16 GPUs. The deployment uses 16 GPUs
in total, while each request follows one selected prefill-to-decode path; all
16 GPUs participate across the complete workload.

nice output. Next I will run the exact same preset against the non-offload
deployment and keep the same seed, duration, prefix groups, and concurrency
points. This offload run proves that the setup survives the pressure and uses
CPU HiCache, but only the matching A/B plots can show whether offloading
actually improves QPS or TTFT enough to justify the extra CPU work.

The repository does not currently contain the raw metric snapshots behind the
1 TB, 1.5 GB/s, under-60-ms, 95%, and 99% observations. I need to retain those
files under the experiment directory before treating these as final published
numbers.

## Qwen3.6 aggregated KEDA autoscaling — August 15–16, 2026

Today I brought up Prometheus-based KEDA autoscaling for the aggregated
Qwen3.6-35B-A3B-FP8 SGLang TP=2 deployment. The frontend remains a single
CPU-only replica. Each aggregated worker performs both prefill and decode and
requests two H100 GPUs.

### Prometheus and KEDA failure investigation

The initial `ScaledObject` was accepted by KEDA and its HPA could read the
DGDSA scale subresource, but the external metric stayed `<unknown>`. KEDA
repeatedly reported:

```text
prometheus metrics 'prometheus' target may be lost, the result is empty
```

This was a metrics-discovery failure rather than a DGDSA or HPA failure. The
important observations were:

- the `DynamoGraphDeploymentScalingAdapter` existed and reported one current
  worker;
- KEDA created `keda-hpa-qwen36-35b-a3b-sglang-worker` successfully;
- Prometheus returned no `dynamo_frontend_active_requests` series at all;
- `qwen32-bench` contained no `PodMonitor`;
- the monitoring Prometheus selected PodMonitors carrying
  `release=monitoring`, while its empty namespace selector already permitted
  cross-namespace discovery.

The fix was to create one frontend-only PodMonitor in `qwen32-bench`, label it
`release: monitoring`, select the DGD's frontend Pod labels, and scrape
`/metrics` on declared container port 8000. The Prometheus resource itself did
not need to be broadened.

The original PromQL also assumed that the frontend-local active-request gauge
carried `dynamo_namespace`. The retained series instead had Prometheus target
labels such as `namespace` and `pod`, so the scaler and dashboard query now
use:

```promql
sum(
  dynamo_frontend_active_requests{
    namespace="qwen32-bench",
    pod=~"qwen36-35b-a3b-fp8-sglang-agg-tp2.*frontend.*"
  }
)
```

The ScaledObject target was also changed from the deprecated
`nvidia.com/v1alpha1` DGDSA API to the API served by this cluster,
`nvidia.com/v1beta1`. The corrected checked-in scaler and complete discovery
procedure are under
[`models/qwen3.6-35B-A3B/sglang/agg-autoscaling/`](models/qwen3.6-35B-A3B/sglang/agg-autoscaling/README.md).

### Successful autoscaling evidence

Once Prometheus retained the frontend metric, the HPA stopped reporting
`<unknown>` and KEDA scaled the worker DGDSA under load. Observed HPA samples
were:

```text
85334m / 16 average at 3 replicas
51200m / 16 average at 5 replicas
36572m / 16 average at 7 replicas
```

The `m` suffix is Kubernetes milli-units, so these values mean approximately
85.334, 51.2, and 36.572 active requests per current worker. Each sample is
consistent with roughly 256 global active requests divided by the current
replica count. This proves the intended `AverageValue` behavior: desired
workers are calculated from approximately
`ceil(global active requests / 16)`, subject to the configured maximum.

The Grafana frontend panel had the same stale `dynamo_namespace` selector and
therefore also returned no data. Its replacement query is the same working
`namespace`/`pod` sum shown above. A second optional panel divides that global
sum by the HPA's current replica count so its value can be compared directly
with the per-worker threshold of 16. Dashboard display against the corrected
Prometheus data source still needs a final saved-panel confirmation.

### GPU capacity ceiling exposed by the scale test

The traffic signal worked, but KEDA requested more workers than the scheduler
could place. Eight is a total worker count, not eight workers in addition to
the seed worker:

```text
8 workers total × 2 GPUs per TP=2 worker = 16 GPUs
```

That maximum is possible only when all 16 allocatable GPUs are free and
dedicated to this deployment. It leaves no GPU headroom. During the test, two
workers were running or starting while six were Pending. KAI Scheduler
reported that neither node had enough GPU resources for another two-GPU Pod.
KEDA is demand-aware but not GPU-capacity-aware; it can update the DGDSA to
eight even when Kubernetes cannot schedule eight workers.

The operational rule is therefore:

```text
maxReplicaCount = guaranteed GPU budget for this deployment / 2
```

For the capacity visible during this run, two workers were the immediately
schedulable ceiling. A maximum of eight should be used only for an exclusive
16-GPU benchmark after all competing GPU workloads are removed. Seven is a
safer exclusive-cluster ceiling when one two-GPU pair should remain available
as rollout or recovery headroom.

### Cleanup finding and remaining work

The first cleanup path removed the KEDA ScaledObject and restored the original
worker count; it did not delete the underlying DGD. That behavior is useful
when removing autoscaling while retaining service, but it is not a full GPU
cleanup.

The later `nvidia-smi` capture did not show an abstract Kubernetes reservation.
It showed live processes named `VLLM::Worker_DP*_EP*`, each retaining roughly
66 GB of VRAM while idle. Those process names do not match this SGLang recipe
and likely belong to another or older VLLM deployment. The two terminal panes
also displayed identical PIDs, so they may have been observing the same node
rather than proving identical occupancy on both nodes.

Before removing anything at the node runtime level, the remaining work is:

1. list every GPU-requesting Pod across all namespaces, including its node,
   phase, GPU request, and image;
2. map the VLLM processes to their container cgroups and Kubernetes Pod/DGD
   ownership;
3. delete the confirmed stale top-level DGD so its controllers do not recreate
   individual Pods;
4. wait for all graph-labeled Pods to disappear and verify GPU release with
   both Kubernetes allocation data and `nvidia-smi`;
5. use `crictl stop`/`rm` only if a process is proven to belong to an orphaned
   container with no live Kubernetes Pod.

Directly killing the visible PIDs is intentionally excluded: a live
controller may restart them, and ownership has not yet been established. The
next session should finish that ownership mapping, confirm the final Grafana
panel, and then set `maxReplicaCount` from the GPU budget actually reserved for
this experiment.

## Documentation and HiCache capacity analysis — August 17, 2026

The last three conversation threads focused on documenting a small cross-node
SGLang P/D canary, explaining the observed 3.10-million-token CPU HiCache
capacity, and preserving the resulting decisions in this progress log. No
cluster workload was deployed or changed as part of these documentation and
capacity-analysis conversations.

### Standalone cross-node SGLang canary analysis

The standalone manifest supplied as `sanjeev.md` was identified as a compact
Qwen3.6 disaggregated-serving canary rather than a full performance topology:

```text
1 Dynamo frontend
1 prefill worker × TP=2 = 2 GPUs
1 decode worker  × TP=2 = 2 GPUs
Total                    = 4 GPUs
```

The prefill and decode workers were selected onto their respective role nodes
and used SGLang disaggregation with NIXL over UCX/RoCE. Its intended use was to
validate model startup, cross-node prefill-to-decode state transfer, UCX GPU
memory registration, and one real request before attempting a larger topology.
It was not a KV-to-disk offload configuration, an autoscaling deployment, or a
meaningful high-availability/performance result by itself.

The raw manifest was converted during that conversation into a self-contained
runbook with:

- cluster-host variables and an `EXP_DIR` under `/ephemeral/shared`;
- namespace, retained PV, and PVC creation;
- safe recovery of a retained PV in `Released` state;
- operator, node-role, GPU, RDMA, cache, and RoCE preflight checks;
- generated storage and `DynamoGraphDeployment` manifests;
- readiness, startup-log, smoke-request, and transfer acceptance gates; and
- normal deployment cleanup separated from optional PVC/PV/namespace teardown.

Local validation at that point proved that all Bash fences were syntactically
valid, both generated YAML manifests parsed, container-time variables remained
literal after shell expansion, and the topology requested four GPUs. It also
exposed two assumptions that must be treated as deployment gates:

1. A `hostPath` PV is not made cross-node by declaring `ReadWriteMany`.
   `/ephemeral/shared/huggingface` must already be a genuinely shared mount at
   the same path on both selected nodes, or the PV must be replaced by the
   cluster's real RWX storage class.
2. The workers referenced the cross-namespace
   `qwen32-bench/qwen-roce` NetworkAttachmentDefinition. Multus authorization,
   the selected `mlx5_8:1` device, and attachment behavior must be verified in
   the target namespace.

The current workspace no longer contains `sanjeev.md`, so that validated
runbook is not a retained repository artifact at this checkpoint. It must be
restored or recreated before claiming that the standalone canary documentation
is checked in. No other recipe was modified during that conversation.

### What the 3.10-million-token CPU cache represents

The `3.10 Mil` value comes from the CPU HiCache capacity metric, not from a
direct setting of 3.10 million tokens. The dashboard query in
[`preflight.md`](models/qwen3.6-35B-A3B/sglang/disagg/tp1-ep2-4p4d/preflight.md)
uses:

```promql
sum by (pod) (
  sglang:hicache_host_total_tokens{...}
)
```

It sums the rank-level series inside one prefill Pod and leaves one value per
Pod. The related `sglang:hicache_host_used_tokens` gauge reports occupied
capacity. Neither metric changes the per-request `--context-length 131072`;
the host tier stores reusable pages from many prompts or conversations rather
than enabling one multi-million-token request.

The CPU tier is sized relative to the GPU cache tier:

```text
host-token capacity ≈ device-token capacity × hicache ratio
```

With the observed per-Pod capacity and the current ratio:

```text
observed host capacity = 3.10 million tokens
current ratio          = 1.2
implied device capacity ≈ 3.10M / 1.2 = 2.583M tokens per Pod
```

To target 10 million host-cache token slots on every prefill Pod while keeping
the model, runtime, parallelism, GPU pool, page size, and memory fraction
unchanged:

```text
required ratio = 1.2 × 10.0M / 3.10M ≈ 3.87
```

`--hicache-ratio 3.9` should therefore produce roughly 10.08 million token
slots per Pod, while `4.0` should provide roughly 10.33 million after the same
linear estimate. The actual value is page- and allocator-rounded and must be
read back from the metric after startup.

The scope of the target matters. Production has four independent prefill Pods,
so four Pods each reporting 3.10 million already provide approximately 12.4
million raw fleet-wide slots. Those are not one shared cache: a prefix stored
on one worker is useful only when KV-aware routing returns matching traffic to
that worker. Duplicate prefixes, uneven routing, eviction, and page alignment
make useful unique capacity lower than the raw sum.

### HiCache and P-to-D transfer are separate mechanisms

The request path has two distinct data movements:

```text
reused prefix on prefill:
GPU L1 <-> local prefill CPU L2 through HiCache direct I/O

new request handoff:
selected prefill worker -> selected decode worker through NIXL/UCX/RDMA
```

`--enable-hierarchical-cache`, `page_first_direct`, and the `direct` I/O backend
provide the local GPU/CPU cache tier for Qwen3.6's hybrid attention and
Mamba/GDN state. `--disaggregation-transfer-backend nixl` moves the live state
between prefill and decode workers; it does not allocate CPU HiCache. Decode
HiCache remains disabled for this pinned hybrid-model/runtime combination.

The frontend's KV-aware router and prefill KV events are therefore essential.
On a GPU hit the selected worker reuses L1 pages directly. On a CPU hit it
copies the required L2 pages back to GPU before continuing prefill. On a miss
it computes the prefix, admits pages according to the configured write policy,
and later evicts finite CPU pages as its working set changes.

### Memory cost of a 10-million-token per-Pod target

Changing the ratio from `1.2` to `4.0` multiplies the host-cache allocation by
approximately:

```text
4.0 / 1.2 = 3.33×
```

The existing prefill resource template requests 128 GiB and limits each worker
Pod to 192 GiB. That template cannot be assumed to fit a 3.33× larger host
pool. Exact sizing requires the current startup lines for every rank, including
`Allocating ... host memory`, plus the non-HiCache Pod baseline. The target
must include runtime RSS, both rank pools, shared-memory use, transfer buffers,
and safety headroom. Four prefill replicas are placed on the prefill role node,
so node-level RAM must cover the sum of all four Pod requests rather than only
one enlarged cache.

The manifest also anchors one resource block and reuses it for prefill and
decode. Raising that shared memory request would reserve the same large amount
for decode even though decode offload is disabled. A 10-million-per-Pod test
should first split prefill and decode resource blocks, then enlarge only
prefill based on measured allocation.

### Write-policy inconsistency found

The runbook and checked-in offload manifest currently disagree:

- the offload manifest embedded in
  [`README.md`](models/qwen3.6-35B-A3B/sglang/disagg/tp1-ep2-4p4d/README.md)
  uses `--hicache-write-policy write_back`;
- [`deploy-kv-offloading.yaml`](models/qwen3.6-35B-A3B/sglang/disagg/tp1-ep2-4p4d/deploy-kv-offloading.yaml)
  uses `--hicache-write-policy write_through`.

The policy does not change `hicache_host_total_tokens`, but it changes when
`used_tokens` grows and how much copy traffic is added. `write_through` places
new reusable pages into L2 promptly, while `write_back` delays the L2 write
until upper-tier eviction. The completed prefill-heavy notes describe a
`write_through` run, which matches the checked-in standalone deployment file,
but the generated runbook manifest must be reconciled before the next A/B run.
Neither file was changed during the capacity-analysis conversation.

### Next actions from these conversations

1. Decide whether “10 million” means fleet-wide raw capacity or capacity on
   every prefill Pod. Query both `sum by (pod)` and a fleet-wide `sum` before
   changing the ratio.
2. Reconcile the README and checked-in deployment to one intentional HiCache
   write policy and record which exact manifest produced each benchmark.
3. Capture per-rank host-allocation startup logs, current Pod memory, node
   allocatable memory, and current total/used-token metrics.
4. If 10 million per Pod is still required, split prefill/decode resource
   blocks and test ratios `2.0`, `3.0`, then `3.9` or `4.0` in the single
   prefill/decode canary before scaling to four prefill replicas.
5. At every stage retain cache capacity/use, OOM status, CPU utilization,
   CPU-to-GPU reload behavior, NIXL transfer health, TTFT, and throughput.
6. Restore the standalone canary runbook if it is still wanted as a retained
   repository artifact; it is absent from the current workspace.
