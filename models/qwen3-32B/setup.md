# Qwen3-32B FP8: two-node cluster setup

This runbook prepares `gpu05` and `gpu06` for the Qwen3-32B-FP8 Dynamo
experiments. It stops at cluster readiness; deployment and benchmark commands
live in the [experiment matrix](experiments/README.md).

Run only one 16-GPU experiment at a time. The eight maintained variants cover
the same four topologies in vLLM and SGLang.

## Command location legend

Every command block in this guide has an execution label immediately above it:

| Label | Meaning |
|---|---|
| **Run on both `gpu05` and `gpu06`** | Open a host shell on each node and run the block separately on both. |
| **Run on `gpu05` only** | Run in a host shell on the prefill/control-plane node. |
| **Run on `gpu06` only** | Run in a host shell on the decode/worker node. |
| **Run on `gpu05` — Kubernetes admin terminal** | Run where the `gpu05` administrator's `kubectl` and `helm` use this cluster's kubeconfig. |

Do not assume that a command applies to both nodes merely because it appears in
a cluster-wide section. Follow the label directly above the block.

## 1. Fixed design

| Item | Setting |
|---|---|
| Prefill node | logical `gpu05`; Kubernetes `inst-1onle-devrel-rdma-pool`; `10.18.96.143`; 8 x H100 80 GB |
| Decode node | logical `gpu06`; Kubernetes `inst-g9dwj-devrel-rdma-pool`; `10.18.96.236`; 8 x H100 80 GB |
| Experiment topology | Defined by the selected experiment manifest |
| Known-good recovery layout | 1 prefill TP=8 + 1 decode TP=8 |
| Model | `Qwen/Qwen3-32B-FP8` |
| Model revision | `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df` |
| Runtime | `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0` |
| Dynamo platform chart | `1.3.0` |
| SGLang in the runtime | `0.5.14` |
| P/D transfer | NIXL over UCX/RDMA |
| Kubernetes namespace | `qwen32-bench` |
| Frontend port | `8000`, private access only |
| Worker page size | 64 tokens in every experiment |
| Maximum model length | 40,960 tokens; no YaRN |
| Readiness-request mode | Non-thinking via `/no_think` |

The experiment manifests use TP=2 workers except for the TP=4 decode worker in
the eight-GPU disaggregation baseline. When roles are pinned to separate nodes,
every P-to-D KV transfer crosses the physical node boundary. The eight-GPU
baseline uses the validated `hostNetwork: true`, `UCX_NET_DEVICES=mlx5_8:1`,
and `UCX_IB_GID_INDEX=3` path. Scaled manifests use pod network isolation to
avoid host-port collisions while retaining RDMA device access.

Dynamo Operator v1.3.0 defaults worker ports to `9090` and `19090`. Replicas of
one component share one pod template, so `replicas: 4` caused host-port
collisions. Experiment 1 instead declares four one-replica components per role
and gives each component unique system, NIXL, forward-pass-metrics, NCCL, and
bootstrap ports. The previously proven 1P/1D TP=8 layout remains the recovery
configuration if the per-component overrides are not preserved by the running
operator.

This is an engineering deployment for this cluster, not a claim that NVIDIA
has published performance results for this exact Qwen + SGLang + 4P/4D TP=2
layout.
The Qwen model card supports this FP8 checkpoint with SGLang, while the
deployment details here are adapted from Dynamo's pinned SGLang interfaces.

The readiness requests deliberately use Qwen's `/no_think` switch. These
guides do not validate separation of `reasoning_content`. If reasoning-mode API
output is required later, add both `--reasoning-parser qwen3` and
`--dyn-reasoning-parser qwen3` identically to prefill and decode in all three
manifests before comparing them.

## 2. Before changing either host

Use the private `10.18.96.x` network for Kubernetes and inference traffic.
Never expose the frontend, NATS, etcd, the Kubernetes API, or metrics endpoints
to the public Internet.

Do not put SSH keys, Hugging Face tokens, kubeadm join tokens, kubeconfigs, or
other credentials in this repository. Use an SSH agent for access. The model
is public, so a Hugging Face token is optional; when used, create it
interactively as shown below.

The Kubernetes bootstrap section is only for fresh or disposable hosts. If a
working Kubernetes cluster already owns these nodes, skip section 4. Inspect
and reuse its GPU/RDMA operators in section 5, then install or validate Dynamo
in section 6. Jump directly to section 7 only when all of those prerequisites
already exist. Do not overwrite site-managed containerd, CNI, GPU Operator, or
Network Operator settings.

**Run on both `gpu05` and `gpu06`:** confirm the identity and hardware in a
separate shell on each host.

```bash
hostname
ip -br -4 addr
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
nvidia-smi topo -m
free -h
df -h /ephemeral /ephemeral/shared
```

Required results:

- exactly eight H100 GPUs on each host;
- NVIDIA driver 580.xx or newer for the CUDA 13 runtime;
- enough system RAM for four TP=2 model workers plus Kubernetes services on
  each node;
- at least 100 GiB free on the shared filesystem;
- `/ephemeral/shared` is the same shared data on both hosts.

Verify the shared mount without placing credentials in it.

**Run on `gpu05` only:** create the marker on the shared mount.

```bash
mkdir -p /ephemeral/shared/qwen3-32b/manifests
date -u --iso-8601=seconds > /ephemeral/shared/qwen3-32b/storage-check
findmnt -T /ephemeral/shared
```

**Run on `gpu06` only:** confirm that the second node sees the marker.

```bash
test -r /ephemeral/shared/qwen3-32b/storage-check
sed -n '1p' /ephemeral/shared/qwen3-32b/storage-check
findmnt -T /ephemeral/shared
```

Stop if the second node does not see the same marker. The static Kubernetes PV
later in this guide is safe only because this path is already a shared mount on
both nodes.

## 3. Validate the private fabric and RDMA on both hosts

**Run on `gpu05` only:** test the route toward `gpu06`.

```bash
ip route get 10.18.96.236
ping -c 4 10.18.96.236
```

**Run on `gpu06` only:** test the route toward `gpu05`.

```bash
ip route get 10.18.96.143
ping -c 4 10.18.96.143
```

**Run on both `gpu05` and `gpu06`:** inspect the local RDMA devices on each
host.

```bash
ibstat
ibv_devinfo -l
ibdev2netdev
rdma link
ls -l /dev/infiniband
lsmod | grep -E 'nvidia_peermem|nv_peer_mem'
```

Every chosen HCA port must be `Active`, and `nvidia_peermem` should be loaded.
If it is installed but not loaded:

**Run on both `gpu05` and `gpu06`, but only when the module is missing:**

```bash
sudo modprobe nvidia_peermem
```

This setup installs NVIDIA's **RDMA Shared Device Plugin**. Its DaemonSet must
be able to share the host RDMA devices with pod network namespaces. On these
hosts that requires `ib_core.netns_mode=1` at boot. `hostNetwork: true` on the
model workers does not remove this requirement from the device-plugin
DaemonSet itself. If the mode is exclusive, the plugin crashes and Kubernetes
reports `rdma/ib=0` even while the physical HCAs remain active.

Configure shared mode one node at a time, beginning with gpu06. Do not try to
change the mode live with `rdma system set` while the devices are in use.

**Run on both `gpu05` and `gpu06`, one node at a time:**

```bash
sudo tee /etc/modprobe.d/99-rdma-shared-netns.conf >/dev/null <<'EOF'
options ib_core netns_mode=1
EOF

sudo tee /etc/default/grub.d/99-rdma-shared-netns.cfg >/dev/null <<'EOF'
GRUB_CMDLINE_LINUX_DEFAULT="${GRUB_CMDLINE_LINUX_DEFAULT} ib_core.netns_mode=1"
EOF

sudo update-initramfs -u
sudo update-grub

grep --line-number 'ib_core\.netns_mode\|netns_mode=1' \
  /etc/default/grub 2>/dev/null || true
grep -R --line-number --include='*.conf' --include='*.cfg' \
  'ib_core\.netns_mode\|netns_mode=1' \
  /etc/modprobe.d /etc/default/grub.d 2>/dev/null || true
```

The final `grep` must show the two active shared-mode settings above. Reboot
gpu06, wait for it to become Kubernetes Ready, and verify it before repeating
the operation on gpu05:

```bash
cat /proc/cmdline
cat /sys/module/ib_core/parameters/netns_mode
rdma system show
```

The required result is that `/proc/cmdline` contains `ib_core.netns_mode=1`,
the sysfs value is `Y`, and `rdma system show` reports `netns shared`. After
both nodes pass, restart `rdma-shared-dp-ds` and require a positive allocatable
`rdma/ib` value on each node before deploying a model.

Do not hard-code `mlx5_0:1` from another machine. UCX is left to choose among
the active devices initially. After the first deployment, the experiment guide
requires confirmation that NIXL instantiated UCX and transferred data over the
RDMA path. If UCX selects the wrong HCA, set `UCX_NET_DEVICES` separately in the
prefill and decode component manifests using the active, non-bonded HCA and
port found above.

## 4. Bootstrap Kubernetes on fresh hosts

These commands pin the Kubernetes minor release to 1.35 and use containerd
with the systemd cgroup driver. Run this subsection on both hosts.

The chosen pod CIDR must not overlap a host, fabric, VPN, or site route. This
command should produce no conflicting route on either node:

**Run on both `gpu05` and `gpu06`:**

```bash
ip route show 192.168.0.0/16
```

If it conflicts, choose an approved non-overlapping CIDR and use that same
CIDR in kubeadm and the CNI configuration.

### 4.1 Host prerequisites

**Run on both `gpu05` and `gpu06`:** configure the kernel prerequisites on
each fresh host.

```bash
sudo swapoff -a
swapon --show

sudo modprobe overlay
sudo modprobe br_netfilter

sudo tee /etc/modules-load.d/kubernetes.conf >/dev/null <<'EOF'
overlay
br_netfilter
EOF

sudo tee /etc/sysctl.d/99-kubernetes-cri.conf >/dev/null <<'EOF'
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward = 1
EOF

sudo sysctl --system
```

`swapon --show` must be empty. Disable the corresponding swap entry in
`/etc/fstab` before rebooting; do not blindly delete unrelated entries.

Install and configure containerd only on a fresh host. If
`/etc/containerd/config.toml` already contains site configuration, have the
cluster administrator merge `SystemdCgroup = true` instead of replacing it.

**Run on both `gpu05` and `gpu06`, fresh hosts only:**

```bash
sudo apt-get update
sudo apt-get install -y containerd ca-certificates curl gpg
sudo install -d -m 0755 /etc/containerd
containerd config default | sudo tee /etc/containerd/config.toml >/dev/null
sudo sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml
sudo systemctl enable --now containerd
sudo systemctl restart containerd
```

Install kubelet, kubeadm, and kubectl from the pinned Kubernetes repository:

**Run on both `gpu05` and `gpu06`:**

```bash
sudo install -d -m 0755 /etc/apt/keyrings
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.35/deb/Release.key \
  | sudo gpg --dearmor --yes -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg

echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.35/deb/ /' \
  | sudo tee /etc/apt/sources.list.d/kubernetes.list >/dev/null

sudo apt-get update
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl
sudo systemctl enable kubelet
```

### 4.2 Initialize `gpu05`

**Run on `gpu05` only:** initialize the Kubernetes control plane.

```bash
sudo kubeadm init \
  --apiserver-advertise-address=10.18.96.143 \
  --pod-network-cidr=192.168.0.0/16 \
  --cri-socket=unix:///run/containerd/containerd.sock
```

Configure kubectl for the non-root administrator account that ran the command:

**Run on `gpu05` only:**

```bash
mkdir -p "$HOME/.kube"
sudo cp -i /etc/kubernetes/admin.conf "$HOME/.kube/config"
sudo chown "$(id -u):$(id -g)" "$HOME/.kube/config"
chmod 0600 "$HOME/.kube/config"
```

Install the pinned Calico CNI:

**Run on `gpu05` — Kubernetes admin terminal:**

```bash
kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.32.1/manifests/calico.yaml

# These hosts have multiple network interfaces. Force Calico to use the same
# private addresses Kubernetes reports for inter-node pod routing.
kubectl set env daemonset/calico-node -n kube-system \
  IP_AUTODETECTION_METHOD=kubernetes-internal-ip

kubectl rollout status daemonset/calico-node -n kube-system --timeout=10m
kubectl wait --for=condition=Ready pod --all -n kube-system --timeout=10m
```

Verify that the Calico node addresses match the Kubernetes `INTERNAL-IP`
addresses (`10.18.96.143` and `10.18.96.236`). Do not continue if Calico has
selected another interface address.

```bash
kubectl get nodes -o wide
kubectl get nodes.crd.projectcalico.org \
  -o custom-columns=NAME:.metadata.name,CALICO-IP:.spec.bgp.ipv4Address
```

Create a short-lived worker join command. Treat its token as a credential and
run the printed command directly in a private terminal on `gpu06`; do not save
it in this repository:

**Run on `gpu05` — Kubernetes admin terminal:** generate the command.

```bash
sudo kubeadm token create --ttl 30m --print-join-command
```

Prefix the printed command with `sudo` and append this CRI socket when running
it on `gpu06`:

**Run on `gpu06` only:** paste the private join command printed by `gpu05`,
prefix it with `sudo`, and append the following option.

```text
--cri-socket=unix:///run/containerd/containerd.sock
```

**Run on `gpu05` — Kubernetes admin terminal:** wait for both nodes after the
`gpu06` join command finishes.

```bash
kubectl get nodes -o wide
kubectl wait --for=condition=Ready \
  node/inst-1onle-devrel-rdma-pool \
  node/inst-g9dwj-devrel-rdma-pool \
  --timeout=10m
```

If the registered node names differ, use the names reported by `kubectl get
nodes` in all subsequent `PREFILL_NODE` and `DECODE_NODE` commands.

## 5. Install Helm and the NVIDIA operators

Use the Kubernetes administrator terminal on `gpu05` for this section. Helm 3
is required.

**Run on `gpu05` — Kubernetes admin terminal:** check whether Helm is already
installed.

```bash
helm version
```

If the preceding command reports that Helm is missing, download and inspect the
official installer before executing it.

**Run on `gpu05` — Kubernetes admin terminal, only when Helm is missing:**

```bash
curl -fsSL -o /tmp/get-helm-3.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
less /tmp/get-helm-3.sh
chmod 0700 /tmp/get-helm-3.sh
/tmp/get-helm-3.sh
helm version
```

The next sequence lets Network Operator own Node Feature Discovery. GPU
Operator is told not to deploy a second NFD instance and not to replace the
already working host driver.

**Run on `gpu05` — Kubernetes admin terminal:** install both NVIDIA operators.

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm upgrade --install network-operator nvidia/network-operator \
  --namespace nvidia-network-operator \
  --create-namespace \
  --version v26.4.0 \
  --set nfd.enabled=true \
  --wait \
  --timeout 15m

helm upgrade --install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --version v26.3.3 \
  --set driver.enabled=false \
  --set nfd.enabled=false \
  --wait \
  --timeout 20m
```

This cluster already has host RDMA drivers, so the `NicClusterPolicy` below
does not install or replace OFED. It publishes shared HCA access as the
Kubernetes extended resource `rdma/ib`. Save as
`/ephemeral/shared/qwen3-32b/manifests/rdma-policy.yaml`:

**Run on `gpu05` only:** create the file on the shared path; do not create a
second copy on `gpu06`.

```bash
tee /ephemeral/shared/qwen3-32b/manifests/rdma-policy.yaml >/dev/null <<'EOF'
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
              "vendors": ["15b3"]
            }
          }
        ]
      }
EOF
```

**Run on `gpu05` — Kubernetes admin terminal:** apply and verify the RDMA
policy.

```bash
kubectl apply -f /ephemeral/shared/qwen3-32b/manifests/rdma-policy.yaml
kubectl get nicclusterpolicy nic-cluster-policy -o yaml
kubectl get pods -n nvidia-network-operator -o wide
kubectl get pods -n gpu-operator -o wide
```

Set the actual Kubernetes node names, label their roles, and inspect the
allocatable resources:

**Run on `gpu05` — Kubernetes admin terminal:**

```bash
export PREFILL_NODE=inst-1onle-devrel-rdma-pool
export DECODE_NODE=inst-g9dwj-devrel-rdma-pool

kubectl label node "$PREFILL_NODE" qwen.nvidia.com/role=prefill --overwrite
kubectl label node "$DECODE_NODE" qwen.nvidia.com/role=decode --overwrite

kubectl get nodes -L qwen.nvidia.com/role -o wide
kubectl describe node "$PREFILL_NODE" | grep -E 'nvidia.com/gpu:|rdma/ib:'
kubectl describe node "$DECODE_NODE" | grep -E 'nvidia.com/gpu:|rdma/ib:'
```

Stop unless each node exposes at least `nvidia.com/gpu: 8` and at least eight
allocatable `rdma/ib` shares. The TP=2 manifest requests two RDMA allocations
per worker, for a total of eight allocations across four workers on each node.
If the installed plugin
exposes a different resource name, change all experiment manifests
consistently; never request two different RDMA resource types for the same pod.

## 6. Install Dynamo 1.3.0

Install the cluster-wide Dynamo platform. The bundled Grove and KAI Scheduler
are enabled for predictable placement of the disaggregated GPU components on
this dedicated experiment cluster.

**Run on `gpu05` — Kubernetes admin terminal:**

```bash
export DYNAMO_CHART_URL=https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-1.3.0.tgz

# Confirm that the pinned chart is reachable before changing the cluster.
helm show chart "$DYNAMO_CHART_URL"

helm upgrade --install dynamo-platform \
  "$DYNAMO_CHART_URL" \
  --namespace dynamo-system \
  --create-namespace \
  --set global.nats.install=false \
  --set global.grove.install=true \
  --set global.kai-scheduler.install=true \
  --wait \
  --timeout 20m
```

These deployments use Kubernetes-native discovery, the TCP request plane, and
worker-to-router ZMQ KV events. NATS and etcd are therefore disabled and are
not required. Do not start separate Docker NATS or etcd instances for these
experiments.

Verify the platform and the API version used by the maintained manifests:

**Run on `gpu05` — Kubernetes admin terminal:**

```bash
kubectl get pods -n dynamo-system -o wide
kubectl get crd | grep -E 'dynamographdeployments|dynamocomponentdeployments'
kubectl get crd dynamographdeployments.nvidia.com \
  -o jsonpath='{.spec.versions[*].name}{"\n"}'
kubectl explain dynamographdeployment.spec.services \
  --api-version=nvidia.com/v1alpha1
```

Every platform pod must be `Running` or successfully `Completed`, and
`kubectl explain` must succeed before continuing.

## 7. Create the experiment namespace

**Run on `gpu05` — Kubernetes admin terminal:** create or reuse the namespace.

```bash
export NAMESPACE=qwen32-bench
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
```

The worker containers require `IPC_LOCK` and `SYS_RESOURCE`. If this cluster
enforces Pod Security Admission, use this dedicated namespace only and have the
cluster administrator approve its policy. For a private disposable experiment
cluster, the required label is:

**Run on `gpu05` — Kubernetes admin terminal, only after approving the policy
relaxation:**

```bash
kubectl label namespace "$NAMESPACE" \
  pod-security.kubernetes.io/enforce=privileged \
  pod-security.kubernetes.io/audit=baseline \
  pod-security.kubernetes.io/warn=baseline \
  --overwrite
```

Do not apply that policy relaxation to a shared or unrelated namespace.

## 8. Expose the existing shared model cache as a PVC

The following static PV is intentionally specific to this two-node lab. It
maps the already-shared `/ephemeral/shared/huggingface` mount into Kubernetes.
For a production cluster, replace it with a real RWX CSI/NFS storage class.

**Run on `gpu05` only:** prepare the underlying shared directory.

```bash
mkdir -p /ephemeral/shared/huggingface
chmod 0770 /ephemeral/shared/huggingface
df -h /ephemeral/shared/huggingface
```

Save as `/ephemeral/shared/qwen3-32b/manifests/model-cache-pv.yaml`:

**Run on `gpu05` only:** create the file on the shared filesystem.

```bash
tee /ephemeral/shared/qwen3-32b/manifests/model-cache-pv.yaml >/dev/null <<'EOF'
apiVersion: v1
kind: PersistentVolume
metadata:
  name: qwen32-model-cache-pv
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: qwen-shared-manual
  volumeMode: Filesystem
  hostPath:
    path: /ephemeral/shared/huggingface
    type: Directory
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
  namespace: qwen32-bench
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: qwen-shared-manual
  volumeName: qwen32-model-cache-pv
EOF
```

**Run on `gpu05` — Kubernetes admin terminal:** create and verify the PV/PVC.

```bash
kubectl apply -f /ephemeral/shared/qwen3-32b/manifests/model-cache-pv.yaml
kubectl wait --for=jsonpath='{.status.phase}'=Bound \
  pvc/model-cache -n "$NAMESPACE" --timeout=2m
kubectl get pv qwen32-model-cache-pv
kubectl get pvc model-cache -n "$NAMESPACE"
```

## 9. Optionally add Hugging Face authentication

`Qwen/Qwen3-32B-FP8` is public. A read token is optional but can avoid anonymous
download limits. The deployment manifests mark this Secret optional.

If using a token, enter it without placing it in shell history:

**Run on `gpu05` — Kubernetes admin terminal:** the token is written directly
to a Kubernetes Secret and is never saved in the Markdown or manifest files.

```bash
read -rsp 'Hugging Face read token: ' HF_TOKEN_INPUT
echo
printf '%s' "$HF_TOKEN_INPUT" \
  | kubectl create secret generic hf-token-secret \
      --namespace "$NAMESPACE" \
      --from-file=HF_TOKEN=/dev/stdin \
      --dry-run=client -o yaml \
  | kubectl apply -f -
unset HF_TOKEN_INPUT
```

Never run `kubectl get secret ... -o yaml` or dump a pod's complete environment
for diagnostics.

## 10. Download the exact model revision once

Save as `/ephemeral/shared/qwen3-32b/manifests/model-download.yaml`:

**Run on `gpu05` only:** create the file on the shared filesystem.

```bash
tee /ephemeral/shared/qwen3-32b/manifests/model-download.yaml >/dev/null <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: qwen32-model-download
  namespace: qwen32-bench
spec:
  backoffLimit: 3
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        qwen.nvidia.com/role: prefill
      tolerations:
        - key: node-role.kubernetes.io/control-plane
          operator: Exists
          effect: NoSchedule
        - key: nvidia.com/gpu
          operator: Equal
          value: "true"
          effect: NoSchedule
      containers:
        - name: download
          image: python:3.10-slim
          command:
            - /bin/sh
            - -c
          args:
            - |
              set -eu
              pip install --no-cache-dir huggingface_hub==1.16.4
              hf download "$MODEL_NAME" --revision "$MODEL_REVISION"
          env:
            - name: MODEL_NAME
              value: Qwen/Qwen3-32B-FP8
            - name: MODEL_REVISION
              value: aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df
            - name: HF_HOME
              value: /model-store
            - name: HF_XET_HIGH_PERFORMANCE
              value: "1"
          envFrom:
            - secretRef:
                name: hf-token-secret
                optional: true
          resources:
            requests:
              cpu: "2"
              memory: 64Gi
            limits:
              cpu: "8"
              memory: 64Gi
          volumeMounts:
            - name: model-cache
              mountPath: /model-store
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache
EOF
```

Apply it and wait. The first download can take longer than 30 minutes on a
slow link:

**Run on `gpu05` — Kubernetes admin terminal:**

```bash
kubectl delete job qwen32-model-download -n "$NAMESPACE" --ignore-not-found
kubectl apply -f /ephemeral/shared/qwen3-32b/manifests/model-download.yaml
kubectl logs -f job/qwen32-model-download -n "$NAMESPACE"
kubectl wait --for=condition=Complete job/qwen32-model-download \
  -n "$NAMESPACE" --timeout=60m
```

Confirm the exact snapshot exists on both hosts without displaying tokens:

**Run on both `gpu05` and `gpu06`:** execute the same `test` command separately
on each host.

```bash
test -d /ephemeral/shared/huggingface/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df
```

Run that `test` command on both `gpu05` and `gpu06`. Both must exit with status
zero. The experiment manifests load this exact local snapshot path rather than
resolving a mutable Hub branch at worker startup. They still expose
`Qwen/Qwen3-32B-FP8` as the OpenAI-compatible served-model name.

## 11. Final cluster gate

**Run on `gpu05` — Kubernetes admin terminal:** verify Kubernetes-side state.

```bash
export NAMESPACE=qwen32-bench

kubectl get nodes -L qwen.nvidia.com/role -o wide
kubectl get nodes \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\tGPU="}{.status.allocatable.nvidia\.com/gpu}{"\tRDMA="}{.status.allocatable.rdma/ib}{"\n"}{end}'
kubectl get pods -A -o wide
kubectl get pvc -n "$NAMESPACE"
kubectl get job qwen32-model-download -n "$NAMESPACE"
kubectl get queues.scheduling.run.ai
kubectl get dynamographdeployments -n "$NAMESPACE"
```

**Run on both `gpu05` and `gpu06`:** verify local hardware and shared storage
from a separate host shell on each node.

```bash
nvidia-smi -L
ibstat
rdma link
free -h
df -h /ephemeral/shared
test -d /ephemeral/shared/huggingface/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df \
  && echo 'Pinned model snapshot: present' \
  || echo 'Pinned model snapshot: MISSING'
```

Do not start an experiment until all of the following are true:

- both Kubernetes nodes are `Ready`;
- each node advertises eight GPUs and at least eight `rdma/ib` shares;
- `gpu05` has `qwen.nvidia.com/role=prefill`;
- `gpu06` has `qwen.nvidia.com/role=decode`;
- Dynamo operator, Grove, and KAI pods are healthy;
- `model-cache` is `Bound` and the pinned model snapshot is present on both
  hosts;
- no other workload is using any of the 16 GPUs;
- no previous Qwen experiment DGD remains in `qwen32-bench`.

## 12. Run the experiments

Use the [experiment matrix](experiments/README.md) to select a backend and
topology. Each folder contains one `deploy.yaml` and a short operator runbook.
Always delete the active DGD and wait for its pods to terminate before applying
the next 16-GPU experiment. Keep model revision, input/output lengths, warmup,
duration, and concurrency identical for framework comparisons.

## References

- [Dynamo v1.3.0 SGLang disaggregation example](https://github.com/ai-dynamo/dynamo/blob/v1.3.0/examples/backends/sglang/deploy/v1beta1/disagg.yaml)
- [Dynamo disaggregated communication guide](https://github.com/ai-dynamo/dynamo/blob/v1.3.0/docs/kubernetes/disagg-communication-guide.md)
- [Dynamo KV-aware routing](https://github.com/ai-dynamo/dynamo/blob/v1.3.0/docs/components/router/router-guide.md)
- [Dynamo and SGLang HiCache](https://github.com/ai-dynamo/dynamo/blob/v1.3.0/docs/backends/sglang/sglang-hicache.md)
- [SGLang v0.5.14 HiCache practices](https://github.com/sgl-project/sglang/blob/v0.5.14/docs/advanced_features/hicache_best_practices.md)
- [Qwen3-32B-FP8 model card](https://huggingface.co/Qwen/Qwen3-32B-FP8)
- [Dynamo release compatibility](https://docs.nvidia.com/dynamo/dev/reference/compatibility)
- [NVIDIA Network Operator 26.4.0](https://docs.nvidia.com/networking/display/kubernetes2640/index.html)
- [NVIDIA GPU Operator installation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html)
