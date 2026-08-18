# Two-node H100 Dynamo cluster setup

We started with plain Ubuntu bare-metal servers. This guide explains how the
two GPU nodes, `gpu05` and `gpu06`, were configured as a Kubernetes cluster for
NVIDIA Dynamo.

This document covers only the base cluster: Kubernetes, Calico, Helm, NVIDIA
GPU and Network Operators, and the Dynamo platform.

## 1. Cluster layout

| Item | `gpu05` | `gpu06` |
| --- | --- | --- |
| Kubernetes name | `inst-1onle-devrel-rdma-pool` | `inst-g9dwj-devrel-rdma-pool` |
| Private IP | `10.18.96.143` | `10.18.96.236` |
| Role | control plane and GPU worker | GPU worker |
| GPUs | 8 x NVIDIA H100 80 GB | 8 x NVIDIA H100 80 GB |

Pinned software:

| Component | Version |
| --- | --- |
| Kubernetes | v1.35 minor release |
| Calico | v3.32.1 |
| NVIDIA Network Operator | v26.4.0 |
| NVIDIA GPU Operator | v26.3.3 |
| NVIDIA Dynamo platform | 1.3.0 |

Command labels:

| Label | Where to run |
| --- | --- |
| **Both nodes** | Run once on `gpu05` and once on `gpu06`. |
| **`gpu05`** | Run in the host shell on `gpu05`. |
| **`gpu06`** | Run in the host shell on `gpu06`. |
| **Admin** | Run on `gpu05` with this cluster's `kubectl` and `helm` context. |

## 2. Validate both hosts

**Both nodes:**

```bash
hostname
ip -br -4 address
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
nvidia-smi topo -m
free -h
df -h /ephemeral /ephemeral/shared
findmnt -T /ephemeral/shared
```

Required result:

- each host has eight H100 GPUs;
- the NVIDIA host driver is already working;
- both hosts can see the same `/ephemeral/shared` filesystem;
- the private node addresses are `10.18.96.143` and `10.18.96.236`.

Check private connectivity:

**`gpu05`:**

```bash
ip route get 10.18.96.236
ping -c 4 10.18.96.236
```

**`gpu06`:**

```bash
ip route get 10.18.96.143
ping -c 4 10.18.96.143
```

Stop if either node cannot reach the other private IP.

## 3. Install Kubernetes

Skip this section if both nodes already belong to a healthy Kubernetes
cluster. Do not overwrite a site-managed container runtime, CNI, or kubelet
configuration.

### 3.1 Prepare both nodes

The Pod CIDR must not overlap any host, fabric, VPN, or site route.

**Both nodes:**

```bash
ip route show 192.168.0.0/16
```

If this prints a conflicting route, select an approved non-overlapping CIDR
and use it consistently in kubeadm and the CNI configuration.

Disable swap and configure the required kernel modules:

**Both nodes:**

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
`/etc/fstab` so swap remains off after reboot.

Install containerd on fresh hosts:

**Both nodes:**

```bash
sudo apt-get update
sudo apt-get install -y containerd ca-certificates curl gpg jq

sudo install -d -m 0755 /etc/containerd
containerd config default | sudo tee /etc/containerd/config.toml >/dev/null
sudo sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' \
  /etc/containerd/config.toml

sudo systemctl enable --now containerd
sudo systemctl restart containerd
```

If containerd already has site configuration, merge `SystemdCgroup = true`
instead of replacing the file.

Install Kubernetes from the v1.35 repository:

**Both nodes:**

```bash
sudo install -d -m 0755 /etc/apt/keyrings

curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.35/deb/Release.key \
  | sudo gpg --dearmor --yes \
      -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg

echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.35/deb/ /' \
  | sudo tee /etc/apt/sources.list.d/kubernetes.list >/dev/null

sudo apt-get update
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl
sudo systemctl enable kubelet
```

### 3.2 Initialize `gpu05`

**`gpu05`:**

```bash
sudo kubeadm init \
  --apiserver-advertise-address=10.18.96.143 \
  --pod-network-cidr=192.168.0.0/16 \
  --cri-socket=unix:///run/containerd/containerd.sock
```

Configure kubectl for the current non-root administrator:

```bash
mkdir -p "$HOME/.kube"
sudo cp -i /etc/kubernetes/admin.conf "$HOME/.kube/config"
sudo chown "$(id -u):$(id -g)" "$HOME/.kube/config"
chmod 0600 "$HOME/.kube/config"
```

### 3.3 Install Calico

**Admin:**

```bash
kubectl apply -f \
  https://raw.githubusercontent.com/projectcalico/calico/v3.32.1/manifests/calico.yaml

kubectl set env daemonset/calico-node -n kube-system \
  IP_AUTODETECTION_METHOD=kubernetes-internal-ip

kubectl rollout status daemonset/calico-node \
  -n kube-system --timeout=10m
kubectl wait --for=condition=Ready pod --all \
  -n kube-system --timeout=10m
```

### 3.4 Join `gpu06`

Generate a short-lived join command. Treat it as a credential and do not save
it in the repository or shared filesystem.

**Admin:**

```bash
sudo kubeadm token create --ttl 30m --print-join-command
```

Run the printed command privately on `gpu06`, prefix it with `sudo`, and
append:

```text
--cri-socket=unix:///run/containerd/containerd.sock
```

Wait for the actual provider node names, not `gpu05` and `gpu06`:

**Admin:**

```bash
kubectl wait --for=condition=Ready \
  node/inst-1onle-devrel-rdma-pool \
  node/inst-g9dwj-devrel-rdma-pool \
  --timeout=10m

kubectl get nodes -o wide
kubectl get nodes.crd.projectcalico.org \
  -o custom-columns=NAME:.metadata.name,CALICO-IP:.spec.bgp.ipv4Address
```

Calico must show `10.18.96.143` and `10.18.96.236`.

### 3.5 Make the control-plane GPU schedulable

This is a dedicated GPU experiment cluster, so `gpu05` must run GPU Operator
DaemonSets and model workloads. Remove the default control-plane taint:

**Admin:**

```bash
kubectl taint node inst-1onle-devrel-rdma-pool \
  node-role.kubernetes.io/control-plane:NoSchedule- || true
```

Do not remove this taint from a general-purpose production control plane.

## 4. Install Helm

Check whether Helm 3 is already installed:

**Admin:**

```bash
helm version
```

If it is missing, download and inspect the official installer before running
it:

```bash
curl -fsSL -o /tmp/get-helm-3.sh \
  https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
less /tmp/get-helm-3.sh
chmod 0700 /tmp/get-helm-3.sh
/tmp/get-helm-3.sh
helm version
```

## 5. Install or upgrade NVIDIA operators

Network Operator owns Node Feature Discovery. GPU Operator reuses the NVIDIA
driver already installed on both hosts.

**Admin:**

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

Warnings about the deprecated `node-role.kubernetes.io/master` affinity key
come from chart templates and do not by themselves mean the installation
failed.

Validate the operators:

```bash
kubectl get pods -n nvidia-network-operator -o wide
kubectl get pods -n gpu-operator -o wide

kubectl get clusterpolicy cluster-policy \
  -o jsonpath='{.status.state}{"\n"}'

kubectl get nodes \
  -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
```

Required result: GPU Operator is ready and both nodes advertise eight GPUs.
The Network Operator is installed here; its RoCE and RDMA resources are
configured in the separate networking guide.

## 6. Install or upgrade Dynamo 1.3.0

Use the verified NGC chart tarball. The previously attempted OCI reference
does not contain the Dynamo 1.3.0 platform chart.

**Admin:**

```bash
export DYNAMO_CHART_URL=https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-1.3.0.tgz

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

This setup uses Kubernetes-native discovery, so separate NATS and etcd
services are not required.

Validate the release:

```bash
helm status dynamo-platform -n dynamo-system
kubectl get pods -n dynamo-system -o wide

kubectl rollout status deployment/queue-controller \
  -n dynamo-system --timeout=10m

kubectl get crd | grep -E \
  'dynamographdeployments|dynamocomponentdeployments'

kubectl get crd dynamographdeployments.nvidia.com \
  -o jsonpath='{.spec.versions[*].name}{"\n"}'
```

If the KAI validation webhook times out during the first Helm run, wait for
`queue-controller` and its Service endpoints to become ready, then rerun the
same `helm upgrade --install` command. Do not create a duplicate Queue.

## 7. Final cluster check

**Admin:**

```bash
kubectl get nodes -o wide
kubectl get nodes \
  -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'

kubectl get pods -n kube-system -o wide
kubectl get pods -n nvidia-network-operator -o wide
kubectl get pods -n gpu-operator -o wide
kubectl get pods -n dynamo-system -o wide

helm list -A
```

The base cluster is ready when:

- both Kubernetes nodes are `Ready` with their correct internal IPs;
- Calico is healthy on both nodes;
- both nodes advertise eight GPUs;
- Network Operator and GPU Operator Pods are healthy;
- the Dynamo Helm release is deployed;
- Dynamo, Grove, and KAI Scheduler Pods are healthy;
- the DynamoGraphDeployment CRD is available.

Continue with the separate RoCE/RDMA guide before deploying a disaggregated
model. A normal reboot does not require rebuilding the cluster: wait for both
nodes to return to `Ready`, then repeat this final check.

## References

- [Kubernetes v1.35 kubeadm installation](https://v1-35.docs.kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/)
- [NVIDIA Network Operator](https://docs.nvidia.com/networking/display/kubernetes2640/getting-started-kubernetes.html)
- [NVIDIA Dynamo Kubernetes deployment](https://docs.nvidia.com/dynamo/latest/kubernetes-deployment/deploy-models/model-deployment-guide)
