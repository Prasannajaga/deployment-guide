# NVIDIA Dynamo v1.3.0 on two 8xH100 nodes

This guide deploys NVIDIA Dynamo with the vLLM backend on the current
reservation:

| Role | Host | Private/fabric IP | GPUs |
|---|---|---:|---:|
| Head | `gpu05` | `10.18.96.143` | 8x H100 |
| Worker | `gpu06` | `10.18.96.236` | 8x H100 |

The commands intentionally pin Dynamo and its runtime image to `v1.3.0`.
NVIDIA has newer releases, but mixing a repository tag and a different image
tag is a common source of hard-to-reproduce failures.

This is a fixed-node Docker deployment suitable for validation and
benchmarking. NVIDIA recommends its Kubernetes operator for a production
multi-node service because Kubernetes adds gang scheduling, restart handling,
networking, and fault tolerance.

Official references:

- [Dynamo multi-node deployment](https://docs.nvidia.com/dynamo/dev/cli/model-deployment/multi-node-deployment)
- [Dynamo vLLM examples](https://docs.nvidia.com/dynamo/dev/recipes/cli-templates/v-llm)
- [Dynamo release artifacts](https://docs.nvidia.com/dynamo/dev/reference/release-artifacts)
- [v1.3.0 multi-node launch script](https://github.com/ai-dynamo/dynamo/blob/v1.3.0/examples/backends/vllm/launch/multi_node_tp.sh)

## 1. Pick the correct scaling topology

Use one of these topologies:

1. **Throughput scaling (recommended when the model fits on one node):** run
   one TP=8 model replica on each node. Dynamo round-robins requests between
   the two replicas. This normally gives much better throughput than stretching
   a small model across all 16 GPUs.
2. **Capacity scaling (this guide's main deployment):** run one TP=16 model
   across both nodes. Use this only when one model copy does not fit on eight
   GPUs. Every transformer layer performs cross-node collectives, so this
   requires a fast and correctly configured interconnect.

Do not evaluate performance with an 8B model at TP=16. The official Llama 8B
default is useful only as a connectivity test.

## 2. Open terminals without exposing credentials

Use an SSH agent or your provider's preconfigured SSH access. Do not put a
private key, Hugging Face token, or other credential in this repository,
shell history, command arguments, or this guide.

Keep at least four terminal sessions available:

- `gpu05`: infrastructure and head logs
- `gpu05`: diagnostics/client requests
- `gpu06`: worker and worker logs
- `gpu06`: diagnostics

Using `tmux` on each node is recommended so a dropped SSH connection does not
interrupt diagnostics.

Confirm the host in every terminal before continuing:

```bash
hostname
```

## 3. Verify both hosts

Run on **both** nodes:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvidia-smi -L
nvidia-smi topo -m
sudo docker --version
sudo docker compose version || true
sudo docker-compose --version 2>/dev/null || true
df -h /ephemeral /ephemeral/shared
```

Required results:

- Eight H100 GPUs on each node.
- Driver `580.00.03` or newer for the CUDA 13 v1.3.0 runtime.
- Docker can use the NVIDIA runtime. Docker Compose is optional; step 7 has a
  direct `docker run` path when the Compose plugin is unavailable.
- `/ephemeral/shared` is mounted on both nodes.

Test the exact Dynamo image on **both** nodes:

```bash
export DYNAMO_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
sudo docker pull "$DYNAMO_IMAGE"
sudo docker run --rm --gpus all --runtime nvidia "$DYNAMO_IMAGE" nvidia-smi -L
```

Stop here if the container does not see all eight GPUs. Fix the NVIDIA
Container Toolkit before debugging Dynamo.

## 4. Verify the private network and select its interface

Never use the public SSH addresses for Dynamo, torch.distributed, NCCL, etcd,
or NATS. Use the `10.18.96.x` network.

On `gpu05`:

```bash
ip -br -4 addr
ip route get 10.18.96.236
ping -c 4 10.18.96.236
```

On `gpu06`:

```bash
ip -br -4 addr
ip route get 10.18.96.143
ping -c 4 10.18.96.143
```

In the `ip route get` output, record the interface after `dev`. The same kind
of interface must carry the `10.18.96.x` traffic on both nodes. The examples
below call it `REPLACE_WITH_FABRIC_INTERFACE`; replace that placeholder before
running them.

Check whether RDMA/InfiniBand is available on both nodes:

```bash
ls -l /dev/infiniband 2>/dev/null || true
ls /sys/class/infiniband 2>/dev/null || true
ibv_devinfo -l 2>/dev/null || true
ibstat 2>/dev/null || true
rdma link 2>/dev/null || true
lsmod | grep -E 'nvidia_peermem|nv_peer_mem' || true
```

An active HCA plus `nvidia_peermem` is the desired GPUDirect RDMA path. The
deployment can fall back to TCP, but TP=16 performance may be poor.

## 5. Verify shared storage

On `gpu05`:

```bash
mkdir -p /ephemeral/shared/dynamo-logs
mkdir -p /ephemeral/shared/huggingface
sudo chown 1000:0 /ephemeral/shared/dynamo-logs /ephemeral/shared/huggingface
sudo chmod 0770 /ephemeral/shared/dynamo-logs /ephemeral/shared/huggingface
date -u > /ephemeral/shared/dynamo-storage-check
```

On `gpu06`:

```bash
test -r /ephemeral/shared/dynamo-storage-check
cat /ephemeral/shared/dynamo-storage-check
```

The model cache is shared to avoid downloading the same weights twice. Do not
store anything here that must survive the reservation: this share is backed by
`gpu05` local NVMe.

## 6. Clone and pin Dynamo on the shared filesystem

Run on `gpu05` only:

```bash
cd /ephemeral/shared
git clone --depth 1 --branch v1.3.0 https://github.com/ai-dynamo/dynamo.git
cd /ephemeral/shared/dynamo
git describe --tags --exact-match
```

Expected output:

```text
v1.3.0
```

If the directory already exists, verify it instead of cloning over it:

```bash
git -C /ephemeral/shared/dynamo status --short
git -C /ephemeral/shared/dynamo describe --tags --exact-match
```

Do not continue if the tag is not `v1.3.0` or the checkout has unexpected
local changes.

## 7. Start etcd and NATS on `gpu05`

First detect which Compose command, if any, is installed:

```bash
sudo docker compose version || true
sudo docker-compose --version 2>/dev/null || true
```

If the first command reports `unknown command: compose` or treats `-f` as a
Docker flag, the Docker Compose v2 plugin is not installed. That is a Docker
installation issue, not a Dynamo error.

### Option A: install/use Docker Compose v2

On Ubuntu or Debian, try the official plugin package:

```bash
sudo apt-get update
sudo apt-get install -y docker-compose-plugin
sudo docker compose version
```

If APT says it cannot locate `docker-compose-plugin`, this host is probably
using the distribution's Docker packages instead of Docker's official APT
repository. Do not replace a working GPU-enabled Docker installation merely
to obtain Compose. Use Option B below. If the legacy standalone command is
already installed, the equivalent syntax is:

```bash
sudo docker-compose -f dev/docker-compose.yml up -d
sudo docker-compose -f dev/docker-compose.yml ps
```

With Compose v2 installed, run on `gpu05`:


```bash
cd /ephemeral/shared/dynamo
sudo docker compose -f dev/docker-compose.yml up -d
sudo docker compose -f dev/docker-compose.yml ps
```

### Option B: no Compose — start the same services directly

This is the fastest fix on the current host. It uses the same images,
configuration file, environment, and ports as Dynamo v1.3.0's
`dev/docker-compose.yml`.

Run on `gpu05`:

```bash
cd /ephemeral/shared/dynamo

sudo docker run -d \
  --name dynamo-nats \
  --restart unless-stopped \
  -p 4222:4222 \
  -p 6222:6222 \
  -p 8222:8222 \
  -v "$PWD/dev/nats-server.conf:/etc/nats/nats-server.conf:ro" \
  nats:2.11.4 \
  -c /etc/nats/nats-server.conf

sudo docker run -d \
  --name dynamo-etcd \
  --restart unless-stopped \
  -p 2379:2379 \
  -p 2380:2380 \
  -e ALLOW_NONE_AUTHENTICATION=yes \
  bitnamilegacy/etcd:3.6.1

sudo docker ps --filter name=dynamo-nats --filter name=dynamo-etcd
```

If Docker says a container name is already in use, inspect rather than create
a duplicate:

```bash
sudo docker ps -a --filter name=dynamo-nats --filter name=dynamo-etcd
sudo docker logs --tail 100 dynamo-nats
sudo docker logs --tail 100 dynamo-etcd
```

The v1.3.0 Compose file starts NATS `2.11.4` and etcd `3.6.1`. Check them
locally:

```bash
curl -fsS http://127.0.0.1:8222/healthz
curl -fsS http://127.0.0.1:2379/health
```

Check them from `gpu06` over the private network:

```bash
curl -fsS http://10.18.96.143:8222/healthz
curl -fsS http://10.18.96.143:2379/health
timeout 3 bash -c 'exec 3<>/dev/tcp/10.18.96.143/4222'
timeout 3 bash -c 'exec 3<>/dev/tcp/10.18.96.143/29500' || true
```

Port `29500` is expected to fail before the distributed worker starts; this
line only confirms the test command works. NATS and etcd must succeed.

The Compose services do not enable authentication. Restrict ports `2379`,
`2380`, `4222`, `6222`, and `8222` to the two private IPs and never expose
them through a public firewall/security group. Allow unrestricted traffic
between the two hosts on the private fabric if possible: torch.distributed,
NCCL, and Dynamo's TCP request plane can use additional dynamically selected
ports.

## 8. Define the per-node environment

The v1.3.0 runtime runs as UID `1000`; the ownership commands in step 5 let it
write model downloads and logs without making the directories world-writable.

Set this on `gpu05`, replacing `REPLACE_WITH_FABRIC_INTERFACE` with the
interface found in step 4:

```bash
export HEAD_IP=10.18.96.143
export NODE_IP=10.18.96.143
export FABRIC_IFACE=REPLACE_WITH_FABRIC_INTERFACE
export DYNAMO_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
export NATS_SERVER="nats://${HEAD_IP}:4222"
export ETCD_ENDPOINTS="http://${HEAD_IP}:2379"
export DYN_TCP_RPC_HOST="$NODE_IP"
export NCCL_SOCKET_IFNAME="$FABRIC_IFACE"
export GLOO_SOCKET_IFNAME="$FABRIC_IFACE"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
```

Set this on `gpu06`, using that node's interface name:

```bash
export HEAD_IP=10.18.96.143
export NODE_IP=10.18.96.236
export FABRIC_IFACE=REPLACE_WITH_FABRIC_INTERFACE
export DYNAMO_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
export NATS_SERVER="nats://${HEAD_IP}:4222"
export ETCD_ENDPOINTS="http://${HEAD_IP}:2379"
export DYN_TCP_RPC_HOST="$NODE_IP"
export NCCL_SOCKET_IFNAME="$FABRIC_IFACE"
export GLOO_SOCKET_IFNAME="$FABRIC_IFACE"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
```

Why all three network settings matter:

- `HEAD_IP` is the torch.distributed rendezvous address.
- `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` prevent the distributed stack
  from selecting the public, Docker, or loopback interface.
- `DYN_TCP_RPC_HOST` makes each Dynamo worker advertise its reachable private
  IP instead of an auto-detected address.

Verify the values on each node:

```bash
printf 'node=%s head=%s iface=%s\n' "$NODE_IP" "$HEAD_IP" "$FABRIC_IFACE"
ip -4 addr show dev "$FABRIC_IFACE"
```

If Docker requires `sudo`, do not rely on `docker run -e VARIABLE` without a
value: `sudo` normally removes user-shell environment variables. The launch
commands below use `-e VARIABLE="$VARIABLE"`, which expands the value before
`sudo docker run` executes and therefore works with either privileged or
unprivileged Docker access.

## 9. Run a one-GPU smoke test first

This isolates the image, model download, and Dynamo frontend from multi-node
NCCL. Run on `gpu05`:

```bash
sudo docker run -d \
  --name dynamo-smoke \
  --gpus all \
  --runtime nvidia \
  --network host \
  --ipc host \
  --shm-size 10g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  --cap-add SYS_PTRACE \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -v /ephemeral/shared/huggingface:/home/dynamo/.cache/huggingface \
  "$DYNAMO_IMAGE" \
  bash -lc 'python3 -m dynamo.frontend --discovery-backend file & exec python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --discovery-backend file --enforce-eager'
```

Follow startup:

```bash
sudo docker logs -f dynamo-smoke
```

Press `Ctrl+C` after the model has loaded; this stops log-following, not the
container.

In another `gpu05` terminal, wait for readiness and test inference:

```bash
until curl -fsS http://127.0.0.1:8000/health; do sleep 5; done

curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Reply with: smoke test passed"}],
    "max_tokens": 32
  }'
```

Stop and remove the smoke-test container:

```bash
sudo docker stop dynamo-smoke
sudo docker rm dynamo-smoke
```

`VLLM_USE_FLASHINFER_SAMPLER=0` avoids a documented CUDA 13/FlashInfer JIT
version-skew failure. Keep it for the first deployment. Remove it only after
you have verified the sampler starts correctly with your exact image.

## 10. Choose a model for TP=16

The official v1.3.0 script defaults to:

```text
meta-llama/Llama-3.1-8B-Instruct
```

That model may require accepting its license and authenticating with Hugging
Face. Authenticate interactively into the shared Hugging Face cache; never put
the token in this file or in `docker run -e HF_TOKEN=...`.

Before authenticating, open
[the Llama 3.1 8B Instruct model page](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
in a browser while signed in to the same Hugging Face account. Accept Meta's
license/access form and wait until that account has access. A successful
`hf auth login` only proves that the token is valid; it does not grant access
to a gated model. A fine-grained token must explicitly include read access to
this repository; a normal read token inherits the repositories available to
its user account.

After accepting the model license, authenticate once on `gpu05` if needed:

Use this path mapping everywhere in this guide:

| Context | Token path |
| --- | --- |
| Host (`gpu05`/`gpu06`) | `/ephemeral/shared/huggingface/token` |
| Dynamo container | `/home/dynamo/.cache/huggingface/token` |

The Docker mount translates the host directory to the container directory:

```text
/ephemeral/shared/huggingface -> /home/dynamo/.cache/huggingface
```

Therefore, `/ephemeral/shared/token` and
`/ephemeral/shared/dynamo/.cache/huggingface/token` are **not** the token paths
created by the commands below. Do not create additional token copies. A single
token under `/ephemeral/shared/huggingface/token`, mounted into each runtime
container, avoids using different credentials on the two nodes.

```bash
sudo docker run --rm -it \
  -e HF_HOME=/home/dynamo/.cache/huggingface \
  -e HF_TOKEN_PATH=/home/dynamo/.cache/huggingface/token \
  -v /ephemeral/shared/huggingface:/home/dynamo/.cache/huggingface \
  "$DYNAMO_IMAGE" \
  hf auth login --force
```

Verify the exact mounted credentials without displaying the token:

```bash
sudo stat -c 'path=%n size=%s owner=%U:%G mode=%a' \
  /ephemeral/shared/huggingface/token

sudo docker run --rm \
  -e HF_HOME=/home/dynamo/.cache/huggingface \
  -e HF_TOKEN_PATH=/home/dynamo/.cache/huggingface/token \
  -v /ephemeral/shared/huggingface:/home/dynamo/.cache/huggingface \
  "$DYNAMO_IMAGE" \
  hf auth whoami

sudo docker run --rm \
  -e HF_HOME=/home/dynamo/.cache/huggingface \
  -e HF_TOKEN_PATH=/home/dynamo/.cache/huggingface/token \
  -v /ephemeral/shared/huggingface:/home/dynamo/.cache/huggingface \
  "$DYNAMO_IMAGE" \
  bash -lc 'test -s "$HF_TOKEN_PATH" || { echo "Hugging Face token file is missing" >&2; exit 1; }; export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; hf download meta-llama/Llama-3.1-8B-Instruct config.json'
```

Do not start TP=16 until both commands succeed. If `whoami` succeeds but the
one-file download returns `401 Unauthorized`, the account/license or token
scope is the problem. If `whoami` fails, repeat the login command using this
exact mount.

The explicit in-container `HF_TOKEN` export is intentional. Hugging Face's CLI
can read the saved token file directly, while Dynamo v1.3.0's ModelExpress
download path expects `HF_TOKEN` in its process environment. The launch
commands below read the mounted token file without printing it and export the
value only inside the container; the token is not embedded in the Docker
command line or Docker container configuration.

`HF_TOKEN_PATH=/home/dynamo/.cache/huggingface/token` is a path **inside the
container**. An empty `echo "$HF_TOKEN_PATH"` on the host is expected and is
not an authentication failure. Do not export the container path on the host
and do not run the final `bash -lc '...'` payload by itself on the host. It
works only as the final part of the complete `sudo docker run` command, after
Docker has mounted the host cache and set the container environment.

For the first cross-node connectivity test, keep the official model. For an
actual capacity test, replace it with a supported model that genuinely needs
more than eight H100s. The model identifier and access must be identical on
both nodes.

Set on **both** nodes:

```bash
export MODEL=meta-llama/Llama-3.1-8B-Instruct
export TENSOR_PARALLEL_SIZE=16
export NNODES=2
```

## 11. Start the TP=16 head container on `gpu05`

The following includes the Docker settings missing from the earlier plan:
host networking, host IPC, the same ulimits as Dynamo's v1.3.0 `container/run.sh`,
the pinned private interfaces, a mounted v1.3.0 workspace, and persistent logs.

Run on `gpu05`:

First confirm the host-side token file exists without displaying it:

```bash
sudo test -s /ephemeral/shared/huggingface/token \
  && echo "Hugging Face token file is present"
```

Then run this **entire block**, including `sudo docker run` and its final
`bash -lc` argument. Do not run the final line separately on the host.

```bash
: "${HEAD_IP:?run step 8 first}"
: "${MODEL:?run step 10 first}"
: "${TENSOR_PARALLEL_SIZE:?run step 10 first}"
: "${NNODES:?run step 10 first}"
: "${DYNAMO_IMAGE:?run step 8 first}"
: "${NATS_SERVER:?run step 8 first}"
: "${ETCD_ENDPOINTS:?run step 8 first}"
: "${DYN_TCP_RPC_HOST:?run step 8 first}"
: "${NCCL_SOCKET_IFNAME:?run step 8 first}"
: "${GLOO_SOCKET_IFNAME:?run step 8 first}"

sudo docker run -d \
  --name dynamo-head \
  --gpus all \
  --runtime nvidia \
  --network host \
  --ipc host \
  --shm-size 10g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  --cap-add SYS_PTRACE \
  -e HEAD_IP="$HEAD_IP" \
  -e MODEL="$MODEL" \
  -e TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE" \
  -e NNODES="$NNODES" \
  -e NATS_SERVER="$NATS_SERVER" \
  -e ETCD_ENDPOINTS="$ETCD_ENDPOINTS" \
  -e DYN_TCP_RPC_HOST="$DYN_TCP_RPC_HOST" \
  -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  -e GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME" \
  -e NCCL_DEBUG="$NCCL_DEBUG" \
  -e NCCL_DEBUG_SUBSYS="$NCCL_DEBUG_SUBSYS" \
  -e NCCL_IB_DISABLE=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e HF_HOME=/home/dynamo/.cache/huggingface \
  -e HF_TOKEN_PATH=/home/dynamo/.cache/huggingface/token \
  -v /ephemeral/shared/dynamo:/workspace:ro \
  -v /ephemeral/shared/huggingface:/home/dynamo/.cache/huggingface \
  -v /ephemeral/shared/dynamo-logs:/logs \
  -w /workspace/examples/backends/vllm \
  "$DYNAMO_IMAGE" \
  bash -lc 'test -s "$HF_TOKEN_PATH" || { echo "Hugging Face token file is missing" >&2; exit 1; }; export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; set -o pipefail; bash launch/multi_node_tp.sh --head --head-ip "$HEAD_IP" 2>&1 | tee /logs/tp16-head.log'
```

Confirm it is waiting for the second node rather than crashing:

```bash
sudo docker ps --filter name=dynamo-head
sudo docker logs --tail 100 dynamo-head
```

## 12. Start the TP=16 worker container on `gpu06`

Run on `gpu06` only after the head container is running:

```bash
: "${HEAD_IP:?run step 8 first}"
: "${MODEL:?run step 10 first}"
: "${TENSOR_PARALLEL_SIZE:?run step 10 first}"
: "${NNODES:?run step 10 first}"
: "${DYNAMO_IMAGE:?run step 8 first}"
: "${NATS_SERVER:?run step 8 first}"
: "${ETCD_ENDPOINTS:?run step 8 first}"
: "${DYN_TCP_RPC_HOST:?run step 8 first}"
: "${NCCL_SOCKET_IFNAME:?run step 8 first}"
: "${GLOO_SOCKET_IFNAME:?run step 8 first}"

sudo docker run -d \
  --name dynamo-worker \
  --gpus all \
  --runtime nvidia \
  --network host \
  --ipc host \
  --shm-size 10g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  --cap-add SYS_PTRACE \
  -e HEAD_IP="$HEAD_IP" \
  -e MODEL="$MODEL" \
  -e TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE" \
  -e NNODES="$NNODES" \
  -e NATS_SERVER="$NATS_SERVER" \
  -e ETCD_ENDPOINTS="$ETCD_ENDPOINTS" \
  -e DYN_TCP_RPC_HOST="$DYN_TCP_RPC_HOST" \
  -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  -e GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME" \
  -e NCCL_DEBUG="$NCCL_DEBUG" \
  -e NCCL_DEBUG_SUBSYS="$NCCL_DEBUG_SUBSYS" \
  -e NCCL_IB_DISABLE=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e HF_HOME=/home/dynamo/.cache/huggingface \
  -e HF_TOKEN_PATH=/home/dynamo/.cache/huggingface/token \
  -v /ephemeral/shared/dynamo:/workspace:ro \
  -v /ephemeral/shared/huggingface:/home/dynamo/.cache/huggingface \
  -v /ephemeral/shared/dynamo-logs:/logs \
  -w /workspace/examples/backends/vllm \
  "$DYNAMO_IMAGE" \
  bash -lc 'test -s "$HF_TOKEN_PATH" || { echo "Hugging Face token file is missing" >&2; exit 1; }; export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; set -o pipefail; bash launch/multi_node_tp.sh --worker --head-ip "$HEAD_IP" 2>&1 | tee /logs/tp16-worker.log'
```

Follow both nodes:

```bash
sudo docker logs -f dynamo-head
```

```bash
sudo docker logs -f dynamo-worker
```

The head runs frontend plus node rank 0. The worker runs node rank 1 with
`--headless`, so it joins the same model instead of registering a second model
endpoint.

## 13. Confirm that the container can see RDMA

Run on both nodes after their container starts:

```bash
sudo docker exec dynamo-head bash -lc 'ls -l /dev/infiniband 2>/dev/null || true; ls /sys/class/infiniband 2>/dev/null || true'
```

Use `dynamo-worker` instead of `dynamo-head` on `gpu06`.

If the host has `/dev/infiniband` but the container does not, stop the
containers and expose the RDMA devices according to the cluster provider's
container policy. On these dedicated temporary nodes, adding `--privileged`
to both `docker run` commands is a diagnostic option, but it grants broad host
device access and should not be the production configuration.

Inspect the NCCL logs. A healthy RDMA run identifies the selected private
interface/HCA and an IB network plugin. If it reports only `NET/Socket`, the
deployment is using TCP.

## 14. Verify the deployment

On `gpu05`, monitor all GPUs:

```bash
watch -n 1 nvidia-smi
```

On `gpu06`, do the same:

```bash
watch -n 1 nvidia-smi
```

All 16 GPUs should allocate model memory. Then, on `gpu05`:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models

curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Explain tensor parallelism in three sentences.\"}],
    \"max_tokens\": 100
  }"
```

Do not expose port `8000` publicly without authentication and TLS. Use an SSH
tunnel, private load balancer, or authenticated gateway for remote clients.

## 15. Recommended scale-out mode: two TP=8 replicas

If the model fits on one node, two TP=8 replicas are usually the better use of
the hardware. Keep the same etcd/NATS and per-node environment from steps 7
and 8.

Start one frontend-only container on `gpu05`:

The frontend does not load model weights, but when a gated Hugging Face model
registers it still downloads model metadata such as `config.json` and policy
files with `ignore_weights=true`. Therefore, the frontend needs the same
authenticated Hugging Face cache as the workers. Keep the token in the mounted
file and export it only inside the container process.

Set the desired frontend port first. Keep `8000` unless another private service
already uses it:

```bash
export DYN_HTTP_PORT=8000

sudo test -s /ephemeral/shared/huggingface/token \
  && echo "Hugging Face token file is present"

sudo docker run -d \
  --name dynamo-frontend \
  --network host \
  -e NATS_SERVER="$NATS_SERVER" \
  -e ETCD_ENDPOINTS="$ETCD_ENDPOINTS" \
  -e DYN_TCP_RPC_HOST="$DYN_TCP_RPC_HOST" \
  -e DYN_HTTP_PORT="$DYN_HTTP_PORT" \
  -e HF_HOME=/home/dynamo/.cache/huggingface \
  -e HF_TOKEN_PATH=/home/dynamo/.cache/huggingface/token \
  -v /ephemeral/shared/huggingface:/home/dynamo/.cache/huggingface \
  "$DYNAMO_IMAGE" \
  bash -lc 'test -s "$HF_TOKEN_PATH" || { echo "Hugging Face token file is missing" >&2; exit 1; }; export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; exec python3 -m dynamo.frontend --http-port "$DYN_HTTP_PORT"'
```

On each node, set a model that fits in eight H100s:

```bash
export MODEL=REPLACE_WITH_MODEL_ID_OR_SHARED_LOCAL_PATH
```

Start a worker on `gpu05` with `DYN_TCP_RPC_HOST=10.18.96.143`, and repeat on
`gpu06` with `DYN_TCP_RPC_HOST=10.18.96.236`:

```bash
sudo docker run -d \
  --name dynamo-replica \
  --gpus all \
  --runtime nvidia \
  --network host \
  --ipc host \
  --shm-size 10g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  --cap-add SYS_PTRACE \
  -e MODEL="$MODEL" \
  -e NATS_SERVER="$NATS_SERVER" \
  -e ETCD_ENDPOINTS="$ETCD_ENDPOINTS" \
  -e DYN_TCP_RPC_HOST="$DYN_TCP_RPC_HOST" \
  -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  -e GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME" \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e HF_HOME=/home/dynamo/.cache/huggingface \
  -e HF_TOKEN_PATH=/home/dynamo/.cache/huggingface/token \
  -v /ephemeral/shared/huggingface:/home/dynamo/.cache/huggingface \
  "$DYNAMO_IMAGE" \
  bash -lc 'test -s "$HF_TOKEN_PATH" || { echo "Hugging Face token file is missing" >&2; exit 1; }; export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; exec python3 -m dynamo.vllm --model "$MODEL" --tensor-parallel-size 8'
```

Both workers register in the shared etcd discovery plane. The default Dynamo
frontend router distributes requests across them. Verify with:

```bash
curl -fsS "http://127.0.0.1:${DYN_HTTP_PORT}/v1/models"
sudo docker logs --tail 100 dynamo-frontend
```

Load-test from a separate client and watch both nodes. Do not judge scaling
from one serial request.

## 16. Clean shutdown and restart

For TP=16, stop the worker first and then the head:

On `gpu06`:

```bash
sudo docker stop dynamo-worker
sudo docker rm dynamo-worker
```

On `gpu05`:

```bash
sudo docker stop dynamo-head
sudo docker rm dynamo-head
```

Stop infrastructure only when no Dynamo deployment is using it. If it was
started with Compose v2:

```bash
cd /ephemeral/shared/dynamo
sudo docker compose -f dev/docker-compose.yml down
```

If it was started with legacy Compose, use:

```bash
cd /ephemeral/shared/dynamo
sudo docker-compose -f dev/docker-compose.yml down
```

If it was started directly with Option B:

```bash
sudo docker stop dynamo-nats dynamo-etcd
sudo docker rm dynamo-nats dynamo-etcd
```

For a clean restart, verify that no previous Python process still owns GPU
memory:

```bash
nvidia-smi
sudo docker ps -a
```

Do not blindly kill host processes. Identify whether a process belongs to a
container or another user first.

## 17. Troubleshooting order

### Hugging Face returns `401 Unauthorized` for a gated model

Run the `hf auth whoami` and one-file `hf download` preflight from step 10.

- If `whoami` fails, the login was not saved in the cache mounted into the
  runtime container. Repeat the exact login command from step 10.
- If `whoami` succeeds but `hf download` fails, sign in to the same account in
  a browser, accept the model's license/access request, and ensure a
  fine-grained token includes this model. Login again after correcting access.
- If both succeed, remove the exited Dynamo container and relaunch it using
  the corrected steps 11 and 12. Those commands export `HF_TOKEN` from the
  mounted token file inside the container before ModelExpress starts.
- If the workers load successfully but a separate `dynamo-frontend` reports
  `401 Unauthorized` with `ignore_weights=true`, the frontend can see the model
  registration but cannot authenticate its metadata download. Inspect only its
  mount and safe path variables:

  ```bash
  sudo docker inspect dynamo-frontend \
    --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'

  sudo docker inspect dynamo-frontend \
    --format '{{range .Config.Env}}{{println .}}{{end}}' \
    | grep -E '^(HF_HOME|HF_TOKEN_PATH)='
  ```

  Expected values are the same as the worker containers:

  ```text
  /ephemeral/shared/huggingface -> /home/dynamo/.cache/huggingface
  HF_HOME=/home/dynamo/.cache/huggingface
  HF_TOKEN_PATH=/home/dynamo/.cache/huggingface/token
  ```

  If any value is missing, remove the failed frontend and rerun the complete
  frontend block from step 15. Restarting the old container cannot add a
  missing mount or environment variable.

  ```bash
  sudo docker rm -f dynamo-frontend
  ```

Do not print the token or pass it as a literal command-line argument.

### `Hugging Face token file is missing`

Use `/ephemeral/shared/huggingface/token` on the host. Docker mounts it as
`/home/dynamo/.cache/huggingface/token` inside the container. Do not use
`/ephemeral/shared/token` or a path under the Dynamo source checkout.

First check only the host file's metadata; this does not display the token:

```bash
sudo stat -c 'path=%n size=%s owner=%U:%G mode=%a' \
  /ephemeral/shared/huggingface/token
```

If a `dynamo-head` container was already created, verify its mount and safe
path variables:

```bash
sudo docker inspect dynamo-head \
  --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'

sudo docker inspect dynamo-head \
  --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -E '^(HF_HOME|HF_TOKEN_PATH)='
```

The expected output includes:

```text
/ephemeral/shared/huggingface -> /home/dynamo/.cache/huggingface
HF_HOME=/home/dynamo/.cache/huggingface
HF_TOKEN_PATH=/home/dynamo/.cache/huggingface/token
```

If the mount or either variable is missing, the existing container was created
with an older or incomplete command. Remove only that failed container and run
the complete step 11 block again:

```bash
sudo docker rm -f dynamo-head
```

For the worker, repeat the same checks with `dynamo-worker`, then use
`sudo docker rm -f dynamo-worker` before rerunning the complete step 12 block.
Never execute the trailing `bash -lc '...'` payload directly in the host shell.

### `--head-ip is required` even though `echo "$HEAD_IP"` works

This happens when the container was launched with `sudo docker run -e HEAD_IP`.
`echo` reads the variable from the current user shell, while `sudo` normally
removes it before starting Docker. Confirm the failed container's value:

```bash
sudo docker inspect dynamo-head \
  --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep '^HEAD_IP='
```

A missing line or empty value confirms the cause. Remove the exited container, re-export the
step 8 and step 10 variables, and use the corrected launch command containing
`-e HEAD_IP="$HEAD_IP"`. The same correction is required for the model,
discovery, and network-interface variables.

### Container exits immediately

```bash
sudo docker ps -a
sudo docker logs --tail 200 dynamo-head
sudo docker logs --tail 200 dynamo-worker
```

Check the first exception, not only the last NCCL timeout.

### `Engine core initialization failed` with FlashInfer/CUDA headers

Confirm both containers have:

```text
VLLM_USE_FLASHINFER_SAMPLER=0
```

This selects vLLM's native sampler and avoids the documented CUDA 13 JIT
header mismatch.

### Head waits forever for the worker

From `gpu06`:

```bash
ip route get 10.18.96.143
timeout 3 bash -c 'exec 3<>/dev/tcp/10.18.96.143/29500'
sudo docker logs --tail 200 dynamo-worker
```

Verify `HEAD_IP`, `NNODES=2`, `TENSOR_PARALLEL_SIZE=16`, the private firewall,
and identical model settings on both nodes.

### NCCL chooses the wrong interface

```bash
sudo docker exec dynamo-head env | grep -E 'NCCL_SOCKET_IFNAME|GLOO_SOCKET_IFNAME|DYN_TCP_RPC_HOST'
sudo docker logs dynamo-head 2>&1 | grep -E 'NCCL|NET/|Socket|IB'
```

Run the equivalent checks on `dynamo-worker`. The interface must be the one
that routes `10.18.96.143 <-> 10.18.96.236`.

### Frontend starts but cannot send requests to a worker

Check that each worker advertises its own private IP:

```bash
sudo docker exec dynamo-head printenv DYN_TCP_RPC_HOST
sudo docker exec dynamo-worker printenv DYN_TCP_RPC_HOST
```

Expected values are `10.18.96.143` and `10.18.96.236`, respectively. Allow
private node-to-node traffic for Dynamo's OS-assigned TCP request-plane ports.

### NATS or etcd connection errors

From both nodes:

```bash
curl -fsS http://10.18.96.143:8222/healthz
curl -fsS http://10.18.96.143:2379/health
```

Confirm that `NATS_SERVER` includes `nats://` and `ETCD_ENDPOINTS` points at
the head node, not `localhost` on `gpu06`.

### OOM on a restart

```bash
sudo docker ps -a
nvidia-smi
```

Stop and remove only the old Dynamo containers. A stale container or orphaned
worker can retain GPU memory.

## 18. Moving from a reservation to production

For production scale, keep the same principles but move to the Dynamo
Kubernetes operator:

1. Pin the Dynamo platform, runtime, and operator to the same release.
2. Use a gang scheduler so every rank is placed atomically.
3. Request the RDMA device resources and use host networking where required.
4. Run authenticated, persistent NATS/etcd rather than the development
   Compose stack.
5. Add readiness/liveness probes, metrics, centralized logs, and a private
   authenticated ingress.
6. Prefer multiple replicas for throughput; use cross-node TP/PP only for
   models that need it.

The manual Docker procedure above is the validation baseline. Do not add
Kubernetes until the one-GPU smoke test and the two-node transport checks pass.
