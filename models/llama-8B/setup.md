# Llama 8B (`meta-llama/Llama-3.1-8B-Instruct`) Deployment Guide

This document tracks the deployment configuration, execution steps, and resource observations for **Llama-3.1-8B-Instruct** using NVIDIA Dynamo v1.3.0 (vLLM backend) on a 2-node 8xH100 cluster.

## Deployment Metadata

| Parameter | Value |
|---|---|
| **Model ID** | `meta-llama/Llama-3.1-8B-Instruct` |
| **Framework** | NVIDIA Dynamo v1.3.0 (vLLM backend) |
| **Container Image** | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0` |
| **Primary Topology** | Multi-Node Capacity Scaling (TP=16 across 2 nodes) |
| **Alternative Topology** | Multi-Replica Throughput Scaling (2x TP=8 single-node replicas) |
| **Head Node (`gpu05`)** | Private IP: `10.18.96.143` (8x H100 80GB) |
| **Worker Node (`gpu06`)** | Private IP: `10.18.96.236` (8x H100 80GB) |
| **Shared Storage** | `/ephemeral/shared/huggingface`, `/ephemeral/shared/dynamo-logs` |
| **Baseline Status** | Deployed & Validated |
| **Observed VRAM Usage** | ~95% per node under baseline TP=16 |

---

## 1. Topology & Resource Selection

This model deployment supports two modes:

1. **Capacity / Baseline Mode (TP=16):** Stretches one model instance across all 16 GPUs (8 GPUs on `gpu05`, 8 GPUs on `gpu06`).
   - *Use Case:* Connectivity validation, baseline latency measurement across multi-node interconnect.
   - *Observation:* Each node occupies roughly 95% of GPU VRAM. Cross-node tensor parallel communication overhead is incurred for every transformer layer.
2. **Throughput Mode (2x TP=8 Replicas):** Runs one independent TP=8 replica on `gpu05` and one on `gpu06`.
   - *Use Case:* Production serving and throughput benchmarking. Eliminates inter-node collective communication during model forward passes.

---

## 2. Cluster & Storage Verification

Run the host and container toolkit checks on **both** `gpu05` and `gpu06`:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
sudo docker run --rm --gpus all --runtime nvidia nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0 nvidia-smi -L
```

Verify shared storage on `gpu05`:

```bash
mkdir -p /ephemeral/shared/dynamo-logs /ephemeral/shared/huggingface
sudo chown 1000:0 /ephemeral/shared/dynamo-logs /ephemeral/shared/huggingface
sudo chmod 0770 /ephemeral/shared/dynamo-logs /ephemeral/shared/huggingface
```

Verify private network connectivity (`10.18.96.x` fabric network):

```bash
# On gpu05
ip route get 10.18.96.236
ping -c 4 10.18.96.236

# On gpu06
ip route get 10.18.96.143
ping -c 4 10.18.96.143
```

Identify the fabric network interface (e.g. `eth0` or `bond0`) from `ip route get` and export it as `FABRIC_IFACE`.

---

## 3. Hugging Face Authentication for Gated Llama 3.1 8B

`meta-llama/Llama-3.1-8B-Instruct` is a gated repository. Accept the Meta Llama 3.1 license on Hugging Face before proceeding.

Authenticate once on `gpu05` into the shared cache directory `/ephemeral/shared/huggingface`:

```bash
sudo docker run --rm -it \
  -e HF_HOME=/home/dynamo/.cache/huggingface \
  -e HF_TOKEN_PATH=/home/dynamo/.cache/huggingface/token \
  -v /ephemeral/shared/huggingface:/home/dynamo/.cache/huggingface \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0 \
  hf auth login --force
```

Pre-flight check (verify account and download model metadata):

```bash
sudo docker run --rm \
  -e HF_HOME=/home/dynamo/.cache/huggingface \
  -e HF_TOKEN_PATH=/home/dynamo/.cache/huggingface/token \
  -v /ephemeral/shared/huggingface:/home/dynamo/.cache/huggingface \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0 \
  bash -lc 'test -s "$HF_TOKEN_PATH" || { echo "HF token missing" >&2; exit 1; }; export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; hf download meta-llama/Llama-3.1-8B-Instruct config.json'
```

---

## 4. Start Control Plane (etcd & NATS)

Start etcd and NATS on `gpu05` using direct `docker run` (Option B):

```bash
cd /ephemeral/shared/dynamo

sudo docker run -d \
  --name dynamo-nats \
  --restart unless-stopped \
  -p 4222:4222 -p 6222:6222 -p 8222:8222 \
  -v "$PWD/dev/nats-server.conf:/etc/nats/nats-server.conf:ro" \
  nats:2.11.4 \
  -c /etc/nats/nats-server.conf

sudo docker run -d \
  --name dynamo-etcd \
  --restart unless-stopped \
  -p 2379:2379 -p 2380:2380 \
  -e ALLOW_NONE_AUTHENTICATION=yes \
  bitnamilegacy/etcd:3.6.1
```

Verify service status from `gpu06`:

```bash
curl -fsS http://10.18.96.143:8222/healthz
curl -fsS http://10.18.96.143:2379/health
```

---

## 5. Environment Export

Export the following environment variables on both nodes. Replace `REPLACE_WITH_FABRIC_INTERFACE` with your actual network interface name.

On `gpu05` (Head):

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
export MODEL=meta-llama/Llama-3.1-8B-Instruct
export TENSOR_PARALLEL_SIZE=16
export NNODES=2
```

On `gpu06` (Worker):

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
export MODEL=meta-llama/Llama-3.1-8B-Instruct
export TENSOR_PARALLEL_SIZE=16
export NNODES=2
```

---

## 6. Multi-Node TP=16 Deployment

### Step 6.1: Start Head Container (`gpu05`)

```bash
sudo docker run -d \
  --name llama8b-tp16-head \
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
  bash -lc 'test -s "$HF_TOKEN_PATH" || { echo "HF token missing" >&2; exit 1; }; export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; set -o pipefail; bash launch/multi_node_tp.sh --head --head-ip "$HEAD_IP" 2>&1 | tee /logs/llama8b-tp16-head.log'
```

### Step 6.2: Start Worker Container (`gpu06`)

```bash
sudo docker run -d \
  --name llama8b-tp16-worker \
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
  bash -lc 'test -s "$HF_TOKEN_PATH" || { echo "HF token missing" >&2; exit 1; }; export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; set -o pipefail; bash launch/multi_node_tp.sh --worker --head-ip "$HEAD_IP" 2>&1 | tee /logs/llama8b-tp16-worker.log'
```

---

## 7. Verification & Inference Testing

Check logs and health status on `gpu05`:

```bash
sudo docker logs -f llama8b-tp16-head

until curl -fsS http://127.0.0.1:8000/health; do sleep 5; done
curl -fsS http://127.0.0.1:8000/v1/models
```

Run test inference:

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain tensor parallelism in three concise sentences."}
    ],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

---

## 8. High-Throughput Alternative (2x TP=8 Replicas)

To deploy two independent TP=8 replicas of `meta-llama/Llama-3.1-8B-Instruct` (one per node):

1. **Start Frontend on `gpu05`:**

```bash
export DYN_HTTP_PORT=8000

sudo docker run -d \
  --name llama8b-frontend \
  --network host \
  -e NATS_SERVER="$NATS_SERVER" \
  -e ETCD_ENDPOINTS="$ETCD_ENDPOINTS" \
  -e DYN_TCP_RPC_HOST="$DYN_TCP_RPC_HOST" \
  -e DYN_HTTP_PORT="$DYN_HTTP_PORT" \
  -e HF_HOME=/home/dynamo/.cache/huggingface \
  -e HF_TOKEN_PATH=/home/dynamo/.cache/huggingface/token \
  -v /ephemeral/shared/huggingface:/home/dynamo/.cache/huggingface \
  "$DYNAMO_IMAGE" \
  bash -lc 'test -s "$HF_TOKEN_PATH" || { echo "HF token missing" >&2; exit 1; }; export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; exec python3 -m dynamo.frontend --http-port "$DYN_HTTP_PORT"'
```

2. **Start Replica 1 on `gpu05` (`DYN_TCP_RPC_HOST=10.18.96.143`):**

```bash
sudo docker run -d \
  --name llama8b-replica-gpu05 \
  --gpus all \
  --runtime nvidia \
  --network host \
  --ipc host \
  --shm-size 10g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  --cap-add SYS_PTRACE \
  -e MODEL="meta-llama/Llama-3.1-8B-Instruct" \
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
  bash -lc 'test -s "$HF_TOKEN_PATH" || { echo "HF token missing" >&2; exit 1; }; export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; exec python3 -m dynamo.vllm --model meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 8'
```

3. **Start Replica 2 on `gpu06` (`DYN_TCP_RPC_HOST=10.18.96.236`):**

```bash
sudo docker run -d \
  --name llama8b-replica-gpu06 \
  --gpus all \
  --runtime nvidia \
  --network host \
  --ipc host \
  --shm-size 10g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  --cap-add SYS_PTRACE \
  -e MODEL="meta-llama/Llama-3.1-8B-Instruct" \
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
  bash -lc 'test -s "$HF_TOKEN_PATH" || { echo "HF token missing" >&2; exit 1; }; export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; exec python3 -m dynamo.vllm --model meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 8'
```

---

## 9. Deployment Observation Log

Record performance, resource utilization, and optimization milestones for Llama 8B deployments in this section.

| Date | Topology | Driver / CUDA | VRAM / GPU | Time-to-First-Token (TTFT) | Output Tokens/sec | Notes |
|---|---|---|---|---|---|---|
| Initial | TP=16 (2 nodes) | 580.00.03 / CUDA 13 | ~95% (~76GB/80GB) | TBD | TBD | Baseline connectivity deployment setup completed. High VRAM footprint per node. |
| Initial | 2x TP=8 (2 nodes) | 580.00.03 / CUDA 13 | TBD | TBD | TBD | Recommended throughput configuration under evaluation. |

---

## 10. Clean Shutdown

To stop the TP=16 containers:

On `gpu06`:
```bash
sudo docker stop llama8b-tp16-worker && sudo docker rm llama8b-tp16-worker
```

On `gpu05`:
```bash
sudo docker stop llama8b-tp16-head && sudo docker rm llama8b-tp16-head
```

To stop etcd and NATS (if no other model is using them):
```bash
sudo docker stop dynamo-nats dynamo-etcd && sudo docker rm dynamo-nats dynamo-etcd
```
