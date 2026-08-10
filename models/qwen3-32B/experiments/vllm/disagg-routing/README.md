# Qwen3-32B-FP8 vLLM disaggregated deployment

This guide starts from the supplied NVIDIA Dynamo recipe and applies the
cluster-specific settings required by NVIDIA's
[Disaggregated Inference Communication Guide](https://docs.nvidia.com/dynamo/dev/knowledge-base/kubernetes/kubernetes-operator/disagg-communication):

- `MODEL_PATH` uses the immutable snapshot on the existing `model-cache` PVC;
- `UCX_NET_DEVICES=mlx5_8:1` selects the active, non-bonded RoCE HCA found on
  both nodes;
- `UCX_IB_ADDR_TYPE=eth` and `UCX_IB_GID_INDEX=3` select the validated RoCE
  address (`10.224.7.x`) instead of relying on UCX auto-selection;
- `hostNetwork: true` makes the host's `rdma7` RoCE netdevice visible in each
  worker network namespace, and `ClusterFirstWithHostNet` retains cluster DNS;
- UCX rendezvous, registration-cache, timeout, keepalive, and bring-up logging
  settings follow the communication guide;
- RDMA resources match tensor parallelism: two per TP=2 prefill worker and four
  for the TP=4 decode worker;
- `IPC_LOCK`, `SYS_RESOURCE`, and 40 GiB shared memory are retained.

The recipe creates:

- two prefill workers with TP=2;
- one decode worker with TP=4;
- one frontend;
- eight allocated H100 GPUs in total.

The original affinity, image, model, and worker topology are retained. It does
not pin prefill and decode to separate nodes. Because host networking
removes per-Pod port isolation, the workers use role-specific system, NIXL
telemetry, NIXL handshake, and forward-pass-metrics ports. Explicit `hostPort`
declarations also keep
the two identical prefill replicas from being scheduled on the same node.

## 1. Create the deployment file

**Run only on the Kubernetes control-plane node (`gpu05`).**

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3-32b/experiments/vllm/disagg-routing

mkdir -p "$EXP_DIR"

tee "$EXP_DIR/deploy.yaml" >/dev/null <<'EOF_DEPLOY_YAML'
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-32b-fp8-vllm-disagg
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
      extraPodSpec:
        hostNetwork: true
        dnsPolicy: ClusterFirstWithHostNet
        affinity:
          podAffinity:
            preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                  - key: nvidia.com/dynamo-component-type
                    operator: In
                    values:
                    - worker
                topologyKey: kubernetes.io/hostname
        mainContainer:
          env:
            - name: SERVED_MODEL_NAME
              value: "Qwen/Qwen3-32B-FP8"
            - name: MODEL_PATH
              value: "/opt/models/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
            - name: HF_HOME
              value: /opt/models
            - name: DYN_SYSTEM_PORT
              value: "9090"
            - name: NIXL_TELEMETRY_PROMETHEUS_PORT
              value: "19090"
            - name: DYN_FORWARDPASS_METRIC_PORT
              value: "20380"
            - name: VLLM_NIXL_SIDE_CHANNEL_PORT
              value: "5600"
            - name: UCX_TLS
              value: "rc_x,rc,cuda_copy,cuda_ipc"
            - name: UCX_NET_DEVICES
              value: "mlx5_8:1"
            - name: UCX_IB_ADDR_TYPE
              value: "eth"
            - name: UCX_IB_GID_INDEX
              value: "3"
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
              --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
              --gpu-memory-utilization 0.90 \
              --max-model-len 8192 \
              --no-enable-prefix-caching \
              --block-size 128
          command:
          - /bin/sh
          - -c
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          workingDir: /workspace/examples/backends/vllm
          ports:
            - name: system
              containerPort: 9090
              hostPort: 9090
            - name: nixl-metrics
              containerPort: 19090
              hostPort: 19090
            - name: fpm
              containerPort: 20380
              hostPort: 20380
            - name: nixl-side
              containerPort: 5600
              hostPort: 5600
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
    VllmDecodeWorker:
      componentType: worker
      subComponentType: decode
      envFromSecret: hf-token-secret
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      sharedMemory:
        size: 40Gi
      extraPodSpec:
        hostNetwork: true
        dnsPolicy: ClusterFirstWithHostNet
        affinity:
          podAffinity:
            preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                  - key: nvidia.com/dynamo-component-type
                    operator: In
                    values:
                    - worker
                topologyKey: kubernetes.io/hostname
        mainContainer:
          env:
            - name: SERVED_MODEL_NAME
              value: "Qwen/Qwen3-32B-FP8"
            - name: MODEL_PATH
              value: "/opt/models/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
            - name: HF_HOME
              value: /opt/models
            - name: DYN_SYSTEM_PORT
              value: "9091"
            - name: NIXL_TELEMETRY_PROMETHEUS_PORT
              value: "19091"
            - name: DYN_FORWARDPASS_METRIC_PORT
              value: "20381"
            - name: VLLM_NIXL_SIDE_CHANNEL_PORT
              value: "5601"
            - name: UCX_TLS
              value: "rc_x,rc,cuda_copy,cuda_ipc"
            - name: UCX_NET_DEVICES
              value: "mlx5_8:1"
            - name: UCX_IB_ADDR_TYPE
              value: "eth"
            - name: UCX_IB_GID_INDEX
              value: "3"
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
              --tensor-parallel-size 4 \
              --data-parallel-size 1 \
              --disaggregation-mode decode \
              --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' \
              --gpu-memory-utilization 0.90 \
              --max-model-len 8192 \
              --no-enable-prefix-caching \
              --block-size 128
          command:
          - /bin/sh
          - -c
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          workingDir: /workspace/examples/backends/vllm
          ports:
            - name: system
              containerPort: 9091
              hostPort: 9091
            - name: nixl-metrics
              containerPort: 19091
              hostPort: 19091
            - name: fpm
              containerPort: 20381
              hostPort: 20381
            - name: nixl-side
              containerPort: 5601
              hostPort: 5601
          securityContext:
            runAsUser: 0
            capabilities:
              add:
                - IPC_LOCK
                - SYS_RESOURCE
      replicas: 1
      resources:
        limits:
          gpu: "4"
          custom:
            rdma/ib: "4"
        requests:
          gpu: "4"
          custom:
            rdma/ib: "4"
EOF_DEPLOY_YAML
```

## 2. Why the earlier deployment failed

The host checks proved that `mlx5_8:1` maps to `rdma7`, and GID index `3` is a
RoCE v2 address on the `10.224.7.x` fabric. The ordinary preflight Pods could
also open the `/dev/infiniband` devices and initialize UCX resources. The
actual failure occurred one step later:

```text
ibv_create_ah(... dgid=::ffff:10.224.7.236 sgid_index=3 ...)
failed: No such device
```

The RDMA device plugin made the verbs character devices available, but the
Calico Pod network namespace contained the overlay `eth0`, not the host's
`rdma7` netdevice. RoCE address-handle creation needs that Ethernet netdevice
to resolve the destination MAC. UCX could therefore see the HCA and GID but
the kernel could not construct the path to the destination.

A simple analogy is that `/dev/infiniband` gave the Pod the controls for the
RDMA engine, while its isolated network namespace did not contain the road
named `rdma7`. `hostNetwork: true` gives the worker the host's network view, so
both the controls and the road are present. `dnsPolicy:
ClusterFirstWithHostNet` then preserves Kubernetes Service-name resolution
even though the Pod uses the host network.

This was proved independently on the prefill and decode nodes: after enabling
the two settings, both checks created the UCX backend, selected
`rc_mlx5/mlx5_8:1`, avoided TCP, and printed `NIXL_UCX_BACKEND_OK`.

The first host-networked model rollout then exposed a separate port collision.
Prefill had already bound the runtime's default forward-pass-metrics ZMQ port,
so decode failed with:

```text
zmq.error.ZMQError: Address already in use (addr='tcp://*:20380')
```

This was not another RDMA failure. Isolated Pods may all bind `20380` because
each has its own network namespace; host-networked workers on the same node
cannot. The deployment therefore sets `DYN_FORWARDPASS_METRIC_PORT=20380` for
prefill and `20381` for decode, matching the role-specific system and NIXL
telemetry port strategy.

The next rollout reached the NIXL handshake thread and exposed the same
host-network effect on its separate side-channel listener:

```text
zmq.error.ZMQError: Address already in use
(addr='tcp://10.18.96.143:5600')
```

Port `5600` is the vLLM NIXL side-channel default. Prefill and decode shared a
host, so the deployment now sets `VLLM_NIXL_SIDE_CHANNEL_PORT=5600` for
prefill and `5601` for decode. The connector configuration also uses the
runtime's role-specific values: `kv_producer` on prefill and `kv_consumer` on
decode instead of the deprecated `kv_both` value. These connector roles do
not set Dynamo's worker topology: the commands must independently include
`--disaggregation-mode prefill` and `--disaggregation-mode decode`.

An intermediate decode command omitted `--disaggregation-mode decode`.
Although its Kubernetes `subComponentType` label said `decode`, the Dynamo
vLLM process defaulted to aggregated mode and published an aggregated model
deployment card. The frontend discovered all three workers but logged the
decode worker set as `chat|completions:aggregated`, so it did not construct a
ready prefill-to-decode pipeline and continued returning HTTP 503. Adding the
decode CLI flag makes the runtime advertise `WorkerType.Decode`, allowing the
frontend to pair it with the two stored prefill endpoints.

The remaining messages were secondary:

- frontend HTTP 503 responses meant no worker was ready;
- `UCX_RCACHE_MAX_UNRELEASED=1024` removes the import-order warning;
- `invalid gid[3] on mlx5_11` is irrelevant because UCX is pinned to
  `mlx5_8:1`;
- `no active primary CUDA context` is expected in the initialization-only
  preflight, which does not allocate a GPU buffer;
- the earlier `kv_both` deprecation warning did not cause backend creation to
  fail, but the production manifest now uses `kv_producer`/`kv_consumer`.

## 3. Stop old workers before preflight

**Run only on `gpu05`.** This preserves the PVC, model snapshot, and Secret.

```bash
export NAMESPACE=qwen32-bench
export EXP_DIR=/ephemeral/shared/qwen3-32b/experiments/vllm/disagg-routing
export DGD=qwen3-32b-fp8-vllm-disagg
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=$DGD"

kubectl delete dynamographdeployment disagg-router-6p-2d \
  -n "$NAMESPACE" --ignore-not-found
kubectl delete dynamographdeployment "$DGD" \
  -n "$NAMESPACE" --ignore-not-found

kubectl wait --for=delete pod \
  -l nvidia.com/dynamo-graph-deployment-name=disagg-router-6p-2d \
  -n "$NAMESPACE" --timeout=5m || true
kubectl wait --for=delete pod -l "$GRAPH_LABEL" \
  -n "$NAMESPACE" --timeout=5m || true

kubectl get pods -n "$NAMESPACE" -o wide
```

Do not continue while an older Qwen worker still owns GPUs or RDMA resources.

## 4. Host and Kubernetes gates

### 4.1 Host gate

**Run the following block separately on both `gpu05` and `gpu06`.**

```bash
hostname
nvidia-smi -L

cat /sys/class/infiniband/mlx5_8/ports/1/state
cat /sys/class/infiniband/mlx5_8/ports/1/link_layer
cat /sys/class/infiniband/mlx5_8/ports/1/gids/3

rdma system show
lsmod | grep -E '^nvidia_peermem|^ib_core|^mlx5_ib'
```

Require eight H100 GPUs, `4: ACTIVE`, `Ethernet`, a nonzero GID at index 3,
shared RDMA namespace mode, and all three kernel modules on both hosts. Stop if
the nodes differ; do not guess another HCA or GID in the model manifest.

### 4.2 Kubernetes gate

**Run only on `gpu05`.**

```bash
kubectl get nodes -L qwen.nvidia.com/role -o wide

kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\tGPU="}{.status.allocatable.nvidia\.com/gpu}{"\tRDMA="}{.status.allocatable.rdma/ib}{"\n"}{end}'

kubectl get clusterpolicy cluster-policy \
  -o jsonpath='GPU_OPERATOR={.status.state}{"\n"}'

kubectl get pods -n gpu-operator -o wide
kubectl get pods -n nvidia-network-operator -o wide
kubectl get daemonset rdma-shared-dp-ds \
  -n nvidia-network-operator
```

Require both nodes `Ready`, `GPU=8`, a nonzero `RDMA` allocation, GPU Operator
state `ready`, and two ready RDMA Shared Device Plugin Pods.

## 5. Storage, manifest, and schema gates

**Run only on `gpu05`.**

```bash
kubectl get pvc model-cache -n "$NAMESPACE"
kubectl get secret hf-token-secret -n "$NAMESPACE"

test -d /ephemeral/shared/huggingface/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df
echo "MODEL_SNAPSHOT_CHECK=$?"

grep -nE 'hostNetwork|dnsPolicy|DYN_SYSTEM_PORT|NIXL_TELEMETRY_PROMETHEUS_PORT|DYN_FORWARDPASS_METRIC_PORT|VLLM_NIXL_SIDE_CHANNEL_PORT|kv_(producer|consumer)|UCX_NET_DEVICES|UCX_IB_GID_INDEX|UCX_IB_ADDR_TYPE|UCX_RCACHE_MAX_UNRELEASED|rdma/ib' \
  "$EXP_DIR/deploy.yaml"

kubectl apply --dry-run=server \
  -n "$NAMESPACE" \
  -f "$EXP_DIR/deploy.yaml"
```

Require a Bound PVC, an existing Secret, `MODEL_SNAPSHOT_CHECK=0`,
`mlx5_8:1`, GID index `3`, Ethernet addressing, registration-cache value
`1024`, host networking with cluster DNS, non-conflicting worker ports,
RDMA=2 for prefill, and RDMA=4 for decode.

## 6. Prove NIXL initialization before loading the model

This is the decisive gate for the earlier failure. It launches one temporary
Pod on each node with the exact runtime image, the validated host-network
configuration, one GPU, one `rdma/ib` resource, and the production UCX
variables.

**Run only on `gpu05`:**

```bash
tee "$EXP_DIR/rdma-preflight.yaml" >/dev/null <<'EOF_RDMA_PREFLIGHT'
apiVersion: v1
kind: List
items:
  - apiVersion: v1
    kind: Pod
    metadata:
      name: qwen-rdma-prefill-check
    spec:
      restartPolicy: Never
      hostNetwork: true
      dnsPolicy: ClusterFirstWithHostNet
      nodeSelector:
        qwen.nvidia.com/role: prefill
      tolerations:
        - key: node-role.kubernetes.io/control-plane
          operator: Exists
          effect: NoSchedule
      containers:
        - name: check
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          command: [/bin/bash, -lc]
          args:
            - |
              set -euxo pipefail
              echo "POD=$POD_NAME NODE=$NODE_NAME"
              ls -la /dev/infiniband
              cat /sys/class/infiniband/mlx5_8/ports/1/state
              cat /sys/class/infiniband/mlx5_8/ports/1/link_layer
              cat /sys/class/infiniband/mlx5_8/ports/1/gids/3
              python3 -c "import os,nixl; nixl.nixl_agent(os.environ['POD_NAME']); print('NIXL_UCX_BACKEND_OK')"
          env: &ucx-env
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: NODE_NAME
              valueFrom:
                fieldRef:
                  fieldPath: spec.nodeName
            - name: UCX_TLS
              value: rc_x,rc,cuda_copy,cuda_ipc
            - name: UCX_NET_DEVICES
              value: mlx5_8:1
            - name: UCX_IB_ADDR_TYPE
              value: eth
            - name: UCX_IB_GID_INDEX
              value: "3"
            - name: UCX_RNDV_SCHEME
              value: get_zcopy
            - name: UCX_RNDV_THRESH
              value: "0"
            - name: UCX_IB_REG_METHODS
              value: odp,rcache
            - name: UCX_RCACHE_MAX_UNRELEASED
              value: "1024"
            - name: UCX_RC_TIMEOUT
              value: 600s
            - name: UCX_KEEPALIVE_INTERVAL
              value: 300s
            - name: UCX_LOG_LEVEL
              value: info
            - name: NIXL_LOG_LEVEL
              value: DEBUG
          securityContext:
            runAsUser: 0
            capabilities:
              add: [IPC_LOCK, SYS_RESOURCE]
          resources:
            requests:
              nvidia.com/gpu: "1"
              rdma/ib: "1"
            limits:
              nvidia.com/gpu: "1"
              rdma/ib: "1"
  - apiVersion: v1
    kind: Pod
    metadata:
      name: qwen-rdma-decode-check
    spec:
      restartPolicy: Never
      hostNetwork: true
      dnsPolicy: ClusterFirstWithHostNet
      nodeSelector:
        qwen.nvidia.com/role: decode
      containers:
        - name: check
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
          command: [/bin/bash, -lc]
          args:
            - |
              set -euxo pipefail
              echo "POD=$POD_NAME NODE=$NODE_NAME"
              ls -la /dev/infiniband
              cat /sys/class/infiniband/mlx5_8/ports/1/state
              cat /sys/class/infiniband/mlx5_8/ports/1/link_layer
              cat /sys/class/infiniband/mlx5_8/ports/1/gids/3
              python3 -c "import os,nixl; nixl.nixl_agent(os.environ['POD_NAME']); print('NIXL_UCX_BACKEND_OK')"
          env: *ucx-env
          securityContext:
            runAsUser: 0
            capabilities:
              add: [IPC_LOCK, SYS_RESOURCE]
          resources:
            requests:
              nvidia.com/gpu: "1"
              rdma/ib: "1"
            limits:
              nvidia.com/gpu: "1"
              rdma/ib: "1"
EOF_RDMA_PREFLIGHT

kubectl delete -f "$EXP_DIR/rdma-preflight.yaml" \
  -n "$NAMESPACE" --ignore-not-found
kubectl apply -f "$EXP_DIR/rdma-preflight.yaml" \
  -n "$NAMESPACE"

kubectl wait --for=jsonpath='{.status.phase}'=Succeeded \
  pod/qwen-rdma-prefill-check pod/qwen-rdma-decode-check \
  -n "$NAMESPACE" --timeout=5m || true

kubectl get pod qwen-rdma-prefill-check qwen-rdma-decode-check \
  -n "$NAMESPACE" -o wide
kubectl logs qwen-rdma-prefill-check -n "$NAMESPACE"
kubectl logs qwen-rdma-decode-check -n "$NAMESPACE"
```

Both Pods must show `Succeeded`, the correct HCA state/link/GID, and
`NIXL_UCX_BACKEND_OK`. Stop if either log contains `Address not valid`,
`NIXL_ERR_BACKEND`, `No such device`, or a TCP fallback.

After both pass:

```bash
kubectl delete -f "$EXP_DIR/rdma-preflight.yaml" \
  -n "$NAMESPACE" --ignore-not-found
```

## 7. Deploy

**Run only on `gpu05`, and only after every preceding gate passes.**

```bash
kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/deploy.yaml"

kubectl get dynamographdeployment "$DGD" \
  -n "$NAMESPACE" -o yaml | sed -n '1,100p'

kubectl get pods -n "$NAMESPACE" \
  -l "$GRAPH_LABEL" \
  -L nvidia.com/dynamo-component-type \
  -o wide -w
```

Expected graph: one frontend, two TP=2 prefill workers, and one TP=4 decode
worker. The worker total is eight H100s. Both prefill replicas declare host
port `9090`, so Kubernetes must place them on different nodes. Decode uses
host port `9091` and can safely share either node with one prefill worker.

The API server may store the submitted `v1alpha1` `spec.services` manifest as
the preferred `spec.components` representation. Inspecting `.spec.services`
alone can therefore return `null`; this is not a deployment error. The
generated Pod specification is the authoritative runtime check.

## 8. Verify the generated worker Pods

**Run only on `gpu05`.**

```bash
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o json |
jq -r '
  .items[] |
  select(.metadata.name | test("vllm(prefill|decode)worker")) |
  .metadata.name as $pod |
  .spec.containers[] |
  .env[]? |
  select(.name | test("^(UCX_|NIXL_)")) |
  "\($pod)\t\(.name)=\(.value)"
' | sort

kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o json |
jq -r '
  .items[] |
  select(.metadata.name | test("vllm(prefill|decode)worker")) |
  [.metadata.name,
   "NODE=" + (.spec.nodeName // "pending"),
   "POD_IP=" + (.status.podIP // "pending"),
   "HOST_NETWORK=" + ((.spec.hostNetwork // false) | tostring),
   "DNS_POLICY=" + (.spec.dnsPolicy // "missing"),
   "HOST_PORTS=" + ([.spec.containers[].ports[]? | select(.hostPort != null) | (.name + "=" + (.hostPort | tostring))] | join(",")),
   "GPU=" + ([.spec.containers[].resources.limits["nvidia.com/gpu"] // empty][0] // "missing"),
   "RDMA=" + ([.spec.containers[].resources.limits["rdma/ib"] // empty][0] // "missing")]
  | @tsv
'
```

Every worker must show `mlx5_8:1`, `eth`, GID index `3`, and the production
UCX settings. Prefill must show GPU/RDMA `2/2`; decode must show `4/4`.
Every worker must show `HOST_NETWORK=true` and
`DNS_POLICY=ClusterFirstWithHostNet`. Prefill workers must expose
system/NIXL-telemetry/FPM/NIXL-side-channel host ports
`9090/19090/20380/5600`; decode must expose `9091/19091/20381/5601`.

## 9. Observe startup and diagnose failures

**Run only on `gpu05`.**

```bash
export GRAPH_LABEL='nvidia.com/dynamo-graph-deployment-name=qwen3-32b-fp8-vllm-disagg'

kubectl logs -n "$NAMESPACE" \
  -l "$GRAPH_LABEL" \
  --all-containers=true \
  --prefix \
  --timestamps \
  --tail=200 \
  --max-log-requests=10 \
  --ignore-errors=true \
  --follow
```

If a Pod is `Pending`, it has no startup log. Show its scheduling events:

```bash
for POD in $(
  kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
    --field-selector=status.phase=Pending -o name
); do
  echo "===== $POD ====="
  kubectl describe -n "$NAMESPACE" "$POD" | sed -n '/Events:/,$p'
done
```

Show NIXL/UCX failures from every current or previously crashed worker:

```bash
for POD in $(
  kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' |
  grep -E 'vllm(prefill|decode)worker'
); do
  echo "===== $POD: current ====="
  kubectl logs "$POD" -n "$NAMESPACE" --all-containers=true \
    --timestamps --tail=3000 2>&1 |
    grep -Ei 'NIXL|UCX|RDMA|TCP|Address not valid|No such device|Traceback|ERROR|failed' || true

  echo "===== $POD: previous ====="
  kubectl logs "$POD" -n "$NAMESPACE" --all-containers=true --previous \
    --timestamps --tail=3000 2>&1 |
    grep -Ei 'NIXL|UCX|RDMA|TCP|Address not valid|No such device|Traceback|ERROR|failed' || true
done
```

Successful bring-up must instantiate/load the UCX backend and must not contain
`NIXL_ERR_BACKEND`, `Address not valid`, `No such device`, or a TCP fallback.
Frontend 503 responses are expected only while workers are still loading.

## 10. Readiness and inference smoke test

Wait for all four graph Pods:

```bash
kubectl wait --for=condition=Ready pod \
  -l "$GRAPH_LABEL" \
  -n "$NAMESPACE" --timeout=45m

kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  -L nvidia.com/dynamo-component-type -o wide
kubectl get services -n "$NAMESPACE" | grep "$DGD"
```

Start the frontend tunnel in one terminal:

```bash
kubectl port-forward -n "$NAMESPACE" \
  service/qwen3-32b-fp8-vllm-disagg-frontend 8000:8000
```

Run in a second terminal:

```bash
curl -fsS http://127.0.0.1:8000/v1/models | jq

curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-32B-FP8",
    "messages": [{"role": "user", "content": "Reply with only: VLLM_RDMA_OK"}],
    "temperature": 0,
    "max_tokens": 32
  }' | jq
```

## 11. Post-deployment RDMA acceptance

Run after at least one successful inference request:

```bash
for POD in $(
  kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o name |
  grep -E 'vllm(prefill|decode)worker' |
  cut -d/ -f2
); do
  echo "===== $POD: devices ====="
  kubectl exec "$POD" -n "$NAMESPACE" -- ls -la /dev/infiniband

  echo "===== $POD: UCX capabilities when ucx_info is installed ====="
  kubectl exec "$POD" -n "$NAMESPACE" -- \
    sh -lc 'command -v ucx_info >/dev/null && ucx_info -d | grep -E "Transport: (rc|rc_x)|memory types" || true'

  echo "===== $POD: backend and transfer evidence ====="
  kubectl logs "$POD" -n "$NAMESPACE" --all-containers=true \
    --timestamps --tail=5000 2>&1 |
    grep -Ei 'Backend UCX|backend plugin: UCX|KV Transfer metrics|successful transfers|falling back to TCP|NIXL_ERR' || true
done
```

Accept the deployment only when:

- all four Pods remain Ready without restarts;
- each worker sees `/dev/infiniband`;
- NIXL reports the UCX backend initialized;
- the inference response is correct;
- logs contain no TCP fallback or NIXL transfer failure;
- if transfer metrics are emitted, successful transfers are nonzero and
  bandwidth is in GB/s rather than MB/s.

## 12. Failure decision tree

- **Temporary preflight fails with `Address not valid` or `No such device`:**
  verify `hostNetwork: true`, `ClusterFirstWithHostNet`, `mlx5_8:1`, GID index
  `3`, and the same nonzero GID as the host. If they all match, investigate
  runtime UCX/host OFED compatibility before loading the model.
- **`/dev/infiniband` is missing:** the Pod did not receive its `rdma/ib`
  resource or the RDMA device-plugin DaemonSet is unhealthy.
- **UCX reports only host memory:** GPUDirect is unavailable; verify
  `nvidia_peermem` and the GPU/RDMA driver stack.
- **A Pod is Pending:** inspect scheduler events; do not look for container
  logs. Typical causes are occupied GPUs, insufficient `rdma/ib`, affinity, or
  a control-plane taint.
- **Frontend returns 503:** inspect worker readiness and NIXL logs. The
  frontend cannot serve until workers register.
- **`NIXL_ERR_BACKEND` continues after the host-network preflight passes:** compare
  the live worker environment and resource limits with the preflight Pod; the
  model is not the cause of backend creation failure.

Do not remove `hostNetwork: true` on this cluster unless a secondary RoCE
network is added to the Pod. Do not remove the explicit port assignments:
host-networked workers on the same node cannot bind identical system,
telemetry, forward-pass-metrics, or NIXL side-channel ports.

## 13. Shutdown & Complete GPU VRAM Cleanup

### 13.1 Delete Kubernetes Resources (`gpu05`)

```bash
export NAMESPACE=qwen32-bench

# Delete all Dynamo Graph Deployments in the namespace
kubectl delete dynamographdeployments --all -n "$NAMESPACE" --ignore-not-found

# Force-delete all lingering pods to immediately release GPU claims
kubectl delete pods --all -n "$NAMESPACE" --force --grace-period=0

# Confirm all pods are deleted
kubectl wait --for=delete pod --all -n "$NAMESPACE" --timeout=1m || true
```

### 13.2 Host GPU VRAM Cleanup (`gpu05` & `gpu06`)

Because `hostNetwork: true` and IPC locks are used, orphaned PyTorch / CUDA worker processes can occasionally persist on the host OS after Pod deletion. Run this on **both `gpu05` and `gpu06`**:

```bash
# 1. Kill any lingering vLLM / Dynamo Python processes on the host
pkill -9 -f "dynamo\.vllm|dynamo\.frontend|vllm" || true

# 2. Kill any processes holding /dev/nvidia* device handles
sudo fuser -v /dev/nvidia* 2>/dev/null | awk '{print $2}' | xargs -r sudo kill -9

# 3. Verify all GPU VRAM is completely freed (0 MiB used)
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
```
