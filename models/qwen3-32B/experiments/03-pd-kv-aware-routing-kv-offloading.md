# Experiment 3: P/D + KV-aware routing + KV offloading

This deployment keeps experiment 2 unchanged and adds SGLang HiCache on the
prefill workers. The offload hierarchy under test is:

```text
prefill GPU HBM (L1) -> pinned host RAM on the same prefill worker (L2)
```

It does not use disk, a shared L3 cache, model-weight CPU offload, or
decode-generated KV offload. Those require different backends and would be
different experiments.

Complete [the cluster setup](../setup.md) and validate
[experiment 2](02-pd-kv-aware-routing.md) first.

## What changes from experiment 2

Only prefill workers add:

```text
--enable-hierarchical-cache
--hicache-ratio 2
--hicache-write-policy write_through
--hicache-io-backend kernel
```

The frontend explicitly fixes the host-tier routing credit at its Dynamo 1.3.0
default, `DYN_ROUTER_HOST_CACHE_HIT_WEIGHT=0.75`. Everything else remains the
same as experiment 2.

## 1. Host-memory safety gate

`--hicache-ratio 2` allocates a pinned host KV pool twice the GPU KV pool for
each SGLang TP rank. There are eight prefill ranks on `gpu05`, so this can use
substantial RAM. Check the host before deploying:

```bash
# Run on gpu05
free -h
grep -E 'MemTotal|MemAvailable|HugePages' /proc/meminfo
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv
```

Also run from the Kubernetes administrator terminal:

```bash
export NAMESPACE=qwen32-bench
kubectl top node gpu05 2>/dev/null || true
kubectl get pods -A -o wide --field-selector spec.nodeName=gpu05
```

Do not deploy if the node cannot hold Kubernetes services, four 40 GiB
`/dev/shm` limits, four model processes, and the HiCache pools with operating
headroom. If ratio 2 is too large, measure the GPU KV pool first and use
`--hicache-size N` with a justified per-rank capacity greater than that pool.
Do not guess `N`, and use the same value for every prefill replica.

## 2. Confirm experiment 2 is stopped

```bash
kubectl get dynamographdeployments -n "$NAMESPACE"
kubectl get pods -n "$NAMESPACE" -o wide
kubectl get pvc model-cache -n "$NAMESPACE"
```

Delete `qwen32-pd-kv` using experiment 2's shutdown steps if it remains. All 16
GPUs must be free before continuing.

## 3. Create the deployment manifest

Save the following as
`/ephemeral/shared/qwen3-32b/manifests/03-pd-kv-aware-routing-kv-offloading.yaml`:

**Run on `gpu05` — Kubernetes admin terminal:** create the deployment file on
the shared filesystem.

```bash
tee /ephemeral/shared/qwen3-32b/manifests/03-pd-kv-aware-routing-kv-offloading.yaml >/dev/null <<'EOF'
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: qwen32-pd-kv-offload
  namespace: qwen32-bench
spec:
  backendFramework: sglang
  env:
    - name: HF_HOME
      value: /model-store
    - name: HF_HUB_OFFLINE
      value: "1"
    - name: TRANSFORMERS_OFFLINE
      value: "1"
  components:
    - name: Frontend
      type: frontend
      replicas: 1
      podTemplate:
        spec:
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
            - name: main
              image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0
              imagePullPolicy: IfNotPresent
              env:
                - name: DYN_HTTP_PORT
                  value: "8000"
                - name: DYN_ROUTER_MODE
                  value: kv
                - name: DYN_ROUTER_USE_KV_EVENTS
                  value: "true"
                - name: DYN_ROUTER_HOST_CACHE_HIT_WEIGHT
                  value: "0.75"
              envFrom:
                - secretRef:
                    name: hf-token-secret
                    optional: true
              ports:
                - name: http
                  containerPort: 8000
              volumeMounts:
                - name: model-cache
                  mountPath: /model-store
                  readOnly: true
          volumes:
            - name: model-cache
              persistentVolumeClaim:
                claimName: model-cache

    - name: prefill
      type: prefill
      replicas: 4
      sharedMemorySize: 40Gi
      podTemplate:
        spec:
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
          terminationGracePeriodSeconds: 120
          containers:
            - name: main
              image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0
              imagePullPolicy: IfNotPresent
              command:
                - python3
                - -m
                - dynamo.sglang
              args:
                - --model-path
                - /model-store/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df
                - --served-model-name
                - Qwen/Qwen3-32B-FP8
                - --tp-size
                - "2"
                - --page-size
                - "64"
                - --context-length
                - "40960"
                - --trust-remote-code
                - --skip-tokenizer-init
                - --mem-fraction-static
                - "0.82"
                - --disaggregation-mode
                - prefill
                - --disaggregation-transfer-backend
                - nixl
                - --disaggregation-bootstrap-port
                - "30001"
                - --host
                - 0.0.0.0
                - --kv-events-config
                - '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}'
                - --enable-hierarchical-cache
                - --hicache-ratio
                - "2"
                - --hicache-write-policy
                - write_through
                - --hicache-io-backend
                - kernel
                - --enable-metrics
                - --disable-piecewise-cuda-graph
              env:
                - name: DYN_SYSTEM_PORT
                  value: "8081"
                - name: GLOO_SOCKET_IFNAME
                  value: eth0
                - name: NCCL_SOCKET_IFNAME
                  value: eth0
                - name: NCCL_IB_DISABLE
                  value: "0"
                - name: SGLANG_DISAGGREGATION_NIXL_BACKEND
                  value: UCX
                - name: UCX_TLS
                  value: rc_x,rc,cuda_copy,cuda_ipc
                - name: UCX_IB_ADDR_TYPE
                  value: eth
                - name: UCX_RNDV_SCHEME
                  value: get_zcopy
                - name: UCX_RNDV_THRESH
                  value: "0"
                - name: UCX_RC_TIMEOUT
                  value: 600s
                - name: UCX_KEEPALIVE_INTERVAL
                  value: 300s
              envFrom:
                - secretRef:
                    name: hf-token-secret
                    optional: true
              resources:
                requests:
                  nvidia.com/gpu: "2"
                  rdma/ib: "2"
                limits:
                  nvidia.com/gpu: "2"
                  rdma/ib: "2"
              securityContext:
                runAsUser: 0
                capabilities:
                  add:
                    - IPC_LOCK
                    - SYS_RESOURCE
              ports:
                - name: metrics
                  containerPort: 8081
                - name: kv-events
                  containerPort: 5557
              volumeMounts:
                - name: model-cache
                  mountPath: /model-store
                  readOnly: true
          volumes:
            - name: model-cache
              persistentVolumeClaim:
                claimName: model-cache

    - name: decode
      type: decode
      replicas: 4
      sharedMemorySize: 40Gi
      podTemplate:
        spec:
          nodeSelector:
            qwen.nvidia.com/role: decode
          tolerations:
            - key: nvidia.com/gpu
              operator: Equal
              value: "true"
              effect: NoSchedule
          terminationGracePeriodSeconds: 120
          containers:
            - name: main
              image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0
              imagePullPolicy: IfNotPresent
              command:
                - python3
                - -m
                - dynamo.sglang
              args:
                - --model-path
                - /model-store/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df
                - --served-model-name
                - Qwen/Qwen3-32B-FP8
                - --tp-size
                - "2"
                - --page-size
                - "64"
                - --context-length
                - "40960"
                - --trust-remote-code
                - --skip-tokenizer-init
                - --mem-fraction-static
                - "0.82"
                - --disaggregation-mode
                - decode
                - --disaggregation-transfer-backend
                - nixl
                - --disaggregation-bootstrap-port
                - "30001"
                - --host
                - 0.0.0.0
                - --kv-events-config
                - '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}'
                - --enable-metrics
                - --disable-piecewise-cuda-graph
              env:
                - name: DYN_SYSTEM_PORT
                  value: "8082"
                - name: GLOO_SOCKET_IFNAME
                  value: eth0
                - name: NCCL_SOCKET_IFNAME
                  value: eth0
                - name: NCCL_IB_DISABLE
                  value: "0"
                - name: SGLANG_DISAGGREGATION_NIXL_BACKEND
                  value: UCX
                - name: UCX_TLS
                  value: rc_x,rc,cuda_copy,cuda_ipc
                - name: UCX_IB_ADDR_TYPE
                  value: eth
                - name: UCX_RNDV_SCHEME
                  value: get_zcopy
                - name: UCX_RNDV_THRESH
                  value: "0"
                - name: UCX_RC_TIMEOUT
                  value: 600s
                - name: UCX_KEEPALIVE_INTERVAL
                  value: 300s
              envFrom:
                - secretRef:
                    name: hf-token-secret
                    optional: true
              resources:
                requests:
                  nvidia.com/gpu: "2"
                  rdma/ib: "2"
                limits:
                  nvidia.com/gpu: "2"
                  rdma/ib: "2"
              securityContext:
                runAsUser: 0
                capabilities:
                  add:
                    - IPC_LOCK
                    - SYS_RESOURCE
              ports:
                - name: metrics
                  containerPort: 8082
                - name: kv-events
                  containerPort: 5557
              volumeMounts:
                - name: model-cache
                  mountPath: /model-store
                  readOnly: true
          volumes:
            - name: model-cache
              persistentVolumeClaim:
                claimName: model-cache
EOF
```

Important negative controls:

- no `--hicache-storage-backend`: host RAM is L2; that flag would add L3;
- no `DYN_SHARED_CACHE_TYPE=hicache`: that configures shared Mooncake lookup;
- no `--cpu-offload-gb`: that offloads model weights, not KV cache;
- no `--disaggregation-decode-enable-offload-kvcache`: it requires a shared L3
  backend and changes the experiment to decode-generated multi-turn offload.

## 4. Apply while watching host RAM

On `gpu05`, keep this visible during model startup:

```bash
watch -n 2 'free -h; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
```

From the Kubernetes terminal:

```bash
kubectl apply -f /ephemeral/shared/qwen3-32b/manifests/03-pd-kv-aware-routing-kv-offloading.yaml

kubectl get pods -n "$NAMESPACE" \
  -l nvidia.com/dynamo-graph-deployment-name=qwen32-pd-kv-offload \
  -o wide -w
```

Once nine pods exist:

```bash
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=qwen32-pd-kv-offload \
  -n "$NAMESPACE" --timeout=45m

kubectl get pods -n "$NAMESPACE" \
  -l nvidia.com/dynamo-graph-deployment-name=qwen32-pd-kv-offload \
  -L nvidia.com/dynamo-component-type -o wide
```

Immediately delete the DGD if `MemAvailable` approaches the host's safety
floor, the OOM killer activates, or system services become unstable. Do not let
a pinned-memory experiment take down the control plane.

## 5. Verify HiCache, KV events, and NIXL

Inspect prefill startup:

```bash
kubectl logs -n "$NAMESPACE" \
  -l nvidia.com/dynamo-component-type=prefill \
  --all-containers --prefix --tail=3000 --max-log-requests=8 \
  | grep -Ei 'hicache|hierarchical|host.*cache|pinned|nixl|ucx|kv.event|zmq'
```

Inspect decode startup separately:

```bash
kubectl logs -n "$NAMESPACE" \
  -l nvidia.com/dynamo-component-type=decode \
  --all-containers --prefix --tail=2500 --max-log-requests=8 \
  | grep -Ei 'nixl|ucx|rdma|kv.event|zmq'
```

Require all four prefill replicas to report a host HiCache pool and all eight
workers to initialize NIXL/UCX and their KV-event publisher. Decode logs must
not claim that decode-side HiCache or an L3 storage backend was enabled.

## 6. Populate and observe the host KV tier

Open the frontend port-forward:

```bash
export FRONTEND_POD="$(kubectl get pods -n "$NAMESPACE" \
  -l 'nvidia.com/dynamo-graph-deployment-name=qwen32-pd-kv-offload,nvidia.com/dynamo-component-type=frontend' \
  -o jsonpath='{.items[0].metadata.name}')"

kubectl port-forward -n "$NAMESPACE" pod/"$FRONTEND_POD" 8000:8000
```

In another terminal, send a repeated long prefix:

```bash
curl -fsS http://127.0.0.1:8000/v1/models

python3 - <<'PY'
import json
import urllib.request

url = "http://127.0.0.1:8000/v1/chat/completions"
prefix = ("This prefix is used to validate pinned-host KV offloading. " * 240)

for number in range(12):
    body = {
        "model": "Qwen/Qwen3-32B-FP8",
        "messages": [{
            "role": "user",
            "content": f"{prefix}\nReturn only the integer {number}. /no_think",
        }],
        "temperature": 0,
        "max_tokens": 16,
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        print(number, response.status)
PY
```

Query frontend routing metrics:

```bash
curl -fsS http://127.0.0.1:8000/metrics \
  | grep -Ei 'kv.*(event|hit|cache)|cached_tokens|host.*cache|router'
```

The KV-event count must increase. After host-tier demotion occurs, frontend or
debug logs should identify host-pinned residency rather than only GPU
residency.

Forward an active prefill pod's metrics endpoint:

```bash
export PREFILL_POD="$(kubectl get pods -n "$NAMESPACE" \
  -l 'nvidia.com/dynamo-graph-deployment-name=qwen32-pd-kv-offload,nvidia.com/dynamo-component-type=prefill' \
  -o jsonpath='{.items[0].metadata.name}')"

kubectl port-forward -n "$NAMESPACE" pod/"$PREFILL_POD" 18081:8081
```

In another terminal:

```bash
curl -fsS http://127.0.0.1:18081/metrics \
  | grep -E 'sglang:(hicache_host_used_tokens|hicache_host_total_tokens|kv_transfer_total_mb|num_.*transfer_failed)'
```

`sglang:hicache_host_total_tokens` must be nonzero. With sufficient repeated
prefix pressure, `sglang:hicache_host_used_tokens` must rise above zero. Query
other prefill pods if the chosen pod did not receive requests.

For definitive event-level evidence, temporarily increase only the prefill log
level in a diagnostic copy of the manifest and look for `BlockStored` or
`BlockRemoved` events whose medium/storage tier is `CPU_PINNED` or
`HostPinned`. Return the log level to its common value before later performance
work.

## 7. Acceptance criteria

Experiment 3 is deployment-ready only when:

- experiment 2 is fully stopped;
- placement, model revision, TP, page size, worker count, and NIXL settings are
  unchanged from experiment 2;
- all nine pods remain Ready and `gpu05` retains safe available RAM;
- each prefill worker reports an initialized HiCache host pool;
- host total-token capacity is nonzero and used-token capacity grows under
  repeated-prefix pressure;
- KV events are applied and host-tier residency is observable;
- NIXL/UCX P-to-D transfers remain healthy with zero failure counters;
- no L3, weight-offload, or decode-generated KV-offload feature is active.

Stop here after recording readiness. Benchmarking is intentionally deferred.

## 8. Shutdown

Stop port-forwards, then remove only this DGD:

```bash
kubectl delete dynamographdeployment qwen32-pd-kv-offload \
  -n "$NAMESPACE" --wait=true
kubectl wait --for=delete pod \
  -l nvidia.com/dynamo-graph-deployment-name=qwen32-pd-kv-offload \
  -n "$NAMESPACE" --timeout=10m
kubectl get pods -n "$NAMESPACE" -o wide
```

On `gpu05`, confirm pinned memory was returned:

```bash
free -h
```

The cache is in process-owned host RAM, so deleting the DGD clears it. The
model PVC and cluster setup remain intact.

## Troubleshooting boundary

- prefill OOM or host pressure: stop the DGD, inspect the logged device/host KV
  pool sizes, and calculate a valid per-rank capacity before retrying;
- host-total metric is zero: confirm the HiCache flags occur only on prefill
  and that the running image is exactly `sglang-runtime:1.3.0`;
- host-used stays zero: use a longer repeated prefix or more unique suffixes,
  then inspect tier transition logs; do not infer offload from RAM use alone;
- events work but host-tier routing is absent: confirm SGLang 0.5.14, host-tier
  event media, and `DYN_ROUTER_HOST_CACHE_HIT_WEIGHT=0.75`;
- transfer failure: fix NIXL/UCX independently. P/D transfer and KV offload are
  separate data paths.

## References

- [Dynamo offloading support matrix](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/router/offloading-support-matrix)
- [Dynamo SGLang HiCache integration](https://github.com/ai-dynamo/dynamo/blob/v1.3.0/docs/backends/sglang/sglang-hicache.md)
- [SGLang v0.5.14 HiCache best practices](https://github.com/sgl-project/sglang/blob/v0.5.14/docs/advanced_features/hicache_best_practices.md)
- [SGLang v0.5.14 HiCache design](https://github.com/sgl-project/sglang/blob/v0.5.14/docs/advanced_features/hicache_design.md)
