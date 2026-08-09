# Experiment 2: P/D disaggregation with KV-aware routing

This deployment keeps experiment 1's model, 4P/4D TP=2 topology, GPU
placement, NIXL/UCX transport, and memory settings unchanged. Its only feature
change is event-driven KV-aware routing. KV offloading remains disabled.

Complete [the cluster setup](../setup.md) and validate
[experiment 1](01-pd-disaggregation.md) first.

## What changes from experiment 1

- Frontend: `DYN_ROUTER_MODE=kv` and `DYN_ROUTER_USE_KV_EVENTS=true`.
- Workers: publish SGLang KV events over ZMQ.
- No `--enable-hierarchical-cache` and no external KV storage backend.

Each pod has its own network namespace, so all replicas can bind ZMQ port 5557
without a host-port collision. In disaggregated mode, reusable-prefix overlap
affects prefill selection; decode selection does not assume transferred blocks
are reusable.

## 1. Confirm a clean cluster

Run on `gpu05`:

```bash
export NAMESPACE=qwen32-bench

kubectl get dynamographdeployments -n "$NAMESPACE"
kubectl get pods -n "$NAMESPACE" -o wide
kubectl get pvc model-cache -n "$NAMESPACE"
```

If `qwen32-pd` or another DGD exists, delete it using that guide's shutdown
steps and wait until all of its experiment pods are gone.

## 2. Create the deployment manifest

Save the following as
`/ephemeral/shared/qwen3-32b/manifests/02-pd-kv-aware-routing.yaml`:

**Run on `gpu05` — Kubernetes admin terminal:** create the deployment file on
the shared filesystem.

```bash
tee /ephemeral/shared/qwen3-32b/manifests/02-pd-kv-aware-routing.yaml >/dev/null <<'EOF'
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: qwen32-pd-kv
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

The decode workers publish events to match Dynamo's pinned SGLang KV-router
launch path. Dynamo's integrated disaggregated router applies reusable-prefix
overlap to the prefill choice and disables that assumption for decode.

## 3. Apply and verify placement

```bash
kubectl apply -f /ephemeral/shared/qwen3-32b/manifests/02-pd-kv-aware-routing.yaml

kubectl get pods -n "$NAMESPACE" \
  -l nvidia.com/dynamo-graph-deployment-name=qwen32-pd-kv \
  -o wide -w
```

Once nine pods exist:

```bash
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=qwen32-pd-kv \
  -n "$NAMESPACE" --timeout=45m

kubectl get pods -n "$NAMESPACE" \
  -l nvidia.com/dynamo-graph-deployment-name=qwen32-pd-kv \
  -L nvidia.com/dynamo-component-type -o wide
```

Require 4P on `gpu05`, 4D on `gpu06`, and the frontend on `gpu05`. Use the
event/describe/log diagnostics from experiment 1 for any pending or restarting
pod.

## 4. Verify NIXL and KV event startup

```bash
kubectl logs -n "$NAMESPACE" \
  -l nvidia.com/dynamo-component-type=prefill \
  --all-containers --prefix --tail=2500 --max-log-requests=8 \
  | grep -Ei 'nixl|ucx|rdma|kv.event|zmq'

kubectl logs -n "$NAMESPACE" \
  -l nvidia.com/dynamo-component-type=decode \
  --all-containers --prefix --tail=2500 --max-log-requests=8 \
  | grep -Ei 'nixl|ucx|rdma|kv.event|zmq'

kubectl logs -n "$NAMESPACE" \
  -l nvidia.com/dynamo-component-type=frontend \
  --all-containers --tail=1000 \
  | grep -Ei 'router|kv|event|prefill'
```

The frontend must start in KV mode, workers must publish `kv-events`, and NIXL
must instantiate UCX. A router silently running prediction-only routing is not
accepted for this experiment.

## 5. Run a repeated-prefix acceptance check

Open the private frontend port-forward:

```bash
export FRONTEND_POD="$(kubectl get pods -n "$NAMESPACE" \
  -l 'nvidia.com/dynamo-graph-deployment-name=qwen32-pd-kv,nvidia.com/dynamo-component-type=frontend' \
  -o jsonpath='{.items[0].metadata.name}')"

kubectl port-forward -n "$NAMESPACE" pod/"$FRONTEND_POD" 8000:8000
```

In another terminal, verify the model and send eight prompts with an identical
long prefix and different suffixes:

```bash
curl -fsS http://127.0.0.1:8000/v1/models

python3 - <<'PY'
import json
import urllib.request

url = "http://127.0.0.1:8000/v1/chat/completions"
prefix = ("This is a stable routing prefix used only for cache validation. " * 160)

for number in range(8):
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

Independent random prompts cannot prove KV-aware behavior; the shared prefix
above is intentional.

Query the frontend metrics after the requests:

```bash
curl -fsS http://127.0.0.1:8000/metrics \
  | grep -E 'dynamo_(component_)?(router_kv_hit_rate|kv_cache_events_applied)|dynamo_frontend_cached_tokens'
```

Metric prefixes can differ slightly between builds. If the exact filter is
empty, first inspect only non-sensitive metric names:

```bash
curl -fsS http://127.0.0.1:8000/metrics \
  | grep -Ei 'kv.*(event|hit|cache)|cached_tokens|router'
```

The applied-event counter must increase above zero. Repeated requests should
also produce cached-token or hit-rate activity. A zero event count means this
is not the requested event-driven routing experiment, even if requests return
successfully.

## 6. Confirm transport health

Forward an active prefill worker's system port:

```bash
export PREFILL_POD="$(kubectl get pods -n "$NAMESPACE" \
  -l 'nvidia.com/dynamo-graph-deployment-name=qwen32-pd-kv,nvidia.com/dynamo-component-type=prefill' \
  -o jsonpath='{.items[0].metadata.name}')"

kubectl port-forward -n "$NAMESPACE" pod/"$PREFILL_POD" 18081:8081
```

In another terminal:

```bash
curl -fsS http://127.0.0.1:18081/metrics \
  | grep -E 'sglang:(kv_transfer_speed_gb_s|kv_transfer_latency_ms|kv_transfer_total_mb|num_.*transfer_failed)'
```

Require nonzero transfer activity and zero transfer/bootstrap failures.

## 7. Acceptance criteria

Experiment 2 is deployment-ready only when:

- experiment 1 is fully stopped;
- all nine pods are Ready with the same placement and GPU allocation as
  experiment 1;
- the frontend reports KV router mode with worker-event consumption enabled;
- both worker roles initialize their ZMQ publishers;
- the router's applied KV-event count is nonzero after repeated-prefix traffic;
- cached-token or KV-hit activity appears for the repeated prefix;
- NIXL uses UCX and transfer failure counters stay zero;
- no HiCache or other KV-offload flag is enabled.

Stop here after recording readiness. Benchmarking is intentionally deferred.

## 8. Shutdown

Stop port-forwards, then run:

```bash
kubectl delete dynamographdeployment qwen32-pd-kv -n "$NAMESPACE" --wait=true
kubectl wait --for=delete pod \
  -l nvidia.com/dynamo-graph-deployment-name=qwen32-pd-kv \
  -n "$NAMESPACE" --timeout=10m
kubectl get pods -n "$NAMESPACE" -o wide
```

Keep the shared setup resources for experiment 3.

## Troubleshooting boundary

- event counter stays zero: confirm `DYN_ROUTER_USE_KV_EVENTS=true`, worker
  `--kv-events-config`, ZMQ startup logs, and pod-to-pod connectivity;
- hit rate stays zero while event count grows: confirm the prompts really share
  more than one 64-token page and that no pod restarted between requests;
- only one worker registers: inspect all four worker logs and the DGD events;
- transfer errors: diagnose the same RDMA/NIXL path as experiment 1 before
  investigating router behavior.

## References

- [Dynamo v1.3.0 SGLang KV-router launcher](https://github.com/ai-dynamo/dynamo/blob/v1.3.0/examples/backends/sglang/launch/disagg_router.sh)
- [Dynamo disaggregated routing architecture](https://github.com/ai-dynamo/dynamo/blob/v1.3.0/docs/components/router/router-disaggregated-serving.md)
- [Dynamo router configuration](https://github.com/ai-dynamo/dynamo/blob/v1.3.0/docs/components/router/router-configuration.md)
