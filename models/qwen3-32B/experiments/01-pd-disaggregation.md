# Experiment 1: P/D disaggregation

This deployment is the control configuration for the three requested
experiments. It separates prefill and decode, transfers KV state with NIXL over
UCX/RDMA, uses round-robin frontend routing, and does not enable KV events or
KV offloading.

Complete [the cluster setup](../setup.md) first.

## Configuration under test

| Component | Placement | Replicas | GPUs per replica | Mode |
|---|---|---:|---:|---|
| Frontend | `gpu05` | 1 | 0 | Round-robin |
| Prefill | `gpu05` | 4 | 2 | SGLang prefill |
| Decode | `gpu06` | 4 | 2 | SGLang decode |

Fixed settings: TP=2, page size 64, model revision
`aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`, memory fraction 0.82, and
40 GiB `/dev/shm` per worker pod.

## 1. Confirm a clean cluster

Run on `gpu05`:

```bash
export NAMESPACE=qwen32-bench

kubectl get dynamographdeployments -n "$NAMESPACE"
kubectl get pods -n "$NAMESPACE" -o wide
kubectl get nodes -L qwen.nvidia.com/role
kubectl get pvc model-cache -n "$NAMESPACE"
```

There must be no active experiment DGD and no unrelated GPU workload. If an
earlier experiment exists, follow its shutdown section and wait for its pods to
terminate. Do not deploy two experiments together.

## 2. Create the deployment manifest

Save the following as
`/ephemeral/shared/qwen3-32b/manifests/01-pd-disaggregation.yaml`:

**Run on `gpu05` — Kubernetes admin terminal:** create the deployment file on
the shared filesystem.

```bash
tee /ephemeral/shared/qwen3-32b/manifests/01-pd-disaggregation.yaml >/dev/null <<'EOF'
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: qwen32-pd
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
                  value: round-robin
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

This manifest intentionally contains no `--kv-events-config`, no HiCache
flags, and no KV storage backend.

## 3. Apply and watch startup

```bash
kubectl apply -f /ephemeral/shared/qwen3-32b/manifests/01-pd-disaggregation.yaml
kubectl get dynamographdeployment qwen32-pd -n "$NAMESPACE" -w
```

In a second terminal:

```bash
kubectl get pods -n "$NAMESPACE" \
  -l nvidia.com/dynamo-graph-deployment-name=qwen32-pd \
  -o wide -w
```

Model initialization can take several minutes. Once nine pods exist, wait for
readiness:

```bash
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=qwen32-pd \
  -n "$NAMESPACE" --timeout=45m
```

If a pod is `Pending` or restarts, inspect it before retrying:

```bash
kubectl get events -n "$NAMESPACE" --sort-by=.lastTimestamp | tail -80
kubectl describe pod POD_NAME -n "$NAMESPACE"
kubectl logs POD_NAME -n "$NAMESPACE" --all-containers --tail=300
```

Do not lower GPU/RDMA requests or remove capabilities merely to make a pod
schedule. Fix the cluster prerequisite that the event or description names.

## 4. Verify topology and P/D transport

Exactly four prefill pods must be on `gpu05`, four decode pods on `gpu06`, and
the frontend on `gpu05`:

```bash
kubectl get pods -n "$NAMESPACE" \
  -l nvidia.com/dynamo-graph-deployment-name=qwen32-pd \
  -L nvidia.com/dynamo-component-type -o wide
```

Inspect NIXL/UCX initialization across both pools:

```bash
kubectl logs -n "$NAMESPACE" \
  -l nvidia.com/dynamo-component-type=prefill \
  --all-containers --prefix --tail=2000 --max-log-requests=8 \
  | grep -Ei 'nixl|ucx|rdma|transport'

kubectl logs -n "$NAMESPACE" \
  -l nvidia.com/dynamo-component-type=decode \
  --all-containers --prefix --tail=2000 --max-log-requests=8 \
  | grep -Ei 'nixl|ucx|rdma|transport'
```

The logs must show the NIXL UCX backend initializing. Treat a TCP fallback,
missing `/dev/infiniband`, bootstrap failure, NCCL error, or repeated worker
registration as a failed deployment.

## 5. Check the API and force a real transfer

Open a private port-forward from `gpu05`; this is for readiness testing only:

```bash
export FRONTEND_POD="$(kubectl get pods -n "$NAMESPACE" \
  -l 'nvidia.com/dynamo-graph-deployment-name=qwen32-pd,nvidia.com/dynamo-component-type=frontend' \
  -o jsonpath='{.items[0].metadata.name}')"

kubectl port-forward -n "$NAMESPACE" pod/"$FRONTEND_POD" 8000:8000
```

In another terminal:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models

curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "Qwen/Qwen3-32B-FP8",
    "messages": [{"role": "user", "content": "Reply with exactly READY. /no_think"}],
    "temperature": 0,
    "max_tokens": 16,
    "stream": false
  }'
```

Send several more requests so every role becomes active:

```bash
for request in 1 2 3 4 5 6 7 8; do
  curl -fsS http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    --data "{\"model\":\"Qwen/Qwen3-32B-FP8\",\"messages\":[{\"role\":\"user\",\"content\":\"Return request number ${request}. /no_think\"}],\"temperature\":0,\"max_tokens\":32}" \
    >/dev/null
done
```

Inspect worker transfer metrics. Choose one active prefill pod, forward its
system port, and query the exact SGLang metrics:

```bash
export PREFILL_POD="$(kubectl get pods -n "$NAMESPACE" \
  -l 'nvidia.com/dynamo-graph-deployment-name=qwen32-pd,nvidia.com/dynamo-component-type=prefill' \
  -o jsonpath='{.items[0].metadata.name}')"

kubectl port-forward -n "$NAMESPACE" pod/"$PREFILL_POD" 18081:8081
```

In another terminal:

```bash
curl -fsS http://127.0.0.1:18081/metrics \
  | grep -E 'sglang:(kv_transfer_speed_gb_s|kv_transfer_latency_ms|kv_transfer_total_mb|num_.*transfer_failed)'
```

At least one worker must report nonzero KV transfer activity and all transfer
failure counters must remain zero. Repeat the port-forward with other prefill
or decode pods if the first pod did not receive a request.

## 6. Acceptance criteria

Experiment 1 is deployment-ready only when:

- all nine pods remain Ready with zero restarts;
- placement is 4P on `gpu05`, 4D on `gpu06`, and frontend on `gpu05`;
- all 16 GPUs are allocated exactly once;
- `/v1/models` lists `Qwen/Qwen3-32B-FP8`;
- the smoke response is valid;
- NIXL selected UCX and the transfer metrics become nonzero;
- transfer/bootstrap failure counters remain zero;
- the frontend is explicitly in `round-robin` mode;
- no KV-event or HiCache flag is present.

Stop here after recording readiness. Benchmarking is intentionally handled
later.

## 7. Shutdown

Stop any port-forward, then remove only this experiment:

```bash
kubectl delete dynamographdeployment qwen32-pd -n "$NAMESPACE" --wait=true
kubectl wait --for=delete pod \
  -l nvidia.com/dynamo-graph-deployment-name=qwen32-pd \
  -n "$NAMESPACE" --timeout=10m
kubectl get pods -n "$NAMESPACE" -o wide
```

Keep the model PVC, model download job, operators, node labels, and namespace;
they are prerequisites for experiments 2 and 3.

## Troubleshooting boundary

- `Pending`: inspect GPU and `rdma/ib` allocatable resources and node labels.
- `ImagePullBackOff`: verify NGC reachability and the exact `1.3.0` image.
- model lookup failure in offline mode: rerun the pinned download job from the
  setup guide; do not change the pinned local snapshot path.
- NIXL bootstrap timeout: verify pod-to-pod networking, HCA state, RDMA device
  exposure, and firewall rules between the two private nodes.
- OOM during bring-up: collect pod logs and node RAM/HBM use. Do not silently
  change `mem-fraction-static`, because it must remain identical in all three
  experiments.

## References

- [Dynamo v1.3.0 SGLang P/D manifest](https://github.com/ai-dynamo/dynamo/blob/v1.3.0/examples/backends/sglang/deploy/v1beta1/disagg.yaml)
- [Dynamo disaggregated serving](https://docs.nvidia.com/dynamo/latest/user-guides/disaggregated-serving)
- [SGLang P/D disaggregation](https://github.com/sgl-project/sglang/blob/v0.5.14/docs/advanced_features/pd_disaggregation.md)
