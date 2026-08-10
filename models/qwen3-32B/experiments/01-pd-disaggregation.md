# Experiment 1: P/D disaggregation with TP=2

This is the TP=2 P/D baseline for the 16×H100 cluster. It runs four prefill
workers on gpu05 and four decode workers on gpu06. Each worker uses two GPUs.
The frontend uses round-robin routing; KV events and KV offloading remain
disabled.

Complete [the cluster setup](../setup.md) first.

## Topology

| Component | Node | Components | Replicas each | GPUs each | TP |
|---|---|---:|---:|---:|---:|
| Frontend | gpu05 | 1 | 1 | 0 | — |
| Prefill | gpu05 | 4 | 1 | 2 | 2 |
| Decode | gpu06 | 4 | 1 | 2 | 2 |

The four logical workers are separate DGD components (`prefill-0` through
`prefill-3`, and likewise for decode), not one component with `replicas: 4`.
This distinction is required with `hostNetwork: true`: every worker on a node
must receive unique host ports.

| Worker index | System | NIXL metrics | Forward-pass metrics | NCCL | Bootstrap |
|---:|---:|---:|---:|---:|---:|
| 0 | 9090 | 19090 | 20380 | 29500 | 30001 |
| 1 | 9091 | 19091 | 20381 | 29501 | 30002 |
| 2 | 9092 | 19092 | 20382 | 29502 | 30003 |
| 3 | 9093 | 19093 | 20383 | 29503 | 30004 |

The same port range can be reused on gpu05 and gpu06 because each node has its
own network namespace. Do not change this back to `replicas: 4`; replicas of a
single component receive the same pod template and would collide again.

## 1. Preflight

**Run on gpu05 only — Kubernetes admin terminal:**

```bash
export NAMESPACE=qwen32-bench
export MANIFEST=/ephemeral/shared/qwen3-32b/manifests/01-pd-disaggregation.yaml

kubectl get nodes -L qwen.nvidia.com/role
kubectl get nodes \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\tGPU="}{.status.allocatable.nvidia\.com/gpu}{"\tRDMA="}{.status.allocatable.rdma/ib}{"\n"}{end}'
kubectl get dgd,pods -n "$NAMESPACE" -o wide
```

Both nodes must report eight GPUs and at least eight `rdma/ib` allocations.
The shared model PVC and pinned model snapshot must already exist.

Verify that the required host ports are not held by unrelated host processes:

```bash
for node in \
  inst-1onle-devrel-rdma-pool \
  inst-g9dwj-devrel-rdma-pool; do
  echo "===== $node ====="
  kubectl debug "node/$node" -it --quiet \
    --image=ubuntu:22.04 --profile=sysadmin -- \
    chroot /host ss -lntup | \
    grep -E ':(909[0-3]|1909[0-3]|2038[0-3]|2950[0-3]|3000[1-4])\\b' || true
done
```

If this prints a listener from the previous deployment, remove the old DGD and
wait for its pods to terminate before continuing.

## 2. Create the TP=2 manifest

**Run on gpu05 only:** the commands below create the complete YAML file. The
unquoted worker heredoc intentionally expands the calculated port values.

```bash
mkdir -p "$(dirname "$MANIFEST")"

tee "$MANIFEST" >/dev/null <<'EOF'
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
              ports:
                - name: http
                  containerPort: 8000
                  protocol: TCP
              volumeMounts:
                - name: model-cache
                  mountPath: /model-store
                  readOnly: true
          volumes:
            - name: model-cache
              persistentVolumeClaim:
                claimName: model-cache
EOF

for ROLE in prefill decode; do
  for INDEX in 0 1 2 3; do
    SYSTEM_PORT=$((9090 + INDEX))
    NIXL_PORT=$((19090 + INDEX))
    FPM_PORT=$((20380 + INDEX))
    NCCL_PORT=$((29500 + INDEX))
    BOOTSTRAP_PORT=$((30001 + INDEX))

    tee -a "$MANIFEST" >/dev/null <<EOF
    - name: ${ROLE}-${INDEX}
      type: ${ROLE}
      replicas: 1
      sharedMemorySize: 40Gi
      podTemplate:
        spec:
          hostNetwork: true
          dnsPolicy: ClusterFirstWithHostNet
          nodeSelector:
            qwen.nvidia.com/role: ${ROLE}
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
                - ${ROLE}
                - --disaggregation-transfer-backend
                - nixl
                - --disaggregation-bootstrap-port
                - "${BOOTSTRAP_PORT}"
                - --nccl-port
                - "${NCCL_PORT}"
                - --host
                - 0.0.0.0
                - --enable-metrics
                - --disable-piecewise-cuda-graph
              env:
                - name: DYN_SYSTEM_PORT
                  value: "${SYSTEM_PORT}"
                - name: NIXL_TELEMETRY_ENABLE
                  value: "n"
                - name: NIXL_TELEMETRY_PROMETHEUS_PORT
                  value: "${NIXL_PORT}"
                - name: DYN_FORWARDPASS_METRIC_PORT
                  value: "${FPM_PORT}"
                - name: NCCL_IB_DISABLE
                  value: "0"
                - name: SGLANG_DISAGGREGATION_NIXL_BACKEND
                  value: UCX
                - name: UCX_NET_DEVICES
                  value: mlx5_8:1
                - name: UCX_IB_GID_INDEX
                  value: "3"
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
              ports:
                - name: system
                  containerPort: ${SYSTEM_PORT}
                  hostPort: ${SYSTEM_PORT}
                  protocol: TCP
                - name: nixl
                  containerPort: ${NIXL_PORT}
                  hostPort: ${NIXL_PORT}
                  protocol: TCP
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
              volumeMounts:
                - name: model-cache
                  mountPath: /model-store
                  readOnly: true
          volumes:
            - name: model-cache
              persistentVolumeClaim:
                claimName: model-cache
EOF
  done
done
```

The operator sets `DYN_COMPONENT` from each component's `type`, so all four
`prefill-*` components register as prefill workers and all four `decode-*`
components register as decode workers despite their unique component names.

## 3. Validate before changing the running deployment

```bash
kubectl apply --dry-run=server -f "$MANIFEST"

python3 - "$MANIFEST" <<'PY'
import sys, yaml

with open(sys.argv[1], encoding="utf-8") as stream:
    doc = yaml.safe_load(stream)

workers = [c for c in doc["spec"]["components"] if c["type"] in {"prefill", "decode"}]
assert len(workers) == 8
assert sum(c["type"] == "prefill" for c in workers) == 4
assert sum(c["type"] == "decode" for c in workers) == 4

for role in ("prefill", "decode"):
    role_workers = [c for c in workers if c["type"] == role]
    system_ports = set()
    nixl_ports = set()
    for component in role_workers:
        assert component["replicas"] == 1
        container = component["podTemplate"]["spec"]["containers"][0]
        args = container["args"]
        assert args[args.index("--tp-size") + 1] == "2"
        assert container["resources"]["limits"]["nvidia.com/gpu"] == "2"
        env = {item["name"]: item["value"] for item in container["env"]}
        system_ports.add(env["DYN_SYSTEM_PORT"])
        nixl_ports.add(env["NIXL_TELEMETRY_PROMETHEUS_PORT"])
    assert len(system_ports) == 4
    assert len(nixl_ports) == 4

print("TP=2 manifest validated: 4 prefill + 4 decode workers with unique ports")
PY
```

Do not apply the manifest if either validation fails.

## 4. Replace the TP=8 deployment

**Run on gpu05 only.** This intentionally stops the current experiment before
starting TP=2 because both deployments need the same GPUs and ports.

```bash
kubectl delete dgd qwen32-pd -n "$NAMESPACE" --ignore-not-found

while kubectl get pods -n "$NAMESPACE" \
  -l nvidia.com/dynamo-graph-deployment-name=qwen32-pd \
  -o name | grep -q .; do
  kubectl get pods -n "$NAMESPACE" \
    -l nvidia.com/dynamo-graph-deployment-name=qwen32-pd -o wide
  sleep 5
done

kubectl apply -f "$MANIFEST"
kubectl get pods -n "$NAMESPACE" \
  -l nvidia.com/dynamo-graph-deployment-name=qwen32-pd \
  -L nvidia.com/dynamo-component-type -o wide -w
```

Expected result: nine pods total—one frontend, four TP=2 prefill workers on
gpu05, and four TP=2 decode workers on gpu06. Model initialization can take
several minutes.

## 5. Prove that the generated ports are unique

After all nine pods have been created, run:

```bash
kubectl get pods -n "$NAMESPACE" \
  -l nvidia.com/dynamo-graph-deployment-name=qwen32-pd \
  -o json | jq -r '
    .items[] |
    select(.metadata.labels["nvidia.com/dynamo-component-type"] != "frontend") |
    [
      .metadata.name,
      .spec.nodeName,
      ([.spec.containers[0].ports[] | "\(.name)=\(.hostPort)"] | join(",")),
      ([.spec.containers[0].env[] |
        select(.name == "DYN_SYSTEM_PORT" or
               .name == "NIXL_TELEMETRY_PROMETHEUS_PORT" or
               .name == "DYN_FORWARDPASS_METRIC_PORT") |
        "\(.name)=\(.value)"] | join(","))
    ] | @tsv' | sort -k2,2 -k1,1
```

On each node, the four workers must show different system and NIXL ports. If
the rendered pods still all show `9090` and `19090`, stop here and capture the
DGD, child DCDs, and pod specifications—the operator did not preserve the
overrides.

Check scheduling and startup:

```bash
kubectl get events -n "$NAMESPACE" --sort-by=.lastTimestamp | tail -100

kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=qwen32-pd \
  -n "$NAMESPACE" --timeout=45m
```

## 6. Follow prefill and decode logs

```bash
PREFILL_POD="$(kubectl get pods -n "$NAMESPACE" \
  -l 'nvidia.com/dynamo-graph-deployment-name=qwen32-pd,nvidia.com/dynamo-component-type=prefill' \
  -o jsonpath='{.items[0].metadata.name}')"

DECODE_POD="$(kubectl get pods -n "$NAMESPACE" \
  -l 'nvidia.com/dynamo-graph-deployment-name=qwen32-pd,nvidia.com/dynamo-component-type=decode' \
  -o jsonpath='{.items[0].metadata.name}')"

echo "PREFILL=$PREFILL_POD"
echo "DECODE=$DECODE_POD"
```

Use two terminals:

```bash
kubectl logs -n "$NAMESPACE" -f "$PREFILL_POD" --all-containers --timestamps
```

```bash
kubectl logs -n "$NAMESPACE" -f "$DECODE_POD" --all-containers --timestamps
```

For combined logs from every worker:

```bash
kubectl logs -n "$NAMESPACE" \
  -l nvidia.com/dynamo-graph-deployment-name=qwen32-pd \
  --all-containers --prefix --timestamps --tail=100 -f \
  --max-log-requests=9
```

## 7. Smoke test

```bash
FRONTEND_POD="$(kubectl get pods -n "$NAMESPACE" \
  -l 'nvidia.com/dynamo-graph-deployment-name=qwen32-pd,nvidia.com/dynamo-component-type=frontend' \
  -o jsonpath='{.items[0].metadata.name}')"

kubectl port-forward -n "$NAMESPACE" pod/"$FRONTEND_POD" 8000:8000
```

In another gpu05 terminal:

```bash
curl -fsS http://127.0.0.1:8000/v1/models | jq

curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-32B-FP8",
    "messages": [{"role": "user", "content": "Reply with only: PD_OK /no_think"}],
    "max_tokens": 16,
    "temperature": 0
  }' | jq
```

## Acceptance gate

Experiment 1 is ready only when:

- all nine pods remain Ready without restart loops;
- placement is 4P on gpu05, 4D on gpu06, and frontend on gpu05;
- each worker reports TP=2 and uses two GPUs;
- the four workers on each node have unique system, NIXL, FPM, NCCL, and
  bootstrap ports;
- logs show NIXL using UCX/RDMA rather than TCP fallback;
- `/v1/models` and the smoke request succeed.

## Troubleshooting

- `Pending` with `didn't have free ports`: inspect the rendered pod ports using
  section 5. One of the components still reused a host port.
- `CrashLoopBackOff` with `Address already in use`: check system, NIXL, FPM,
  NCCL, and bootstrap ports, including host processes outside Kubernetes.
- `NIXL_ERR_BACKEND`: confirm `mlx5_8:1`, GID index 3, RDMA resources, and the
  host-network diagnostic still succeed on both nodes.
- Only one worker per role registers: verify every component has the correct
  `type: prefill` or `type: decode`; do not override `DYN_COMPONENT` manually.
- OOM: inspect the actual two-GPU allocation and model logs before lowering
  `--mem-fraction-static`; keep any change identical across all eight workers.

## Shutdown

```bash
kubectl delete dgd qwen32-pd -n "$NAMESPACE" --ignore-not-found
kubectl get pods -n "$NAMESPACE" -w
```

The namespace, model PVC, Dynamo platform, GPU Operator, Network Operator, and
node labels are shared prerequisites; do not remove them between experiments.
