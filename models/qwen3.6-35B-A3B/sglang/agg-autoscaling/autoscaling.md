# KEDA autoscaling and live Pod dashboard

This runbook adds Prometheus-based KEDA autoscaling to the aggregated SGLang
deployment in this directory. The frontend stays at one replica. The TP=2
worker service starts at one replica, never scales below one, and can scale to
eight replicas when the frontend active-request count grows.

Each worker requests two GPUs. A scale from one to eight workers therefore
changes worker GPU demand from 2 to 16 GPUs. KEDA can request replicas that the
cluster cannot schedule, so set `maxReplicaCount` below eight when less GPU
capacity is available.

The flow follows NVIDIA's
[Dynamo autoscaling guide](https://docs.nvidia.com/dynamo/latest/kubernetes-deployment/operate/autoscaling)
and uses `dynamo_frontend_active_requests`, Dynamo 1.3's gauge for requests from
frontend entry through completion. KEDA targets the worker's
`DynamoGraphDeploymentScalingAdapter` (DGDSA), not the DGD or generated worker
Deployment directly.

## 1. Set variables

Run all cluster commands from a Kubernetes administrator shell. Keep generated
manifests and test artifacts in the shared ephemeral recipe directory:

```bash
export NAMESPACE=dynamo-bench
export RECIPE_ROOT=/ephemeral/shared/qwen3.6-35b-a3b
export EXP_DIR="${RECIPE_ROOT}/sglang/agg-autoscaling"
export DEPLOYMENT=qwen36-35b-a3b-fp8-sglang-agg-tp2
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
export FRONTEND_SERVICE="${DEPLOYMENT}-frontend"
export SCALING_ADAPTER="${DEPLOYMENT}-sglangworker"
export SCALED_OBJECT=qwen36-35b-a3b-sglang-worker
export KEDA_HPA="keda-hpa-${SCALED_OBJECT}"
export FRONTEND_POD_REGEX="${DEPLOYMENT}.*frontend.*"
export MODEL=Qwen/Qwen3.6-35B-A3B-FP8
export PREFLIGHT_DEPLOYMENT="${DEPLOYMENT}-preflight"
export PREFLIGHT_GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${PREFLIGHT_DEPLOYMENT}"

export MONITOR_NAMESPACE=monitoring
export PROM_SERVICE=monitoring-kube-prometheus-prometheus
export PROM_SERVICE_PORT=9090
export PROMETHEUS_SERVER="http://${PROM_SERVICE}.${MONITOR_NAMESPACE}.svc:${PROM_SERVICE_PORT}"
export GRAFANA_SERVICE=monitoring-grafana

mkdir -p "$EXP_DIR"
```

`FRONTEND_POD_REGEX` selects only this DGD's frontend scrape target. Section 5
verifies that Prometheus actually retains a matching series before KEDA is
created.

## 2. Check the required controllers

The Dynamo deployment must already have been created from
[README.md](README.md). Check the DGD, its scaling adapter, KEDA, Prometheus,
and Grafana:

```bash
kubectl get dgd "$DEPLOYMENT" -n "$NAMESPACE"
kubectl get dgdsa -n "$NAMESPACE"
kubectl get crd scaledobjects.keda.sh
kubectl get pods -n keda
kubectl get service -n "$MONITOR_NAMESPACE" \
  "$PROM_SERVICE" "$GRAFANA_SERVICE"
```

The expected adapter is `$SCALING_ADAPTER`, derived from the DGD name and the
lowercase `SglangWorker` service name. Require it to exist and report one
replica:

```bash
kubectl get dgdsa "$SCALING_ADAPTER" -n "$NAMESPACE" -o wide
kubectl get dgdsa "$SCALING_ADAPTER" -n "$NAMESPACE" \
  -o jsonpath='{.spec.replicas}{" desired, "}{.status.replicas}{" current\n"}'
```

If no adapter exists, confirm that the applied DGD contains the following
worker fields, then reapply `$EXP_DIR/deploy.yaml`:

```yaml
SglangWorker:
  componentType: worker
  scalingAdapter:
    enabled: true
  replicas: 1
```

Do not add the deprecated `spec.services[X].autoscaling` field and do not
target the generated Kubernetes Deployment with KEDA.

### Install KEDA only when it is absent

Skip this block when `scaledobjects.keda.sh` and healthy KEDA Pods already
exist. Otherwise install KEDA once for the cluster:

```bash
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm upgrade --install keda kedacore/keda \
  --namespace keda \
  --create-namespace \
  --wait

kubectl get pods -n keda
kubectl get apiservice v1beta1.external.metrics.k8s.io
```

The external-metrics APIService must be available. If another metrics adapter
already owns that APIService, resolve that cluster-wide ownership conflict
before continuing; do not run two autoscalers against the same DGDSA.

### Install monitoring only when it is absent

This recipe expects the existing kube-prometheus-stack named `monitoring`.
When the cluster has no Prometheus Operator, Prometheus, or Grafana, install
the stack once with cross-namespace PodMonitor discovery enabled:

```bash
helm repo add prometheus-community \
  https://prometheus-community.github.io/helm-charts
helm repo update
helm upgrade --install monitoring \
  prometheus-community/kube-prometheus-stack \
  --namespace "$MONITOR_NAMESPACE" \
  --create-namespace \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
  --set-json 'prometheus.prometheusSpec.podMonitorNamespaceSelector={}' \
  --wait
```

If an existing monitoring stack uses different names, change the four
monitoring variables in section 1 instead of installing a duplicate stack.

## 3. Run the preflight canary

The README has already created `$EXP_DIR/deploy.yaml`. Derive the canary from
that file so its runtime, model, TP size, resources, and scaling-adapter fields
cannot drift from production. Only the DGD name changes. The canary temporarily
uses two additional GPUs and is always removed when this subshell exits.

```bash
(
  set -euo pipefail

  cleanup_preflight() {
    kubectl delete dgd "$PREFLIGHT_DEPLOYMENT" -n "$NAMESPACE" \
      --ignore-not-found --wait=false >/dev/null || true
    kubectl wait -n "$NAMESPACE" --for=delete pod \
      -l "$PREFLIGHT_GRAPH_LABEL" --timeout=300s >/dev/null || true
  }
  trap cleanup_preflight EXIT

  command -v kubectl awk jq >/dev/null
  test -s "$EXP_DIR/deploy.yaml"
  cleanup_preflight

  awk -v name="$PREFLIGHT_DEPLOYMENT" '
    !renamed && /^  name:/ {
      print "  name: " name
      renamed = 1
      next
    }
    { print }
    END { if (!renamed) exit 1 }
  ' "$EXP_DIR/deploy.yaml" |
    tee "$EXP_DIR/preflight.yaml" >/dev/null

  kubectl apply --dry-run=server -n "$NAMESPACE" \
    -f "$EXP_DIR/preflight.yaml"
  kubectl apply -n "$NAMESPACE" -f "$EXP_DIR/preflight.yaml"

  pod_count=0
  for _ in {1..60}; do
    pod_count="$(kubectl get pods -n "$NAMESPACE" \
      -l "$PREFLIGHT_GRAPH_LABEL" -o name | wc -l)"
    test "$pod_count" -eq 2 && break
    sleep 2
  done
  test "$pod_count" -eq 2

  kubectl wait -n "$NAMESPACE" --for=condition=Ready pod \
    -l "$PREFLIGHT_GRAPH_LABEL" --timeout=1800s

  kubectl get dgdsa "${PREFLIGHT_DEPLOYMENT}-sglangworker" \
    -n "$NAMESPACE" -o name

  preflight_frontend_pod="$(kubectl get pods -n "$NAMESPACE" \
    -l "$PREFLIGHT_GRAPH_LABEL,nvidia.com/dynamo-component-type=frontend" \
    -o jsonpath='{.items[0].metadata.name}')"
  test -n "$preflight_frontend_pod"

  kubectl exec -n "$NAMESPACE" "$preflight_frontend_pod" -- \
    env "MODEL=$MODEL" python3 -c '
import json
import os
import urllib.request

body = json.dumps({
    "model": os.environ["MODEL"],
    "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
    "chat_template_kwargs": {"enable_thinking": False},
    "temperature": 0,
    "max_tokens": 16,
}).encode()

request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=300) as response:
    result = json.load(response)

content = result["choices"][0]["message"]["content"].strip()
response_model = result.get("model")
if response_model != os.environ["MODEL"] or content != "ready":
    raise SystemExit(
        f"unexpected response: model={response_model!r}, content={content!r}"
    )
print(content)
'

  kubectl logs -n "$NAMESPACE" -l "$PREFLIGHT_GRAPH_LABEL" \
    --all-containers --prefix --tail=1500 |
    tee "$EXP_DIR/preflight-startup.log"

  if grep -Ei \
    'traceback|assertionerror|notimplementederror|out of memory|unsupported' \
    "$EXP_DIR/preflight-startup.log"; then
    echo "Preflight failed: fatal signature found" >&2
    exit 1
  fi

  restart_count="$(kubectl get pods -n "$NAMESPACE" \
    -l "$PREFLIGHT_GRAPH_LABEL" -o json |
    jq '[.items[].status.containerStatuses[]?.restartCount] | add // 0')"
  test "$restart_count" -eq 0

  kubectl get pods -n "$NAMESPACE" -l "$PREFLIGHT_GRAPH_LABEL" \
    -o custom-columns='POD:.metadata.name,NODE:.spec.nodeName,READY:.status.containerStatuses[*].ready,RESTARTS:.status.containerStatuses[*].restartCount,GPU:.spec.containers[*].resources.requests.nvidia\.com/gpu'

  kubectl delete dgd "$PREFLIGHT_DEPLOYMENT" -n "$NAMESPACE" \
    --ignore-not-found --wait=false
  kubectl wait -n "$NAMESPACE" --for=delete pod \
    -l "$PREFLIGHT_GRAPH_LABEL" --timeout=300s
  trap - EXIT
)
```

Continue only when the subshell exits successfully. Its `EXIT` trap deletes
the canary even when validation fails, and waits for the two GPUs to be
released. The manifest and startup log remain in `$EXP_DIR` for diagnosis.

## 4. Verify Dynamo metrics discovery

Dynamo labels its managed Pods for metrics discovery. Some operator and chart
combinations create PodMonitor resources automatically and some do not.
Confirm that metrics are enabled and inspect the active monitors before
creating an additional monitor:

```bash
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  -L nvidia.com/metrics-enabled,nvidia.com/dynamo-component-type
kubectl get podmonitor -A
kubectl get prometheus -n "$MONITOR_NAMESPACE" \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.podMonitorSelector}{"\t"}{.spec.podMonitorNamespaceSelector}{"\n"}{end}'
```

Do not create a second PodMonitor when the operator-created one already
produces a healthy frontend target. Duplicate monitors double-scrape the same
series and make rate calculations misleading.

If the namespace has no PodMonitor, first prove that the expected selector
finds exactly one frontend Pod and that this Pod declares port 8000:

```bash
kubectl get pods -n "$NAMESPACE" \
  -l "$GRAPH_LABEL,nvidia.com/dynamo-component-type=frontend" \
  -o wide

frontend_pod="$(kubectl get pods -n "$NAMESPACE" \
  -l "$GRAPH_LABEL,nvidia.com/dynamo-component-type=frontend" \
  -o jsonpath='{.items[0].metadata.name}')"
test -n "$frontend_pod"
kubectl get pod "$frontend_pod" -n "$NAMESPACE" \
  -o jsonpath='{range .spec.containers[*]}{.name}{range .ports[*]}{"\t"}{.name}{"="}{.containerPort}{end}{"\n"}{end}'
```

When port 8000 is present, create one frontend-only PodMonitor. Its
`release: monitoring` label matches the kube-prometheus-stack selector shown
in section 2; `portNumber` avoids depending on the container port's name:

```bash
tee "$EXP_DIR/frontend-podmonitor.yaml" >/dev/null <<EOF
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: ${DEPLOYMENT}-frontend
  namespace: ${NAMESPACE}
  labels:
    release: monitoring
spec:
  selector:
    matchLabels:
      nvidia.com/dynamo-graph-deployment-name: ${DEPLOYMENT}
      nvidia.com/dynamo-component-type: frontend
  podMetricsEndpoints:
    - portNumber: 8000
      path: /metrics
      interval: 15s
EOF

kubectl apply --dry-run=server \
  -f "$EXP_DIR/frontend-podmonitor.yaml"
kubectl apply -f "$EXP_DIR/frontend-podmonitor.yaml"
kubectl get podmonitor "${DEPLOYMENT}-frontend" \
  -n "$NAMESPACE" -o wide
```

If the Prometheus resource uses a different non-empty
`spec.podMonitorSelector`, copy its required labels to the PodMonitor instead
of broadening the Prometheus selector. An empty
`podMonitorNamespaceSelector: {}` already permits discovery across namespaces.

Check the frontend endpoint directly. Keep the port-forward running in this
terminal:

```bash
kubectl port-forward -n "$NAMESPACE" \
  "service/$FRONTEND_SERVICE" 8000:8000 \
  --address 127.0.0.1
```

In another terminal, issue the README smoke test once, then inspect the metric:

```bash
curl -fsS http://127.0.0.1:8000/metrics |
  grep '^dynamo_frontend_active_requests'
```

Labeled Dynamo series may not exist until the first matching request has been
served. A zero value after that request is healthy; an absent metric family or
an HTTP failure is not.

## 5. Validate the Prometheus query

Forward the in-cluster Prometheus Service in a separate terminal:

```bash
kubectl port-forward -n "$MONITOR_NAMESPACE" \
  "service/$PROM_SERVICE" "9095:$PROM_SERVICE_PORT" \
  --address 127.0.0.1
```

Run these checks from another terminal on the same host. Inspect the labels
that Prometheus actually retained; Dynamo 1.3 frontend-local metrics such as
`dynamo_frontend_active_requests` do not themselves carry a
`dynamo_namespace` label, while a PodMonitor adds the `namespace` and `pod`
target labels used below:

```bash
export PROM=http://127.0.0.1:9095
curl -fsS "$PROM/-/ready"

curl -fsS -G "$PROM/api/v1/query" \
  --data-urlencode "query=dynamo_frontend_active_requests{namespace=\"$NAMESPACE\"}" |
jq -r '.data.result[] | [.metric.namespace, .metric.pod, .metric.model, .value[1]] | @tsv'
```

Require a row whose `pod` belongs to this DGD's frontend and whose model is
`$MODEL`. Prove that the selector matches at least one retained series, then
test the exact query used by KEDA:

```bash
series_count="$(
  curl -fsS -G "$PROM/api/v1/query" \
    --data-urlencode "query=count(dynamo_frontend_active_requests{namespace=\"$NAMESPACE\",pod=~\"$FRONTEND_POD_REGEX\"})" |
    jq -er '.data.result[0].value[1] | tonumber'
)"
test "$series_count" -gt 0

curl -fsS -G "$PROM/api/v1/query" \
  --data-urlencode "query=sum(dynamo_frontend_active_requests{namespace=\"$NAMESPACE\",pod=~\"$FRONTEND_POD_REGEX\"})" |
jq -e '
  .status == "success" and
  (.data.result | length) == 1 and
  (.data.result[0].value[1] | tonumber) >= 0
'
```

The result must be one numeric vector, normally `0` while idle. If the observed
`namespace` or `pod` labels differ, update the selector rather than guessing a
`dynamo_namespace` value. If direct frontend metrics work but Prometheus has no
series, fix PodMonitor selection or target health first. Do not apply the
ScaledObject until `series_count` is greater than zero: KEDA cannot scale from
a metric Prometheus does not retain.

## 6. Create and apply the KEDA ScaledObject

The Prometheus query returns one global queue-like total and targets 16 active
requests per worker. Explicit `metricType: AverageValue` is critical: KEDA and
the HPA use it to calculate `ceil(active requests / 16)` independently of the
current replica count. `Value` would instead scale that global total relative
to the current replica count and can over-scale. KEDA's generated HPA can add
up to two workers every 30 seconds and removes at most one worker per minute
after a five-minute stabilization window. If Prometheus is unavailable, the
HPA retains its current recommendation until metrics recover.

```bash
tee "$EXP_DIR/scaledobject.yaml" >/dev/null <<EOF
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: ${SCALED_OBJECT}
  namespace: ${NAMESPACE}
spec:
  scaleTargetRef:
    apiVersion: nvidia.com/v1beta1
    kind: DynamoGraphDeploymentScalingAdapter
    name: ${SCALING_ADAPTER}
  minReplicaCount: 1
  maxReplicaCount: 6
  pollingInterval: 15
  advanced:
    restoreToOriginalReplicaCount: true
    horizontalPodAutoscalerConfig:
      behavior:
        scaleUp:
          stabilizationWindowSeconds: 0
          selectPolicy: Max
          policies:
            - type: Pods
              value: 2
              periodSeconds: 30
        scaleDown:
          stabilizationWindowSeconds: 200
          selectPolicy: Min
          policies:
            - type: Pods
              value: 1
              periodSeconds: 60
  triggers:
    - type: prometheus
      metricType: AverageValue
      metadata:
        serverAddress: ${PROMETHEUS_SERVER}
        query: |
          sum(dynamo_frontend_active_requests{
            namespace="${NAMESPACE}",
            pod=~"${FRONTEND_POD_REGEX}"
          })
        threshold: "16"
        ignoreNullValues: "false"
EOF

kubectl apply --dry-run=server -f "$EXP_DIR/scaledobject.yaml"
kubectl apply -f "$EXP_DIR/scaledobject.yaml"
kubectl wait -n "$NAMESPACE" \
  --for=condition=Ready "scaledobject/$SCALED_OBJECT" \
  --timeout=120s
```

Verify that KEDA accepted the object and created an HPA that targets the
DGDSA:

```bash
kubectl get scaledobject "$SCALED_OBJECT" -n "$NAMESPACE"
kubectl describe scaledobject "$SCALED_OBJECT" -n "$NAMESPACE"
kubectl get hpa "$KEDA_HPA" -n "$NAMESPACE" -o wide
kubectl get hpa "$KEDA_HPA" -n "$NAMESPACE" \
  -o jsonpath='{.spec.scaleTargetRef.kind}{"/"}{.spec.scaleTargetRef.name}{" metricType="}{.spec.metrics[0].external.target.type}{"\n"}'
```

Require `READY=True`, an HPA target of
`DynamoGraphDeploymentScalingAdapter/$SCALING_ADAPTER`, and minimum/current
replicas of one. The generated HPA must report `metricType=AverageValue`; stop
if it reports `Value`. The HPA metric can briefly show `<unknown>` before KEDA's
first poll; it must become numeric before load testing.

## 7. Watch autoscaling Pods live

Open two observer terminals before generating load.

Terminal A shows every frontend and worker Pod transition:

```bash
kubectl get pods -n "$NAMESPACE" \
  -l "$GRAPH_LABEL" \
  -L nvidia.com/dynamo-component-type \
  -o wide \
  --watch
```

Terminal B shows the scaling control plane:

```bash
kubectl get dgdsa,scaledobject,hpa -n "$NAMESPACE" --watch
```

For a compact worker-only count in a third terminal:

```bash
watch -n 2 "kubectl get pods -n '$NAMESPACE' \
  -l '$GRAPH_LABEL,nvidia.com/dynamo-component-type=worker' \
  --no-headers | awk '{count[\$3]++} END {for (state in count) print state, count[state]}'"
```

New Pods first appear as `Pending` or `ContainerCreating`, then become ready
after loading the model. Scaling is not complete merely because the DGDSA or
HPA desired count changed; require the new worker Pods to reach `Ready`.

## 8. Configure the Grafana autoscaling panels

Discover the actual Grafana port and keep a local-only port-forward running:

```bash
export GRAFANA_SERVICE_PORT="$(kubectl get service \
  -n "$MONITOR_NAMESPACE" "$GRAFANA_SERVICE" \
  -o jsonpath='{.spec.ports[0].port}')"

kubectl port-forward -n "$MONITOR_NAMESPACE" \
  "service/$GRAFANA_SERVICE" "3000:$GRAFANA_SERVICE_PORT" \
  --address 127.0.0.1
```

Open `http://127.0.0.1:3000` and sign in with the cluster's existing SSO or
Grafana administrator credentials. Do not expose Grafana on `0.0.0.0`. When
the `kubectl` shell is on a remote administration host, use an SSH agent and a
local tunnel from the workstation before opening the URL.

Create a dashboard named **Qwen3.6 SGLang KEDA Autoscaling**, set its time range
to **Last 15 minutes**, and set refresh to **5s**. Add these panels using the
cluster Prometheus data source and the Code query editor. For every query
below, set **Query type** to **Range** rather than **Instant** so Grafana draws
the autoscaling history.

| Panel | Query | Visualization | Display |
|---|---|---|---|
| Worker replicas | A: Ready Pods | Time series | Line, step interpolation |
| Worker replicas | B: HPA Current | Time series, same panel | Line, step interpolation |
| Worker replicas | C: HPA Desired | Time series, same panel | Line, step interpolation |
| Frontend active requests | A: Active Requests | Time series | Line with threshold at `16` |
| Worker Pod phases | A: count by phase | Time series | Stacked bars |
| Worker GPU utilization | A: Average GPU | Time series | Line, percent unit |

### Verify the retained labels before creating panels

The DGD inserts a numeric component index into generated Pod names. For this
deployment, worker Pods look like
`qwen36-35b-a3b-fp8-sglang-agg-tp2-0-sglangworker-...`; the panel selectors
therefore match `tp2-[0-9]+-sglangworker-.*` rather than
`tp2-sglangworker.*`.

In Grafana **Explore**, select the same Prometheus data source used by the
dashboard, turn on **Instant**, and run each discovery query separately.

Worker Pod discovery must return the running worker Pod:

```promql
count by (pod) (
  kube_pod_info{
    namespace="dynamo-bench",
    pod=~"qwen36-35b-a3b-fp8-sglang-agg-tp2-[0-9]+-sglangworker-.*"
  }
)
```

HPA discovery must return `keda-hpa-qwen36-35b-a3b-sglang-worker` after the
ScaledObject has created its HPA:

```promql
kube_horizontalpodautoscaler_status_current_replicas{
  namespace="dynamo-bench"
}
```

Frontend metric discovery must return the `namespace`, `pod`, and `model`
labels used to identify the series selected by Panel 2:

```promql
count by (namespace, pod, model) (
  dynamo_frontend_active_requests
)
```

Optional DCGM discovery shows whether this Prometheus uses the
`exported_namespace`/`exported_pod` label pair or the standard
`namespace`/`pod` pair:

```promql
count by (namespace, pod, exported_namespace, exported_pod) (
  DCGM_FI_DEV_GPU_UTIL
)
```

Only turn **Instant** off and return to **Range** after these discovery queries
produce data. An empty worker or HPA discovery query indicates missing
kube-state-metrics data in the selected Prometheus source, not a visualization
setting problem.

### Panel 1: Worker replicas

Visualization: **Time series**. Put all three queries in this one panel. Set
unit to **short**, minimum to `0`, maximum to `8`, line interpolation to
**Step after**, and legend placement to **Bottom**.

Query A — Ready workers:

- Visualization: **Time series**, in Panel 1.
- Query type: **Range**.
- Legend: `Ready Pods`.

```promql
sum(
  kube_pod_status_ready{
    namespace="dynamo-bench",
    pod=~"qwen36-35b-a3b-fp8-sglang-agg-tp2-[0-9]+-sglangworker-.*",
    condition="true"
  }
)
```

Query B — Current HPA replicas:

- Visualization: **Time series**, in the same Panel 1.
- Query type: **Range**.
- Legend: `HPA Current`.

```promql
max(
  kube_horizontalpodautoscaler_status_current_replicas{
    namespace="dynamo-bench",
    horizontalpodautoscaler="keda-hpa-qwen36-35b-a3b-sglang-worker"
  }
)
```

Query C — Desired HPA replicas:

- Visualization: **Time series**, in the same Panel 1.
- Query type: **Range**.
- Legend: `HPA Desired`.

```promql
max(
  kube_horizontalpodautoscaler_status_desired_replicas{
    namespace="dynamo-bench",
    horizontalpodautoscaler="keda-hpa-qwen36-35b-a3b-sglang-worker"
  }
)
```

This panel makes the startup gap visible: `HPA Desired` rises first, `HPA
Current` follows, and `Ready Pods` catches up after model initialization.

### Panel 2: Frontend active requests

Query A — Frontend active requests:

- Visualization: **Time series**.
- Query type: **Range**.
- Legend: `Active Requests`.
- Unit: **short**.
- Standard options: minimum `0`.
- Threshold: add a horizontal threshold at `16`.

```promql
sum(
  dynamo_frontend_active_requests{
    namespace="dynamo-bench",
    pod=~"qwen36-35b-a3b-fp8-sglang-agg-tp2.*frontend.*"
  }
)
```

Because the trigger uses `metricType: AverageValue`, KEDA calculates desired
replicas from the global active-request count divided by 16.

### Panel 3: Worker Pod phases

Query A — Number of workers in each Pod phase:

- Visualization: **Time series**. This query returns a count for each phase,
  so a State timeline is not appropriate for this exact query.
- Query type: **Range**.
- Legend: `{{phase}}`.
- Unit: **short**.
- Draw style: **Bars**.
- Bar alignment: `0`.
- Stacking: **Normal**.
- Standard options: minimum `0`, maximum `8`.

```promql
sum by (phase) (
  kube_pod_status_phase{
    namespace="dynamo-bench",
    pod=~"qwen36-35b-a3b-fp8-sglang-agg-tp2-[0-9]+-sglangworker-.*",
    phase=~"Pending|Running|Failed"
  } == 1
)
```

### Panel 4: Worker GPU utilization (optional)

Add this panel only when DCGM Exporter is installed and the query returns
series.

Query A — Average GPU utilization across worker GPUs:

- Visualization: **Time series**.
- Query type: **Range**.
- Legend: `Average GPU Utilization`.
- Unit: **Percent (0-100)**.
- Standard options: minimum `0`, maximum `100`.

```promql
avg(
  DCGM_FI_DEV_GPU_UTIL{
    exported_namespace="dynamo-bench",
    exported_pod=~"qwen36-35b-a3b-fp8-sglang-agg-tp2-[0-9]+-sglangworker-.*"
  }
  or
  DCGM_FI_DEV_GPU_UTIL{
    namespace="dynamo-bench",
    pod=~"qwen36-35b-a3b-fp8-sglang-agg-tp2-[0-9]+-sglangworker-.*"
  }
)
```

Save the dashboard before starting load. An empty Panel 1 or Panel 3 usually
means Grafana selected a Prometheus data source without kube-state-metrics; an
empty Panel 2 means the frontend target or metric labels need correction.

## 9. Generate load and observe scale-up

Keep both `kubectl` watches and Grafana running. Do not send high-concurrency
load through `kubectl port-forward`; it is a debugging tunnel and can reset or
lose its selected Pod under this workload. Create the request body on the
administration host:

```bash
tee "$EXP_DIR/autoscaling-request.json" >/dev/null <<'EOF'
{
  "model": "Qwen/Qwen3.6-35B-A3B-FP8",
  "messages": [
    {
      "role": "user",
      "content": "Write a detailed technical explanation of distributed inference, including scheduling, batching, tensor parallelism, queueing, and failure recovery."
    }
  ],
  "chat_template_kwargs": {"enable_thinking": false},
  "temperature": 0.7,
  "max_tokens": 1024
}
EOF
```

Create a ConfigMap from that request body:

```bash
kubectl create configmap qwen36-autoscaling-request \
  -n "$NAMESPACE" \
  --from-file=request.json="$EXP_DIR/autoscaling-request.json" \
  --dry-run=client -o yaml |
tee "$EXP_DIR/load-request-configmap.yaml" >/dev/null

kubectl apply -f "$EXP_DIR/load-request-configmap.yaml"
```

Generate 256 concurrent requests inside the cluster with eight load Pods and
32 request loops per Pod:

```bash
tee "$EXP_DIR/load-generator.yaml" >/dev/null <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qwen36-autoscaling-load
  namespace: dynamo-bench
spec:
  replicas: 8
  selector:
    matchLabels:
      app: qwen36-autoscaling-load
  template:
    metadata:
      labels:
        app: qwen36-autoscaling-load
    spec:
      containers:
        - name: load
          image: curlimages/curl:8.12.1
          command:
            - /bin/sh
            - -c
          args:
            - |
              for slot in $(seq 1 32); do
                (
                  while true; do
                    curl -sS -o /dev/null \
                      --connect-timeout 5 \
                      --max-time 300 \
                      -H 'Content-Type: application/json' \
                      --data-binary @/load/request.json \
                      http://qwen36-35b-a3b-fp8-sglang-agg-tp2-frontend:8000/v1/chat/completions \
                      || true
                  done
                ) &
              done
              wait
          resources:
            requests:
              cpu: 100m
              memory: 64Mi
            limits:
              cpu: "1"
              memory: 256Mi
          volumeMounts:
            - name: request
              mountPath: /load
              readOnly: true
      volumes:
        - name: request
          configMap:
            name: qwen36-autoscaling-request
EOF

kubectl apply --dry-run=server -f "$EXP_DIR/load-generator.yaml"
kubectl apply -f "$EXP_DIR/load-generator.yaml"
kubectl get pods -n "$NAMESPACE" -l app=qwen36-autoscaling-load -w
```

Expected order:

1. Grafana active requests rise above `16`.
2. KEDA updates the generated HPA.
3. The HPA increases the DGDSA desired replicas.
4. The DGD creates additional TP=2 worker Pods.
5. New workers request two GPUs each, load the model, and become ready.
6. Active requests fall as ready capacity increases and the test is stopped.

Worker startup includes scheduling and loading a 37.5 GB checkpoint, so Pod
readiness can lag the KEDA decision by minutes. A `Pending` Pod accompanied by
an insufficient-GPU scheduling event is a capacity limit, not a KEDA failure:

```bash
kubectl get events -n "$NAMESPACE" \
  --sort-by='.lastTimestamp' |
tail -n 40
```

## 10. Observe scale-down to one

Leave the watches and Grafana open. Stop and remove the in-cluster load
generator first:

```bash
kubectl delete deployment qwen36-autoscaling-load \
  -n "$NAMESPACE" --ignore-not-found
kubectl delete configmap qwen36-autoscaling-request \
  -n "$NAMESPACE" --ignore-not-found
```

Active requests should return to zero. The HPA retains capacity through the
configured five-minute stabilization window, then removes at most one worker
per minute until exactly one worker remains.

Verify the final steady state:

```bash
kubectl get dgdsa "$SCALING_ADAPTER" -n "$NAMESPACE" \
  -o jsonpath='{.spec.replicas}{" desired, "}{.status.replicas}{" current\n"}'
kubectl get pods -n "$NAMESPACE" \
  -l "$GRAPH_LABEL,nvidia.com/dynamo-component-type=worker" \
  -o wide
```

Pass criteria:

- worker desired/current replicas rose above one during load;
- each added worker requested two GPUs and became ready;
- Grafana showed active requests and desired/current/ready replica transitions;
- the worker service returned to one ready replica after load and the
  stabilization period;
- the frontend stayed at one replica throughout.

## Troubleshooting

First establish whether the object exists. A missing HPA is expected when the
ScaledObject is missing because KEDA owns and generates that HPA:

```bash
kubectl get scaledobject "$SCALED_OBJECT" -n "$NAMESPACE"
kubectl get hpa "$KEDA_HPA" -n "$NAMESPACE"
```

If both commands return `NotFound`, recreate the object with the two
`kubectl apply` commands in section 6. `NotFound` is not a Prometheus-scaler
error and KEDA does not spontaneously delete a ScaledObject. A log line saying
`Successfully finalized ScaledObject` confirms that Kubernetes deletion was
already requested and KEDA merely completed its cleanup, including deletion
of the generated HPA.

Once the object exists, inspect the complete scaling path. Limit the operator
log window so output from a previously deleted ScaledObject is not mistaken
for current state:

```bash
kubectl describe scaledobject "$SCALED_OBJECT" -n "$NAMESPACE"
kubectl describe hpa "$KEDA_HPA" -n "$NAMESPACE"
kubectl describe dgdsa "$SCALING_ADAPTER" -n "$NAMESPACE"
kubectl logs -n keda deployment/keda-operator --since=10m --tail=300
```

- `ScaledObject` not ready: verify its target name and the served DGDSA API
  version with `kubectl api-resources | grep -i scalingadapter`. This recipe
  targets the cluster's served `nvidia.com/v1beta1` DGDSA API.
- HPA metric is `<unknown>`: query the exact PromQL in section 5 and verify
  KEDA can resolve `$PROMETHEUS_SERVER` from inside the cluster.
- KEDA reports `result is empty`: the selector matched no retained series.
  Re-run `series_count` in section 5 and inspect `namespace`, `pod`, and
  `model`; do not add a `dynamo_namespace` filter to this frontend-local
  metric.
- Desired replicas rise but Pods do not appear: inspect DGD and operator
  events/logs.
- Pods remain `Pending`: inspect scheduler events and reduce
  `maxReplicaCount` to match free GPU pairs.
- Scaling oscillates: increase the scale-down stabilization window or request
  threshold; never attach a second HPA, KEDA object, or Planner to this DGDSA.
- Active requests are positive but no scale-up occurs: confirm the query
  returns one scalar and its `namespace` and `pod` labels match retained data.

## Remove autoscaling while keeping one replica

Delete the ScaledObject first. `restoreToOriginalReplicaCount: true` asks KEDA
to restore the original one-worker count. Then explicitly set the adapter to
one and verify it before leaving the deployment running:

```bash
kubectl delete -f "$EXP_DIR/scaledobject.yaml" --ignore-not-found
kubectl scale dgdsa "$SCALING_ADAPTER" -n "$NAMESPACE" --replicas=1
kubectl get dgdsa "$SCALING_ADAPTER" -n "$NAMESPACE" -o wide
```

Do not uninstall a shared KEDA or monitoring stack as part of recipe cleanup.
To remove the complete model deployment, continue with the cleanup section in
[README.md](README.md).
