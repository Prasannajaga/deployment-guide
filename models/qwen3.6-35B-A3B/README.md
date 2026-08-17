# Qwen3.6-35B-A3B-FP8 serving recipes

These recipes deploy `Qwen/Qwen3.6-35B-A3B-FP8` on the repository's two-node,
16 x H100 80 GB cluster with NVIDIA Dynamo 1.3.0, SGLang 0.5.14, and vLLM
0.23.0. They include functional baselines plus a controlled eight-GPU vLLM
TP1-versus-TP2 disaggregated benchmark. Run only one deployment at a time.

## Why this layout

The checkpoint is a 37.5 GB multimodal sparse MoE with 35B total parameters
and about 3B active parameters per token. Its language model has 40 layers,
16 attention heads, 2 KV heads, hidden size 2,048, 256 experts, 8 selected
experts per token, and MoE intermediate size 512. Three of every four layers
use Gated DeltaNet linear attention; every fourth layer uses full attention.
The native context limit is 262,144 tokens.

TP=2 is the initial parallelism setting because it divides the attention
heads, KV heads, hidden size, and MoE intermediate size exactly. TP greater
than 2 would replicate the model's two KV heads across ranks. TP=1 is feasible
on an H100, but would not establish a tensor-parallel baseline.

Expert parallelism is intentionally disabled in the original baselines: those
manifests do not set `--enable-ep-moe`, `--ep-size`, or an expert-parallel
attention/data-parallel layout. The separate SGLang TP1-attention + EP2 recipe
adds EP, DP attention, and KV-aware routing as an explicitly labeled experiment.
Speculative decoding, KV offloading, and Mamba `extra_buffer` scheduling remain
disabled. The vLLM disaggregated experiment enables KV-aware routing identically
in both compared variants.

These first recipes validate text/chat serving. Although the checkpoint also
contains a vision encoder, native multimodal P/D requires
`--enable-multimodal` on both roles and has different processing and transfer
behavior. Add an aggregated multimodal and native EP/D or dedicated E/P/D
pair as a separate experiment rather than mixing it into this TP baseline.

The Dynamo 1.3.0 SGLang image contains SGLang 0.5.14, while Qwen3.6 requires
SGLang 0.5.10 or newer. The Dynamo 1.3.0 vLLM image contains vLLM 0.23.0,
including the hybrid/Mamba NIXL prefix-caching work. The vLLM recipes make HMA
and Mamba `align` cache mode explicit and require a startup/log transfer gate;
see [`vllm/disagg/README.md`](vllm/disagg/README.md).

## Baseline matrix

| Recipe | Topology | GPU use | Placement |
|---|---|---:|---|
| `sglang/agg` | 8 aggregated workers x TP=2 | 16 | 4 workers per node by GPU capacity |
| `sglang/disagg` | 4 prefill x TP=2 + 4 decode x TP=2 | 16 | Prefill and decode on separate 8-GPU nodes |
| `sglang/disagg/tp1-ep2-4p4d` | 4 prefill x effective TP=1/EP=2 + 4 decode x effective TP=1/EP=2 | 16 | KV-aware; one 2-GPU worker group per replica |
| `vllm/agg` | 4 aggregated workers x TP=2 | 8 | Functional aggregated baseline |
| `vllm/disagg/tp1-4p4d` | 4 prefill x TP=1 + 4 decode x TP=1 | 8 | 4 prefill GPUs + 4 decode GPUs |
| `vllm/disagg/tp2-2p2d` | 2 prefill x TP=2 + 2 decode x TP=2 | 8 | 4 prefill GPUs + 4 decode GPUs |

All recipes use a 131,072-token context baseline, 85% GPU-memory fraction,
Dynamo-native Qwen reasoning/tool parsing, and the pinned model revision
`95a723d08a9490559dae23d0cff1d9466213d989`. SGLang uses 64-token pages; vLLM
uses 128-token blocks following the repository's vLLM KV-aware convention.

## Prerequisites

Use the existing cluster namespace and `model-cache` PVC. The checkpoint is
public, so these manifests do not require `hf-token-secret`; the download Job
will consume it only when the secret already exists.

```bash
export NAMESPACE=qwen32-bench
kubectl get pvc model-cache -n "$NAMESPACE"
kubectl get nodes -L qwen.nvidia.com/role \
  -o custom-columns='NODE:.metadata.name,ROLE:.metadata.labels.qwen\.nvidia\.com/role,GPU:.status.allocatable.nvidia\.com/gpu,RDMA:.status.allocatable.rdma/ib'
```

The disaggregated recipe additionally requires:

- one 8-GPU node labeled `qwen.nvidia.com/role=prefill`;
- one 8-GPU node labeled `qwen.nvidia.com/role=decode`;
- the `qwen-roce` NetworkAttachmentDefinition in the deployment namespace;
- `mlx5_8:1` exposed inside worker Pods and at least 16 total `rdma/ib`
  resources across the two nodes.

It deliberately uses the cluster's pod-native RoCE network rather than
`hostNetwork`. This gives every replica its own port namespace and avoids the
host-port collisions documented for the earlier Qwen3-32B deployment. Do not
set a host GID index in these Pods; UCX must select the Pod-specific GID for
the attached MacVLAN interface.

## Download the pinned model

```bash
kubectl apply -n "$NAMESPACE" -f model-cache/model-download.yaml
kubectl wait -n "$NAMESPACE" \
  --for=condition=Complete job/qwen36-35b-a3b-fp8-download \
  --timeout=3600s
```

## Deploy

Aggregated baseline:

```bash
kubectl apply --dry-run=server -n "$NAMESPACE" -f sglang/agg/deploy.yaml
kubectl apply -n "$NAMESPACE" -f sglang/agg/deploy.yaml
```

Disaggregated baseline:

```bash
kubectl get network-attachment-definition qwen-roce -n "$NAMESPACE"
kubectl apply --dry-run=server -n "$NAMESPACE" -f sglang/disagg/deploy.yaml
kubectl apply -n "$NAMESPACE" -f sglang/disagg/deploy.yaml
```

The vLLM aggregate and controlled TP comparison have complete runbooks and
in-cluster AIPerf Jobs under [`vllm/`](vllm/). Follow those runbooks rather than
mixing their eight-GPU settings with the 16-GPU SGLang baselines.

## Acceptance checks

Set `DEPLOYMENT` to `qwen36-35b-a3b-fp8-sglang-agg-tp2` or
`qwen36-35b-a3b-fp8-sglang-disagg-tp2`, then check the deployment:

```bash
export GRAPH_LABEL="nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT}"
kubectl get pods -n "$NAMESPACE" -l "$GRAPH_LABEL" -o wide
kubectl logs -n "$NAMESPACE" -l "$GRAPH_LABEL" \
  --all-containers --prefix --tail=500
kubectl port-forward -n "$NAMESPACE" \
  service/"${DEPLOYMENT}-frontend" 8000:8000
```

In another terminal, issue a deterministic non-thinking smoke test:

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'EOF'
{
  "model": "Qwen/Qwen3.6-35B-A3B-FP8",
  "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
  "chat_template_kwargs": {"enable_thinking": false},
  "temperature": 0,
  "max_tokens": 16
}
EOF
```

For disaggregation, also require successful UCX/NIXL initialization and a
real prefill-to-decode transfer in the logs. A Pod reaching `Running` without
a completed NIXL transfer is not sufficient acceptance.

## Cleanup

```bash
kubectl delete dynamographdeployment.nvidia.com "$DEPLOYMENT" \
  -n "$NAMESPACE" --wait=false --ignore-not-found
```

## References

- [Qwen model checkpoint](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)
- [SGLang Qwen3.6 cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.6)
- [Dynamo SGLang backend](https://docs.nvidia.com/dynamo/latest/knowledge-base/modular-components/backends/sg-lang/overview)
- [Dynamo parser configuration](https://docs.nvidia.com/dynamo/latest/user-guides/parsing/parser-configuration)
- [Open vLLM hybrid-cache disaggregation issue](https://github.com/ai-dynamo/dynamo/issues/10741)
