# Cluster Progress & Model Benchmarks

## Llama 8B (`meta-llama/Llama-3.1-8B-Instruct`)

I started off with the baseline deployment of Llama 8B using **TP=16** stretched across both nodes (`gpu05` and `gpu06`). While it worked to validate our multi-node setup and network connectivity.

it was pretty messy each node ended up occupying roughly 95% of its GPU VRAM (~76GB/80GB), and inter-node network communication became a bottleneck for every single layer!

![Llama 8B TP=16 NVIDIA-SMI Memory Usage](assets/llama8B-TP16.png)

### **TP=16 (Baseline):** Runs 1 single model instance across 16 GPUs

**Weight Distribution:** Llama 8B in FP16 precision takes ~16GB of total model weights. At TP=16, these weights are sharded across 16 GPUs, so **each individual GPU holds only ~1GB of model weights!** The rest of the 76GB+ VRAM allocated per GPU (as shown in the `nvidia-smi` output above) is pre-allocated by vLLM for KV cache and execution buffers.

  Because each GPU does tiny 1GB math operations, GPU computation finishes in microseconds while inter-node network communication between `gpu05` and `gpu06` stalls every transformer layer.

For a smaller model like 8B, scaling down to lower TP sizes (such as TP=8 for 2 single-node replicas on local NVLink, or TP=4 / TP=2 / TP=1 for multi-replica serving) would eliminate inter-node network communication and yield significantly higher request throughput. However, since we aren't experimenting further or benchmarking Llama 8B, we won't be deploying these lower TP configurations.

I'm not trying to play around with this model with higher throughput. I've done this only to test and verify that multi-node TP deployment is working properly with our current setup here, so we focus on the bigger models now!

## GLM-5.2 (FP8)

Evaluating feasibility for GLM-5.2 FP8 (~744–750 GB model weights) on our 16× H100 80GB cluster (1,280 GB total VRAM):

**Aggregated TP16 (1 Model Copy across 16 GPUs) — Feasible**  
Running a single model copy across all 16 GPUs works because the ~744 GB model weights split to roughly 46.5 GB per GPU (~744 GB total). This leaves about 530 GB of gross VRAM across the cluster for KV cache and runtime execution buffers.

**Disaggregated Prefill + Decode (8 H100 Prefill + 8 H100 Decode) — Not Feasible**  
Disaggregated P/D requires two complete model copies (one for prefill and one for decode), totaling ~1,488 GB in weights alone—which exceeds our cluster's 1,280 GB total VRAM. Additionally, an 8× H100 node only has 640 GB VRAM, but one copy requires ~744 GB (~93 GB/GPU), so an 8-GPU group cannot hold the model. This is why NVIDIA's reference P/D recipes target H200 nodes (141GB VRAM per GPU).
