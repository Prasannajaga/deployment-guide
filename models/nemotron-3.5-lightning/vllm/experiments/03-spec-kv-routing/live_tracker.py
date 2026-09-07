#!/usr/bin/env python3
"""Live terminal telemetry tracker for Experiment 3 (vLLM speculative decoding x KV routing)."""

from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request


PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://127.0.0.1:9090")
NAMESPACE = os.environ.get("NAMESPACE", "qwen32-bench")
DEPLOYMENT = os.environ.get("DEPLOYMENT", "nemotron35-vllm-e3")
CELL = os.environ.get("EXPERIMENT_CELL", os.environ.get("CELL", "D")).upper()
REFRESH_INTERVAL = int(os.environ.get("REFRESH_INTERVAL", "2"))

SCOPE = f'pod=~"{DEPLOYMENT}-.*",namespace="{NAMESPACE}"'


def query_prometheus(query: str) -> list[dict]:
    url = f"{PROMETHEUS_URL}/api/v1/query?{urllib.parse.urlencode({'query': query})}"
    req = urllib.request.Request(url, headers={"User-Agent": "nemotron-tracker/3.0"})
    try:
        with urllib.request.urlopen(req, timeout=4) as response:
            if response.status == 200:
                payload = json.loads(response.read().decode("utf-8"))
                if payload.get("status") == "success":
                    return payload.get("data", {}).get("result", [])
    except Exception:
        pass
    return []


def get_scalar(queries: list[str], default: float = 0.0) -> float:
    for q in queries:
        res = query_prometheus(q)
        if res and "value" in res[0]:
            try:
                val = float(res[0]["value"][1])
                if not math.isnan(val) and not math.isinf(val):
                    return val
            except (ValueError, IndexError, TypeError):
                continue
    return default


def get_vector(queries: list[str]) -> list[dict]:
    for q in queries:
        res = query_prometheus(q)
        if res:
            return res
    return []


def main() -> None:
    print("\033[?25l", end="")  # Hide cursor
    try:
        while True:
            # 1. Target Health
            up_targets = query_prometheus(f'up{{{SCOPE}}}')
            total_up = sum(1 for t in up_targets if t.get("value", [0, "0"])[1] == "1")

            # 2. Token Throughput
            gen_tokens = get_scalar([
                f'sum(rate(dynamo_frontend_output_tokens_total{{{SCOPE}}}[1m]))',
                f'sum(rate({{__name__="vllm:generation_tokens_total",{SCOPE}}}[1m]))',
                f'sum(rate(vllm_generation_tokens_total{{{SCOPE}}}[1m]))',
                f'sum(increase(dynamo_frontend_output_tokens_total{{{SCOPE}}}[1m])) / 60.0',
                f'sum(increase({{__name__="vllm:generation_tokens_total",{SCOPE}}}[1m])) / 60.0',
            ], default=0.0)

            prompt_tokens = get_scalar([
                f'sum(rate(dynamo_frontend_request_tokens_total{{{SCOPE}}}[1m]))',
                f'sum(rate({{__name__="vllm:prompt_tokens_total",{SCOPE}}}[1m]))',
                f'sum(rate(vllm_prompt_tokens_total{{{SCOPE}}}[1m]))',
            ], default=0.0)

            # 3. MTP Speculative Metrics
            mtp_accepted = get_scalar([
                f'sum(rate({{__name__="vllm:spec_decode_num_accepted_tokens_total",{SCOPE}}}[1m]))',
                f'sum(rate(vllm_spec_decode_num_accepted_tokens_total{{{SCOPE}}}[1m]))',
                f'sum(rate({{__name__=~".*spec_decode_num_accepted_tokens.*",{SCOPE}}}[1m]))',
            ], default=0.0)

            mtp_draft = get_scalar([
                f'sum(rate({{__name__="vllm:spec_decode_num_draft_tokens_total",{SCOPE}}}[1m]))',
                f'sum(rate(vllm_spec_decode_num_draft_tokens_total{{{SCOPE}}}[1m]))',
                f'sum(rate({{__name__=~".*spec_decode_num_draft_tokens.*",{SCOPE}}}[1m]))',
            ], default=0.0)

            mtp_rate = (mtp_accepted / mtp_draft * 100.0) if mtp_draft > 0 else 0.0

            # 4. KV-Aware Routing & Events
            router_hit_rate = get_scalar([
                f'histogram_quantile(0.50, sum by (le) (rate(dynamo_component_router_kv_hit_rate_bucket{{{SCOPE}}}[1m]))) * 100.0',
                'histogram_quantile(0.50, sum by (le) (rate(dynamo_component_router_kv_hit_rate_bucket[1m]))) * 100.0',
            ], default=0.0)

            worker_cache_hits = get_scalar([
                f'sum(rate({{__name__="vllm:prefix_cache_hits_total",{SCOPE}}}[1m]))',
                f'sum(rate(vllm_prefix_cache_hits_total{{{SCOPE}}}[1m]))',
                'sum(rate({__name__=~".*prefix_cache_hits.*"}[1m]))',
            ], default=0.0)

            worker_cache_queries = get_scalar([
                f'sum(rate({{__name__="vllm:prefix_cache_queries_total",{SCOPE}}}[1m]))',
                f'sum(rate(vllm_prefix_cache_queries_total{{{SCOPE}}}[1m]))',
                'sum(rate({__name__=~".*prefix_cache_queries.*"}[1m]))',
            ], default=0.0)

            worker_prefix_hit_rate = (
                (worker_cache_hits / worker_cache_queries * 100.0)
                if worker_cache_queries > 0
                else 0.0
            )

            kv_events_rate = get_scalar([
                f'sum(rate(dynamo_component_kv_cache_events_applied{{{SCOPE}}}[1m]))',
                'sum(rate(dynamo_component_kv_cache_events_applied[1m]))',
            ], default=0.0)

            # 5. Worker Queues & KV Cache
            running_res = get_vector([
                f'sum by (pod) ({{__name__="vllm:num_requests_running",{SCOPE}}})',
                f'sum by (pod) (vllm_num_requests_running{{{SCOPE}}})',
            ])

            waiting_res = get_vector([
                f'sum by (pod) ({{__name__="vllm:num_requests_waiting",{SCOPE}}})',
                f'sum by (pod) (vllm_num_requests_waiting{{{SCOPE}}})',
            ])

            kv_cache_res = get_vector([
                f'{{__name__="vllm:kv_cache_usage_perc",{SCOPE}}}',
                f'vllm_kv_cache_usage_perc{{{SCOPE}}}',
            ])

            # 6. Common Unfiltered DCGM GPU Telemetry
            gpu_util_raw = get_vector(['DCGM_FI_DEV_GPU_UTIL'])
            gpu_mem_raw = get_vector(['DCGM_FI_DEV_FB_USED'])
            gpu_power_raw = get_vector(['DCGM_FI_DEV_POWER_USAGE'])

            gpu_data: dict[str, dict[str, float]] = {}
            for r in gpu_util_raw:
                m = r.get("metric", {})
                gpu_id = m.get("gpu", m.get("device", m.get("UUID", "0")))
                key = f"GPU {gpu_id}"
                if "instance" in m:
                    node_short = m['instance'].split(':')[0].split('.')[-1]
                    key = f"GPU {gpu_id} (node .{node_short})"
                try:
                    gpu_data.setdefault(key, {})["util"] = float(r["value"][1])
                except (ValueError, IndexError, TypeError):
                    pass

            for r in gpu_mem_raw:
                m = r.get("metric", {})
                gpu_id = m.get("gpu", m.get("device", m.get("UUID", "0")))
                key = f"GPU {gpu_id}"
                if "instance" in m:
                    node_short = m['instance'].split(':')[0].split('.')[-1]
                    key = f"GPU {gpu_id} (node .{node_short})"
                try:
                    gpu_data.setdefault(key, {})["mem"] = float(r["value"][1])
                except (ValueError, IndexError, TypeError):
                    pass

            for r in gpu_power_raw:
                m = r.get("metric", {})
                gpu_id = m.get("gpu", m.get("device", m.get("UUID", "0")))
                key = f"GPU {gpu_id}"
                if "instance" in m:
                    node_short = m['instance'].split(':')[0].split('.')[-1]
                    key = f"GPU {gpu_id} (node .{node_short})"
                try:
                    gpu_data.setdefault(key, {})["power"] = float(r["value"][1])
                except (ValueError, IndexError, TypeError):
                    pass

            # Map Nemotron Pods
            pod_data: dict[str, dict[str, float]] = {}
            for r in up_targets:
                pod = r["metric"].get("pod", "unknown")
                if DEPLOYMENT in pod:
                    pod_data.setdefault(pod, {})["up"] = float(r["value"][1])

            for r in running_res:
                pod = r["metric"].get("pod", "unknown")
                if DEPLOYMENT in pod:
                    pod_data.setdefault(pod, {})["running"] = float(r["value"][1])
            for r in waiting_res:
                pod = r["metric"].get("pod", "unknown")
                if DEPLOYMENT in pod:
                    pod_data.setdefault(pod, {})["waiting"] = float(r["value"][1])
            for r in kv_cache_res:
                pod = r["metric"].get("pod", "unknown")
                if DEPLOYMENT in pod:
                    val = float(r["value"][1])
                    pod_data.setdefault(pod, {})["kv_cache"] = val * 100.0 if val <= 1.0 else val

            # Render Dashboard
            now_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            lines = [
                "\033[H\033[J",
                "=" * 88,
                f"  NEMOTRON-3.5 VLLM TELEMETRY DASHBOARD (Cell {CELL} | Namespace: {NAMESPACE})",
                f"  Deployment: {DEPLOYMENT}  |  Scraped Pods: {total_up}/6  |  Time: {now_utc}",
                "=" * 88,
                "",
                "--- SYSTEM THROUGHPUT & METRICS ---",
                f"  Generation Throughput : {gen_tokens:10.1f} tokens/sec",
                f"  Prompt Throughput     : {prompt_tokens:10.1f} tokens/sec",
            ]

            if CELL in ("B", "D"):
                if mtp_draft > 0:
                    lines.append(f"  MTP Acceptance Rate   : {mtp_rate:9.1f} %  (accepted: {mtp_accepted:.1f}/s, draft: {mtp_draft:.1f}/s)")
                else:
                    lines.append("  MTP Acceptance Rate   :      0.0 %  (Waiting for active generation requests...)")
            else:
                lines.append(f"  MTP Speculative       :   DISABLED  (Expected: Cell {CELL} is non-speculative)")

            if CELL in ("C", "D"):
                lines.append(f"  Router Predicted Hits : {router_hit_rate:9.1f} %  (Worker Prefix Hit Rate: {worker_prefix_hit_rate:.1f}%)")
                lines.append(f"  ZMQ KV Events Applied : {kv_events_rate:10.1f} events/sec")
            else:
                lines.append(f"  KV-Aware Routing      :   DISABLED  (Expected: Cell {CELL} is Round-Robin)")

            lines.extend([
                "",
                "--- NEMOTRON-3.5 WORKER QUEUES & KV CACHE (TP1 x 4) ---",
                f"  {'POD NAME':<46} {'RUNNING':>10} {'WAITING':>10} {'KV CACHE':>12}",
                "  " + "-" * 82,
            ])

            nemotron_workers = [p for p in sorted(pod_data.keys()) if "worker" in p and DEPLOYMENT in p]
            if not nemotron_workers:
                lines.append("  Waiting for Nemotron-3.5 worker metrics...")
            else:
                for w in nemotron_workers:
                    d = pod_data[w]
                    r_val = f"{int(d.get('running', 0))}"
                    w_val = f"{int(d.get('waiting', 0))}"
                    kv_val = f"{d.get('kv_cache', 0.0):.1f}%"
                    lines.append(f"  {w:<46} {r_val:>10} {w_val:>10} {kv_val:>12}")

            lines.extend([
                "",
                "--- HARDWARE GPU TELEMETRY (DCGM ALL GPUS) ---",
                f"  {'GPU DEVICE':<26} {'UTILIZATION':>14} {'VRAM USED (MB)':>16} {'POWER (W)':>14}",
                "  " + "-" * 74,
            ])

            if not gpu_data:
                lines.append("  Waiting for DCGM exporter GPU samples...")
            else:
                for g in sorted(gpu_data.keys()):
                    gd = gpu_data[g]
                    util_str = f"{gd.get('util', 0.0):.1f}%"
                    mem_str = f"{gd.get('mem', 0.0):.0f}"
                    power_str = f"{gd.get('power', 0.0):.1f}W"
                    lines.append(f"  {g:<26} {util_str:>14} {mem_str:>16} {power_str:>14}")

            lines.extend(["=" * 88, "Press Ctrl+C to exit."])
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()
            time.sleep(REFRESH_INTERVAL)
    except KeyboardInterrupt:
        pass
    finally:
        print("\033[?25h")


if __name__ == "__main__":
    main()
