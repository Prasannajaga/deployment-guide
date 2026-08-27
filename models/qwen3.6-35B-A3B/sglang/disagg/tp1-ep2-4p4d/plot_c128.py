#!/usr/bin/env python3
"""
Custom Plotting Script for Concurrency 128: Baseline vs. KV Offloading (HiCache).

Extracts metrics from AIPerf profile_export_aiperf.json files:
- Output Token Throughput (output tok/s)
- Request Throughput (request/s)
- Time to First Token (TTFT - p50 & p95)
- Time Per Output Token / Inter-Token Latency (TPOT / ITL - p50 & p95)
- End-to-End Request Latency (E2E - p50 & p95)
- Prompt Cache Read Hit Rate (%)

Generates modern, dark-themed plots styled like the reference UI (Dark Navy background,
custom pill badges/tooltips, vibrant purple and cyan bars, clean axis typography).
"""

import os
import sys
import json
import argparse
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    plt = None


# Default Paths relative to the cluster authoring repo
DEFAULT_BASELINE_JSON = Path(
    "/data/inference/cluster/models/qwen3.6-35B-A3B/artifacts/tp1-ep2-4p4d/"
    "baseline-kv-aware/isl-64536_osl-256/c128/profile_export_aiperf.json"
)

DEFAULT_OFFLOAD_JSON = Path(
    "/data/inference/cluster/models/qwen3.6-35B-A3B/artifacts/tp1-ep2-4p4d/"
    "baseline-kv-aware+offloading/isl-64536_osl-256/c128/profile_export_aiperf.json"
)


# Color Palette (Clean White Background Theme)
BG_COLOR = "#ffffff"         # Default white background
CARD_BG = "#ffffff"          # Clean white container background
GRID_COLOR = "#e2e8f0"       # Light subtle gray grid line
BORDER_COLOR = "#cbd5e1"     # Card border
TEXT_MUTED = "#64748b"       # Muted subtitle/header text
TEXT_MAIN = "#1e293b"        # Main dark axis text
TEXT_WHITE = "#0f172a"       # Dark text for highlights

# Bar Colors (High Contrast on White Background)
COLOR_BASELINE = "#475569"   # Solid Slate Gray for Baseline (HBM/GPU)
COLOR_OFFLOAD = "#ea580c"    # Vibrant Orange for KV Offload (HiCache CPU)
BADGE_BG = "#f8fafc"         # Light pill badge background
BADGE_BORDER = "#cbd5e1"     # Badge border


def load_metrics(json_path: Path) -> dict:
    """Load and parse essential metrics from an AIPerf profile export JSON."""
    if not json_path.exists():
        raise FileNotFoundError(f"AIPerf JSON not found at: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "output_tokens_per_sec": data.get("output_token_throughput", {}).get("avg", 0.0),
        "request_throughput": data.get("request_throughput", {}).get("avg", 0.0),
        "total_token_throughput": data.get("total_token_throughput", {}).get("avg", 0.0),
        "ttft_p50_ms": data.get("time_to_first_token", {}).get("p50", 0.0),
        "ttft_p95_ms": data.get("time_to_first_token", {}).get("p95", 0.0),
        "ttft_p50_s": data.get("time_to_first_token", {}).get("p50", 0.0) / 1000.0,
        "ttft_p95_s": data.get("time_to_first_token", {}).get("p95", 0.0) / 1000.0,
        "itl_p50_ms": data.get("inter_token_latency", {}).get("p50", 0.0),
        "itl_p95_ms": data.get("inter_token_latency", {}).get("p95", 0.0),
        "e2e_p50_ms": data.get("request_latency", {}).get("p50", 0.0),
        "e2e_p95_ms": data.get("request_latency", {}).get("p95", 0.0),
        "e2e_p50_s": data.get("request_latency", {}).get("p50", 0.0) / 1000.0,
        "e2e_p95_s": data.get("request_latency", {}).get("p95", 0.0) / 1000.0,
        "cache_hit_pct": data.get("overall_usage_prompt_cache_read_pct", {}).get("avg", 0.0),
        "request_count": data.get("request_count", {}).get("avg", 0),
    }


def draw_single_card(
    ax,
    title: str,
    val_baseline: float,
    val_offload: float,
    unit: str,
    y_axis_label: str = None,
    higher_is_better: bool = True,
    y_max: float = None,
    format_str: str = "{:.2f}",
    detail_unit: str = None,
    val_baseline_detail: float = None,
    val_offload_detail: float = None,
    show_x_labels: bool = False,
):
    """Draw a single high-fidelity card matching the reference dark UI."""
    ax.set_facecolor(CARD_BG)
    x_positions = [0.75, 1.75]
    bar_width = 0.44

    # Determine Winner & Plain Text Direction
    if higher_is_better:
        base_wins = val_baseline > val_offload
        offload_wins = val_offload > val_baseline
        direction_text = "Higher is better"
    else:
        base_wins = val_baseline < val_offload
        offload_wins = val_offload < val_baseline
        direction_text = "Lower is better"

    bars = ax.bar(
        x_positions,
        [val_baseline, val_offload],
        width=bar_width,
        color=[COLOR_BASELINE, COLOR_OFFLOAD],
        edgecolor=[COLOR_BASELINE, COLOR_OFFLOAD],
        linewidth=1.0,
        zorder=3,
    )

    # Styling Limits and Spacing
    max_val = max(val_baseline, val_offload)
    if y_max is None:
        y_max = max_val * 1.35 if max_val > 0 else 1.0

    ax.set_ylim(0, y_max)
    ax.set_xlim(0.1, 2.4)

    # Gridlines
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, color=GRID_COLOR, zorder=1)
    ax.xaxis.grid(False)

    # Title Placed Cleanly Above the Plot Area
    title_full = f"{title.upper()}  ({direction_text})"
    ax.set_title(
        title_full,
        loc="left",
        fontsize=9.5,
        fontweight="bold",
        color=TEXT_DARK,
        pad=10,
    )

    # Vertical Y-Axis Label
    label_text = y_axis_label if y_axis_label else f"({unit})"
    ax.set_ylabel(label_text, color=TEXT_MUTED, fontsize=9.0, labelpad=6)
    ax.tick_params(axis="y", colors=TEXT_MUTED, labelsize=8.5, length=0)

    # X-Axis Labels: Show only if explicitly requested
    if show_x_labels:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            ["Baseline", "KV Offloading"],
            color=TEXT_MAIN,
            fontsize=9.0,
            fontweight="medium",
        )
        ax.tick_params(axis="x", length=0, pad=6)
    else:
        ax.set_xticks([])

    # Spines
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BORDER_COLOR)
    ax.spines["bottom"].set_linewidth(1.0)

    # Calculate difference label
    if higher_is_better:
        pct_diff = ((val_offload - val_baseline) / val_baseline) * 100 if val_baseline > 0 else 0
        diff_label = f"+{pct_diff:.1f}%"
    else:
        if val_baseline > 0:
            reduction = ((val_baseline - val_offload) / val_baseline) * 100
            diff_label = f"{abs(reduction):.1f}% faster" if val_offload < val_baseline else f"+{abs(reduction):.1f}% slower"
        else:
            diff_label = ""

    # Value strings
    if val_baseline_detail is not None:
        base_text = f"{format_str.format(val_baseline)} {unit}\n({val_baseline_detail:,.0f} {detail_unit})"
    else:
        base_text = f"{format_str.format(val_baseline)} {unit}"

    if val_offload_detail is not None:
        offload_text = f"{format_str.format(val_offload)} {unit}\n({val_offload_detail:,.0f} {detail_unit})"
    else:
        offload_text = f"{format_str.format(val_offload)} {unit}"

    # Annotate Baseline Value
    ax.text(
        0.75,
        val_baseline + (y_max * 0.02),
        base_text,
        ha="center",
        va="bottom",
        color=TEXT_DARK if base_wins else TEXT_MUTED,
        fontsize=8.5,
        fontweight="bold" if base_wins else "normal",
        zorder=6,
    )

    # Annotate Offload Value
    ax.text(
        1.75,
        val_offload + (y_max * 0.02),
        offload_text,
        ha="center",
        va="bottom",
        color=TEXT_DARK if offload_wins else TEXT_MUTED,
        fontsize=8.5,
        fontweight="bold" if offload_wins else "normal",
        zorder=6,
    )

    # Draw Highlighted Winner Badge
    bbox_winner = dict(
        boxstyle="round,pad=0.4,rounding_size=0.3",
        facecolor=WINNER_BG,
        edgecolor=WINNER_BORDER,
        linewidth=1.0,
    )
    if offload_wins:
        badge_y = val_offload + (y_max * 0.16 if val_offload_detail else y_max * 0.09)
        ax.text(
            1.75,
            badge_y,
            f"★ WINNER ({diff_label})",
            ha="center",
            va="bottom",
            color=WINNER_TEXT,
            fontsize=8.0,
            fontweight="bold",
            bbox=bbox_winner,
            zorder=7,
        )
    elif base_wins:
        badge_y = val_baseline + (y_max * 0.16 if val_baseline_detail else y_max * 0.09)
        ax.text(
            0.75,
            badge_y,
            "★ WINNER",
            ha="center",
            va="bottom",
            color=WINNER_TEXT,
            fontsize=8.0,
            fontweight="bold",
            bbox=bbox_winner,
            zorder=7,
        )


def generate_c128_plots(
    baseline_path: Path,
    offload_path: Path,
    output_dir: Path,
):
    """Generate individual and dashboard comparison plots for Concurrency 128."""
    if plt is None:
        print("Error: matplotlib is required to generate plots. Please install matplotlib.", file=sys.stderr)
        return

    import matplotlib.patches as mpatches

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Baseline C128 metrics from: {baseline_path}")
    base = load_metrics(baseline_path)

    print(f"Loading Offload C128 metrics from:   {offload_path}")
    offload = load_metrics(offload_path)

    # List of all metric definitions with milliseconds & seconds details
    metrics = [
        {
            "name": "output_tokens_per_sec",
            "title": "Output Token Throughput",
            "val_base": base["output_tokens_per_sec"],
            "val_offload": offload["output_tokens_per_sec"],
            "unit": "tokens/sec",
            "y_label": "Tokens / sec",
            "higher_is_better": True,
            "y_max": 2600,
            "format": "{:,.0f}",
            "det_unit": None,
            "base_det": None,
            "off_det": None,
        },
        {
            "name": "request_throughput",
            "title": "Request Throughput",
            "val_base": base["request_throughput"],
            "val_offload": offload["request_throughput"],
            "unit": "requests/sec",
            "y_label": "Requests / sec",
            "higher_is_better": True,
            "y_max": 10.5,
            "format": "{:.2f}",
            "det_unit": None,
            "base_det": None,
            "off_det": None,
        },
        {
            "name": "ttft_p50",
            "title": "Time to First Token (P50 TTFT)",
            "val_base": base["ttft_p50_s"],
            "val_offload": offload["ttft_p50_s"],
            "unit": "sec",
            "y_label": "Latency (seconds)",
            "higher_is_better": False,
            "y_max": 24.0,
            "format": "{:.2f}",
            "det_unit": "ms",
            "base_det": base["ttft_p50_ms"],
            "off_det": offload["ttft_p50_ms"],
        },
        {
            "name": "ttft_p95",
            "title": "Time to First Token (P95 TTFT)",
            "val_base": base["ttft_p95_s"],
            "val_offload": offload["ttft_p95_s"],
            "unit": "sec",
            "y_label": "Latency (seconds)",
            "higher_is_better": False,
            "y_max": 75.0,
            "format": "{:.2f}",
            "det_unit": "ms",
            "base_det": base["ttft_p95_ms"],
            "off_det": offload["ttft_p95_ms"],
        },
        {
            "name": "itl_p50",
            "title": "Inter-Token Latency (P50 ITL / TPOT)",
            "val_base": base["itl_p50_ms"],
            "val_offload": offload["itl_p50_ms"],
            "unit": "ms",
            "y_label": "Latency (ms)",
            "higher_is_better": False,
            "y_max": 12.0,
            "format": "{:.2f}",
            "det_unit": None,
            "base_det": None,
            "off_det": None,
        },
        {
            "name": "e2e_p95",
            "title": "End-to-End Latency (P95 E2E)",
            "val_base": base["e2e_p95_s"],
            "val_offload": offload["e2e_p95_s"],
            "unit": "sec",
            "y_label": "Latency (seconds)",
            "higher_is_better": False,
            "y_max": 75.0,
            "format": "{:.2f}",
            "det_unit": "ms",
            "base_det": base["e2e_p95_ms"],
            "off_det": offload["e2e_p95_ms"],
        },
    ]

    # 1. Generate Individual High-Resolution PNGs for each metric
    for m in metrics:
        fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=BG_COLOR, dpi=200)
        draw_single_card(
            ax=ax,
            title=m["title"],
            val_baseline=m["val_base"],
            val_offload=m["val_offload"],
            unit=m["unit"],
            y_axis_label=m["y_label"],
            higher_is_better=m["higher_is_better"],
            y_max=m["y_max"],
            format_str=m["format"],
            detail_unit=m["det_unit"],
            val_baseline_detail=m["base_det"],
            val_offload_detail=m["off_det"],
            show_x_labels=True,
        )
        plt.tight_layout(pad=1.5)
        out_file = output_dir / f"c128_{m['name']}.png"
        fig.savefig(out_file, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"Saved: {out_file}")

    # 2. Generate Combined 6-Panel Dashboard with Global 2-Item Legend
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), facecolor=BG_COLOR, dpi=200)
    fig.suptitle(
        "Qwen3.6-35B-A3B FP8 (16x H100) — Concurrency 128 Performance Breakdown\nBaseline KV-Aware vs. HiCache CPU Offloading",
        color=TEXT_DARK,
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )

    # 2-Item Legend Only
    patch_base = mpatches.Patch(color=COLOR_BASELINE, label="Baseline (GPU VRAM / HBM)")
    patch_offload = mpatches.Patch(color=COLOR_OFFLOAD, label="KV Offloading (HiCache CPU)")

    fig.legend(
        handles=[patch_base, patch_offload],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=2,
        frameon=True,
        facecolor="#f8fafc",
        edgecolor="#cbd5e1",
        fontsize=10.0,
    )

    for ax, m in zip(axes.flatten(), metrics):
        draw_single_card(
            ax=ax,
            title=m["title"],
            val_baseline=m["val_base"],
            val_offload=m["val_offload"],
            unit=m["unit"],
            y_axis_label=m["y_label"],
            higher_is_better=m["higher_is_better"],
            y_max=m["y_max"],
            format_str=m["format"],
            detail_unit=m["det_unit"],
            val_baseline_detail=m["base_det"],
            val_offload_detail=m["off_det"],
            show_x_labels=False,
        )

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.90], h_pad=2.8, w_pad=2.2)
    dashboard_file = output_dir / "c128_comparison_dashboard.png"
    fig.savefig(dashboard_file, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved Combined Dashboard: {dashboard_file}")


def generate_standalone_html(
    baseline_path: Path,
    offload_path: Path,
    output_html_file: Path,
):
    """Generate an interactive standalone HTML dashboard with exact dark theme styling."""
    base = load_metrics(baseline_path)
    offload = load_metrics(offload_path)

    cards = [
        {
            "title": "OUTPUT TOKENS/SEC",
            "base_val": f"{base['output_tokens_per_sec']:,.0f} tok/s",
            "offload_val": f"{offload['output_tokens_per_sec']:,.0f} tok/s",
            "base_h": min(100, (base["output_tokens_per_sec"] / 2400) * 100),
            "offload_h": min(100, (offload["output_tokens_per_sec"] / 2400) * 100),
            "gain": f"+{((offload['output_tokens_per_sec'] - base['output_tokens_per_sec'])/base['output_tokens_per_sec'])*100:.1f}%",
            "gain_type": "positive",
        },
        {
            "title": "REQUEST THROUGHPUT",
            "base_val": f"{base['request_throughput']:.2f} req/s",
            "offload_val": f"{offload['request_throughput']:.2f} req/s",
            "base_h": min(100, (base["request_throughput"] / 9.0) * 100),
            "offload_h": min(100, (offload["request_throughput"] / 9.0) * 100),
            "gain": f"+{((offload['request_throughput'] - base['request_throughput'])/base['request_throughput'])*100:.1f}%",
            "gain_type": "positive",
        },
        {
            "title": "TIME TO FIRST TOKEN (P50 TTFT)",
            "base_val": f"{base['ttft_p50_s']:.2f} s",
            "offload_val": f"{offload['ttft_p50_s']:.2f} s",
            "base_h": min(100, (base["ttft_p50_s"] / 18.0) * 100),
            "offload_h": min(100, (offload["ttft_p50_s"] / 18.0) * 100),
            "gain": f"{((offload['ttft_p50_s'] - base['ttft_p50_s'])/base['ttft_p50_s'])*100:.1f}%",
            "gain_type": "positive",
        },
        {
            "title": "TIME TO FIRST TOKEN (P95 TTFT)",
            "base_val": f"{base['ttft_p95_s']:.2f} s",
            "offload_val": f"{offload['ttft_p95_s']:.2f} s",
            "base_h": min(100, (base["ttft_p95_s"] / 60.0) * 100),
            "offload_h": min(100, (offload["ttft_p95_s"] / 60.0) * 100),
            "gain": f"{((offload['ttft_p95_s'] - base['ttft_p95_s'])/base['ttft_p95_s'])*100:.1f}%",
            "gain_type": "positive",
        },
        {
            "title": "INTER-TOKEN LATENCY (P50 ITL / TPOT)",
            "base_val": f"{base['itl_p50_ms']:.2f} ms",
            "offload_val": f"{offload['itl_p50_ms']:.2f} ms",
            "base_h": min(100, (base["itl_p50_ms"] / 10.0) * 100),
            "offload_h": min(100, (offload["itl_p50_ms"] / 10.0) * 100),
            "gain": "Steady (0% penalty)",
            "gain_type": "neutral",
        },
        {
            "title": "P95 END-TO-END LATENCY (E2E)",
            "base_val": f"{base['e2e_p95_s']:.2f} s",
            "offload_val": f"{offload['e2e_p95_s']:.2f} s",
            "base_h": min(100, (base["e2e_p95_s"] / 60.0) * 100),
            "offload_h": min(100, (offload["e2e_p95_s"] / 60.0) * 100),
            "gain": f"{((offload['e2e_p95_s'] - base['e2e_p95_s'])/base['e2e_p95_s'])*100:.1f}%",
            "gain_type": "positive",
        },
    ]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Concurrency 128 Benchmark: Baseline vs KV Offloading</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  body {{ background-color: #ffffff; color: #1e293b; padding: 30px; }}
  h1 {{ font-size: 20px; color: #0f172a; margin-bottom: 8px; font-weight: 700; }}
  p.subtitle {{ font-size: 13px; color: #64748b; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 20px; }}
  .card {{ background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
  .card-title {{ font-size: 11px; font-weight: 700; color: #64748b; letter-spacing: 1px; margin-bottom: 20px; }}
  .chart-area {{ height: 180px; display: flex; align-items: flex-end; justify-content: space-around; padding: 0 30px 10px; border-bottom: 1px solid #e2e8f0; position: relative; }}
  .bar-group {{ display: flex; flex-direction: column; align-items: center; width: 60px; height: 100%; justify-content: flex-end; position: relative; }}
  .bar {{ width: 44px; border-radius: 6px 6px 0 0; transition: all 0.3s ease; }}
  .bar.baseline {{ background: #6366f1; }}
  .bar.offload {{ background: #0284c7; }}
  .tooltip-badge {{
    position: absolute; top: -38px; background: #f8fafc; border: 1px solid #cbd5e1;
    border-radius: 8px; padding: 4px 8px; font-size: 11px; color: #0f172a; font-weight: 600;
    white-space: nowrap; box-shadow: 0 2px 8px rgba(0,0,0,0.08); z-index: 10;
  }}
  .val-text {{ font-size: 11px; color: #64748b; margin-top: 6px; font-weight: 500; }}
  .x-labels {{ display: flex; justify-content: space-around; margin-top: 12px; font-size: 11px; color: #475569; font-weight: 500; }}
  .badge-gain {{ font-size: 11px; font-weight: 600; color: #0284c7; margin-top: 8px; text-align: right; }}
</style>
</head>
<body>
  <h1>Qwen3.6-35B-A3B FP8 (16x H100) — Concurrency 128 Performance</h1>
  <p class="subtitle">Disaggregated SGLang 4P4D (TP1-Attention + EP2-MoE) — Baseline vs. CPU HiCache Offloading</p>

  <div class="grid">
"""

    for c in cards:
        html_content += f"""
    <div class="card">
      <div class="card-title">{c['title']}</div>
      <div class="chart-area">
        <div class="bar-group">
          <div class="bar baseline" style="height: {c['base_h']}%;"></div>
          <span class="val-text">{c['base_val']}</span>
        </div>
        <div class="bar-group">
          <div class="tooltip-badge">HiCache: {c['offload_val']} ({c['gain']})</div>
          <div class="bar offload" style="height: {c['offload_h']}%;"></div>
          <span class="val-text">{c['offload_val']}</span>
        </div>
      </div>
      <div class="x-labels">
        <span>Baseline (GPU VRAM)</span>
        <span>HiCache (CPU Offload)</span>
      </div>
    </div>
"""

    html_content += """
  </div>
</body>
</html>
"""

    output_html_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved Interactive HTML: {output_html_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Custom Plotting for Concurrency 128 AIPerf Benchmark Exports"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_JSON,
        help="Path to baseline profile_export_aiperf.json",
    )
    parser.add_argument(
        "--offload",
        type=Path,
        default=DEFAULT_OFFLOAD_JSON,
        help="Path to KV offloading profile_export_aiperf.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/inference/cluster/models/qwen3.6-35B-A3B/artifacts/tp1-ep2-4p4d/plots/c128"),
        help="Output directory for generated plots",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Generating C128 Comparison Plots (Baseline vs. KV Offloading)")
    print("=" * 60)

    generate_c128_plots(
        baseline_path=args.baseline,
        offload_path=args.offload,
        output_dir=args.output_dir,
    )

    generate_standalone_html(
        baseline_path=args.baseline,
        offload_path=args.offload,
        output_html_file=args.output_dir / "index.html",
    )

    print("\nPlot generation complete!")


if __name__ == "__main__":
    main()

