#!/usr/bin/env python3
"""
Custom Plotting and Comparison Script for SGLang Disaggregated Serving:
TP1 (4 Prefill + 4 Decode) vs. TP2 (2 Prefill + 2 Decode) on 8x NVIDIA H100 GPUs.

Extracts metrics from AIPerf profile_export_aiperf.json files across concurrencies (c1 to c128):
- Output Token Throughput (output tok/s)
- Request Throughput (requests/s)
- Total Token Throughput (total tok/s)
- Time to First Token (TTFT - p50, p95, p99)
- Inter-Token Latency (ITL / TPOT - p50, p95, p99)
- End-to-End Request Latency (E2E - p50, p95, p99)
- Per-User Output Token Throughput (tok/s/user)

Generates:
1. Concurrency 128 individual metric card PNGs
2. Concurrency 128 combined 6-panel comparison dashboard
3. Multi-concurrency scaling sweep curve plots (c1 to c128)
4. Multi-concurrency 6-panel overview dashboard
5. Standalone responsive HTML comparison dashboard
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.patches as mpatches
except ImportError:
    plt = None


# Default Paths relative to the cluster authoring repo
DEFAULT_TP1_BENCHMARKS = Path(
    "/data/inference/cluster/benchmarks/qwen3.6-35B-A3B/tp1-4p4d"
)

DEFAULT_TP2_BENCHMARKS = Path(
    "/data/inference/cluster/benchmarks/qwen3.6-35B-A3B/tp2-2p2d"
)

DEFAULT_OUTPUT_DIR = Path(
    "/data/inference/cluster/benchmarks/qwen3.6-35B-A3B/tp1-4p4d/plots"
)

CONCURRENCIES = [1, 4, 8, 16, 32, 64, 128]

# Professional High-Contrast Palette (Clean White Background Theme)
BG_COLOR = "#ffffff"         # Pure white background
CARD_BG = "#ffffff"          # White plot/card background
GRID_COLOR = "#e2e8f0"       # Subtle slate gridline
BORDER_COLOR = "#cbd5e1"     # Card border
TEXT_MUTED = "#64748b"       # Muted subtitle/header text
TEXT_MAIN = "#1e293b"        # Main dark axis text
TEXT_DARK = "#0f172a"        # Deep charcoal text for highlights

# Bar and Curve Colors (High Contrast on White Background)
COLOR_TP1 = "#475569"        # Solid Slate Gray for TP1 (4P4D: 4 Prefill / 4 Decode TP1)
COLOR_TP2 = "#ea580c"        # Vibrant Orange for TP2 (2P2D: 2 Prefill / 2 Decode TP2)

# Winner Badges
WINNER_BG = "#ecfdf5"        # Emerald green light background
WINNER_BORDER = "#a7f3d0"    # Emerald green border
WINNER_TEXT = "#047857"      # Dark emerald green text

BADGE_BG = "#f8fafc"         # Neutral badge background
BADGE_BORDER = "#cbd5e1"     # Neutral badge border


def load_metrics(json_path: Path) -> dict:
    """Load and parse essential metrics from an AIPerf profile export JSON."""
    if not json_path.exists():
        raise FileNotFoundError(f"AIPerf JSON not found at: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ttft = data.get("time_to_first_token", {})
    itl = data.get("inter_token_latency", {})
    req_lat = data.get("request_latency", {})
    ttst = data.get("time_to_second_token", {})
    req_tput = data.get("request_throughput", {})
    out_tput = data.get("output_token_throughput", {})
    tot_tput = data.get("total_token_throughput", {})
    user_tput = data.get("output_token_throughput_per_user", {})

    return {
        "output_tokens_per_sec": out_tput.get("avg", 0.0),
        "request_throughput": req_tput.get("avg", 0.0),
        "total_token_throughput": tot_tput.get("avg", 0.0),
        "ttft_p50_ms": ttft.get("p50", 0.0),
        "ttft_p95_ms": ttft.get("p95", 0.0),
        "ttft_p99_ms": ttft.get("p99", 0.0),
        "ttft_p50_s": ttft.get("p50", 0.0) / 1000.0,
        "ttft_p95_s": ttft.get("p95", 0.0) / 1000.0,
        "ttft_p99_s": ttft.get("p99", 0.0) / 1000.0,
        "itl_p50_ms": itl.get("p50", 0.0),
        "itl_p95_ms": itl.get("p95", 0.0),
        "itl_p99_ms": itl.get("p99", 0.0),
        "ttst_p50_ms": ttst.get("p50", 0.0),
        "ttst_p95_ms": ttst.get("p95", 0.0),
        "e2e_p50_ms": req_lat.get("p50", 0.0),
        "e2e_p95_ms": req_lat.get("p95", 0.0),
        "e2e_p99_ms": req_lat.get("p99", 0.0),
        "e2e_p50_s": req_lat.get("p50", 0.0) / 1000.0,
        "e2e_p95_s": req_lat.get("p95", 0.0) / 1000.0,
        "e2e_p99_s": req_lat.get("p99", 0.0) / 1000.0,
        "user_tput_avg": user_tput.get("avg", 0.0),
        "user_tput_p50": user_tput.get("p50", 0.0),
        "request_count": data.get("request_count", {}).get("avg", 0),
    }


def load_all_concurrencies(base_dir: Path, concurrencies: List[int] = CONCURRENCIES) -> Dict[int, dict]:
    """Load benchmark metrics across all concurrency subdirectories."""
    results = {}
    for c in concurrencies:
        json_file = base_dir / f"c{c}" / "profile_export_aiperf.json"
        if json_file.exists():
            results[c] = load_metrics(json_file)
        else:
            print(f"Warning: Concurrency c{c} JSON not found at: {json_file}", file=sys.stderr)
    return results


def draw_single_card(
    ax,
    title: str,
    val_tp1: float,
    val_tp2: float,
    unit: str,
    y_axis_label: str = None,
    higher_is_better: bool = True,
    y_max: float = None,
    format_str: str = "{:.2f}",
    detail_unit: str = None,
    val_tp1_detail: float = None,
    val_tp2_detail: float = None,
    show_x_labels: bool = False,
    tp1_label: str = "TP1 (4P4D)",
    tp2_label: str = "TP2 (2P2D)",
):
    """Draw a single high-fidelity card comparing TP1 vs TP2."""
    ax.set_facecolor(CARD_BG)
    x_positions = [0.75, 1.75]
    bar_width = 0.44

    # Determine Winner & Plain Text Direction
    if higher_is_better:
        tp1_wins = val_tp1 > val_tp2
        tp2_wins = val_tp2 > val_tp1
        direction_text = "Higher is better"
    else:
        tp1_wins = val_tp1 < val_tp2
        tp2_wins = val_tp2 < val_tp1
        direction_text = "Lower is better"

    ax.bar(
        x_positions,
        [val_tp1, val_tp2],
        width=bar_width,
        color=[COLOR_TP1, COLOR_TP2],
        edgecolor=[COLOR_TP1, COLOR_TP2],
        linewidth=1.0,
        zorder=3,
    )

    # Scaling Limits and Spacing
    max_val = max(val_tp1, val_tp2)
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

    # X-Axis Labels
    if show_x_labels:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [tp1_label, tp2_label],
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

    # Calculate difference label relative to winner
    if higher_is_better:
        if tp1_wins and val_tp2 > 0:
            pct_lead = ((val_tp1 - val_tp2) / val_tp2) * 100
            diff_label = f"+{pct_lead:.1f}%"
        elif tp2_wins and val_tp1 > 0:
            pct_lead = ((val_tp2 - val_tp1) / val_tp1) * 100
            diff_label = f"+{pct_lead:.1f}%"
        else:
            diff_label = ""
    else:
        if tp1_wins and val_tp2 > 0:
            pct_reduction = ((val_tp2 - val_tp1) / val_tp2) * 100
            diff_label = f"{pct_reduction:.1f}% faster"
        elif tp2_wins and val_tp1 > 0:
            pct_reduction = ((val_tp1 - val_tp2) / val_tp1) * 100
            diff_label = f"{pct_reduction:.1f}% faster"
        else:
            diff_label = ""

    # Value strings
    if val_tp1_detail is not None:
        tp1_text = f"{format_str.format(val_tp1)} {unit}\n({val_tp1_detail:,.0f} {detail_unit})"
    else:
        tp1_text = f"{format_str.format(val_tp1)} {unit}"

    if val_tp2_detail is not None:
        tp2_text = f"{format_str.format(val_tp2)} {unit}\n({val_tp2_detail:,.0f} {detail_unit})"
    else:
        tp2_text = f"{format_str.format(val_tp2)} {unit}"

    # Annotate TP1 Value
    ax.text(
        0.75,
        val_tp1 + (y_max * 0.02),
        tp1_text,
        ha="center",
        va="bottom",
        color=TEXT_DARK if tp1_wins else TEXT_MUTED,
        fontsize=8.5,
        fontweight="bold" if tp1_wins else "normal",
        zorder=6,
    )

    # Annotate TP2 Value
    ax.text(
        1.75,
        val_tp2 + (y_max * 0.02),
        tp2_text,
        ha="center",
        va="bottom",
        color=TEXT_DARK if tp2_wins else TEXT_MUTED,
        fontsize=8.5,
        fontweight="bold" if tp2_wins else "normal",
        zorder=6,
    )

    # Draw Highlighted Winner Badge
    bbox_winner = dict(
        boxstyle="round,pad=0.4,rounding_size=0.3",
        facecolor=WINNER_BG,
        edgecolor=WINNER_BORDER,
        linewidth=1.0,
    )
    if tp1_wins:
        badge_y = val_tp1 + (y_max * 0.16 if val_tp1_detail else y_max * 0.09)
        ax.text(
            0.75,
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
    elif tp2_wins:
        badge_y = val_tp2 + (y_max * 0.16 if val_tp2_detail else y_max * 0.09)
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


def generate_c128_plots(
    tp1_c128_path: Path,
    tp2_c128_path: Path,
    output_dir: Path,
):
    """Generate individual cards and 6-panel dashboard for Concurrency 128."""
    if plt is None:
        print("Error: matplotlib is required. Please install matplotlib.", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading TP1 C128 metrics from: {tp1_c128_path}")
    tp1 = load_metrics(tp1_c128_path)

    print(f"Loading TP2 C128 metrics from: {tp2_c128_path}")
    tp2 = load_metrics(tp2_c128_path)

    metrics = [
        {
            "name": "output_tokens_per_sec",
            "title": "Output Token Throughput",
            "val_tp1": tp1["output_tokens_per_sec"],
            "val_tp2": tp2["output_tokens_per_sec"],
            "unit": "tokens/sec",
            "y_label": "Tokens / sec",
            "higher_is_better": True,
            "y_max": 11500,
            "format": "{:,.0f}",
            "det_unit": None,
            "tp1_det": None,
            "tp2_det": None,
        },
        {
            "name": "request_throughput",
            "title": "Request Throughput",
            "val_tp1": tp1["request_throughput"],
            "val_tp2": tp2["request_throughput"],
            "unit": "requests/sec",
            "y_label": "Requests / sec",
            "higher_is_better": True,
            "y_max": 23.0,
            "format": "{:.2f}",
            "det_unit": None,
            "tp1_det": None,
            "tp2_det": None,
        },
        {
            "name": "ttft_p50",
            "title": "Time to First Token (P50 TTFT)",
            "val_tp1": tp1["ttft_p50_s"],
            "val_tp2": tp2["ttft_p50_s"],
            "unit": "sec",
            "y_label": "Latency (seconds)",
            "higher_is_better": False,
            "y_max": 1.4,
            "format": "{:.2f}",
            "det_unit": "ms",
            "tp1_det": tp1["ttft_p50_ms"],
            "tp2_det": tp2["ttft_p50_ms"],
        },
        {
            "name": "ttft_p95",
            "title": "Time to First Token (P95 TTFT)",
            "val_tp1": tp1["ttft_p95_s"],
            "val_tp2": tp2["ttft_p95_s"],
            "unit": "sec",
            "y_label": "Latency (seconds)",
            "higher_is_better": False,
            "y_max": 15.5,
            "format": "{:.2f}",
            "det_unit": "ms",
            "tp1_det": tp1["ttft_p95_ms"],
            "tp2_det": tp2["ttft_p95_ms"],
        },
        {
            "name": "itl_p50",
            "title": "Inter-Token Latency (P50 ITL / TPOT)",
            "val_tp1": tp1["itl_p50_ms"],
            "val_tp2": tp2["itl_p50_ms"],
            "unit": "ms",
            "y_label": "Latency (ms)",
            "higher_is_better": False,
            "y_max": 14.0,
            "format": "{:.2f}",
            "det_unit": None,
            "tp1_det": None,
            "tp2_det": None,
        },
        {
            "name": "e2e_p95",
            "title": "End-to-End Latency (P95 E2E)",
            "val_tp1": tp1["e2e_p95_s"],
            "val_tp2": tp2["e2e_p95_s"],
            "unit": "sec",
            "y_label": "Latency (seconds)",
            "higher_is_better": False,
            "y_max": 19.5,
            "format": "{:.2f}",
            "det_unit": "ms",
            "tp1_det": tp1["e2e_p95_ms"],
            "tp2_det": tp2["e2e_p95_ms"],
        },
    ]

    # 1. Individual High-Resolution PNGs
    for m in metrics:
        fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=BG_COLOR, dpi=200)
        draw_single_card(
            ax=ax,
            title=m["title"],
            val_tp1=m["val_tp1"],
            val_tp2=m["val_tp2"],
            unit=m["unit"],
            y_axis_label=m["y_label"],
            higher_is_better=m["higher_is_better"],
            y_max=m["y_max"],
            format_str=m["format"],
            detail_unit=m["det_unit"],
            val_tp1_detail=m["tp1_det"],
            val_tp2_detail=m["tp2_det"],
            show_x_labels=True,
            tp1_label="TP1 (4P4D)",
            tp2_label="TP2 (2P2D)",
        )
        plt.tight_layout(pad=1.5)
        out_file = output_dir / f"c128_{m['name']}.png"
        fig.savefig(out_file, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"Saved: {out_file}")

    # 2. Combined 6-Panel Dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), facecolor=BG_COLOR, dpi=200)
    fig.suptitle(
        "Qwen3.6-35B-A3B FP8 (8x H100) — Concurrency 128 Architecture Comparison\nTP1 (4 Prefill + 4 Decode) vs. TP2 (2 Prefill + 2 Decode)",
        color=TEXT_DARK,
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )

    patch_tp1 = mpatches.Patch(color=COLOR_TP1, label="TP1: 4P4D (4 Prefill Pods + 4 Decode Pods, TP=1)")
    patch_tp2 = mpatches.Patch(color=COLOR_TP2, label="TP2: 2P2D (2 Prefill Pods + 2 Decode Pods, TP=2)")

    fig.legend(
        handles=[patch_tp1, patch_tp2],
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
            val_tp1=m["val_tp1"],
            val_tp2=m["val_tp2"],
            unit=m["unit"],
            y_axis_label=m["y_label"],
            higher_is_better=m["higher_is_better"],
            y_max=m["y_max"],
            format_str=m["format"],
            detail_unit=m["det_unit"],
            val_tp1_detail=m["tp1_det"],
            val_tp2_detail=m["tp2_det"],
            show_x_labels=False,
            tp1_label="TP1 (4P4D)",
            tp2_label="TP2 (2P2D)",
        )

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.90], h_pad=2.8, w_pad=2.2)
    dashboard_file = output_dir / "c128_comparison_dashboard.png"
    fig.savefig(dashboard_file, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved Combined Dashboard: {dashboard_file}")


def generate_scaling_plots(
    tp1_data: Dict[int, dict],
    tp2_data: Dict[int, dict],
    output_dir: Path,
):
    """Generate multi-concurrency scaling sweep plots (c1 to c128)."""
    if plt is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    concurrencies = sorted(list(set(tp1_data.keys()) & set(tp2_data.keys())))
    if not concurrencies:
        print("No matching concurrencies found for scaling plots.", file=sys.stderr)
        return

    # Helper function to plot a single curve
    def plot_curve(
        ax,
        title: str,
        y_label: str,
        tp1_vals: List[float],
        tp2_vals: List[float],
        higher_is_better: bool = True,
        log_x: bool = True,
    ):
        ax.set_facecolor(CARD_BG)
        ax.plot(
            concurrencies,
            tp1_vals,
            marker="o",
            markersize=6,
            linewidth=2.2,
            color=COLOR_TP1,
            label="TP1 (4P4D)",
            zorder=4,
        )
        ax.plot(
            concurrencies,
            tp2_vals,
            marker="s",
            markersize=6,
            linewidth=2.2,
            color=COLOR_TP2,
            label="TP2 (2P2D)",
            zorder=4,
        )

        if log_x:
            ax.set_xscale("log", base=2)
            ax.set_xticks(concurrencies)
            ax.set_xticklabels([f"c{c}" for c in concurrencies], color=TEXT_MAIN, fontsize=8.5)

        ax.set_ylabel(y_label, color=TEXT_MUTED, fontsize=9.0)
        ax.set_xlabel("Concurrency", color=TEXT_MUTED, fontsize=9.0)
        ax.tick_params(axis="both", colors=TEXT_MUTED, labelsize=8.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7, color=GRID_COLOR, zorder=1)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4, color=GRID_COLOR, zorder=1)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(BORDER_COLOR)

        dir_str = "Higher is better" if higher_is_better else "Lower is better"
        ax.set_title(f"{title.upper()}  ({dir_str})", loc="left", fontsize=9.5, fontweight="bold", color=TEXT_DARK, pad=8)
        ax.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=8.5, loc="best")

    # Metrics to plot
    curves = [
        {
            "name": "scaling_output_tokens_per_sec",
            "title": "Output Token Throughput Scaling",
            "y_label": "Output Tokens / sec",
            "tp1": [tp1_data[c]["output_tokens_per_sec"] for c in concurrencies],
            "tp2": [tp2_data[c]["output_tokens_per_sec"] for c in concurrencies],
            "higher": True,
        },
        {
            "name": "scaling_request_throughput",
            "title": "Request Throughput Scaling",
            "y_label": "Requests / sec",
            "tp1": [tp1_data[c]["request_throughput"] for c in concurrencies],
            "tp2": [tp2_data[c]["request_throughput"] for c in concurrencies],
            "higher": True,
        },
        {
            "name": "scaling_ttft_p50",
            "title": "Time to First Token (P50 TTFT) Scaling",
            "y_label": "TTFT P50 (ms)",
            "tp1": [tp1_data[c]["ttft_p50_ms"] for c in concurrencies],
            "tp2": [tp2_data[c]["ttft_p50_ms"] for c in concurrencies],
            "higher": False,
        },
        {
            "name": "scaling_ttft_p95",
            "title": "Time to First Token (P95 TTFT) Scaling",
            "y_label": "TTFT P95 (ms)",
            "tp1": [tp1_data[c]["ttft_p95_ms"] for c in concurrencies],
            "tp2": [tp2_data[c]["ttft_p95_ms"] for c in concurrencies],
            "higher": False,
        },
        {
            "name": "scaling_itl_p50",
            "title": "Inter-Token Latency (P50 ITL) Scaling",
            "y_label": "ITL P50 (ms)",
            "tp1": [tp1_data[c]["itl_p50_ms"] for c in concurrencies],
            "tp2": [tp2_data[c]["itl_p50_ms"] for c in concurrencies],
            "higher": False,
        },
        {
            "name": "scaling_e2e_p95",
            "title": "End-to-End Latency (P95 E2E) Scaling",
            "y_label": "E2E P95 (seconds)",
            "tp1": [tp1_data[c]["e2e_p95_s"] for c in concurrencies],
            "tp2": [tp2_data[c]["e2e_p95_s"] for c in concurrencies],
            "higher": False,
        },
    ]

    # 1. Individual Curve PNGs
    for c in curves:
        fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=BG_COLOR, dpi=200)
        plot_curve(
            ax=ax,
            title=c["title"],
            y_label=c["y_label"],
            tp1_vals=c["tp1"],
            tp2_vals=c["tp2"],
            higher_is_better=c["higher"],
        )
        plt.tight_layout(pad=1.5)
        out_file = output_dir / f"{c['name']}.png"
        fig.savefig(out_file, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"Saved: {out_file}")

    # 2. Combined 6-Panel Scaling Overview Dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), facecolor=BG_COLOR, dpi=200)
    fig.suptitle(
        "Qwen3.6-35B-A3B FP8 (8x H100) — Concurrency Scaling Sweep (c1 to c128)\nTP1 (4P4D) vs. TP2 (2P2D) Throughput & Latency Trajectories",
        color=TEXT_DARK,
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )

    for ax, c in zip(axes.flatten(), curves):
        plot_curve(
            ax=ax,
            title=c["title"],
            y_label=c["y_label"],
            tp1_vals=c["tp1"],
            tp2_vals=c["tp2"],
            higher_is_better=c["higher"],
        )

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.94], h_pad=2.8, w_pad=2.2)
    scaling_dashboard_file = output_dir / "scaling_comparison_dashboard.png"
    fig.savefig(scaling_dashboard_file, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved Scaling Dashboard: {scaling_dashboard_file}")


def generate_standalone_html(
    tp1_data: Dict[int, dict],
    tp2_data: Dict[int, dict],
    output_html_file: Path,
):
    """Generate an interactive standalone HTML dashboard comparing TP1 vs TP2."""
    tp1_c128 = tp1_data.get(128, {})
    tp2_c128 = tp2_data.get(128, {})

    cards = [
        {
            "title": "OUTPUT TOKEN THROUGHPUT",
            "tp1_val": f"{tp1_c128.get('output_tokens_per_sec', 0):,.0f} tok/s",
            "tp2_val": f"{tp2_c128.get('output_tokens_per_sec', 0):,.0f} tok/s",
            "tp1_h": min(100, (tp1_c128.get('output_tokens_per_sec', 0) / 10000) * 100),
            "tp2_h": min(100, (tp2_c128.get('output_tokens_per_sec', 0) / 10000) * 100),
            "lead": f"+{((tp1_c128.get('output_tokens_per_sec', 1) - tp2_c128.get('output_tokens_per_sec', 1)) / tp2_c128.get('output_tokens_per_sec', 1)) * 100:.1f}% for TP1",
            "winner": "TP1",
        },
        {
            "title": "REQUEST THROUGHPUT",
            "tp1_val": f"{tp1_c128.get('request_throughput', 0):.2f} req/s",
            "tp2_val": f"{tp2_c128.get('request_throughput', 0):.2f} req/s",
            "tp1_h": min(100, (tp1_c128.get('request_throughput', 0) / 20.0) * 100),
            "tp2_h": min(100, (tp2_c128.get('request_throughput', 0) / 20.0) * 100),
            "lead": f"+{((tp1_c128.get('request_throughput', 1) - tp2_c128.get('request_throughput', 1)) / tp2_c128.get('request_throughput', 1)) * 100:.1f}% for TP1",
            "winner": "TP1",
        },
        {
            "title": "TIME TO FIRST TOKEN (P50 TTFT)",
            "tp1_val": f"{tp1_c128.get('ttft_p50_s', 0):.2f} s ({tp1_c128.get('ttft_p50_ms', 0):.0f} ms)",
            "tp2_val": f"{tp2_c128.get('ttft_p50_s', 0):.2f} s ({tp2_c128.get('ttft_p50_ms', 0):.0f} ms)",
            "tp1_h": min(100, (tp1_c128.get('ttft_p50_s', 0) / 1.2) * 100),
            "tp2_h": min(100, (tp2_c128.get('ttft_p50_s', 0) / 1.2) * 100),
            "lead": f"{((tp2_c128.get('ttft_p50_s', 1) - tp1_c128.get('ttft_p50_s', 1)) / tp2_c128.get('ttft_p50_s', 1)) * 100:.1f}% faster on TP1",
            "winner": "TP1",
        },
        {
            "title": "TIME TO FIRST TOKEN (P95 TTFT)",
            "tp1_val": f"{tp1_c128.get('ttft_p95_s', 0):.2f} s",
            "tp2_val": f"{tp2_c128.get('ttft_p95_s', 0):.2f} s",
            "tp1_h": min(100, (tp1_c128.get('ttft_p95_s', 0) / 14.0) * 100),
            "tp2_h": min(100, (tp2_c128.get('ttft_p95_s', 0) / 14.0) * 100),
            "lead": f"{((tp2_c128.get('ttft_p95_s', 1) - tp1_c128.get('ttft_p95_s', 1)) / tp2_c128.get('ttft_p95_s', 1)) * 100:.1f}% faster on TP1 (8.2x)",
            "winner": "TP1",
        },
        {
            "title": "INTER-TOKEN LATENCY (P50 ITL / TPOT)",
            "tp1_val": f"{tp1_c128.get('itl_p50_ms', 0):.2f} ms",
            "tp2_val": f"{tp2_c128.get('itl_p50_ms', 0):.2f} ms",
            "tp1_h": min(100, (tp1_c128.get('itl_p50_ms', 0) / 13.0) * 100),
            "tp2_h": min(100, (tp2_c128.get('itl_p50_ms', 0) / 13.0) * 100),
            "lead": f"{((tp1_c128.get('itl_p50_ms', 1) - tp2_c128.get('itl_p50_ms', 1)) / tp1_c128.get('itl_p50_ms', 1)) * 100:.1f}% lower on TP2",
            "winner": "TP2",
        },
        {
            "title": "END-TO-END LATENCY (P95 E2E)",
            "tp1_val": f"{tp1_c128.get('e2e_p95_s', 0):.2f} s",
            "tp2_val": f"{tp2_c128.get('e2e_p95_s', 0):.2f} s",
            "tp1_h": min(100, (tp1_c128.get('e2e_p95_s', 0) / 18.0) * 100),
            "tp2_h": min(100, (tp2_c128.get('e2e_p95_s', 0) / 18.0) * 100),
            "lead": f"{((tp2_c128.get('e2e_p95_s', 1) - tp1_c128.get('e2e_p95_s', 1)) / tp2_c128.get('e2e_p95_s', 1)) * 100:.1f}% faster on TP1",
            "winner": "TP1",
        },
    ]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SGLang Disaggregated Architecture: TP1 (4P4D) vs. TP2 (2P2D)</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  body {{ background-color: #ffffff; color: #1e293b; padding: 30px; }}
  h1 {{ font-size: 22px; color: #0f172a; margin-bottom: 6px; font-weight: 700; }}
  p.subtitle {{ font-size: 13.5px; color: #64748b; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 20px; margin-bottom: 30px; }}
  .card {{ background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
  .card-title {{ font-size: 11px; font-weight: 700; color: #64748b; letter-spacing: 1px; margin-bottom: 20px; }}
  .chart-area {{ height: 180px; display: flex; align-items: flex-end; justify-content: space-around; padding: 0 30px 10px; border-bottom: 1px solid #e2e8f0; position: relative; }}
  .bar-group {{ display: flex; flex-direction: column; align-items: center; width: 80px; height: 100%; justify-content: flex-end; position: relative; }}
  .bar {{ width: 50px; border-radius: 6px 6px 0 0; transition: all 0.3s ease; }}
  .bar.tp1 {{ background: {COLOR_TP1}; }}
  .bar.tp2 {{ background: {COLOR_TP2}; }}
  .tooltip-badge {{
    position: absolute; top: -38px; background: #f8fafc; border: 1px solid #cbd5e1;
    border-radius: 8px; padding: 4px 8px; font-size: 11px; color: #0f172a; font-weight: 600;
    white-space: nowrap; box-shadow: 0 2px 8px rgba(0,0,0,0.08); z-index: 10;
  }}
  .val-text {{ font-size: 11px; color: #64748b; margin-top: 6px; font-weight: 500; text-align: center; }}
  .x-labels {{ display: flex; justify-content: space-around; margin-top: 12px; font-size: 11px; color: #475569; font-weight: 600; }}
  .table-container {{ margin-top: 30px; overflow-x: auto; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
  table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 12.5px; }}
  th, td {{ padding: 10px 14px; border-bottom: 1px solid #e2e8f0; }}
  th {{ background: #f8fafc; color: #475569; font-weight: 600; }}
  tr:hover {{ background-color: #f8fafc; }}
  .badge-winner {{ display: inline-block; background: {WINNER_BG}; border: 1px solid {WINNER_BORDER}; color: {WINNER_TEXT}; padding: 2px 6px; border-radius: 4px; font-size: 10.5px; font-weight: 700; }}
</style>
</head>
<body>
  <h1>Qwen3.6-35B-A3B FP8 (8x NVIDIA H100) — Architecture Comparison</h1>
  <p class="subtitle">Disaggregated SGLang 8-GPU Cluster: TP1 (4P4D: 4 Prefill + 4 Decode) vs. TP2 (2P2D: 2 Prefill + 2 Decode)</p>

  <div class="grid">
"""

    for c in cards:
        html_content += f"""
    <div class="card">
      <div class="card-title">{c['title']}</div>
      <div class="chart-area">
        <div class="bar-group">
          <div class="bar tp1" style="height: {c['tp1_h']}%;"></div>
          <span class="val-text">{c['tp1_val']}</span>
        </div>
        <div class="bar-group">
          <div class="tooltip-badge">{c['lead']}</div>
          <div class="bar tp2" style="height: {c['tp2_h']}%;"></div>
          <span class="val-text">{c['tp2_val']}</span>
        </div>
      </div>
      <div class="x-labels">
        <span>TP1 (4P4D)</span>
        <span>TP2 (2P2D)</span>
      </div>
    </div>
"""

    html_content += """
  </div>

  <div class="table-container">
    <h2 style="font-size: 15px; color: #0f172a; margin-bottom: 14px; font-weight: 700;">Complete Concurrency Scaling Sweep (c1 to c128)</h2>
    <table>
      <thead>
        <tr>
          <th>Concurrency</th>
          <th>TP1 Output Tok/s</th>
          <th>TP2 Output Tok/s</th>
          <th>TP1 Req/s</th>
          <th>TP2 Req/s</th>
          <th>TP1 TTFT P50</th>
          <th>TP2 TTFT P50</th>
          <th>TP1 TTFT P95</th>
          <th>TP2 TTFT P95</th>
          <th>TP1 ITL P50</th>
          <th>TP2 ITL P50</th>
          <th>Outcome / Analysis</th>
        </tr>
      </thead>
      <tbody>
"""

    for conc in CONCURRENCIES:
        d1 = tp1_data.get(conc, {})
        d2 = tp2_data.get(conc, {})
        t1_out = d1.get("output_tokens_per_sec", 0)
        t2_out = d2.get("output_tokens_per_sec", 0)
        t1_req = d1.get("request_throughput", 0)
        t2_req = d2.get("request_throughput", 0)
        t1_ttft50 = d1.get("ttft_p50_ms", 0)
        t2_ttft50 = d2.get("ttft_p50_ms", 0)
        t1_ttft95 = d1.get("ttft_p95_ms", 0)
        t2_ttft95 = d2.get("ttft_p95_ms", 0)
        t1_itl50 = d1.get("itl_p50_ms", 0)
        t2_itl50 = d2.get("itl_p50_ms", 0)

        if conc <= 32:
            status_badge = f'<span class="badge-winner">TP2 Faster ITL ({t2_itl50:.2f}ms)</span>'
        else:
            diff = ((t1_out - t2_out) / t2_out) * 100 if t2_out > 0 else 0
            status_badge = f'<span class="badge-winner">TP1 +{diff:.1f}% Throughput</span>'

        html_content += f"""
        <tr>
          <td><strong>c{conc}</strong></td>
          <td>{t1_out:,.1f}</td>
          <td>{t2_out:,.1f}</td>
          <td>{t1_req:.2f}</td>
          <td>{t2_req:.2f}</td>
          <td>{t1_ttft50:.1f} ms</td>
          <td>{t2_ttft50:.1f} ms</td>
          <td>{t1_ttft95:.1f} ms</td>
          <td>{t2_ttft95:.1f} ms</td>
          <td>{t1_itl50:.2f} ms</td>
          <td>{t2_itl50:.2f} ms</td>
          <td>{status_badge}</td>
        </tr>
"""

    html_content += """
      </tbody>
    </table>
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
        description="Custom Plotting and Comparison for SGLang Disaggregated Serving: TP1 (4P4D) vs. TP2 (2P2D)"
    )
    parser.add_argument(
        "--tp1-dir",
        type=Path,
        default=DEFAULT_TP1_BENCHMARKS,
        help="Path to TP1 benchmarks directory containing c1..c128 subfolders",
    )
    parser.add_argument(
        "--tp2-dir",
        type=Path,
        default=DEFAULT_TP2_BENCHMARKS,
        help="Path to TP2 benchmarks directory containing c1..c128 subfolders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated plots and HTML dashboard",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Generating SGLang Disaggregated Serving Comparison: TP1 (4P4D) vs. TP2 (2P2D)")
    print("=" * 70)

    tp1_data = load_all_concurrencies(args.tp1_dir)
    tp2_data = load_all_concurrencies(args.tp2_dir)

    # 1. Generate Concurrency 128 Plots
    if 128 in tp1_data and 128 in tp2_data:
        c128_dir = args.output_dir / "c128"
        generate_c128_plots(
            tp1_c128_path=args.tp1_dir / "c128" / "profile_export_aiperf.json",
            tp2_c128_path=args.tp2_dir / "c128" / "profile_export_aiperf.json",
            output_dir=c128_dir,
        )

    # 2. Generate Scaling Sweep Plots (c1 to c128)
    scaling_dir = args.output_dir / "scaling"
    generate_scaling_plots(
        tp1_data=tp1_data,
        tp2_data=tp2_data,
        output_dir=scaling_dir,
    )

    # 3. Generate Standalone HTML Dashboard
    generate_standalone_html(
        tp1_data=tp1_data,
        tp2_data=tp2_data,
        output_html_file=args.output_dir / "index.html",
    )

    print("\nComparison plots and dashboard generation complete!")


if __name__ == "__main__":
    main()

