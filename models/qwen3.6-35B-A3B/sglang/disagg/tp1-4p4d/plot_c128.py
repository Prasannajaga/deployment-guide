#!/usr/bin/env python3
"""
Custom Plotting Script for Concurrency 128:
SGLang Disaggregated TP1 (4 Prefill + 4 Decode) vs. TP2 (2 Prefill + 2 Decode) on 8x NVIDIA H100 GPUs.

Extracts metrics from AIPerf profile_export_aiperf.json files:
- Output Token Throughput (output tok/s)
- Request Throughput (requests/s)
- Time to First Token (TTFT - p50, p95, p99)
- Time Per Output Token / Inter-Token Latency (ITL / TPOT - p50, p95, p99)
- End-to-End Request Latency (E2E - p50, p95, p99)

Generates:
1. Individual high-resolution PNG cards for Concurrency 128
2. Combined 6-panel comparison dashboard
3. Standalone interactive HTML dashboard
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
    import matplotlib.patches as mpatches
except ImportError:
    plt = None


# Default Paths relative to the cluster authoring repo
DEFAULT_TP1_JSON = Path(
    "/data/inference/cluster/benchmarks/qwen3.6-35B-A3B/tp1-4p4d/c128/profile_export_aiperf.json"

DEFAULT_TP2_JSON = Path(
    "/data/inference/cluster/benchmarks/qwen3.6-35B-A3B/tp2-2p2d/c128/profile_export_aiperf.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/inference/cluster/benchmarks/qwen3.6-35B-A3B/tp1-4p4d/plots/c128"
)

BG_COLOR = "#ffffff"         # Default white background
CARD_BG = "#ffffff"          # Clean white container background
GRID_COLOR = "#e2e8f0"       # Light subtle gray grid line
BORDER_COLOR = "#cbd5e1"     # Card border
TEXT_MUTED = "#64748b"       # Muted subtitle/header text
TEXT_MAIN = "#1e293b"        # Main dark axis text
TEXT_DARK = "#0f172a"        # Dark text for highlights

# Bar Colors (High Contrast on White Background)
COLOR_TP1 = "#475569"        # Solid Slate Gray for TP1 (4P4D: 4 Prefill / 4 Decode TP1)
COLOR_TP2 = "#ea580c"        # Vibrant Orange for TP2 (2P2D: 2 Prefill / 2 Decode TP2)

# Winner Badges
WINNER_BG = "#ecfdf5"        # Emerald green light background
WINNER_BORDER = "#a7f3d0"    # Emerald green border
WINNER_TEXT = "#047857"      # Dark emerald green text


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
    """Draw a single high-fidelity card matching modern clean UI."""
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

    # Styling Limits and Spacing
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

    # Difference calculation
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
    tp1_path: Path,
    tp2_path: Path,
    output_dir: Path,
):
    """Generate individual and dashboard comparison plots for Concurrency 128."""
    if plt is None:
        print("Error: matplotlib is required. Please install matplotlib.", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading TP1 C128 metrics from: {tp1_path}")
    tp1 = load_metrics(tp1_path)

    print(f"Loading TP2 C128 metrics from: {tp2_path}")
    tp2 = load_metrics(tp2_path)

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

    # 1. Generate Individual PNGs
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

    # 2. Generate Combined 6-Panel Dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), facecolor=BG_COLOR, dpi=200)
    fig.suptitle(
        "Qwen3.6-35B-A3B FP8 (8x H100) — Concurrency 128 Performance Breakdown\nTP1 (4 Prefill + 4 Decode) vs. TP2 (2 Prefill + 2 Decode)",
        color=TEXT_DARK,
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )

    patch_tp1 = mpatches.Patch(color=COLOR_TP1, label="TP1 (4P4D: 4 Prefill + 4 Decode Pods, TP=1)")
    patch_tp2 = mpatches.Patch(color=COLOR_TP2, label="TP2 (2P2D: 2 Prefill + 2 Decode Pods, TP=2)")

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


def main():
    parser = argparse.ArgumentParser(
        description="Custom Plotting for Concurrency 128 AIPerf Benchmark Exports (TP1 vs TP2)"
    )
    parser.add_argument(
        "--tp1",
        type=Path,
        default=DEFAULT_TP1_JSON,
        help="Path to TP1 profile_export_aiperf.json",
    )
    parser.add_argument(
        "--tp2",
        type=Path,
        default=DEFAULT_TP2_JSON,
        help="Path to TP2 profile_export_aiperf.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated plots",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Generating C128 Comparison Plots (TP1-4P4D vs. TP2-2P2D)")
    print("=" * 60)

    generate_c128_plots(
        tp1_path=args.tp1,
        tp2_path=args.tp2,
        output_dir=args.output_dir,
    )

    print("\nC128 Plot generation complete!")


if __name__ == "__main__":
    main()

