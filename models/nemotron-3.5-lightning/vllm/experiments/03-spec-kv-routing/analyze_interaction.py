#!/usr/bin/env python3
"""Compute the 2x2 speculative-decoding x KV-routing interaction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_METRIC = "output_token_throughput"


def metric_value(path: Path, metric: str) -> float:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    value = payload
    for segment in metric.split("."):
        if not isinstance(value, dict) or segment not in value:
            raise KeyError(f"{path}: missing metric path {metric!r}")
        value = value[segment]
    if isinstance(value, dict):
        value = value.get("avg")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{path}: {metric!r} did not resolve to a number")
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate (T_D - T_C - T_B + T_A) and normalize it by T_A. "
            "Each input is an AIPerf profile_export_aiperf.json file."
        )
    )
    for cell in "abcd":
        parser.add_argument(f"--{cell}", type=Path, required=True)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    values = {
        cell.upper(): metric_value(getattr(args, cell), args.metric)
        for cell in "abcd"
    }
    baseline = values["A"]
    if baseline == 0:
        raise ZeroDivisionError("cell A baseline metric is zero")

    interaction = values["D"] - values["C"] - values["B"] + baseline
    result = {
        "metric": args.metric,
        "cells": values,
        "interaction": interaction,
        "normalized_interaction": interaction / baseline,
        "normalized_interaction_percent": 100.0 * interaction / baseline,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
