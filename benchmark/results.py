"""Results processing and statistics calculation."""

import math
from typing import Any


def calculate_statistics(tps_list: list[dict[str, Any]]) -> dict[str, float]:
    """Calculate statistics from a list of TPS results.

    Args:
        tps_list: List of result dictionaries with 'tps' key

    Returns:
        dict: Dictionary with avg_tps, min_tps, max_tps, std_dev
    """
    if not tps_list:
        return {
            "avg_tps": 0.0,
            "min_tps": 0.0,
            "max_tps": 0.0,
            "std_dev": 0.0
        }

    avg_tps = sum(r['tps'] for r in tps_list) / len(tps_list)
    min_tps = min(r['tps'] for r in tps_list)
    max_tps = max(r['tps'] for r in tps_list)

    if len(tps_list) > 1:
        mean = avg_tps
        variance = sum((r['tps'] - mean) ** 2 for r in tps_list) / len(tps_list)
        std_dev = math.sqrt(variance)
    else:
        std_dev = 0.0

    return {
        "avg_tps": avg_tps,
        "min_tps": min_tps,
        "max_tps": max_tps,
        "std_dev": std_dev
    }


def format_result_row(ctx: int, stats: dict[str, float], vram: int | None) -> str:
    """Format a single result row for display.

    Args:
        ctx: Context size
        stats: Statistics dictionary from calculate_statistics()
        vram: VRAM usage in bytes

    Returns:
        str: Formatted result row string
    """
    ctx_str = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
    vram_str = f"{vram / 1024 / 1024:.1f} MiB" if vram else "N/A"

    return (
        f"  Context: {ctx_str} | "
        f"Avg: {stats['avg_tps']:.2f} TPS | "
        f"Min: {stats['min_tps']:.2f} TPS | "
        f"Max: {stats['max_tps']:.2f} TPS | "
        f"StdDev: {stats['std_dev']:.2f} TPS | "
        f"VRAM: {vram_str}"
    )


def format_recommendations(results: dict[str, list[dict[str, Any]]], min_tps: float = 0) -> list[dict[str, Any]]:
    """Generate top 3 recommendations based on results.

    Args:
        results: Dictionary of results per model
        min_tps: Minimum TPS threshold for recommendations

    Returns:
        list: List of recommended result entries
    """
    all_runs = []
    for model_name, runs in results.items():
        for run in runs:
            run_with_model = dict(run)
            run_with_model['model_name'] = model_name
            all_runs.append(run_with_model)

    valid = [r for r in all_runs if r.get('avg_tps', 0) >= min_tps]
    if not valid:
        valid = [r for r in all_runs if r.get('avg_tps', 0) > 0]

    valid.sort(key=lambda x: (x.get('ctx', 0), x.get('avg_tps', 0)), reverse=True)
    return valid[:3]
