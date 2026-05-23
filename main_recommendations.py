"""Expert-guided recommendation system for roo_bench."""

from benchmark.result import BenchmarkMetrics, BenchmarkResult
from i18n import get_text


def _mode_from_metric(metric: BenchmarkMetrics) -> str:
    """Return recommendation mode for a metric."""
    if metric.mode in ('architect', 'code', 'debug'):
        return metric.mode
    if metric.ctx >= 65536:
        return 'architect'
    if metric.ctx >= 16384:
        return 'code'
    return 'debug'


def _recommendation_score(mode: str, metric: BenchmarkMetrics) -> float:
    """Score a metric for mode-specific recommendations."""
    expert = metric.expert_score if metric.expert_score is not None else 50.0
    ctx_score = min(metric.ctx / 131072, 2.0)
    speed_score = min(metric.avg_tps / 50, 2.0)

    if mode == 'architect':
        return expert * 10 + ctx_score * 25 + speed_score * 5
    if mode == 'code':
        return expert * 10 + speed_score * 30 + ctx_score * 8
    return expert * 10 + speed_score * 18 + ctx_score * 18


def _build_mode_recommendations(
    all_results: list[BenchmarkResult],
    test_models: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build top model recommendations per Roo mode."""
    from typing import Any

    model_lookup: dict[str, dict[str, Any]] = {m['name']: m for m in test_models}
    best_by_mode_and_model: dict[str, dict[str, BenchmarkMetrics]] = {
        'architect': {},
        'code': {},
        'debug': {},
    }

    for result in all_results:
        model_name = result.model_name
        for metric in result.results:
            mode = _mode_from_metric(metric)
            current = best_by_mode_and_model[mode].get(model_name)
            if current is None or _recommendation_score(mode, metric) > _recommendation_score(mode, current):
                best_by_mode_and_model[mode][model_name] = metric

    recommendations: dict[str, list[dict[str, Any]]] = {}
    for mode, model_metrics in best_by_mode_and_model.items():
        ranked: list[dict[str, Any]] = []
        for model_name, metric in model_metrics.items():
            model_data = model_lookup.get(model_name, {})
            ranked.append({
                'model_name': model_name,
                'metric': metric,
                'model_data': model_data,
                'score': _recommendation_score(mode, metric),
            })
        ranked.sort(key=lambda item: item['score'], reverse=True)
        recommendations[mode] = ranked[:3]

    return recommendations


def _format_ctx(ctx: int) -> str:
    """Format context size for display."""
    return f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)


def _recommendation_reason(mode: str, metric: BenchmarkMetrics) -> str:
    """Explain why a metric is recommended for a mode."""
    score_text = (
        f"expert score {int(metric.expert_score)}/100"
        if metric.expert_score is not None
        else "no expert score, using performance fallback"
    )
    ctx_text = f"context {_format_ctx(metric.ctx)}"
    speed_text = f"{metric.avg_tps:.1f} tok/s"

    if mode == 'architect':
        return (
            f"{score_text}; strongest fit favors large {ctx_text} "
            f"for project-wide analysis, with {speed_text} generation."
        )
    if mode == 'code':
        return (
            f"{score_text}; best coding fit balances response quality "
            f"with high generation speed ({speed_text}) at {ctx_text}."
        )
    return (
        f"{score_text}; debug fit balances enough {ctx_text} "
        f"for traces/logs with practical speed ({speed_text})."
    )


def print_expert_recommendations(
    all_results: list[BenchmarkResult],
    test_models: list[dict[str, Any]],
) -> None:
    """Print top 3 expert-guided recommendations for Architect, Coder, and Debug."""
    recommendations = _build_mode_recommendations(all_results, test_models)
    mode_titles = [
        ('architect', get_text('architect_mode')),
        ('code', get_text('code_mode')),
        ('debug', get_text('debug_mode')),
    ]

    print("\n" + "=" * 60)
    print(get_text("recommendations_header"))
    print("=" * 60)

    for mode, title in mode_titles:
        print(f"\n  {title}")
        recs = recommendations.get(mode, [])
        if not recs:
            print("    No completed tests for this mode.")
            continue

        for rank, rec in enumerate(recs, 1):
            metric = rec['metric']
            model_data = rec['model_data']
            model_name = rec['model_name']
            params = model_data.get('params', 'N/A')
            quant = model_data.get('quant', 'N/A')
            size_gb = model_data.get('size_gb', 'N/A')
            score = (
                f"{int(metric.expert_score)}/100"
                if metric.expert_score is not None
                else "N/A"
            )
            prefix = "  ★" if rank == 1 else "   "

            print(f"{prefix} {rank}. {model_name} ({params}, {quant}, {size_gb} GB)")
            print(
                f"      Recommended: ctx={_format_ctx(metric.ctx)}, "
                f"temperature={metric.temperature:.2f}, "
                f"speed={metric.avg_tps:.1f} tok/s, expert_score={score}"
            )
            print(f"      Why: {_recommendation_reason(mode, metric)}")
