"""Console output formatting and display utilities."""

from i18n import get_text


def print_model_list(models: list):
    """Print formatted model list to console.

    Args:
        models: List of model dictionaries
    """
    print(get_text("available_models"))
    for i, m in enumerate(models):
        max_ctx_str = f"{m['max_ctx'] // 1024}K" if m['max_ctx'] >= 1024 else str(m['max_ctx'])

        print(get_text("model_list_header",
            index=i,
            name=m['name'],
            params=m['params'],
            size_gb=m['size_gb'],
            max_ctx_str=max_ctx_str,
            vision=m.get('vision', '❓'),
            tools=m.get('tools', '❓'),
            thinking=m.get('thinking', '❌')))


def print_benchmark_progress(model_name: str, context_size: int, tps: float, vram=None):
    """Print benchmark progress information.

    Args:
        model_name: Model name
        context_size: Context size
        tps: Tokens per second
        vram: VRAM usage in bytes (optional)
    """
    ctx_str = f"{context_size // 1024}K" if context_size >= 1024 else str(context_size)
    vram_str = f"{vram / 1024 / 1024:.1f} MiB" if vram else "N/A"
    print(f"  Context: {ctx_str} | TPS: {tps:.2f} | VRAM: {vram_str}")


def print_results_table(results: dict):
    """Print formatted results table.

    Args:
        results: Dictionary of results per model
    """
    print("\n" + "="*60)
    print(get_text("recommendations_header"))
    print("="*60)

    for model_name, runs in results.items():
        if not runs:
            print(get_text("no_successful_runs", model_name=model_name))
            continue

        # Output results for each model
        print("\n" + "="*60)
        print(get_text("results_header", model_name=model_name))
        print("="*60)

        for run in runs:
            ctx = run['ctx']
            ctx_str = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
            avg_tps = run['avg_tps']
            min_tps = run['min_tps']
            max_tps = run['max_tps']
            std_dev = run['std_dev']
            vram = run['vram']
            vram_str = f"{vram / 1024 / 1024:.1f} MiB" if vram else "N/A"

            print(get_text("result_row",
                ctx=ctx_str,
                avg_tps=avg_tps,
                min_tps=min_tps,
                max_tps=max_tps,
                std_dev=std_dev,
                vram=vram_str))


def print_recommendations(results: dict):
    """Print recommendations output.

    Args:
        results: Dictionary of results per model
    """
    print("\n" + "="*60)
    print(get_text("recommendations_header"))
    print("="*60)

    for model_name, runs in results.items():
        if not runs:
            continue

        print(f"\nModel: {model_name}")
        for run in runs:
            ctx = run['ctx']
            ctx_str = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
            print(f"  Context: {ctx_str} | Avg TPS: {run['avg_tps']:.2f}")
