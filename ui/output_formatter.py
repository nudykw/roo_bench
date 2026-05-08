"""Console output formatting and display utilities."""

from typing import List
from i18n import get_text
from benchmark.result import BenchmarkResult


def _format_moe_display(moe_data) -> str:
    """Format MoE status for display.
    
    Args:
        moe_data: MoE data from model cache (dict, bool, or None)
        
    Returns:
        str: Display string with icon
    """
    if moe_data is None:
        return "❓"
    elif moe_data is False:
        return "⬛"
    elif isinstance(moe_data, dict):
        # It's MoE
        num_experts = moe_data.get('num_experts', 'N/A')
        if isinstance(num_experts, int) and num_experts > 0:
            return f"🟡({num_experts})"
        return "🟡"
    elif moe_data is True:
        return "🟡"
    return "❓"


def print_model_list(models: list):
    """Print formatted model list to console.

    Args:
        models: List of model dictionaries
    """
    print(get_text("available_models"))
    for i, m in enumerate(models):
        max_ctx_str = f"{m['max_ctx'] // 1024}K" if m['max_ctx'] >= 1024 else str(m['max_ctx'])
        
        # Get MoE data from model metadata (if available)
        moe_data = m.get('moe', None)
        moe_str = _format_moe_display(moe_data)

        # Ensure size_gb is a float for format string {size_gb:4.1f}
        size_gb = m.get('size_gb', 0)
        if not isinstance(size_gb, (int, float)):
            size_gb = 0.0

        print(get_text("model_list_header",
            index=i,
            name=m['name'],
            params=m['params'],
            size_gb=size_gb,
            max_ctx_str=max_ctx_str,
            moe_str=moe_str,
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


def print_results_table(results: List[BenchmarkResult]):
    """Print formatted results table.

    Args:
        results: List of BenchmarkResult objects
    """
    print("\n" + "="*60)
    print(get_text("recommendations_header"))
    print("="*60)

    for result in results:
        model_name = result.model_name
        if not result.results:
            print(get_text("no_successful_runs", model_name=model_name))
            continue

        # Output results for each model
        print("\n" + "="*60)
        print(get_text("results_header", model_name=model_name))
        print("="*60)

        for run in result.results:
            ctx = run.ctx
            ctx_str = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
            avg_tps = run.avg_tps
            min_tps = run.min_tps
            max_tps = run.max_tps
            std_dev = run.std_dev
            vram = run.vram
            vram_str = f"{vram / 1024 / 1024:.1f} MiB" if vram else "N/A"

            print(get_text("result_row",
                ctx=ctx_str,
                avg_tps=avg_tps,
                min_tps=min_tps,
                max_tps=max_tps,
                std_dev=std_dev,
                vram=vram_str))


def print_recommendations(results: List[BenchmarkResult]):
    """Print recommendations output.

    Args:
        results: List of BenchmarkResult objects
    """
    print("\n" + "="*60)
    print(get_text("recommendations_header"))
    print("="*60)

    for result in results:
        model_name = result.model_name
        if not result.results:
            continue

        print(f"\nModel: {model_name}")
        for run in result.results:
            ctx = run.ctx
            ctx_str = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
            print(f"  Context: {ctx_str} | Avg TPS: {run.avg_tps:.2f}")


def format_tokens_info(prompt_tokens: int, response_tokens: int) -> str:
    """Format token information for console output.
    
    Args:
        prompt_tokens: Number of tokens in the prompt
        response_tokens: Number of tokens in the response
        
    Returns:
        str: Formatted token information string
    """
    return f"| 📊 Tokens: {prompt_tokens}/{response_tokens}"


def update_tokens_display(prompt_tokens: int, response_tokens: int, estimated_response_tokens: int = 0, response_len: int = 0, indent: str = "         ", is_done: bool = False) -> None:
    """Update token display in real-time during streaming.
    
    This function prints/updates the token count on a single line.
    During streaming (is_done=False), it shows estimated progress based on response length.
    When done (is_done=True), it shows real token counts from the API.
    
    Args:
        prompt_tokens: Current number of prompt tokens evaluated (0 during streaming)
        response_tokens: Current number of response tokens from API (0 during streaming)
        estimated_response_tokens: Estimated response tokens based on text length
        response_len: Raw response text length in characters
        indent: Leading whitespace for alignment
        is_done: If True, this is the final chunk with real token counts
    """
    if is_done:
        # Show real token counts when done
        print(f"{indent}📊 Tokens: {prompt_tokens}/{response_tokens}", end="\r", flush=True)
    elif estimated_response_tokens > 0 or response_len > 0:
        # Show estimated tokens during streaming with response length
        print(f"{indent}📊 Tokens: ~{prompt_tokens}/{estimated_response_tokens} (len={response_len})", end="\r", flush=True)
