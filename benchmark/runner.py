"""Core benchmark execution orchestration."""

from api.ollama_client import OllamaClient
from api.capabilities_fetcher import CapabilitiesFetcher
from system.restart_manager import restart_ollama
from benchmark.results import calculate_statistics
from i18n import get_text


class BenchmarkRunner:
    """Orchestrates benchmark execution for models."""

    def __init__(self, ollama_client: OllamaClient, context_sizes: list, num_runs: int = 3):
        """Initialize benchmark runner.

        Args:
            ollama_client: OllamaClient instance
            context_sizes: List of context sizes to test
            num_runs: Number of runs per configuration
        """
        self.ollama_client = ollama_client
        self.context_sizes = context_sizes
        self.num_runs = num_runs
        self.capabilities_fetcher = CapabilitiesFetcher()

    def filter_contexts(self, max_ctx: int) -> list:
        """Filter context sizes based on model's maximum context.

        Args:
            max_ctx: Model's maximum context size

        Returns:
            list: Filtered list of valid context sizes
        """
        # Filter contexts: take only those that are LESS THAN OR EQUAL to maximum
        valid_contexts = [c for c in self.context_sizes if c <= max_ctx]

        # If the model supports some "non-standard" max context (e.g. 128000),
        # which is not in our list, but it's less than 256K, we can add it for testing
        if max_ctx not in valid_contexts and 0 < max_ctx <= 262144:
            valid_contexts.append(max_ctx)
            valid_contexts.sort()

        return valid_contexts

    def run_for_model(self, model: dict) -> tuple:
        """Run benchmarks for a single model across valid contexts.

        Args:
            model: Model dictionary with name, max_ctx, etc.

        Returns:
            tuple: (model_name, results, error)
                model_name: Model name
                results: List of result dictionaries
                error: Error message if any
        """
        model_name = model["name"]
        max_ctx = model["max_ctx"]
        max_ctx_str = f"{max_ctx // 1024}K" if max_ctx >= 1024 else str(max_ctx)

        print(get_text("testing_model", model_name=model_name))
        print(get_text("model_size", size_gb=model['size_gb'], max_ctx_str=max_ctx_str))

        results = []
        valid_contexts = self.filter_contexts(max_ctx)

        if not valid_contexts:
            print(get_text("skipping_no_contexts"))
            return model_name, results, None

        for ctx in valid_contexts:
            restart_ollama(None, getattr(self, 'no_restart', False))

            print(get_text("warming_up", ctx=ctx))
            avg_tps, vram, tps_list, error_msg = self.ollama_client.run_generation(
                model_name, ctx, self.num_runs
            )

            if error_msg:
                print(get_text("benchmark_failed", error_msg=error_msg))
                print(get_text("stopping_tests", model_name=model_name))
                break

            # Show results of each run
            print(get_text("benchmark_runs_header"))
            for run in tps_list:
                run_num = run['run']
                tps = run['tps']
                vram_run = run['vram']
                if vram_run:
                    vram_str = f"{vram_run / 1024 / 1024:.1f} MiB"
                else:
                    vram_str = "N/A"
                print(f"   Run {run_num}: {tps:.2f} TPS (VRAM: {vram_str})")

            # Show summary
            print(get_text("benchmark_summary"))
            print(f"   Average: {avg_tps:.2f} TPS")
            print(f"   Min: {min(r['tps'] for r in tps_list):.2f} TPS")
            print(f"   Max: {max(r['tps'] for r in tps_list):.2f} TPS")

            # Calculate std dev
            mean = avg_tps
            variance = sum((r['tps'] - mean) ** 2 for r in tps_list) / len(tps_list)
            std_dev = variance ** 0.5
            print(f"   Std Dev: {std_dev:.2f} TPS")

            # Save averaged results
            results.append({
                "ctx": ctx,
                "avg_tps": avg_tps,
                "min_tps": min(r['tps'] for r in tps_list),
                "max_tps": max(r['tps'] for r in tps_list),
                "std_dev": std_dev,
                "vram": vram
            })

        return model_name, results, None
