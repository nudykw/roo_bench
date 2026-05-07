"""Core benchmark execution orchestration."""

from api.base_client import BaseApiClient
from system.restart_manager import restart_ollama, RestartMethod
from benchmark.results import calculate_statistics
from i18n import get_text


class BenchmarkRunner:
    """Orchestrates benchmark execution for models."""

    def __init__(self, ollama_client: BaseApiClient, context_sizes: list, num_runs: int = 3,
                 restart_method: str = 'manual', no_restart: bool = False, disable_thinking: bool = True):
        """Initialize benchmark runner.

        Args:
            ollama_client: BaseApiClient instance (LocalApiClient or RemoteApiClient)
            context_sizes: List of context sizes to test
            num_runs: Number of runs per configuration
            restart_method: Ollama restart method
            no_restart: If True, restart is not performed
            disable_thinking: If True, disables thinking mode to prevent reasoning loops
        """
        self.ollama_client = ollama_client
        self.context_sizes = context_sizes
        self.num_runs = num_runs
        self.no_restart = no_restart
        self.disable_thinking = disable_thinking
        self.ssh_client = ollama_client.ssh_client  # Get SSH client from the API client
        
        # Map string method to RestartMethod enum
        method_map = {
            'systemctl': RestartMethod.SYSTEMCTL,
            'docker': RestartMethod.DOCKER,
            'kill_start': RestartMethod.KILL_START,
            'manual': RestartMethod.MANUAL,
            'ssh': RestartMethod.SSH
        }
        self.restart_method = method_map.get(restart_method, RestartMethod.MANUAL)

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

        print("\n============================================================\n");
        print(get_text("testing_model", model_name=model_name))
        # Format size_gb: if it's a number, format to 1 decimal; if "N/A", keep as string
        size_gb_val = model['size_gb']
        if isinstance(size_gb_val, (int, float)):
            size_gb_str = f"{size_gb_val:.1f}"
        else:
            size_gb_str = str(size_gb_val)
        print(get_text("model_size", size_gb=size_gb_str, max_ctx_str=max_ctx_str))

        results = []
        valid_contexts = self.filter_contexts(max_ctx)

        if not valid_contexts:
            print(get_text("skipping_no_contexts"))
            return model_name, results, None

        for ctx in valid_contexts:
            restart_ollama(
                self.restart_method,
                self.no_restart,
                ssh_client=self.ssh_client
            )

            # Display model's default num_ctx (informational only)
            # Actual n_ctx will be verified during generation via /api/ps
            try:
                current_num_ctx = self.ollama_client.get_current_num_ctx(model_name)
                print(get_text("current_num_ctx", num_ctx=current_num_ctx))
                if current_num_ctx != ctx:
                    print(get_text("ctx_info_only", actual=current_num_ctx, expected=ctx))
            except Exception as e:
                print(f"   ⚠️  Error getting model settings: {e}")

            print(get_text("warming_up", ctx=ctx))
            avg_tps, vram, tps_list, error_msg = self.ollama_client.run_generation(
                model_name, ctx, self.num_runs, self.disable_thinking
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
            
            # Check n_ctx AFTER generation (to avoid blocking Ctrl+C)
            try:
                actual_ctx = self.ollama_client.get_actual_num_ctx(model_name)
                if actual_ctx > 0:
                    print(get_text("actual_n_ctx_after_gen", ctx=actual_ctx, expected=ctx))
                    if actual_ctx == ctx:
                        print(get_text("ctx_verified_after_gen", actual=actual_ctx, expected=ctx))
                    else:
                        print(get_text("ctx_mismatch_after_gen", actual=actual_ctx, expected=ctx))
            except Exception:
                pass

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

        # Unload model from VRAM after all context tests are done
        print(f"\n   🧹 Unloading model '{model_name}' from VRAM...")
        try:
            self.ollama_client.unload_model(model_name)
        except Exception as e:
            print(f"   ⚠️  Warning: Could not unload model: {e}")

        return model_name, results, None
