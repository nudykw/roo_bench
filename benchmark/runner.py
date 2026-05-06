"""Core benchmark execution orchestration."""

from api.ollama_client import OllamaClient
from system.restart_manager import restart_ollama, RestartMethod
from benchmark.results import calculate_statistics
from i18n import get_text


class BenchmarkRunner:
    """Orchestrates benchmark execution for models."""

    def __init__(self, ollama_client: OllamaClient, context_sizes: list, num_runs: int = 3,
                 restart_method: str = 'systemctl', no_restart: bool = False,
                 ssh_host: str = None, ssh_user: str = None,
                 ssh_port: int = 22, ssh_key: str = None):
        """Initialize benchmark runner.

        Args:
            ollama_client: OllamaClient instance
            context_sizes: List of context sizes to test
            num_runs: Number of runs per configuration
            restart_method: Ollama restart method
            no_restart: If True, restart is not performed
            ssh_host: SSH host (for SSH method)
            ssh_user: SSH user (for SSH method)
            ssh_port: SSH port (for SSH method)
            ssh_key: Path to SSH private key (for SSH method)
        """
        self.ollama_client = ollama_client
        self.context_sizes = context_sizes
        self.num_runs = num_runs
        self.no_restart = no_restart
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.ssh_port = ssh_port
        self.ssh_key = ssh_key
        
        # Map string method to RestartMethod enum
        method_map = {
            'systemctl': RestartMethod.SYSTEMCTL,
            'docker': RestartMethod.DOCKER,
            'kill_start': RestartMethod.KILL_START,
            'manual': RestartMethod.MANUAL,
            'ssh': RestartMethod.SSH
        }
        self.restart_method = method_map.get(restart_method, RestartMethod.SYSTEMCTL)

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
            restart_ollama(
                self.restart_method,
                self.no_restart,
                ssh_host=self.ssh_host,
                ssh_user=self.ssh_user,
                ssh_port=self.ssh_port,
                ssh_key=self.ssh_key
            )

            # Display current num_ctx after restart
            try:
                current_num_ctx = self.ollama_client.get_current_num_ctx(model_name)
                print(get_text("current_num_ctx", num_ctx=current_num_ctx))
            except Exception:
                pass

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
