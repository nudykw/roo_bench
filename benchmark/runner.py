"""Core benchmark execution orchestration."""

import logging
from typing import Any

from api.base_client import BaseApiClient
from benchmark.expert_evaluator_types import ExpertEvaluationEntry
from benchmark.result import BenchmarkResult, ModelInfo
from system.gpu_monitor import get_vram_stats
from system.restart_manager import RestartMethod

# Setup logging
logger = logging.getLogger('roo_bench.benchmark')

# Default temperature values for testing
DEFAULT_TEMPERATURES = [0.0, 0.66, 1.0]


# Re-export for backward compatibility
from benchmark.runner_chains import run_all_chains, run_chain  # noqa: E402, F401
from benchmark.runner_independent import (  # noqa: E402, F401
    run_all_independent_prompts,
    run_independent_prompts,
)
from benchmark.runner_model import run_for_model  # noqa: E402, F401


class BenchmarkRunner:
    """Orchestrates benchmark execution for models."""

    def __init__(self, ollama_client: BaseApiClient, context_sizes: list, num_runs: int = 3,
                 restart_method: str = 'manual', no_restart: bool = False, disable_thinking: bool = True,
                 prompt_loader=None, temperature_test_values: list = None, expert_evaluator=None,
                 num_predict: int = 12000, independent_top: int | None = None,
                 chunks_top: int | None = None, prompts_top: int | None = None,
                 user_context_sizes: list[int] | None = None):
        """Initialize benchmark runner.

        Args:
            ollama_client: BaseApiClient instance (LocalApiClient or RemoteApiClient)
            context_sizes: List of default context sizes to test
            num_runs: Number of runs per configuration
            restart_method: Ollama restart method
            no_restart: If True, restart is not performed
            disable_thinking: If True, disables thinking mode to prevent reasoning loops
            prompt_loader: PromptLoader instance for loading prompts
            temperature_test_values: List of temperature values for testing (default: None)
            expert_evaluator: Optional ExpertEvaluator instance for response quality assessment
            num_predict: Maximum number of tokens to generate (default: 12000, use -1 for unlimited)
            user_context_sizes: User-specified context sizes from --context-sizes CLI arg (None if not specified)
        """
        self.ollama_client = ollama_client
        self.context_sizes = context_sizes
        self.user_context_sizes = user_context_sizes
        self.num_runs = num_runs
        self.no_restart = no_restart
        self.disable_thinking = disable_thinking
        self.ssh_client = ollama_client.ssh_client  # Get SSH client from the API client
        self.num_predict = num_predict  # Store num_predict for use in run_generation calls

        # Map string method to RestartMethod enum
        method_map = {
            'systemctl': RestartMethod.SYSTEMCTL,
            'docker': RestartMethod.DOCKER,
            'kill_start': RestartMethod.KILL_START,
            'manual': RestartMethod.MANUAL,
            'ssh': RestartMethod.SSH
        }
        self.restart_method = method_map.get(restart_method, RestartMethod.MANUAL)

        self.prompt_loader = prompt_loader
        self.temperature_test_values = temperature_test_values or DEFAULT_TEMPERATURES
        self.expert_evaluator = expert_evaluator
        self._response_store: list[ExpertEvaluationEntry] = []
        self.independent_top = independent_top  # Limit prompts per mode for independent tests
        self.chunks_top = chunks_top  # Limit number of chains
        self.prompts_top = prompts_top  # Universal limit - overrides independent_top and chunks_top

        # Priority: prompts_top > independent_top / chunks_top
        if prompts_top is not None and prompts_top > 0:
            self.effective_top = prompts_top
            if independent_top is not None or chunks_top is not None:
                logger.warning(
                    "Both --prompts-top and --independent-top/--chunks-top specified. "
                    "Using --prompts-top for all filtering."
                )
        else:
            self.effective_top = independent_top

    def get_test_params(self, model_names: list, expert_model: str | None = None) -> dict:
        """Get test parameters as a dictionary.

        Args:
            model_names: List of model names being tested
            expert_model: Expert model name (or None)

        Returns:
            dict: Test parameters including context sizes, num runs, etc.
        """
        # Use user_context_sizes if specified, otherwise fall back to default context_sizes
        effective_context_sizes = self.user_context_sizes if self.user_context_sizes is not None else self.context_sizes
        return {
            "context_sizes": effective_context_sizes,
            "num_runs": self.num_runs,
            "num_predict": self.num_predict,
            "temperature_test_values": self.temperature_test_values,
            "used_prompt_ids": self.get_used_prompt_ids(),
            "used_chain_ids": self.get_used_chain_ids(),
            "independent_top": self.independent_top,
            "chunks_top": self.chunks_top,
            "prompts_top": self.prompts_top,
            "effective_top": self.effective_top,
            "disable_thinking": self.disable_thinking,
            "models": model_names,
            "expert_model": expert_model
        }

    def get_used_independent_prompts(self) -> list[dict[str, Any]]:
        """Return independent prompts that will be used by all-independent mode."""
        if not self.prompt_loader:
            return []

        all_prompts = self.prompt_loader.get_all_independent_prompts_ordered()
        if self.effective_top is None or self.effective_top <= 0:
            return all_prompts

        prompts_by_mode: dict[str, list[dict[str, Any]]] = {}
        for prompt in all_prompts:
            prompts_by_mode.setdefault(prompt.get('mode'), []).append(prompt)

        filtered_prompts: list[dict[str, Any]] = []
        for mode in ['architect', 'code', 'debug']:
            filtered_prompts.extend(prompts_by_mode.get(mode, [])[:self.effective_top])
        return filtered_prompts

    def get_used_prompt_ids(self) -> list[str]:
        """Return independent prompt IDs used by all-independent mode."""
        return [p.get('id') for p in self.get_used_independent_prompts() if p.get('id')]

    def get_used_chain_ids(self) -> list[str]:
        """Return all configured chain IDs."""
        if not self.prompt_loader:
            return []
        return [c.get('id') for c in self.prompt_loader.get_chains() if c.get('id')]

    def run_expert_evaluation(self) -> None:
        """Run expert evaluation on all stored responses.

        This is called AFTER all model testing is complete.
        Scores are assigned directly via entry.metrics_ref.expert_score in evaluate_batch().
        No return value needed — BenchmarkMetrics objects are mutated in place.
        """
        logger.info(
            "[Expert] run_expert_evaluation: expert_evaluator=%s _response_store size=%d",
            "set" if self.expert_evaluator else "NONE",
            len(self._response_store)
        )
        if not self.expert_evaluator or not self._response_store:
            logger.warning("[Expert] Skipping evaluation: evaluator=%s store_empty=%s",
                          self.expert_evaluator is None, len(self._response_store) == 0)
            return

        print("\n" + "=" * 60)
        print("Starting expert evaluation...")
        print(f"Total responses to evaluate: {len(self._response_store)}")
        print("=" * 60)

        model_groups: dict[str, list[ExpertEvaluationEntry]] = {}
        for entry in self._response_store:
            if entry.model_name not in model_groups:
                model_groups[entry.model_name] = []
            model_groups[entry.model_name].append(entry)

        for model_name, entries in model_groups.items():
            print(f"\nEvaluating responses for: {model_name}")
            self.expert_evaluator.evaluate_batch(entries)

        print("\n" + "=" * 60)
        print("Expert evaluation completed.")
        print("=" * 60)

    def run_expert_evaluation_for_model(self, model_name: str) -> None:
        """Run expert evaluation only for responses from one tested model."""
        if not self.expert_evaluator:
            return

        entries = [e for e in self._response_store if e.model_name == model_name]
        logger.info("[DEBUG] run_expert_evaluation_for_model: model=%s, total_store=%d, entries_for_model=%d", model_name, len(self._response_store), len(entries))
        if not entries:
            # Собираем расширенную диагностическую информацию
            models_in_store = set(e.model_name for e in self._response_store)
            store_by_model: dict[str, int] = {}
            for e in self._response_store:
                store_by_model[e.model_name] = store_by_model.get(e.model_name, 0) + 1

            # Детали по каждому entry в хранилище (если есть)
            store_details = ""
            if self._response_store:
                detail_lines = []
                for i, e in enumerate(self._response_store, 1):
                    detail_lines.append(
                        f"  [{i}] model={e.model_name}, prompt_id={e.prompt_id!r}, "
                        f"mode={e.mode!r}, chain_id={e.chain_id!r}, "
                        f"ctx={e.ctx}, temp={e.temperature}, "
                        f"response_len={len(e.response)}, "
                        f"has_metrics={e.metrics_ref is not None}"
                    )
                store_details = "\n" + "\n".join(detail_lines)
            else:
                store_details = "\n  (store is completely empty)"

            # Конфигурация тестирования
            effective_ctx = self.user_context_sizes if self.user_context_sizes is not None else self.context_sizes
            
            error_details = (
                f"\n{'='*70}\n"
                f"[ERROR] Expert evaluation interrupted for model={model_name}\n"
                f"{'='*70}\n"
                f"Reason: No stored responses found for the tested model.\n"
                f"\n"
                f"--- Test Configuration ---\n"
                f"Tested model: {model_name}\n"
                f"Context sizes configured: {effective_ctx}\n"
                f"User context sizes: {self.user_context_sizes}\n"
                f"Default context sizes: {self.context_sizes}\n"
                f"Temperature values: {self.temperature_test_values}\n"
                f"Num runs: {self.num_runs}\n"
                f"Num predict: {self.num_predict}\n"
                f"Disable thinking: {self.disable_thinking}\n"
                f"Independent top: {self.independent_top}\n"
                f"Prompt loader: {'set' if self.prompt_loader else 'NONE'}\n"
                f"Expert evaluator: {'set' if self.expert_evaluator else 'NONE'}\n"
                f"API client type: {type(self.ollama_client).__name__}\n"
                f"\n"
                f"--- Response Store Status ---\n"
                f"Total responses in store: {len(self._response_store)}\n"
                f"Models in store: {models_in_store if models_in_store else 'NONE'}\n"
                f"Responses by model: {store_by_model if store_by_model else 'EMPTY'}\n"
                f"\n"
                f"--- Store Entries Detail ---{store_details}\n"
                f"\n"
                f"--- Possible Causes ---\n"
                f"1. Model failed to generate any responses (check API errors above)\n"
                f"2. Condition 'tps_list and tps_list[0].get(\"response\")' was False\n"
                f"3. All requests returned empty/None responses\n"
                f"4. Model was unloaded before responses were stored\n"
                f"{'='*70}"
            )
            
            logger.error("[Expert] %s", error_details)
            print(error_details)
            
            raise RuntimeError(
                f"No stored responses for model={model_name}. "
                f"Total stored: {len(self._response_store)}, "
                f"Models in store: {models_in_store}. "
                "Benchmark interrupted — see diagnostic output above for details."
            )

        print("\n" + "=" * 60)
        print(f"Starting expert evaluation for: {model_name}")
        print(f"Total responses to evaluate: {len(entries)}")
        print("=" * 60)
        self.expert_evaluator.evaluate_batch(entries)
        self._response_store = [e for e in self._response_store if e.model_name != model_name]
        print("Expert evaluation completed for model.")
        print("=" * 60)

    def _unload_tested_model(self, model_name: str) -> None:
        """Unload a tested model from VRAM and keep benchmark flow alive."""
        print(f"\n   🧹 Unloading model '{model_name}' from VRAM...")
        try:
            self.ollama_client.unload_model(model_name)
        except Exception as e:
            print(f"   ⚠️  Warning: Could not unload model: {e}")

    def _print_vram_stats(self, model_name: str) -> None:
        """Print system resource statistics (CPU, RAM, VRAM) for the model.

        Args:
            model_name: Name of the model being tested.
        """
        from system.system_monitor import ResourceMonitor
        import time

        print(f"\n   📊 System Resource Statistics for {model_name}:")

        # Check if we're connected to a remote server via SSH
        if self.ssh_client and self.ssh_client.is_configured:
            # Use SSH monitoring for remote server
            # Get VRAM total once
            vram_total_bytes = None
            vram_samples = []

            try:
                result = self.ssh_client.execute(
                    "nvidia-smi --query-gpu=memory.total --format=csv,nounits,noheader",
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    vram_total_bytes = int(result.stdout.strip()) * 1024 * 1024
            except Exception:
                pass

            # Collect VRAM samples via SSH
            for _ in range(5):
                try:
                    result = self.ssh_client.execute(
                        "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader",
                        timeout=5,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        lines = result.stdout.strip().split("\n")
                        v = int(lines[0].strip()) * 1024 * 1024
                        vram_samples.append(v)
                except Exception:
                    pass
                time.sleep(0.2)

            # Get CPU/RAM stats via SSH
            resource_monitor = ResourceMonitor('remote', 0.2, self.ssh_client)
            resource_monitor.start_monitoring(duration=0.5)
            time.sleep(0.6)
            resource_monitor.stop_monitoring()
            resource_stats = resource_monitor.get_aggregated_stats()

            # CPU stats
            if resource_stats and 'cpu' in resource_stats:
                cpu = resource_stats['cpu']
                print(f"      CPU: {cpu.get('avg', 0):.1f}% (min: {cpu.get('min', 0):.1f}%, max: {cpu.get('max', 0):.1f}%)")

            # RAM stats
            if resource_stats and 'ram' in resource_stats:
                ram = resource_stats['ram']
                print(f"      RAM: {ram.get('avg_percent', 0):.1f}% (min: {ram.get('min_percent', 0):.1f}%, max: {ram.get('max_percent', 0):.1f}%)")

            # VRAM stats
            if vram_samples:
                avg_vram = sum(vram_samples) / len(vram_samples)
                min_vram = min(vram_samples)
                max_vram = max(vram_samples)
                print(f"      VRAM: {avg_vram / 1024 / 1024:.1f} MiB (min: {min_vram / 1024 / 1024:.1f} MiB, max: {max_vram / 1024 / 1024:.1f} MiB)")
                if vram_total_bytes:
                    print(f"      VRAM Total: {vram_total_bytes / 1024 / 1024:.1f} MiB")
            else:
                print("      VRAM: N/A (GPU not available)")
        else:
            # Use local monitoring
            # Get VRAM stats
            vram_stats = get_vram_stats(samples=5, interval=0.2)

            # Get CPU/RAM stats
            resource_monitor = ResourceMonitor('local', 0.2)
            resource_monitor.start_monitoring(duration=0.5)
            time.sleep(0.6)
            resource_monitor.stop_monitoring()
            resource_stats = resource_monitor.get_aggregated_stats()

            # CPU stats
            if resource_stats and 'cpu' in resource_stats:
                cpu = resource_stats['cpu']
                print(f"      CPU: {cpu.get('avg', 0):.1f}% (min: {cpu.get('min', 0):.1f}%, max: {cpu.get('max', 0):.1f}%)")

            # RAM stats
            if resource_stats and 'ram' in resource_stats:
                ram = resource_stats['ram']
                print(f"      RAM: {ram.get('avg_percent', 0):.1f}% (min: {ram.get('min_percent', 0):.1f}%, max: {ram.get('max_percent', 0):.1f}%)")

            # VRAM stats
            if vram_stats:
                print(f"      VRAM: {vram_stats['avg'] / 1024 / 1024:.1f} MiB (min: {vram_stats['min'] / 1024 / 1024:.1f} MiB, max: {vram_stats['max'] / 1024 / 1024:.1f} MiB)")
                if vram_stats['total']:
                    print(f"      VRAM Total: {vram_stats['total'] / 1024 / 1024:.1f} MiB")
            else:
                print("      VRAM: N/A (GPU not available)")

    def _store_response(self, entry: ExpertEvaluationEntry) -> None:
        """Store response for later expert evaluation.

        Args:
            entry: Evaluation entry with response data.
        """
        logger.info(
            "[Expert] _store_response: model=%s prompt_id=%r mode=%r response_len=%d has_metrics_ref=%s",
            entry.model_name, entry.prompt_id, entry.mode,
            len(entry.response), entry.metrics_ref is not None
        )
        self._response_store.append(entry)

    def filter_contexts(self, max_ctx: int) -> list:
        """Filter context sizes based on model's maximum context.

        Args:
            max_ctx: Model's maximum context size

        Returns:
            list: Filtered list of valid context sizes
        """
        MIN_CONTEXT = 32_768  # 32K minimum for default list only

        if self.user_context_sizes is not None:
            # User explicitly specified context sizes - only use those
            # No max_ctx filter for user values - allow testing contexts beyond model's declared max
            valid_contexts = [c for c in self.user_context_sizes if 0 < c]
            logger.info(f"[DIAGNOSIS] filter_contexts: user-specified sizes={self.user_context_sizes}, valid_contexts={valid_contexts} (no max_ctx filter)")
        else:
            # Default mode - use default list and add max_ctx if not already present
            valid_contexts = [c for c in self.context_sizes if MIN_CONTEXT <= c <= max_ctx]

            # If the model supports some "non-standard" max context (e.g. 128000),
            # which is not in our list, but it's >= 32K and <= 262144, add it for testing
            if max_ctx not in valid_contexts and MIN_CONTEXT <= max_ctx <= 262144:
                logger.info(f"[DIAGNOSIS] Adding max_ctx={max_ctx} to valid_contexts (not in list)")
                valid_contexts.append(max_ctx)
                valid_contexts.sort()

            logger.info(f"[DIAGNOSIS] filter_contexts: default sizes={self.context_sizes}, max_ctx={max_ctx}, valid_contexts={valid_contexts}")
        logger.info(f"[DIAGNOSIS] filter_contexts result: {valid_contexts}")
        return valid_contexts

    # Delegate methods - forward to submodules for backward compatibility
    def run_independent_prompts(self, model: ModelInfo, mode: str, temperature: float | None = None) -> tuple[BenchmarkResult | None, str | None]:
        """Run benchmarks using independent prompts for a specific mode."""
        return run_independent_prompts(self, model, mode, temperature)  # type: ignore[misc]

    def run_all_independent_prompts(self, model: ModelInfo, temperature: float | None = None) -> tuple[BenchmarkResult | None, str | None]:
        """Run ALL independent prompts for a model across contexts and temperatures."""
        return run_all_independent_prompts(self, model, temperature)  # type: ignore[misc]

    def run_chain(self, model: ModelInfo, chain: dict, temperature: float | None = None) -> tuple[BenchmarkResult | None, str | None]:
        """Run benchmarks using a prompt chain (Architect -> Code -> Debug)."""
        return run_chain(self, model, chain, temperature)  # type: ignore[misc]

    def run_all_chains(self, model: ModelInfo, temperature: float | None = None) -> tuple[BenchmarkResult | None, str | None]:
        """Run ALL chains for a model."""
        return run_all_chains(self, model, temperature)  # type: ignore[misc]

    def run_all_combined(self, model: ModelInfo, temperature: float | None = None) -> tuple[BenchmarkResult | None, str | None]:
        """Run ALL independent prompts followed by ALL chains for a model.
        
        Execution order: Contexts -> Temperatures -> Independent prompts -> Chains
        """
        from benchmark.runner_combined import run_all_combined as _run_all_combined
        return _run_all_combined(self, model, temperature)  # type: ignore[misc]

    def run_for_model(self, model: ModelInfo, temperature: float | None = None) -> tuple[BenchmarkResult | None, str | None]:
        """Run benchmarks for a single model across valid contexts."""
        return run_for_model(self, model, temperature)  # type: ignore[misc]
