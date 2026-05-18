"""Core benchmark execution orchestration."""

import logging
from typing import Any

from api.base_client import BaseApiClient
from benchmark.expert_evaluator_types import ExpertEvaluationEntry
from benchmark.result import BenchmarkMetrics, BenchmarkResult, ModelInfo
from i18n import get_text
from system.gpu_monitor import get_vram_stats
from system.restart_manager import RestartMethod, restart_ollama
from ui.output_formatter import format_tokens_info, update_tokens_display

# Setup logging
logger = logging.getLogger('roo_bench.benchmark')

# Default temperature values for testing
DEFAULT_TEMPERATURES = [0.0, 0.66, 1.0]


class BenchmarkRunner:
    """Orchestrates benchmark execution for models."""

    def __init__(self, ollama_client: BaseApiClient, context_sizes: list, num_runs: int = 3,
                 restart_method: str = 'manual', no_restart: bool = False, disable_thinking: bool = True,
                 prompt_loader=None, temperature_test_values: list = None, expert_evaluator=None,
                 num_predict: int = 12000, independent_top: int | None = None,
                 user_context_sizes: str = None):
        """Initialize benchmark runner.

        Args:
            ollama_client: BaseApiClient instance (LocalApiClient or RemoteApiClient)
            context_sizes: List of context sizes to test
            num_runs: Number of runs per configuration
            restart_method: Ollama restart method
            no_restart: If True, restart is not performed
            disable_thinking: If True, disables thinking mode to prevent reasoning loops
            prompt_loader: PromptLoader instance for loading prompts
            temperature_test_values: List of temperature values for testing (default: None)
            expert_evaluator: Optional ExpertEvaluator instance for response quality assessment
            num_predict: Maximum number of tokens to generate (default: 12000, use -1 for unlimited)
            user_context_sizes: Raw string from --context-sizes CLI arg (None if not specified)
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

    def get_test_params(self, model_names: list, expert_model: str | None = None) -> dict:
        """Get test parameters as a dictionary.
        
        Args:
            model_names: List of model names being tested
            expert_model: Expert model name (or None)
            
        Returns:
            dict: Test parameters including context sizes, num runs, etc.
        """
        return {
            "context_sizes": self.context_sizes,
            "num_runs": self.num_runs,
            "num_predict": self.num_predict,
            "temperature_test_values": self.temperature_test_values,
            "used_prompt_ids": self.get_used_prompt_ids(),
            "used_chain_ids": self.get_used_chain_ids(),
            "independent_top": self.independent_top,
            "disable_thinking": self.disable_thinking,
            "models": model_names,
            "expert_model": expert_model
        }

    def get_used_independent_prompts(self) -> list[dict[str, Any]]:
        """Return independent prompts that will be used by all-independent mode."""
        if not self.prompt_loader:
            return []

        all_prompts = self.prompt_loader.get_all_independent_prompts_ordered()
        if self.independent_top is None or self.independent_top <= 0:
            return all_prompts

        prompts_by_mode: dict[str, list[dict[str, Any]]] = {}
        for prompt in all_prompts:
            prompts_by_mode.setdefault(prompt.get('mode'), []).append(prompt)

        filtered_prompts: list[dict[str, Any]] = []
        for mode in ['architect', 'code', 'debug']:
            filtered_prompts.extend(prompts_by_mode.get(mode, [])[:self.independent_top])
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
        if not entries:
            logger.warning("[Expert] No stored responses for model=%s", model_name)
            return

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
        
        # Get VRAM stats
        vram_stats = get_vram_stats(samples=5, interval=0.2)
        
        # Get CPU/RAM stats
        resource_monitor = ResourceMonitor('local', 0.2)
        resource_monitor.start_monitoring(duration=0.5)
        import time
        time.sleep(0.6)
        resource_monitor.stop_monitoring()
        resource_stats = resource_monitor.get_aggregated_stats()
        
        print(f"\n   📊 System Resource Statistics for {model_name}:")
        
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
        MIN_CONTEXT = 32_768  # 32K minimum
    
        if self.user_context_sizes is not None:
            # User explicitly specified context sizes - only use those
            # Check each one against model's max_ctx
            valid_contexts = [c for c in self.context_sizes if MIN_CONTEXT <= c <= max_ctx]
            logger.info(f"[DIAGNOSIS] filter_contexts: user-specified sizes={self.context_sizes}, max_ctx={max_ctx}, valid_contexts={valid_contexts}")
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

    def run_independent_prompts(self, model: ModelInfo, mode: str, temperature: float | None = None) -> tuple[BenchmarkResult | None, str | None]:
        """Run benchmarks using independent prompts for a specific mode.

        Args:
            model: ModelInfo instance containing model metadata.
            mode: One of 'architect', 'code', 'debug'.
            temperature: Temperature value to use (None for default).

        Returns:
            tuple: (BenchmarkResult, None) on success, or (None, error_message) on error.
        """
        model_name = model.name
        max_ctx = model.max_ctx
        prompts = self.prompt_loader.get_independent_prompts(mode)
        
        # Apply independent_top filter if specified
        if self.independent_top is not None and self.independent_top > 0:
            prompts = prompts[:self.independent_top]
            logger.info("[independent_top] Limited %s prompts to first %d: %d prompts",
                       mode, self.independent_top, len(prompts))
        
        all_metrics: list[BenchmarkMetrics] = []
        error_message: str | None = None
        
        # Display model name at the start
        print(f"\n🤖 Testing model: {model_name}")
        logger.info(f"🤖 Testing model: {model_name}")
        
        # Restart Ollama before testing
        restart_ollama(
            self.restart_method,
            self.no_restart,
            ssh_client=self.ssh_client
        )
        
        # Filter contexts (from 32K to MaxModelContext)
        valid_contexts = self.filter_contexts(max_ctx)
        
        # Get default temperature if not specified
        if temperature is None:
            temps = self.temperature_test_values or [1.0, 0.7, 0.3, 0.0]
        else:
            temps = [temperature]
        
        # NEW ORDER: Contexts → Temperatures → Prompts
        for ctx in valid_contexts:
            logger.info(f"   📏 Context size: {ctx}")
            print(f"   📏 Context: {ctx // 1024}K" if ctx >= 1024 else f"   📏 Context: {ctx}")
            
            for temp in temps:
                logger.info(f"   🌡️  Temperature: {temp:.2f}")
                print(f"      🌡️  Temperature: {temp:.2f}")
                
                for prompt_data in prompts:
                    prompt = prompt_data['prompt']
                    prompt_id = prompt_data['id']
                    prompt_name = prompt_data['name']
                    prompt_metadata = {
                        'type': 'independent',
                        'mode': mode,
                        'id': prompt_id,
                        'name': prompt_name
                    }
                    
                    logger.info(f"📝 Running prompt: {prompt_name} (ID: {prompt_id})")
                    print(f"         Running prompt: {prompt_name} (ID: {prompt_id})")
                    logger.debug(f"📝 Prompt text: {prompt}")
                    
                    # Create token update callback for this prompt
                    current_tokens = {'prompt': 0, 'response': 0}
                    
                    def token_callback(prompt_tokens, response_tokens, estimated_response_tokens=0, response_len=0, is_done=False, current_tps=0.0, cpu_percent=0.0, ram_percent=0.0, vram_percent=0.0, gpu_percent=0.0):
                        current_tokens['prompt'] = prompt_tokens
                        current_tokens['response'] = response_tokens
                        update_tokens_display(prompt_tokens, response_tokens, estimated_response_tokens, response_len, indent="         ", is_done=is_done, current_tps=current_tps, cpu_percent=cpu_percent, ram_percent=ram_percent, vram_percent=vram_percent, gpu_percent=gpu_percent)
                    
                    try:
                        result = self.ollama_client.run_generation(
                            model_name,
                            ctx,  # Use current context size
                            self.num_runs,
                            self.disable_thinking,
                            temperature=temp,
                            prompt=prompt,
                            prompt_metadata=prompt_metadata,
                            num_predict=self.num_predict,
                            on_token_update=token_callback
                        )
                        # Handle both old and new return formats for backward compatibility
                        if len(result) == 6:
                            avg_tps, vram, tps_list, error, _, used_temp = result
                            resource_stats = None
                        else:
                            avg_tps, vram, tps_list, error, _, used_temp, resource_stats = result
                        
                        if error:
                            print(f"         Error running prompt {prompt_id}: {error}")
                            # Log error but don't stop the entire benchmark
                            continue
                        
                        # Log response
                        if tps_list and tps_list[0].get('response'):
                            response = tps_list[0]['response']
                            logger.info(f"   🤖 Model response (first 500 chars): {response[:500]}...")
                        
                        # Show metrics
                        duration = tps_list[0].get('total_duration', 0) / 1e9
                        prompt_tokens = tps_list[0].get('prompt_eval_count', 0)
                        response_tokens = tps_list[0].get('eval_count', 0)
                        # Clear the streaming token display and show final metrics
                        print(f"            Duration: {duration:.2f}s | Prompt: {prompt_tokens} | Response: {response_tokens}")
                        # Move to next line after streaming display
                        print()
                        
                        # Calculate std_dev
                        if len(tps_list) > 1:
                            mean = avg_tps
                            variance = sum((run['tps'] - mean) ** 2 for run in tps_list) / len(tps_list)
                            std_dev = variance ** 0.5
                        else:
                            std_dev = 0.0
                        
                        # Create BenchmarkMetrics instance
                        response = tps_list[0]['response'] if tps_list and tps_list[0].get('response') else None
                        metrics = BenchmarkMetrics(
                            prompt_id=prompt_id,
                            prompt_name=prompt_name,
                            ctx=ctx,
                            temperature=used_temp,
                            duration_sec=duration,
                            prompt_tokens=prompt_tokens,
                            response_tokens=response_tokens,
                            avg_tps=avg_tps,
                            min_tps=min(run['tps'] for run in tps_list),
                            max_tps=max(run['tps'] for run in tps_list),
                            std_dev=std_dev,
                            vram=vram,
                            mode=mode,
                            response=response,
                            # Добавляем статистику ресурсов
                            cpu_stats=resource_stats.get('cpu') if resource_stats else None,
                            ram_stats=resource_stats.get('ram') if resource_stats else None,
                            vram_stats=resource_stats.get('vram') if resource_stats else None,
                            avg_cpu_percent=resource_stats.get('cpu', {}).get('avg') if resource_stats else None,
                            max_cpu_percent=resource_stats.get('cpu', {}).get('max') if resource_stats else None,
                            avg_ram_percent=resource_stats.get('ram', {}).get('avg_percent') if resource_stats else None,
                            max_ram_percent=resource_stats.get('ram', {}).get('max_percent') if resource_stats else None,
                            avg_vram_percent=resource_stats.get('vram', {}).get('avg_percent') if resource_stats else None,
                            max_vram_percent=resource_stats.get('vram', {}).get('max_percent') if resource_stats else None,
                        )
                        all_metrics.append(metrics)
                        
                        if tps_list and tps_list[0].get('response'):
                            response = tps_list[0]['response']
                            store_entry = ExpertEvaluationEntry(
                                model_name=model_name,
                                ctx=ctx,
                                temperature=used_temp,
                                prompt_id=prompt_id,
                                prompt_name=prompt_name,
                                mode=mode,
                                chain_id=None,
                                chain_name=None,
                                response=response,
                                avg_tps=avg_tps,
                                metrics_ref=metrics
                            )
                            self._store_response(store_entry)
                        
                    except Exception as e:
                        # Log critical error that stopped the test
                        error_message = f"Critical error during {prompt_id} testing: {e}"
                        print(f"         ❌ {error_message}")
                        # Return error so main.py knows the test failed
                        self._unload_tested_model(model_name)
                        return None, error_message
        
        self._unload_tested_model(model_name)
        self._print_vram_stats(model_name)
        
        # Create the final BenchmarkResult object
        benchmark_result = BenchmarkResult(model=model, results=all_metrics)
        return benchmark_result, None
    
    def run_chain(self, model: ModelInfo, chain: dict, temperature: float | None = None) -> tuple[BenchmarkResult | None, str | None]:
        """Run benchmarks using a prompt chain (Architect -> Code -> Debug).

        Args:
            model: ModelInfo instance containing model metadata.
            chain: Chain dictionary from prompt loader.
            temperature: Temperature value to use (None for default).

        Returns:
            tuple: (BenchmarkResult, None) on success, or (None, error_message) on error.
        """
        model_name = model.name
        max_ctx = model.max_ctx
        chain_id = chain.get('id', 'unknown')
        chain_name = chain.get('name', 'Unknown Chain')
        
        all_metrics: list[BenchmarkMetrics] = []
        error_message: str | None = None
        
        # Display model name at the start
        print(f"\n🤖 Testing model: {model_name}")
        logger.info(f"🤖 Testing model: {model_name}")
        
        print(f"   Running chain: {chain_name} ({chain_id})")
        
        # Restart Ollama before testing
        restart_ollama(
            self.restart_method,
            self.no_restart,
            ssh_client=self.ssh_client
        )
        
        # Build chain context
        chain_prompts = self.prompt_loader.build_chain_context(chain)
        
        # Filter contexts
        valid_contexts = self.filter_contexts(max_ctx)
        
        # Get default temperature if not specified
        if temperature is None:
            temps = self.temperature_test_values or [1.0, 0.7, 0.3, 0.0]
        else:
            temps = [temperature]
        
        # Contexts → Temperatures → Chain modes
        for ctx in valid_contexts:
            logger.info(f"   📏 Context size: {ctx}")
            print(f"   📏 Context: {ctx // 1024}K" if ctx >= 1024 else f"   📏 Context: {ctx}")
            
            for temp in temps:
                logger.info(f"   🌡️  Temperature: {temp:.2f}")
                print(f"      🌡️  Temperature: {temp:.2f}")
                
                chain_responses: dict[str, str] = {}
                
                for mode in ['architect', 'code', 'debug']:
                    if mode not in chain_prompts:
                        continue
                    
                    prompt_data = chain_prompts[mode]
                    prompt = prompt_data['prompt']
                    prompt_id = prompt_data['id']
                    prompt_name = prompt_data['name']
                    
                    prompt_metadata = {
                        'type': 'chain',
                        'chain_id': chain_id,
                        'chain_name': chain_name,
                        'mode': mode,
                        'id': prompt_id,
                        'name': prompt_name
                    }
                    
                    logger.info(f"📝 Running chain [{mode}]: {prompt_name} (ID: {prompt_id})")
                    print(f"         Running chain [{mode}]: {prompt_name} (ID: {prompt_id})")
                    
                    # Create token update callback for this chain
                    def chain_token_callback(prompt_tokens, response_tokens, estimated_response_tokens=0, response_len=0, is_done=False, current_tps=0.0, cpu_percent=0.0, ram_percent=0.0, vram_percent=0.0, gpu_percent=0.0):
                        update_tokens_display(prompt_tokens, response_tokens, estimated_response_tokens, response_len, indent="         ", is_done=is_done, current_tps=current_tps, cpu_percent=cpu_percent, ram_percent=ram_percent, vram_percent=vram_percent, gpu_percent=gpu_percent)
                    
                    try:
                        result = self.ollama_client.run_generation(
                            model_name,
                            ctx,
                            self.num_runs,
                            self.disable_thinking,
                            temperature=temp,
                            prompt=prompt,
                            prompt_metadata=prompt_metadata,
                            num_predict=self.num_predict,
                            on_token_update=chain_token_callback
                        )
                        # Handle both old and new return formats for backward compatibility
                        if len(result) == 6:
                            avg_tps, vram, tps_list, error, _, used_temp = result
                            resource_stats = None
                        else:
                            avg_tps, vram, tps_list, error, _, used_temp, resource_stats = result
                        
                        if error:
                            print(f"         Error in chain [{mode}]: {error}")
                            break
                        
                        duration = tps_list[0].get('total_duration', 0) / 1e9
                        prompt_tokens = tps_list[0].get('prompt_eval_count', 0)
                        response_tokens = tps_list[0].get('eval_count', 0)
                        # Move to next line after streaming display
                        print()
                        
                        if len(tps_list) > 1:
                            mean = avg_tps
                            variance = sum((run['tps'] - mean) ** 2 for run in tps_list) / len(tps_list)
                            std_dev = variance ** 0.5
                        else:
                            std_dev = 0.0
                        
                        response = tps_list[0]['response'] if tps_list and tps_list[0].get('response') else None
                        metrics = BenchmarkMetrics(
                            mode=mode,
                            prompt_id=prompt_id,
                            prompt_name=prompt_name,
                            ctx=ctx,
                            temperature=used_temp,
                            duration_sec=duration,
                            prompt_tokens=prompt_tokens,
                            response_tokens=response_tokens,
                            avg_tps=avg_tps,
                            min_tps=min(run['tps'] for run in tps_list),
                            max_tps=max(run['tps'] for run in tps_list),
                            std_dev=std_dev,
                            vram=vram,
                            chain_id=chain_id,
                            chain_name=chain_name,
                            response=response,
                            # Добавляем статистику ресурсов
                            cpu_stats=resource_stats.get('cpu') if resource_stats else None,
                            ram_stats=resource_stats.get('ram') if resource_stats else None,
                            vram_stats=resource_stats.get('vram') if resource_stats else None,
                            avg_cpu_percent=resource_stats.get('cpu', {}).get('avg') if resource_stats else None,
                            max_cpu_percent=resource_stats.get('cpu', {}).get('max') if resource_stats else None,
                            avg_ram_percent=resource_stats.get('ram', {}).get('avg_percent') if resource_stats else None,
                            max_ram_percent=resource_stats.get('ram', {}).get('max_percent') if resource_stats else None,
                            avg_vram_percent=resource_stats.get('vram', {}).get('avg_percent') if resource_stats else None,
                            max_vram_percent=resource_stats.get('vram', {}).get('max_percent') if resource_stats else None,
                        )
                        all_metrics.append(metrics)
                        
                        if response:
                            response = tps_list[0]['response']
                            chain_responses[mode] = response
                            
                            current_chain_context = {}
                            if mode == 'code' and 'architect' in chain_responses:
                                current_chain_context['architect_response'] = chain_responses['architect']
                            elif mode == 'debug' and 'code' in chain_responses:
                                current_chain_context['code_response'] = chain_responses['code']
                            
                            store_entry = ExpertEvaluationEntry(
                                model_name=model_name,
                                ctx=ctx,
                                temperature=used_temp,
                                prompt_id=prompt_id,
                                prompt_name=prompt_name,
                                mode=mode,
                                chain_id=chain_id,
                                chain_name=chain_name,
                                response=response,
                                avg_tps=avg_tps,
                                metrics_ref=metrics,
                                chain_context=current_chain_context
                            )
                            self._store_response(store_entry)
                        
                    except Exception as e:
                        error_message = f"Critical error during chain [{mode}] testing: {e}"
                        print(f"         ❌ {error_message}")
                        self._unload_tested_model(model_name)
                        return None, error_message
        
        self._unload_tested_model(model_name)
        self._print_vram_stats(model_name)
        
        benchmark_result = BenchmarkResult(model=model, results=all_metrics)
        return benchmark_result, None

    def run_for_model(self, model: ModelInfo, temperature: float | None = None) -> tuple[BenchmarkResult | None, str | None]:
        """Run benchmarks for a single model across valid contexts.

        Args:
            model: ModelInfo instance containing model metadata.
            temperature: Temperature value to use (None for default).

        Returns:
            tuple: (BenchmarkResult, None) on success, or (None, error_message) on error.
        """
        model_name = model.name
        max_ctx = model.max_ctx
        max_ctx_str = f"{max_ctx // 1024}K" if max_ctx >= 1024 else str(max_ctx)

        print("\n============================================================\n")
        print(f"🤖 Testing model: {model_name}")
        print(get_text("testing_model", model_name=model_name))
        
        size_gb_str = f"{model.size_gb:.1f}"
        print(get_text("model_size", size_gb=size_gb_str, max_ctx_str=max_ctx_str))

        all_metrics: list[BenchmarkMetrics] = []
        error_message: str | None = None
        valid_contexts = self.filter_contexts(max_ctx)

        if not valid_contexts:
            print(get_text("skipping_no_contexts"))
            return BenchmarkResult(model=model, results=[]), None

        for ctx in valid_contexts:
            restart_ollama(
                self.restart_method,
                self.no_restart,
                ssh_client=self.ssh_client
            )

            try:
                current_num_ctx = self.ollama_client.get_current_num_ctx(model_name)
                print(get_text("current_num_ctx", num_ctx=current_num_ctx))
                if current_num_ctx != ctx:
                    print(get_text("ctx_info_only", actual=current_num_ctx, expected=ctx))
            except Exception as e:
                print(f"   ⚠️  Error getting model settings: {e}")

            print(get_text("warming_up", ctx=ctx))
            
            if temperature is None:
                default_temp = self.ollama_client.get_default_temperature(model_name)
                temps = [default_temp, default_temp * 2/3, default_temp * 1/3]
            else:
                temps = [temperature]
            
            for temp in temps:
                logger.info(f"   🌡️  Running with temperature: {temp:.2f}")
                print(f"   🌡️  Temperature: {temp:.2f}")
                print(f"   📦 Model: {model_name}")
                
                # Create token update callback for run_for_model
                def run_for_model_token_callback(prompt_tokens, response_tokens, estimated_response_tokens=0, response_len=0, is_done=False, current_tps=0.0, cpu_percent=0.0, ram_percent=0.0, vram_percent=0.0, gpu_percent=0.0):
                    update_tokens_display(prompt_tokens, response_tokens, estimated_response_tokens, response_len, indent="   ", is_done=is_done, current_tps=current_tps, cpu_percent=cpu_percent, ram_percent=ram_percent, vram_percent=vram_percent, gpu_percent=gpu_percent)

                try:
                    result = self.ollama_client.run_generation(
                        model_name, ctx, self.num_runs, self.disable_thinking, temperature=temp,
                        num_predict=self.num_predict,
                        on_token_update=run_for_model_token_callback
                    )
                    # Handle both old and new return formats for backward compatibility
                    if len(result) == 6:
                        avg_tps, vram, tps_list, error_msg, _, used_temp = result
                        resource_stats = None
                    else:
                        avg_tps, vram, tps_list, error_msg, _, used_temp, resource_stats = result

                    if error_msg:
                        print(get_text("benchmark_failed", error_msg=error_msg))
                        print(get_text("stopping_tests", model_name=model_name))
                        break

                    print(get_text("benchmark_runs_header"))
                    for run in tps_list:
                        run_num = run['run']
                        tps = run['tps']
                        vram_run = run['vram']
                        if vram_run:
                            vram_str = f"{vram_run / 1024 / 1024:.1f} MiB"
                        else:
                            vram_str = "N/A"
                        prompt_tokens = run.get('prompt_eval_count', 0)
                        response_tokens = run.get('eval_count', 0)
                        duration = run.get('total_duration', 0) / 1e9
                        tokens_info = format_tokens_info(prompt_tokens, response_tokens)
                        print(f"   Run {run_num}: {tps:.2f} TPS (VRAM: {vram_str}) {tokens_info}")
                        print(f"      Duration: {duration:.2f}s")
                    
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

                    print(get_text("benchmark_summary"))
                    print(f"   Average: {avg_tps:.2f} TPS")
                    print(f"   Min: {min(r['tps'] for r in tps_list):.2f} TPS")
                    print(f"   Max: {max(r['tps'] for r in tps_list):.2f} TPS")

                    mean = avg_tps
                    variance = sum((r['tps'] - mean) ** 2 for r in tps_list) / len(tps_list)
                    std_dev = variance ** 0.5
                    print(f"   Std Dev: {std_dev:.2f} TPS")

                    response = tps_list[0]['response'] if tps_list and tps_list[0].get('response') else None
                    metrics = BenchmarkMetrics(
                        ctx=ctx,
                        temperature=used_temp,
                        duration_sec=tps_list[0].get('total_duration', 0) / 1e9 if tps_list else 0,
                        prompt_tokens=tps_list[0].get('prompt_eval_count', 0) if tps_list else 0,
                        response_tokens=tps_list[0].get('eval_count', 0) if tps_list else 0,
                        avg_tps=avg_tps,
                        min_tps=min(r['tps'] for r in tps_list),
                        max_tps=max(r['tps'] for r in tps_list),
                        std_dev=std_dev,
                        vram=vram,
                        response=response,
                        # Добавляем статистику ресурсов
                        cpu_stats=resource_stats.get('cpu') if resource_stats else None,
                        ram_stats=resource_stats.get('ram') if resource_stats else None,
                        vram_stats=resource_stats.get('vram') if resource_stats else None,
                        avg_cpu_percent=resource_stats.get('cpu', {}).get('avg') if resource_stats else None,
                        max_cpu_percent=resource_stats.get('cpu', {}).get('max') if resource_stats else None,
                        avg_ram_percent=resource_stats.get('ram', {}).get('avg_percent') if resource_stats else None,
                        max_ram_percent=resource_stats.get('ram', {}).get('max_percent') if resource_stats else None,
                        avg_vram_percent=resource_stats.get('vram', {}).get('avg_percent') if resource_stats else None,
                        max_vram_percent=resource_stats.get('vram', {}).get('max_percent') if resource_stats else None,
                    )
                    all_metrics.append(metrics)
                    
                    _resp_val = response
                    logger.debug(
                        "[Expert] run_for_model check: tps_list_len=%d response_present=%s response_len=%d",
                        len(tps_list),
                        _resp_val is not None,
                        len(_resp_val) if _resp_val else 0
                    )
                    if tps_list and tps_list[0].get('response'):
                        response = tps_list[0]['response']
                        store_entry = ExpertEvaluationEntry(
                            model_name=model_name,
                            ctx=ctx,
                            temperature=used_temp,
                            prompt_id='',
                            prompt_name='',
                            mode=None,
                            chain_id=None,
                            chain_name=None,
                            response=response,
                            avg_tps=avg_tps,
                            metrics_ref=metrics
                        )
                        self._store_response(store_entry)
                    
                except Exception as e:
                    error_message = f"Critical error during context {ctx} testing: {e}"
                    print(f"   ❌ {error_message}")
                    self._unload_tested_model(model_name)
                    return None, error_message

        self._unload_tested_model(model_name)
        self._print_vram_stats(model_name)

        benchmark_result = BenchmarkResult(model=model, results=all_metrics)
        return benchmark_result, None
    
    def run_all_independent_prompts(self, model: ModelInfo, temperature: float | None = None) -> tuple[BenchmarkResult | None, str | None]:
        """Run ALL independent prompts for a model across contexts and temperatures.
        
        Execution order:
            Model → Contexts → Temperatures → ALL prompts (architect → code → debug)
        
        Args:
            model: ModelInfo instance containing model metadata.
            temperature: Temperature value to use (None for default).
        
        Returns:
            tuple: (BenchmarkResult, None) on success, or (None, error_message) on error.
        """
        model_name = model.name
        max_ctx = model.max_ctx
        all_metrics: list[BenchmarkMetrics] = []
        error_message: str | None = None
        
        print(f"\n🤖 Testing model: {model_name}")
        logger.info(f"🤖 Testing model: {model_name}")
        logger.info(
            "[Expert] run_all_independent_prompts START: expert_evaluator=%s _response_store_size=%d",
            self.expert_evaluator is not None, len(self._response_store)
        )
        
        # DIAGNOSIS: Log temperature-related values
        logger.info("[DIAGNOSIS] run_all_independent_prompts: temperature arg=%s, self.temperature_test_values=%s", temperature, self.temperature_test_values)
        
        restart_ollama(
            self.restart_method,
            self.no_restart,
            ssh_client=self.ssh_client
        )
        
        valid_contexts = self.filter_contexts(max_ctx)
        
        # DIAGNOSIS: Use self.temperature_test_values if available, otherwise fallback to hardcoded
        if temperature is None and (not self.temperature_test_values or len(self.temperature_test_values) == 0):
            logger.info("[DIAGNOSIS] Using hardcoded fallback temps because temperature_test_values is empty")
            temps = [1.0, 0.7, 0.3, 0.0]
        elif temperature is not None:
            logger.info("[DIAGNOSIS] Using temperature arg: %s", temperature)
            temps = [temperature]
        elif self.temperature_test_values and len(self.temperature_test_values) > 0:
            logger.info("[DIAGNOSIS] Using self.temperature_test_values: %s", self.temperature_test_values)
            temps = self.temperature_test_values
        else:
            logger.info("[DIAGNOSIS] Fallback to hardcoded temps: [1.0, 0.7, 0.3, 0.0]")
            temps = [1.0, 0.7, 0.3, 0.0]
        
        all_prompts = self.prompt_loader.get_all_independent_prompts_ordered()
        
        # Apply independent_top filter if specified
        if self.independent_top is not None and self.independent_top > 0:
            prompts_by_mode: dict[str, list[dict[str, Any]]] = {}
            for p in all_prompts:
                mode = p['mode']
                if mode not in prompts_by_mode:
                    prompts_by_mode[mode] = []
                prompts_by_mode[mode].append(p)
            
            filtered_prompts = []
            for mode in ['architect', 'code', 'debug']:
                mode_prompts = prompts_by_mode.get(mode, [])
                filtered_prompts.extend(mode_prompts[:self.independent_top])
            
            all_prompts = filtered_prompts
            logger.info("[independent_top] Limited to first %d prompts per mode: %d total prompts",
                       self.independent_top, len(all_prompts))
        
        logger.debug(
            "[Expert] run_all_independent_prompts: model=%s all_prompts_count=%d valid_contexts=%s",
            model_name, len(all_prompts), valid_contexts
        )

        for ctx in valid_contexts:
            logger.info(f"   📏 Context size: {ctx}")
            print(f"   📏 Context: {ctx // 1024}K" if ctx >= 1024 else f"   📏 Context: {ctx}")
            
            for temp in temps:
                logger.info(f"   🌡️  Temperature: {temp:.2f}")
                print(f"      🌡️  Temperature: {temp:.2f}")
                
                for prompt_data in all_prompts:
                    prompt = prompt_data['prompt']
                    prompt_id = prompt_data['id']
                    prompt_name = prompt_data['name']
                    mode = prompt_data['mode']
                    
                    prompt_metadata = {
                        'type': 'independent',
                        'mode': mode,
                        'id': prompt_id,
                        'name': prompt_name
                    }
                    
                    logger.info(f"📝 Running [{mode}] {prompt_name} (ID: {prompt_id})")
                    print(f"         [{mode}] {prompt_name} (ID: {prompt_id})")
                    
                    # Create token update callback for this prompt
                    def independent_token_callback(prompt_tokens, response_tokens, estimated_response_tokens=0, response_len=0, is_done=False, current_tps=0.0, cpu_percent=0.0, ram_percent=0.0, vram_percent=0.0, gpu_percent=0.0):
                        update_tokens_display(prompt_tokens, response_tokens, estimated_response_tokens, response_len, indent="         ", is_done=is_done, current_tps=current_tps, cpu_percent=cpu_percent, ram_percent=ram_percent, vram_percent=vram_percent, gpu_percent=gpu_percent)
                    
                    try:
                        avg_tps, vram, tps_list, error, _, used_temp, _ = self.ollama_client.run_generation(
                            model_name,
                            ctx,
                            self.num_runs,
                            self.disable_thinking,
                            temperature=temp,
                            prompt=prompt,
                            prompt_metadata=prompt_metadata,
                            num_predict=self.num_predict,
                            on_token_update=independent_token_callback
                        )
                        
                        if error:
                            print(f"         Error: {error}")
                            continue
                        
                        # Move to next line after streaming display
                        print()
                        
                        if len(tps_list) > 1:
                            mean = avg_tps
                            variance = sum((run['tps'] - mean) ** 2 for run in tps_list) / len(tps_list)
                            std_dev = variance ** 0.5
                        else:
                            std_dev = 0.0
                        
                        response = tps_list[0]['response'] if tps_list and tps_list[0].get('response') else None
                        metrics = BenchmarkMetrics(
                            mode=mode,
                            prompt_id=prompt_id,
                            prompt_name=prompt_name,
                            ctx=ctx,
                            temperature=used_temp,
                            duration_sec=tps_list[0].get('total_duration', 0) / 1e9 if tps_list else 0,
                            prompt_tokens=tps_list[0].get('prompt_eval_count', 0) if tps_list else 0,
                            response_tokens=tps_list[0].get('eval_count', 0) if tps_list else 0,
                            avg_tps=avg_tps,
                            min_tps=min(run['tps'] for run in tps_list),
                            max_tps=max(run['tps'] for run in tps_list),
                            std_dev=std_dev,
                            vram=vram,
                            response=response,
                        )
                        all_metrics.append(metrics)
                        
                        _resp_val_ai = response
                        logger.info(
                            "[Expert] run_all_ind check: prompt_id=%r tps_list_len=%d response_present=%s response_len=%d",
                            prompt_id, len(tps_list), _resp_val_ai is not None,
                            len(_resp_val_ai) if _resp_val_ai else 0
                        )
                        if response:
                            store_entry = ExpertEvaluationEntry(
                                model_name=model_name,
                                ctx=ctx,
                                temperature=used_temp,
                                prompt_id=prompt_id,
                                prompt_name=prompt_name,
                                mode=mode,
                                chain_id=None,
                                chain_name=None,
                                response=response,
                                avg_tps=avg_tps,
                                metrics_ref=metrics
                            )
                            self._store_response(store_entry)
                        
                    except Exception as e:
                        error_message = f"Critical error during prompt {prompt_id} testing: {e}"
                        print(f"   ❌ {error_message}")
                        self._unload_tested_model(model_name)
                        return None, error_message
        
        logger.info(
            "[Expert] run_all_independent_prompts END: model=%s _response_store_size=%d all_metrics_count=%d",
            model_name, len(self._response_store), len(all_metrics)
        )
        
        self._unload_tested_model(model_name)
        self._print_vram_stats(model_name)
        
        benchmark_result = BenchmarkResult(model=model, results=all_metrics)
        return benchmark_result, None
    
    def run_all_chains(self, model: ModelInfo, temperature: float | None = None) -> tuple[BenchmarkResult | None, str | None]:
        """Run ALL chains for a model.
        
        Execution order:
            Model → Chains (sequentially) → Contexts → Temperatures → Modes
        
        Args:
            model: ModelInfo instance containing model metadata.
            temperature: Temperature value to use (None for default).
        
        Returns:
            tuple: (BenchmarkResult, None) on success, or (None, error_message) on error.
        """
        model_name = model.name
        max_ctx = model.max_ctx
        all_metrics: list[BenchmarkMetrics] = []
        error_message: str | None = None
        
        print(f"\n🤖 Testing model: {model_name}")
        logger.info(f"🤖 Testing model: {model_name}")
        
        restart_ollama(
            self.restart_method,
            self.no_restart,
            ssh_client=self.ssh_client
        )
        
        valid_contexts = self.filter_contexts(max_ctx)
        
        if temperature is None:
            temps = self.temperature_test_values or [1.0, 0.7, 0.3, 0.0]
        else:
            temps = [temperature]
        
        chains = self.prompt_loader.get_chains()
        
        for chain in chains:
            chain_id = chain.get('id', 'unknown')
            chain_name = chain.get('name', 'Unknown Chain')
            
            print(f"\n   🔗 Running chain: {chain_name} ({chain_id})")
            logger.info(f"🔗 Running chain: {chain_name} ({chain_id})")
            
            chain_prompts = self.prompt_loader.build_chain_context(chain)
            
            for ctx in valid_contexts:
                logger.info(f"   📏 Context size: {ctx}")
                print(f"      📏 Context: {ctx // 1024}K" if ctx >= 1024 else f"      📏 Context: {ctx}")
                
                for temp in temps:
                    logger.info(f"   🌡️  Temperature: {temp:.2f}")
                    print(f"         🌡️  Temperature: {temp:.2f}")
                    
                    for mode in ['architect', 'code', 'debug']:
                        if mode not in chain_prompts:
                            continue
                        
                        prompt_data = chain_prompts[mode]
                        prompt = prompt_data['prompt']
                        prompt_id = prompt_data['id']
                        prompt_name = prompt_data['name']
                        
                        prompt_metadata = {
                            'type': 'chain',
                            'chain_id': chain_id,
                            'chain_name': chain_name,
                            'mode': mode,
                            'id': prompt_id,
                            'name': prompt_name
                        }
                        
                        logger.info(f"📝 Chain [{mode}]: {prompt_name} (ID: {prompt_id})")
                        print(f"            [{mode}] {prompt_name}")
                        
                        # Create token update callback for this chain
                        def chain_token_callback_2(prompt_tokens, response_tokens, estimated_response_tokens=0, response_len=0, is_done=False, current_tps=0.0, cpu_percent=0.0, ram_percent=0.0, vram_percent=0.0, gpu_percent=0.0):
                            update_tokens_display(prompt_tokens, response_tokens, estimated_response_tokens, response_len, indent="            ", is_done=is_done, current_tps=current_tps, cpu_percent=cpu_percent, ram_percent=ram_percent, vram_percent=vram_percent, gpu_percent=gpu_percent)
                        
                        try:
                            avg_tps, vram, tps_list, error, _, used_temp, resource_stats = self.ollama_client.run_generation(
                                                                            model_name,
                                                                ctx,
                                                                self.num_runs,
                                                                self.disable_thinking,
                                                                temperature=temp,
                                                                prompt=prompt,
                                                                prompt_metadata=prompt_metadata,
                                                                num_predict=self.num_predict,
                                                                on_token_update=chain_token_callback_2
                                                            )
                            
                            if error:
                                print(f"            Error: {error}")
                                break
                            
                            duration = tps_list[0].get('total_duration', 0) / 1e9
                            prompt_tokens = tps_list[0].get('prompt_eval_count', 0)
                            response_tokens = tps_list[0].get('eval_count', 0)
                            # Move to next line after streaming display
                            print()
                            
                            if len(tps_list) > 1:
                                mean = avg_tps
                                variance = sum((run['tps'] - mean) ** 2 for run in tps_list) / len(tps_list)
                                std_dev = variance ** 0.5
                            else:
                                std_dev = 0.0
                            
                            response = tps_list[0]['response'] if tps_list and tps_list[0].get('response') else None
                            metrics = BenchmarkMetrics(
                                chain_id=chain_id,
                                chain_name=chain_name,
                                mode=mode,
                                prompt_id=prompt_id,
                                prompt_name=prompt_name,
                                ctx=ctx,
                                temperature=used_temp,
                                duration_sec=duration,
                                prompt_tokens=prompt_tokens,
                                response_tokens=response_tokens,
                                avg_tps=avg_tps,
                                min_tps=min(run['tps'] for run in tps_list),
                                max_tps=max(run['tps'] for run in tps_list),
                                std_dev=std_dev,
                                vram=vram,
                                response=response,
                                # Добавляем статистику ресурсов
                                cpu_stats=resource_stats.get('cpu') if resource_stats else None,
                                ram_stats=resource_stats.get('ram') if resource_stats else None,
                                vram_stats=resource_stats.get('vram') if resource_stats else None,
                                avg_cpu_percent=resource_stats.get('cpu', {}).get('avg') if resource_stats else None,
                                max_cpu_percent=resource_stats.get('cpu', {}).get('max') if resource_stats else None,
                                avg_ram_percent=resource_stats.get('ram', {}).get('avg_percent') if resource_stats else None,
                                max_ram_percent=resource_stats.get('ram', {}).get('max_percent') if resource_stats else None,
                                avg_vram_percent=resource_stats.get('vram', {}).get('avg_percent') if resource_stats else None,
                                max_vram_percent=resource_stats.get('vram', {}).get('max_percent') if resource_stats else None,
                            )
                            all_metrics.append(metrics)
                            
                            if response:
                                store_entry = ExpertEvaluationEntry(
                                    model_name=model_name,
                                    ctx=ctx,
                                    temperature=used_temp,
                                    prompt_id=prompt_id,
                                    prompt_name=prompt_name,
                                    mode=mode,
                                    chain_id=chain_id,
                                    chain_name=chain_name,
                                    response=response,
                                    avg_tps=avg_tps,
                                    metrics_ref=metrics
                                )
                                self._store_response(store_entry)
                            
                        except Exception as e:
                            error_message = f"Critical error during chain [{mode}] testing: {e}"
                            print(f"            ❌ {error_message}")
                            self._unload_tested_model(model_name)
                            return None, error_message
        
        self._unload_tested_model(model_name)
        
        self._print_vram_stats(model_name)
        benchmark_result = BenchmarkResult(model=model, results=all_metrics)
        return benchmark_result, None
