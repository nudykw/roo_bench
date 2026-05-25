"""Independent prompts benchmark execution methods."""

import logging
from typing import Any

from benchmark.expert_evaluator_types import ExpertEvaluationEntry
from benchmark.result import BenchmarkMetrics, BenchmarkResult, ModelInfo
from system.restart_manager import restart_ollama
from ui.output_formatter import update_tokens_display

logger = logging.getLogger('roo_bench.benchmark')


def run_independent_prompts(
    self: Any,  # type: ignore[reportSelfClsParameterType]
    model: ModelInfo,
    mode: str,
    temperature: float | None = None,
) -> tuple[BenchmarkResult | None, str | None]:
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
    prompts = self.prompt_loader.get_independent_prompts(mode)  # type: ignore[attr-defined]

    # Apply prompts_top filter if specified (takes priority over independent_top)
    if self.prompts_top is not None and self.prompts_top > 0:  # type: ignore[attr-defined]
        prompts = prompts[:self.prompts_top]  # type: ignore[attr-defined]
        logger.info("[prompts_top] Limited %s prompts to first %d: %d prompts",
                   mode, self.prompts_top, len(prompts))  # type: ignore[attr-defined]
    elif self.independent_top is not None and self.independent_top > 0:  # type: ignore[attr-defined]
        prompts = prompts[:self.independent_top]  # type: ignore[attr-defined]
        logger.info("[independent_top] Limited %s prompts to first %d: %d prompts",
                   mode, self.independent_top, len(prompts))  # type: ignore[attr-defined]

    all_metrics: list[BenchmarkMetrics] = []
    error_message: str | None = None

    # Display model name at the start
    print(f"\n🤖 Testing model: {model_name}")
    logger.info(f"🤖 Testing model: {model_name}")

    # Restart Ollama before testing
    restart_ollama(
        self.restart_method,  # type: ignore[attr-defined]
        self.no_restart,  # type: ignore[attr-defined]
        ssh_client=self.ssh_client  # type: ignore[attr-defined]
    )

    # Filter contexts (from 32K to MaxModelContext)
    valid_contexts = self.filter_contexts(max_ctx)  # type: ignore[attr-defined]

    # Get default temperature if not specified
    if temperature is None:
        temps = self.temperature_test_values or [1.0, 0.7, 0.3, 0.0]  # type: ignore[attr-defined]
    else:
        temps = [temperature]

    # NEW ORDER: Contexts -> Temperatures -> Prompts
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

                def token_callback(
                    prompt_tokens, response_tokens, estimated_response_tokens=0,
                    response_len=0, is_done=False, current_tps=0.0,
                    cpu_percent=0.0, ram_percent=0.0, vram_percent=0.0,
                    gpu_percent=0.0,
                ):
                    current_tokens['prompt'] = prompt_tokens
                    current_tokens['response'] = response_tokens
                    update_tokens_display(
                        prompt_tokens, response_tokens, estimated_response_tokens,
                        response_len, indent="         ", is_done=is_done,
                        current_tps=current_tps, cpu_percent=cpu_percent,
                        ram_percent=ram_percent, vram_percent=vram_percent,
                        gpu_percent=gpu_percent,
                    )

                try:
                    result = self.ollama_client.run_generation(  # type: ignore[attr-defined]
                        model_name,
                        ctx,  # Use current context size
                        self.num_runs,  # type: ignore[attr-defined]
                        self.disable_thinking,  # type: ignore[attr-defined]
                        temperature=temp,
                        prompt=prompt,
                        prompt_metadata=prompt_metadata,
                        num_predict=self.num_predict,  # type: ignore[attr-defined]
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

                    # Print resource statistics if available
                    if resource_stats:
                        if 'cpu' in resource_stats:
                            cpu = resource_stats['cpu']
                            print(f"            CPU: {cpu.get('avg', 0):.1f}% (min: {cpu.get('min', 0):.1f}%, max: {cpu.get('max', 0):.1f}%)")
                        if 'ram' in resource_stats:
                            ram = resource_stats['ram']
                            print(f"            RAM: {ram.get('avg_percent', 0):.1f}% (min: {ram.get('min_percent', 0):.1f}%, max: {ram.get('max_percent', 0):.1f}%)")
                        if 'vram' in resource_stats:
                            vram_data = resource_stats['vram']
                            if vram_data.get('total', 0) > 0:
                                vram_used = vram_data.get('used_current', 0) / 1024 / 1024
                                vram_total = vram_data.get('total', 0) / 1024 / 1024
                                print(f"            VRAM: {vram_data.get('percent_current', 0):.1f}% "
                                      f"(avg: {vram_data.get('avg_percent', 0):.1f}%, "
                                      f"max: {vram_data.get('max_percent', 0):.1f}%) — "
                                      f"{vram_used:.1f} MiB / {vram_total:.1f} MiB")
                            else:
                                print(f"            VRAM: {vram_data.get('percent_current', 0):.1f}% (avg: {vram_data.get('avg_percent', 0):.1f}%, max: {vram_data.get('max_percent', 0):.1f}%)")
                        if 'gpu' in resource_stats:
                            gpu = resource_stats['gpu']
                            print(f"            GPU: {gpu.get('avg', 0):.1f}% (min: {gpu.get('min', 0):.1f}%, max: {gpu.get('max', 0):.1f}%)")

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

                    # DIAGNOSTIC: Log response storage decision
                    logger.info(
                        "[Expert] run_independent_prompts store check: "
                        "model=%s prompt_id=%s mode=%s "
                        "tps_list_len=%d "
                        "has_first_entry=%s "
                        "has_response_key=%s "
                        "response_len=%d "
                        "response_preview=%s",
                        model_name, prompt_id, mode,
                        len(tps_list),
                        len(tps_list) > 0,
                        'response' in tps_list[0] if tps_list else False,
                        len(tps_list[0].get('response', '')) if tps_list else 0,
                        tps_list[0].get('response', '')[:100] if tps_list else None,
                    )

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
                        self._store_response(store_entry)  # type: ignore[attr-defined]

                except Exception as e:
                    # Log critical error that stopped the test
                    error_message = f"Critical error during {prompt_id} testing: {e}"
                    print(f"         ❌ {error_message}")
                    # Return error so main.py knows the test failed
                    self._unload_tested_model(model_name)  # type: ignore[attr-defined]
                    return None, error_message

    self._unload_tested_model(model_name)  # type: ignore[attr-defined]
    self._print_vram_stats(model_name)  # type: ignore[attr-defined]

    # Create the final BenchmarkResult object
    benchmark_result = BenchmarkResult(model=model, results=all_metrics)
    return benchmark_result, None


def run_all_independent_prompts(
    self: Any,  # type: ignore[reportSelfClsParameterType]
    model: ModelInfo,
    temperature: float | None = None,
) -> tuple[BenchmarkResult | None, str | None]:
    """Run ALL independent prompts for a model across contexts and temperatures.

    Execution order:
        Model -> Contexts -> Temperatures -> ALL prompts (architect -> code -> debug)

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
        self.expert_evaluator is not None, len(self._response_store)  # type: ignore[attr-defined]
    )

    # DIAGNOSIS: Log temperature-related values
    logger.info("[DIAGNOSIS] run_all_independent_prompts: temperature arg=%s, self.temperature_test_values=%s", temperature, self.temperature_test_values)  # type: ignore[attr-defined]

    restart_ollama(
        self.restart_method,  # type: ignore[attr-defined]
        self.no_restart,  # type: ignore[attr-defined]
        ssh_client=self.ssh_client  # type: ignore[attr-defined]
    )

    valid_contexts = self.filter_contexts(max_ctx)  # type: ignore[attr-defined]

    # DIAGNOSIS: Use self.temperature_test_values if available, otherwise fallback to hardcoded
    if temperature is None and (not self.temperature_test_values or len(self.temperature_test_values) == 0):  # type: ignore[attr-defined]
        logger.info("[DIAGNOSIS] Using hardcoded fallback temps because temperature_test_values is empty")
        temps = [1.0, 0.7, 0.3, 0.0]
    elif temperature is not None:
        logger.info("[DIAGNOSIS] Using temperature arg: %s", temperature)
        temps = [temperature]
    elif self.temperature_test_values and len(self.temperature_test_values) > 0:  # type: ignore[attr-defined]
        logger.info("[DIAGNOSIS] Using self.temperature_test_values: %s", self.temperature_test_values)
        temps = self.temperature_test_values  # type: ignore[attr-defined]
    else:
        logger.info("[DIAGNOSIS] Fallback to hardcoded temps: [1.0, 0.7, 0.3, 0.0]")
        temps = [1.0, 0.7, 0.3, 0.0]

    all_prompts = self.prompt_loader.get_all_independent_prompts_ordered()  # type: ignore[attr-defined]

    # Apply prompts_top filter if specified (takes priority over independent_top)
    if self.prompts_top is not None and self.prompts_top > 0:  # type: ignore[attr-defined]
        prompts_by_mode: dict[str, list[dict[str, Any]]] = {}
        for p in all_prompts:
            mode = p['mode']
            if mode not in prompts_by_mode:
                prompts_by_mode[mode] = []
            prompts_by_mode[mode].append(p)

        filtered_prompts = []
        for mode in ['architect', 'code', 'debug']:
            mode_prompts = prompts_by_mode.get(mode, [])
            filtered_prompts.extend(mode_prompts[:self.prompts_top])  # type: ignore[attr-defined]

        all_prompts = filtered_prompts
        logger.info("[prompts_top] Limited to first %d prompts per mode: %d total prompts",
                   self.prompts_top, len(all_prompts))  # type: ignore[attr-defined]
    elif self.independent_top is not None and self.independent_top > 0:  # type: ignore[attr-defined]
        prompts_by_mode: dict[str, list[dict[str, Any]]] = {}
        for p in all_prompts:
            mode = p['mode']
            if mode not in prompts_by_mode:
                prompts_by_mode[mode] = []
            prompts_by_mode[mode].append(p)

        filtered_prompts = []
        for mode in ['architect', 'code', 'debug']:
            mode_prompts = prompts_by_mode.get(mode, [])
            filtered_prompts.extend(mode_prompts[:self.independent_top])  # type: ignore[attr-defined]

        all_prompts = filtered_prompts
        logger.info("[independent_top] Limited to first %d prompts per mode: %d total prompts",
                   self.independent_top, len(all_prompts))  # type: ignore[attr-defined]

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
                def independent_token_callback(
                    prompt_tokens, response_tokens, estimated_response_tokens=0,
                    response_len=0, is_done=False, current_tps=0.0,
                    cpu_percent=0.0, ram_percent=0.0, vram_percent=0.0,
                    gpu_percent=0.0,
                ):
                    update_tokens_display(
                        prompt_tokens, response_tokens, estimated_response_tokens,
                        response_len, indent="         ", is_done=is_done,
                        current_tps=current_tps, cpu_percent=cpu_percent,
                        ram_percent=ram_percent, vram_percent=vram_percent,
                        gpu_percent=gpu_percent,
                    )

                try:
                    avg_tps, vram, tps_list, error, _, used_temp, _ = self.ollama_client.run_generation(  # type: ignore[attr-defined]
                        model_name,
                        ctx,
                        self.num_runs,  # type: ignore[attr-defined]
                        self.disable_thinking,  # type: ignore[attr-defined]
                        temperature=temp,
                        prompt=prompt,
                        prompt_metadata=prompt_metadata,
                        num_predict=self.num_predict,  # type: ignore[attr-defined]
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
                        self._store_response(store_entry)  # type: ignore[attr-defined]

                except Exception as e:
                    error_message = f"Critical error during prompt {prompt_id} testing: {e}"
                    print(f"   ❌ {error_message}")
                    self._unload_tested_model(model_name)  # type: ignore[attr-defined]
                    return None, error_message

    logger.info(
        "[Expert] run_all_independent_prompts END: model=%s _response_store_size=%d all_metrics_count=%d",
        model_name, len(self._response_store), len(all_metrics)  # type: ignore[attr-defined]
    )

    self._unload_tested_model(model_name)  # type: ignore[attr-defined]
    self._print_vram_stats(model_name)  # type: ignore[attr-defined]

    benchmark_result = BenchmarkResult(model=model, results=all_metrics)
    return benchmark_result, None
