"""Prompt chains benchmark execution methods.
Added debug logging to trace chain execution and used_chain_ids population.
"""

import logging
from typing import Any

from benchmark.expert_evaluator_types import ExpertEvaluationEntry
from benchmark.result import BenchmarkMetrics, BenchmarkResult, ModelInfo
from system.restart_manager import restart_ollama
from ui.output_formatter import update_tokens_display

logger = logging.getLogger('roo_bench.benchmark')


def run_chain(
    self: Any,  # type: ignore[reportSelfClsParameterType]
    model: ModelInfo,
    chain: dict,
    temperature: float | None = None,
) -> tuple[BenchmarkResult | None, str | None]:
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
        self.restart_method,  # type: ignore[attr-defined]
        self.no_restart,  # type: ignore[attr-defined]
        ssh_client=self.ssh_client  # type: ignore[attr-defined]
    )

    # Build chain context
    chain_prompts = self.prompt_loader.build_chain_context(chain)  # type: ignore[attr-defined]

    # Filter contexts
    valid_contexts = self.filter_contexts(max_ctx)  # type: ignore[attr-defined]

    # Get default temperature if not specified
    if temperature is None:
        temps = self.temperature_test_values or [1.0, 0.7, 0.3, 0.0]  # type: ignore[attr-defined]
    else:
        temps = [temperature]

    # Contexts -> Temperatures -> Chain modes
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
                def chain_token_callback(
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
                    result = self.ollama_client.run_generation(  # type: ignore[attr-defined]
                        model_name,
                        ctx,
                        self.num_runs,  # type: ignore[attr-defined]
                        self.disable_thinking,  # type: ignore[attr-defined]
                        temperature=temp,
                        prompt=prompt,
                        prompt_metadata=prompt_metadata,
                        num_predict=self.num_predict,  # type: ignore[attr-defined]
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
                        self._store_response(store_entry)  # type: ignore[attr-defined]

                except Exception as e:
                    error_message = f"Critical error during chain [{mode}] testing: {e}"
                    print(f"         ❌ {error_message}")
                    self._unload_tested_model(model_name)  # type: ignore[attr-defined]
                    return None, error_message

    self._unload_tested_model(model_name)  # type: ignore[attr-defined]
    self._print_vram_stats(model_name)  # type: ignore[attr-defined]

    benchmark_result = BenchmarkResult(model=model, results=all_metrics)
    return benchmark_result, None


def run_all_chains(
    self: Any,  # type: ignore[reportSelfClsParameterType]
    model: ModelInfo,
    temperature: float | None = None,
) -> tuple[BenchmarkResult | None, str | None]:
    """Run ALL chains for a model.

    Execution order (NEW):
        Model -> Contexts -> Temperatures -> Chains -> Modes

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
        self.restart_method,  # type: ignore[attr-defined]
        self.no_restart,  # type: ignore[attr-defined]
        ssh_client=self.ssh_client  # type: ignore[attr-defined]
    )

    valid_contexts = self.filter_contexts(max_ctx)  # type: ignore[attr-defined]

    if temperature is None:
        temps = self.temperature_test_values or [1.0, 0.7, 0.3, 0.0]  # type: ignore[attr-defined]
    else:
        temps = [temperature]

    chains = self.prompt_loader.get_chains()  # type: ignore[attr-defined]
    logger.info("[DEBUG] run_all_chains: found %d chains: %s", len(chains), [c.get('id') for c in chains])
    if not chains:
        logger.warning("[DEBUG] run_all_chains: NO CHAINS FOUND - skipping chain tests")
        return

    # NEW ORDER: Contexts -> Temperatures -> Chains -> Modes
    for ctx in valid_contexts:
        logger.info(f"   📏 Context size: {ctx}")
        print(f"      📏 Context: {ctx // 1024}K" if ctx >= 1024 else f"      📏 Context: {ctx}")

        for temp in temps:
            logger.info(f"   🌡️  Temperature: {temp:.2f}")
            print(f"         🌡️  Temperature: {temp:.2f}")

            for chain in chains:
                chain_id = chain.get('id', 'unknown')
                chain_name = chain.get('name', 'Unknown Chain')
                logger.info("[DEBUG] Processing chain: %s", chain_id)

                print(f"\n      🔗 Chain: {chain_name} ({chain_id})")
                logger.info(f"🔗 Chain: {chain_name} ({chain_id})")

                chain_prompts = self.prompt_loader.build_chain_context(chain)  # type: ignore[attr-defined]

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
                    def chain_token_callback_2(
                        prompt_tokens, response_tokens, estimated_response_tokens=0,
                        response_len=0, is_done=False, current_tps=0.0,
                        cpu_percent=0.0, ram_percent=0.0, vram_percent=0.0,
                        gpu_percent=0.0,
                    ):
                        update_tokens_display(
                            prompt_tokens, response_tokens, estimated_response_tokens,
                            response_len, indent="            ", is_done=is_done,
                            current_tps=current_tps, cpu_percent=cpu_percent,
                            ram_percent=ram_percent, vram_percent=vram_percent,
                            gpu_percent=gpu_percent,
                        )

                    try:
                        avg_tps, vram, tps_list, error, _, used_temp, resource_stats = self.ollama_client.run_generation(  # type: ignore[attr-defined]
                            model_name,
                            ctx,
                            self.num_runs,  # type: ignore[attr-defined]
                            self.disable_thinking,  # type: ignore[attr-defined]
                            temperature=temp,
                            prompt=prompt,
                            prompt_metadata=prompt_metadata,
                            num_predict=self.num_predict,  # type: ignore[attr-defined]
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
                            logger.info("[DEBUG] Storing response: model=%s prompt_id=%s mode=%s chain_id=%s", model_name, prompt_id, mode, chain_id)
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
                            self._store_response(store_entry)  # type: ignore[attr-defined]

                    except Exception as e:
                        error_message = f"Critical error during chain [{mode}] testing: {e}"
                        print(f"            ❌ {error_message}")
                        self._unload_tested_model(model_name)  # type: ignore[attr-defined]
                        return None, error_message

    self._unload_tested_model(model_name)  # type: ignore[attr-defined]

    # Debug: log total responses stored for this model
    total_stored = len([e for e in self._response_store if e.model_name == model_name])  # type: ignore[attr-defined]
    logger.info("[DEBUG] run_all_chains completed for %s: total responses in store=%d, chain results collected=%d", model_name, total_stored, len(all_metrics))

    self._print_vram_stats(model_name)  # type: ignore[attr-defined]
    benchmark_result = BenchmarkResult(model=model, results=all_metrics)
    return benchmark_result, None
