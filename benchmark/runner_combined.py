"""Combined independent + chains benchmark execution for --all mode.

This module provides a unified execution order:
    Model -> Contexts -> Temperatures -> Independent prompts -> Chains

This ensures that for each Context/Temperature combination,
independent tests run first, followed immediately by chain tests.
"""

import logging
from typing import Any

from benchmark.expert_evaluator_types import ExpertEvaluationEntry
from benchmark.result import BenchmarkMetrics, BenchmarkResult, ModelInfo
from system.restart_manager import restart_ollama
from ui.output_formatter import update_tokens_display

logger = logging.getLogger('roo_bench.benchmark')


def _build_chain_token_callback(indent: str = "            ") -> Any:
    """Create a token update callback for chain tests."""
    def callback(
        prompt_tokens: int,
        response_tokens: int,
        estimated_response_tokens: int = 0,
        response_len: int = 0,
        is_done: bool = False,
        current_tps: float = 0.0,
        cpu_percent: float = 0.0,
        ram_percent: float = 0.0,
        vram_percent: float = 0.0,
        gpu_percent: float = 0.0,
    ) -> None:
        update_tokens_display(
            prompt_tokens, response_tokens, estimated_response_tokens, response_len,
            indent=indent, is_done=is_done, current_tps=current_tps,
            cpu_percent=cpu_percent, ram_percent=ram_percent,
            vram_percent=vram_percent, gpu_percent=gpu_percent,
        )
    return callback  # type: ignore[return-value]


def _run_chain_mode(
    self: Any,  # type: ignore[reportSelfClsParameterType]
    model: ModelInfo,
    chain: dict,
    ctx: int,
    temp: float,
) -> tuple[list[BenchmarkMetrics], str | None]:
    """Run a single chain for a specific context and temperature.

    Args:
        self: BenchmarkRunner instance
        model: ModelInfo instance
        chain: Chain dictionary
        ctx: Context size
        temp: Temperature value

    Returns:
        tuple: (list of BenchmarkMetrics, error_message or None)
    """
    model_name = model.name
    chain_id = chain.get('id', 'unknown')
    chain_name = chain.get('name', 'Unknown Chain')
    chain_metrics: list[BenchmarkMetrics] = []
    error_message: str | None = None

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

        on_token_update = _build_chain_token_callback()

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
                on_token_update=on_token_update
            )

            if error:
                print(f"            Error: {error}")
                break

            duration = tps_list[0].get('total_duration', 0) / 1e9
            prompt_tokens = tps_list[0].get('prompt_eval_count', 0)
            response_tokens = tps_list[0].get('eval_count', 0)
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
            chain_metrics.append(metrics)

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
            return [], error_message

    return chain_metrics, None


def _run_independent_prompts_for_ctx_temp(
    self: Any,  # type: ignore[reportSelfClsParameterType]
    model: ModelInfo,
    ctx: int,
    temp: float,
    all_prompts: list[dict[str, Any]],
) -> tuple[list[BenchmarkMetrics], str | None]:
    """Run independent prompts for a specific context and temperature.

    Args:
        self: BenchmarkRunner instance
        model: ModelInfo instance
        ctx: Context size
        temp: Temperature value
        all_prompts: List of prompt dictionaries

    Returns:
        tuple: (list of BenchmarkMetrics, error_message or None)
    """
    model_name = model.name
    prompt_metrics: list[BenchmarkMetrics] = []
    error_message: str | None = None

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

        logger.info(f"📝 Running prompt: {prompt_name} (ID: {prompt_id})")
        print(f"         Running prompt: {prompt_name} (ID: {prompt_id})")
        logger.debug(f"📝 Prompt text: {prompt}")

        current_tokens = {'prompt': 0, 'response': 0}

        def token_callback(
            prompt_tokens: int,
            response_tokens: int,
            estimated_response_tokens: int = 0,
            response_len: int = 0,
            is_done: bool = False,
            current_tps: float = 0.0,
            cpu_percent: float = 0.0,
            ram_percent: float = 0.0,
            vram_percent: float = 0.0,
            gpu_percent: float = 0.0,
        ) -> None:
            current_tokens['prompt'] = prompt_tokens
            current_tokens['response'] = response_tokens
            update_tokens_display(
                prompt_tokens, response_tokens, estimated_response_tokens, response_len,
                indent="         ", is_done=is_done, current_tps=current_tps,
                cpu_percent=cpu_percent, ram_percent=ram_percent,
                vram_percent=vram_percent, gpu_percent=gpu_percent,
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
                on_token_update=token_callback
            )

            # Handle both old and new return formats
            if len(result) == 6:
                avg_tps, vram, tps_list, error, _, used_temp = result
                resource_stats = None
            else:
                avg_tps, vram, tps_list, error, _, used_temp, resource_stats = result

            if error:
                print(f"         Error running prompt {prompt_id}: {error}")
                continue

            if tps_list and tps_list[0].get('response'):
                response = tps_list[0]['response']
                logger.info(f"   🤖 Model response (first 500 chars): {response[:500]}...")

            duration = tps_list[0].get('total_duration', 0) / 1e9
            prompt_tokens = tps_list[0].get('prompt_eval_count', 0)
            response_tokens = tps_list[0].get('eval_count', 0)
            print(f"            Duration: {duration:.2f}s | Prompt: {prompt_tokens} | Response: {response_tokens}")
            print()

            if len(tps_list) > 1:
                mean = avg_tps
                variance = sum((run['tps'] - mean) ** 2 for run in tps_list) / len(tps_list)
                std_dev = variance ** 0.5
            else:
                std_dev = 0.0

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
                        print(f"            VRAM: {vram_data.get('percent_current', 0):.1f}% (avg: {vram_data.get('avg_percent', 0):.1f}%, max: {vram_data.get('max_percent', 0):.1f}%) — {vram_data.get('used_current', 0) / 1024 / 1024:.1f} MiB / {vram_data.get('total', 0) / 1024 / 1024:.1f} MiB")
                    else:
                        print(f"            VRAM: {vram_data.get('percent_current', 0):.1f}% (avg: {vram_data.get('avg_percent', 0):.1f}%, max: {vram_data.get('max_percent', 0):.1f}%)")
                if 'gpu' in resource_stats:
                    gpu = resource_stats['gpu']
                    print(f"            GPU: {gpu.get('avg', 0):.1f}% (min: {gpu.get('min', 0):.1f}%, max: {gpu.get('max', 0):.1f}%)")

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
            prompt_metrics.append(metrics)

            logger.info(
                "[Expert] run_combined store check: model=%s prompt_id=%s mode=%s response_len=%d",
                model_name, prompt_id, mode,
                len(tps_list[0].get('response', '')) if tps_list else 0,
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
            error_message = f"Critical error during {prompt_id} testing: {e}"
            print(f"         ❌ {error_message}")
            self._unload_tested_model(model_name)  # type: ignore[attr-defined]
            return [], error_message

    return prompt_metrics, None


def run_all_combined(
    self: Any,  # type: ignore[reportSelfClsParameterType]
    model: ModelInfo,
    temperature: float | None = None,
) -> tuple[BenchmarkResult | None, str | None]:
    """Run ALL independent prompts followed by ALL chains for a model.

    Execution order (NEW):
        Model -> Contexts -> Temperatures -> Independent prompts -> Chains

    This ensures that for each Context/Temperature combination,
    independent tests run first, followed immediately by chain tests.

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
        "[Expert] run_all_combined START: expert_evaluator=%s _response_store_size=%d",
        self.expert_evaluator is not None, len(self._response_store)  # type: ignore[attr-defined]
    )

    restart_ollama(
        self.restart_method,  # type: ignore[attr-defined]
        self.no_restart,  # type: ignore[attr-defined]
        ssh_client=self.ssh_client  # type: ignore[attr-defined]
    )

    valid_contexts = self.filter_contexts(max_ctx)  # type: ignore[attr-defined]

    # Get temperature values
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

    # Get all independent prompts
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
        logger.info("[prompts_top] Limited independent to first %d prompts per mode: %d total prompts",
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

    # Get all chains
    chains = self.prompt_loader.get_chains()  # type: ignore[attr-defined]

    # Apply prompts_top filter to chains (takes priority over chunks_top)
    if self.prompts_top is not None and self.prompts_top > 0:  # type: ignore[attr-defined]
        chains = chains[:self.prompts_top]  # type: ignore[attr-defined]
        logger.info("[prompts_top] Limited chains to first %d: %d total chains",
                   self.prompts_top, len(chains))  # type: ignore[attr-defined]
    elif self.chunks_top is not None and self.chunks_top > 0:  # type: ignore[attr-defined]
        chains = chains[:self.chunks_top]  # type: ignore[attr-defined]
        logger.info("[chunks_top] Limited chains to first %d: %d total chains",
                   self.chunks_top, len(chains))  # type: ignore[attr-defined]

    logger.info("[DEBUG] run_all_combined: found %d chains: %s", len(chains), [c.get('id') for c in chains])

    logger.debug(
        "[Expert] run_all_combined: model=%s all_prompts_count=%d valid_contexts=%s",
        model_name, len(all_prompts), valid_contexts
    )

    # NEW ORDER: Contexts -> Temperatures -> Independent -> Chains
    for ctx in valid_contexts:
        logger.info(f"   📏 Context size: {ctx}")
        print(f"   📏 Context: {ctx // 1024}K" if ctx >= 1024 else f"   📏 Context: {ctx}")

        for temp in temps:
            logger.info(f"   🌡️  Temperature: {temp:.2f}")
            print(f"      🌡️  Temperature: {temp:.2f}")

            # Step 1: Run independent prompts for this Context/Temperature
            prompt_metrics, prompt_error = _run_independent_prompts_for_ctx_temp(
                self, model, ctx, temp, all_prompts
            )
            all_metrics.extend(prompt_metrics)
            if prompt_error:
                error_message = prompt_error
                break

            # Step 2: Run all chains for this Context/Temperature
            for chain in chains:
                chain_metrics, chain_error = _run_chain_mode(self, model, chain, ctx, temp)  # type: ignore[arg-type]
                all_metrics.extend(chain_metrics)
                if chain_error:
                    error_message = chain_error
                    break

            if error_message:
                break

        if error_message:
            break

    self._unload_tested_model(model_name)  # type: ignore[attr-defined]

    # Debug: log total responses stored for this model
    total_stored = len([e for e in self._response_store if e.model_name == model_name])  # type: ignore[attr-defined]
    logger.info("[DEBUG] run_all_combined completed for %s: total responses in store=%d, results collected=%d", model_name, total_stored, len(all_metrics))

    self._print_vram_stats(model_name)  # type: ignore[attr-defined]
    benchmark_result = BenchmarkResult(model=model, results=all_metrics)
    return benchmark_result, None
