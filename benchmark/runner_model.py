"""Single model benchmark execution methods."""

import logging
from typing import Any

from benchmark.expert_evaluator_types import ExpertEvaluationEntry
from benchmark.result import BenchmarkMetrics, BenchmarkResult, ModelInfo
from i18n import get_text
from system.restart_manager import restart_ollama
from ui.output_formatter import format_tokens_info, update_tokens_display

logger = logging.getLogger('roo_bench.benchmark')


def run_for_model(
    self: Any,  # type: ignore[reportSelfClsParameterType]
    model: ModelInfo,
    temperature: float | None = None,
) -> tuple[BenchmarkResult | None, str | None]:
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
    valid_contexts = self.filter_contexts(max_ctx)  # type: ignore[attr-defined]

    if not valid_contexts:
        print(get_text("skipping_no_contexts"))
        return BenchmarkResult(model=model, results=[]), None

    for ctx in valid_contexts:
        restart_ollama(
            self.restart_method,  # type: ignore[attr-defined]
            self.no_restart,  # type: ignore[attr-defined]
            ssh_client=self.ssh_client  # type: ignore[attr-defined]
        )

        try:
            current_num_ctx = self.ollama_client.get_current_num_ctx(model_name)  # type: ignore[attr-defined]
            print(get_text("current_num_ctx", num_ctx=current_num_ctx))
            if current_num_ctx != ctx:
                print(get_text("ctx_info_only", actual=current_num_ctx, expected=ctx))
        except Exception as e:
            print(f"   ⚠️  Error getting model settings: {e}")

        print(get_text("warming_up", ctx=ctx))

        if temperature is None:
            default_temp = self.ollama_client.get_default_temperature(model_name)  # type: ignore[attr-defined]
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
                result = self.ollama_client.run_generation(  # type: ignore[attr-defined]
                    model_name, ctx, self.num_runs, self.disable_thinking, temperature=temp,
                    num_predict=self.num_predict,  # type: ignore[attr-defined]
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
                    actual_ctx = self.ollama_client.get_actual_num_ctx(model_name)  # type: ignore[attr-defined]
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

                # Print resource statistics if available
                if resource_stats:
                    if 'cpu' in resource_stats:
                        cpu = resource_stats['cpu']
                        print(f"      CPU: {cpu.get('avg', 0):.1f}% (min: {cpu.get('min', 0):.1f}%, max: {cpu.get('max', 0):.1f}%)")
                    if 'ram' in resource_stats:
                        ram = resource_stats['ram']
                        print(f"      RAM: {ram.get('avg_percent', 0):.1f}% (min: {ram.get('min_percent', 0):.1f}%, max: {ram.get('max_percent', 0):.1f}%)")
                    if 'vram' in resource_stats:
                        vram_data = resource_stats['vram']
                        if vram_data.get('total', 0) > 0:
                            print(f"      VRAM: {vram_data.get('percent_current', 0):.1f}% (avg: {vram_data.get('avg_percent', 0):.1f}%, max: {vram_data.get('max_percent', 0):.1f}%) — {vram_data.get('used_current', 0) / 1024 / 1024:.1f} MiB / {vram_data.get('total', 0) / 1024 / 1024:.1f} MiB")
                        else:
                            print(f"      VRAM: {vram_data.get('percent_current', 0):.1f}% (avg: {vram_data.get('avg_percent', 0):.1f}%, max: {vram_data.get('max_percent', 0):.1f}%)")
                    if 'gpu' in resource_stats:
                        gpu = resource_stats['gpu']
                        print(f"      GPU: {gpu.get('avg', 0):.1f}% (min: {gpu.get('min', 0):.1f}%, max: {gpu.get('max', 0):.1f}%)")
                else:
                    print("      Resource stats: Not available")

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
                    store_entry = ExpertEvaluationEntry(  # type: ignore[name-defined]
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
                    self._store_response(store_entry)  # type: ignore[attr-defined]

            except Exception as e:
                error_message = f"Critical error during context {ctx} testing: {e}"
                print(f"   ❌ {error_message}")
                self._unload_tested_model(model_name)  # type: ignore[attr-defined]
                return None, error_message

    self._unload_tested_model(model_name)  # type: ignore[attr-defined]
    self._print_vram_stats(model_name)  # type: ignore[attr-defined]

    benchmark_result = BenchmarkResult(model=model, results=all_metrics)
    return benchmark_result, None
