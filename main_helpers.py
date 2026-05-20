"""Helper functions for main workflow in roo_bench."""

import os
from argparse import Namespace
from typing import Any

from benchmark.result import BenchmarkResult, ModelInfo
from benchmark.runner import BenchmarkRunner
from export.merge_utils import (
    load_run_config,
    merge_results,
    model_exists_in_results,
    run_configs_match,
    save_results_file,
)
from export.retest_dialog import (
    RetestDecision,
    prompt_retest_decision,
    should_skip_model,
    should_stop_testing,
)

# Default output file for benchmark results
DEFAULT_OUTPUT_FILE = "benchmark_results.json"

SAVE_MODE_DISABLED = "disabled"
SAVE_MODE_NEW = "new"
SAVE_MODE_OVERWRITE = "overwrite"
SAVE_MODE_MERGE = "merge"


def _minimal_prompt(prompt: dict[str, Any]) -> dict[str, Any]:
    """Keep only prompt fields needed in benchmark_results.json."""
    return {
        'id': prompt.get('id'),
        'name': prompt.get('name'),
        'prompt': prompt.get('prompt'),
    }


def _minimal_chain(chain: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': chain.get('id'),
        'name': chain.get('name'),
        'prompts': {
            mode: _minimal_prompt(prompt)
            for mode, prompt in chain.get('prompts', {}).items()
        }
    }


def build_used_prompts_config(
    prompt_loader: Any,
    args: Namespace,
    benchmark_runner: BenchmarkRunner,
) -> dict[str, Any] | None:
    """Build prompts_config containing only prompts used by this run."""
    if not prompt_loader:
        return None

    if args.chain:
        chain = prompt_loader.get_chain_by_id(args.chain)
        if not chain:
            return None
        return {'chains': [_minimal_chain(chain)]}

    if args.chains:
        return {
            'chains': [_minimal_chain(chain) for chain in prompt_loader.get_chains()]
        }

    if args.independent or prompt_loader.data.get('independent'):
        independent: dict[str, list[dict[str, Any]]] = {}
        for prompt in benchmark_runner.get_used_independent_prompts():
            mode = prompt.get('mode')
            if not mode:
                continue
            independent.setdefault(mode, []).append(_minimal_prompt(prompt))
        return {'independent': independent}

    return None


def build_run_config(
    prompt_loader: Any,
    args: Namespace,
    benchmark_runner: BenchmarkRunner,
    test_models: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build effective run config used for merge compatibility."""
    context_sizes: set[int] = set()
    for model in test_models:
        context_sizes.update(benchmark_runner.filter_contexts(model.get('max_ctx', 131072)))

    used_prompt_ids: list[str] = []
    used_chain_ids: list[str] = []

    if args.chain:
        chain = prompt_loader.get_chain_by_id(args.chain) if prompt_loader else None
        if chain and chain.get('id'):
            used_chain_ids = [chain['id']]
    elif args.chains:
        used_chain_ids = benchmark_runner.get_used_chain_ids()
    elif args.independent or (prompt_loader and prompt_loader.data.get('independent')):
        used_prompt_ids = benchmark_runner.get_used_prompt_ids()

    return {
        'context_sizes': sorted(context_sizes),
        'temperature_test_values': sorted(benchmark_runner.temperature_test_values or []),
        'used_prompt_ids': sorted(used_prompt_ids),
        'used_chain_ids': sorted(used_chain_ids),
    }


def _prompt_existing_results_action(results_file: str, can_merge: bool) -> str:
    """Ask what to do with an existing results file before tests start."""
    print(f"\nResults file already exists: {results_file}")
    if can_merge:
        print("Compatible run config found. Choose: [m]erge, [o]verwrite, [a]bort")
        valid = {
            "m": SAVE_MODE_MERGE, "merge": SAVE_MODE_MERGE,
            "o": SAVE_MODE_OVERWRITE, "overwrite": SAVE_MODE_OVERWRITE,
            "a": "abort", "abort": "abort",
        }
    else:
        print("Run config is not compatible for merge. Choose: [o]verwrite, [a]bort")
        valid = {
            "o": SAVE_MODE_OVERWRITE, "overwrite": SAVE_MODE_OVERWRITE,
            "a": "abort", "abort": "abort",
        }

    while True:
        response = input("> ").strip().lower()
        if response in valid:
            return valid[response]
        print("Please enter one of the listed options.")


def prepare_results_output_file(
    results_file: str,
    run_config: dict[str, Any],
    prompts_config: dict[str, Any],
) -> str:
    """Prepare result-file save mode before benchmark starts."""
    if not results_file:
        return SAVE_MODE_DISABLED

    if not os.path.exists(results_file):
        return SAVE_MODE_NEW

    existing_run_config = load_run_config(results_file)
    can_merge = bool(existing_run_config) and run_configs_match(
        existing_run_config, run_config
    )
    action = _prompt_existing_results_action(results_file, can_merge)
    if action == "abort":
        print("Benchmark aborted before tests started.")
        return "abort"

    if action == SAVE_MODE_OVERWRITE:
        save_results_file(
            results_file, [], prompts_config=prompts_config, run_config=run_config
        )
    return action


def persist_model_result(
    save_mode: str,
    results_file: str,
    benchmark_result: BenchmarkResult,
    all_results: list[BenchmarkResult],
    prompts_config: dict[str, Any],
    run_config: dict[str, Any],
) -> None:
    """Persist one finished model according to selected save mode."""
    if save_mode == SAVE_MODE_DISABLED or not results_file or not benchmark_result:
        return

    if save_mode == SAVE_MODE_MERGE:
        merge_results(
            results_file, benchmark_result,
            prompts_config=prompts_config, run_config=run_config
        )
    else:
        save_results_file(
            results_file, all_results,
            prompts_config=prompts_config, run_config=run_config
        )
    print(
        f"   💾 Results saved atomically after model: {benchmark_result.model.name}"
    )


def make_model_info(model_data: dict[str, Any]) -> ModelInfo:
    """Convert discovered model dict to ModelInfo."""
    from benchmark.result import Capability

    moe_val = model_data.get('moe')
    moe_dict = moe_val if isinstance(moe_val, dict) else None
    return ModelInfo(
        name=model_data['name'],
        size_gb=float(model_data['size_gb'])
        if model_data.get('size_gb') and model_data.get('size_gb') != 'N/A'
        else 0.0,
        params=model_data.get('params', 'N/A'),
        quant=model_data.get('quant', 'N/A'),
        architecture=model_data.get('architecture', 'N/A'),
        max_ctx=model_data.get('max_ctx', 131072),
        moe=moe_dict,
        vision=Capability.VISION if model_data.get('vision') == '✅' else Capability.TOOLS,
        tools=Capability.TOOLS if model_data.get('tools') == '✅' else Capability.THINKING,
        thinking=Capability.THINKING if model_data.get('thinking') == '✅' else Capability.AUDIO,
    )


def _check_model_retest(
    model_name: str,
    tested_count: int,
    total_count: int,
    results_file: str,
    decision_state: dict[str, bool],
) -> tuple[bool, bool]:
    """Check if model exists in results and prompt for retest decision."""
    if decision_state.get('skip_all', False):
        return False, False
    if decision_state.get('test_all', False):
        return True, False

    if not model_exists_in_results(results_file, model_name):
        return True, False

    decision = prompt_retest_decision(model_name, tested_count, total_count)

    if should_skip_model(decision, tested_count, total_count):
        if should_stop_testing(decision):
            return False, True
        return False, False

    if decision == RetestDecision.YES_ALL:
        decision_state['test_all'] = True
    elif decision == RetestDecision.NO_ALL:
        decision_state['skip_all'] = True

    return True, False


def collect_model_data(
    m: dict[str, Any],
    cached: dict[str, Any] | None,
    use_cache: bool,
    ollama_client: Any,
    capabilities_fetcher: Any,
) -> dict[str, Any]:
    """Collect and enrich model data from cache or API."""
    if cached and use_cache:
        cached_size_gb = cached.get('size_gb')
        if cached_size_gb == "N/A":
            size_bytes = m.get("size", 0)
            m["size_gb"] = round(size_bytes / (1024**3), 1) if size_bytes > 0 else "N/A"
        else:
            m["size_gb"] = cached_size_gb
        m["params"] = cached.get('params', 'N/A')
        m["quant"] = cached.get('quant', 'N/A')
        m["architecture"] = cached.get('architecture', 'N/A')
        m["max_ctx"] = cached.get('max_ctx', 131072)
        m["moe"] = cached.get('moe', None)
        caps = cached.get(
            'capabilities',
            {'vision': False, 'tools': False, 'thinking': False, 'audio': False}
        )
        if isinstance(m["size_gb"], (int, float)):
            m["size_gb"] = round(m["size_gb"], 1)
    else:
        size_bytes = m.get("size", 0)
        m["size_gb"] = round(size_bytes / (1024**3), 1) if size_bytes > 0 else "N/A"

        details = m.get("details", {})
        m["params"] = details.get("parameter_size", "N/A") if details else "N/A"

        m["quant"] = (
            (details.get("quantization_level")
             or details.get("quantization_format")
             or "N/A") if details else "N/A"
        )

        try:
            model_info = ollama_client.get_model_info(m["name"])
            architecture = (
                model_info.get('model_info', {}).get('general.architecture', 'N/A')
            )
            m["architecture"] = architecture if architecture else 'N/A'
        except Exception:
            m["architecture"] = 'N/A'

        try:
            model_info = ollama_client.get_model_info(m["name"])
            max_ctx = 131072
            caps = {'vision': False, 'tools': False, 'thinking': False, 'audio': False}
            if model_info:
                top_level_caps = model_info.get("capabilities", [])
                if isinstance(top_level_caps, list) and len(top_level_caps) > 0:
                    caps_list_str = ' '.join(top_level_caps).lower()
                    caps = {
                        'vision': 'vision' in caps_list_str or 'image' in caps_list_str,
                        'tools': 'tools' in caps_list_str or 'function' in caps_list_str,
                        'thinking': 'thinking' in caps_list_str or 'reason' in caps_list_str,
                        'audio': 'audio' in caps_list_str or 'whisper' in caps_list_str,
                    }
                else:
                    model_info_dict = model_info.get("model_info", {})
                    caps = ollama_client.get_capabilities_from_model_info(model_info_dict)

                params_str = model_info.get("parameters", "")
                if params_str:
                    for line in params_str.split('\n'):
                        line = line.strip()
                        if line.startswith('num_ctx:'):
                            try:
                                max_ctx = int(line.split(':')[1].strip())
                            except (ValueError, IndexError):
                                pass

                model_info_dict = model_info.get("model_info", {})
                for key, val in model_info_dict.items():
                    if 'context_length' in key.lower() or 'num_ctx' in key.lower():
                        try:
                            max_ctx = int(val)
                        except (ValueError, TypeError):
                            pass

            m["max_ctx"] = max_ctx
            capabilities_fetcher.add_model_metadata(m["name"], model_info)
        except Exception:
            m["max_ctx"] = 131072
            caps = {'vision': False, 'tools': False, 'thinking': False, 'audio': False}

    m["vision"] = "✅" if caps.get("vision", False) else "❌"
    m["tools"] = "✅" if caps.get("tools", False) else "❌"
    m["thinking"] = "✅" if caps.get("thinking", False) else "❌"
    capabilities_fetcher.add_model_from_api(m["name"], caps)
    return m
