"""Main entry point for roo_bench - orchestration of all modules."""

import sys
import os
import signal
import logging
import curses
from i18n import set_language, get_text
from config import OllamaConfig
from cli import parse_args, get_context_sizes, get_temperature_test_values
from constants import CONTEXT_SIZES, DEFAULT_OLLAMA_URL
from api.factory import ApiClientFactory
from system.restart_manager import restart_ollama, RestartMethod
from system.gpu_monitor import check_gpu_available, get_vram_usage
from benchmark.runner import BenchmarkRunner
from benchmark.results import calculate_statistics, format_result_row, format_recommendations
from benchmark.result import ModelInfo, BenchmarkResult, Capability
from ui.curses_selector import interactive_model_select, select_expert_model
from ui.output_formatter import print_model_list, print_results_table
from export.result_saver import save_results, ResultSaver
from export.retest_dialog import RetestDecision, prompt_retest_decision, should_skip_model, should_stop_testing
from export.merge_utils import (
    load_run_config,
    merge_results,
    model_exists_in_results,
    run_configs_match,
    save_results_file,
)
from export.expert_results import ExpertResultsManager
from prompts.loader import PromptLoader

# Setup logging
logger = logging.getLogger('roo_bench')

# Default output file for benchmark results
DEFAULT_OUTPUT_FILE = "benchmark_results.json"


SAVE_MODE_DISABLED = "disabled"
SAVE_MODE_NEW = "new"
SAVE_MODE_OVERWRITE = "overwrite"
SAVE_MODE_MERGE = "merge"


def validate_expert_prompts() -> bool:
    """Check if expert evaluation prompts are available.

    Returns:
        bool: True if prompts file exists and is readable, False otherwise.
    """
    if os.path.exists('prompts/analysis_prompt.jsonc'):
        return True
    logger.warning(
        "Expert evaluation prompts not found. Expert-Evaluator will use default templates."
    )
    return False


def prompt_user(message: str) -> bool:
    """Prompt user for yes/no confirmation.

    Args:
        message: Message to display

    Returns:
        bool: True if user confirms, False otherwise
    """
    response = input(f"{message} (y/n): ").strip().lower()
    return response in ['y', 'yes']


def prompt_output_filename(default: str = DEFAULT_OUTPUT_FILE) -> str:
    """Prompt user for output filename with default value.
    
    Args:
        default: Default filename to use if user presses Enter
        
    Returns:
        str: The filename entered by user or default
    """
    try:
        response = input(f"\n{get_text('ask_save_filename')} ").strip()
        return response if response else default
    except (EOFError, KeyboardInterrupt):
        return default


def _prompt_existing_results_action(results_file: str, can_merge: bool) -> str:
    """Ask what to do with an existing results file before tests start."""
    print(f"\nResults file already exists: {results_file}")
    if can_merge:
        print("Compatible run config found. Choose: [m]erge, [o]verwrite, [a]bort")
        valid = {"m": SAVE_MODE_MERGE, "merge": SAVE_MODE_MERGE,
                 "o": SAVE_MODE_OVERWRITE, "overwrite": SAVE_MODE_OVERWRITE,
                 "a": "abort", "abort": "abort"}
    else:
        print("Run config is not compatible for merge. Choose: [o]verwrite, [a]bort")
        valid = {"o": SAVE_MODE_OVERWRITE, "overwrite": SAVE_MODE_OVERWRITE,
                 "a": "abort", "abort": "abort"}

    while True:
        response = input("> ").strip().lower()
        if response in valid:
            return valid[response]
        print("Please enter one of the listed options.")


def _minimal_prompt(prompt: dict) -> dict:
    """Keep only prompt fields needed in benchmark_results.json."""
    return {
        'id': prompt.get('id'),
        'name': prompt.get('name'),
        'prompt': prompt.get('prompt'),
    }


def build_used_prompts_config(prompt_loader: PromptLoader, args, benchmark_runner: BenchmarkRunner) -> dict:
    """Build prompts_config containing only prompts used by this run."""
    if not prompt_loader:
        return None

    if args.chain:
        chain = prompt_loader.get_chain_by_id(args.chain)
        if not chain:
            return None
        return {
            'chains': [_minimal_chain(chain)]
        }

    if args.chains:
        return {
            'chains': [_minimal_chain(chain) for chain in prompt_loader.get_chains()]
        }

    if args.independent or prompt_loader.data.get('independent'):
        independent = {}
        for prompt in benchmark_runner.get_used_independent_prompts():
            mode = prompt.get('mode')
            if not mode:
                continue
            independent.setdefault(mode, []).append(_minimal_prompt(prompt))
        return {'independent': independent}

    return None


def _minimal_chain(chain: dict) -> dict:
    return {
        'id': chain.get('id'),
        'name': chain.get('name'),
        'prompts': {
            mode: _minimal_prompt(prompt)
            for mode, prompt in chain.get('prompts', {}).items()
        }
    }


def build_run_config(prompt_loader: PromptLoader, args, benchmark_runner: BenchmarkRunner,
                     test_models: list) -> dict:
    """Build effective run config used for merge compatibility."""
    context_sizes = set()
    for model in test_models:
        context_sizes.update(benchmark_runner.filter_contexts(model.get('max_ctx', 131072)))

    used_prompt_ids = []
    used_chain_ids = []

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


def prepare_results_output_file(results_file: str, run_config: dict,
                                prompts_config: dict) -> str:
    """Prepare result-file save mode before benchmark starts."""
    if not results_file:
        return SAVE_MODE_DISABLED

    if not os.path.exists(results_file):
        return SAVE_MODE_NEW

    existing_run_config = load_run_config(results_file)
    can_merge = bool(existing_run_config) and run_configs_match(existing_run_config, run_config)
    action = _prompt_existing_results_action(results_file, can_merge)
    if action == "abort":
        print("Benchmark aborted before tests started.")
        return "abort"

    if action == SAVE_MODE_OVERWRITE:
        save_results_file(results_file, [], prompts_config=prompts_config, run_config=run_config)
    return action


def persist_model_result(save_mode: str, results_file: str, benchmark_result: BenchmarkResult,
                         all_results: list, prompts_config: dict, run_config: dict) -> None:
    """Persist one finished model according to selected save mode."""
    if save_mode == SAVE_MODE_DISABLED or not results_file or not benchmark_result:
        return

    if save_mode == SAVE_MODE_MERGE:
        merge_results(results_file, benchmark_result, prompts_config=prompts_config, run_config=run_config)
    else:
        save_results_file(results_file, all_results, prompts_config=prompts_config, run_config=run_config)
    print(f"   💾 Results saved atomically after model: {benchmark_result.model.name}")


def make_model_info(model_data: dict) -> ModelInfo:
    """Convert discovered model dict to ModelInfo."""
    moe_val = model_data.get('moe')
    moe_dict = moe_val if isinstance(moe_val, dict) else None
    return ModelInfo(
        name=model_data['name'],
        size_gb=float(model_data['size_gb'])
        if model_data.get('size_gb') and model_data.get('size_gb') != 'N/A' else 0.0,
        params=model_data.get('params', 'N/A'),
        quant=model_data.get('quant', 'N/A'),
        architecture=model_data.get('architecture', 'N/A'),
        max_ctx=model_data.get('max_ctx', 131072),
        moe=moe_dict,
        vision=Capability.VISION if model_data.get('vision') == '✅' else Capability.TOOLS,
        tools=Capability.TOOLS if model_data.get('tools') == '✅' else Capability.THINKING,
        thinking=Capability.THINKING if model_data.get('thinking') == '✅' else Capability.AUDIO,
    )


def _mode_from_metric(metric) -> str:
    """Return recommendation mode for a metric."""
    if metric.mode in ('architect', 'code', 'debug'):
        return metric.mode
    if metric.ctx >= 65536:
        return 'architect'
    if metric.ctx >= 16384:
        return 'code'
    return 'debug'


def _recommendation_score(mode: str, metric) -> float:
    """Score a metric for mode-specific recommendations."""
    expert = metric.expert_score if metric.expert_score is not None else 50.0
    ctx_score = min(metric.ctx / 131072, 2.0)
    speed_score = min(metric.avg_tps / 50, 2.0)

    if mode == 'architect':
        return expert * 10 + ctx_score * 25 + speed_score * 5
    if mode == 'code':
        return expert * 10 + speed_score * 30 + ctx_score * 8
    return expert * 10 + speed_score * 18 + ctx_score * 18


def _build_mode_recommendations(all_results: list, test_models: list) -> dict:
    """Build top model recommendations per Roo mode."""
    model_lookup = {m['name']: m for m in test_models}
    best_by_mode_and_model = {
        'architect': {},
        'code': {},
        'debug': {},
    }

    for result in all_results:
        model_name = result.model_name
        for metric in result.results:
            mode = _mode_from_metric(metric)
            current = best_by_mode_and_model[mode].get(model_name)
            if current is None or _recommendation_score(mode, metric) > _recommendation_score(mode, current):
                best_by_mode_and_model[mode][model_name] = metric

    recommendations = {}
    for mode, model_metrics in best_by_mode_and_model.items():
        ranked = []
        for model_name, metric in model_metrics.items():
            model_data = model_lookup.get(model_name, {})
            ranked.append({
                'model_name': model_name,
                'metric': metric,
                'model_data': model_data,
                'score': _recommendation_score(mode, metric),
            })
        ranked.sort(key=lambda item: item['score'], reverse=True)
        recommendations[mode] = ranked[:3]

    return recommendations


def _format_ctx(ctx: int) -> str:
    return f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)


def _recommendation_reason(mode: str, metric) -> str:
    """Explain why a metric is recommended for a mode."""
    score_text = (
        f"expert score {metric.expert_score:.1f}/100"
        if metric.expert_score is not None else "no expert score, using performance fallback"
    )
    ctx_text = f"context {_format_ctx(metric.ctx)}"
    speed_text = f"{metric.avg_tps:.1f} tok/s"

    if mode == 'architect':
        return f"{score_text}; strongest fit favors large {ctx_text} for project-wide analysis, with {speed_text} generation."
    if mode == 'code':
        return f"{score_text}; best coding fit balances response quality with high generation speed ({speed_text}) at {ctx_text}."
    return f"{score_text}; debug fit balances enough {ctx_text} for traces/logs with practical speed ({speed_text})."


def print_expert_recommendations(all_results: list, test_models: list) -> None:
    """Print top 3 expert-guided recommendations for Architect, Coder, and Debug."""
    recommendations = _build_mode_recommendations(all_results, test_models)
    mode_titles = [
        ('architect', get_text('architect_mode')),
        ('code', get_text('code_mode')),
        ('debug', get_text('debug_mode')),
    ]

    print("\n" + "="*60)
    print(get_text("recommendations_header"))
    print("="*60)

    for mode, title in mode_titles:
        print(f"\n  {title}")
        recs = recommendations.get(mode, [])
        if not recs:
            print("    No completed tests for this mode.")
            continue

        for rank, rec in enumerate(recs, 1):
            metric = rec['metric']
            model_data = rec['model_data']
            model_name = rec['model_name']
            params = model_data.get('params', 'N/A')
            quant = model_data.get('quant', 'N/A')
            size_gb = model_data.get('size_gb', 'N/A')
            score = f"{metric.expert_score:.1f}/100" if metric.expert_score is not None else "N/A"
            prefix = "  ★" if rank == 1 else "   "

            print(f"{prefix} {rank}. {model_name} ({params}, {quant}, {size_gb} GB)")
            print(
                f"      Recommended: ctx={_format_ctx(metric.ctx)}, "
                f"temperature={metric.temperature:.2f}, "
                f"speed={metric.avg_tps:.1f} tok/s, expert_score={score}"
            )
            print(f"      Why: {_recommendation_reason(mode, metric)}")


def get_ollama_config(args) -> OllamaConfig:
    """Get Ollama configuration from CLI arguments.

    Args:
        args: Parsed arguments from argparse

    Returns:
        OllamaConfig: Configuration object
    """
    cli_config = {
        'ollama_url': getattr(args, 'ollama_url', None),
        'ollama_port': getattr(args, 'ollama_port', None),
        'ollama_api_key': getattr(args, 'ollama_api_key', None),
        'ollama_timeout': getattr(args, 'ollama_timeout', None),
        'config': getattr(args, 'config', None)
    }
    return OllamaConfig(cli_config)


def run_benchmark_workflow(config: OllamaConfig, args):
    """Main benchmark workflow orchestration.

    Args:
        config: OllamaConfig object
        args: Parsed arguments
    """
    from i18n import get_text
    
    try:
        _run_benchmark_workflow_impl(config, args)
    except KeyboardInterrupt:
        print("\n" + get_text("benchmark_interrupted"))
        sys.exit(0)


def _check_model_retest(model_name: str, tested_count: int, total_count: int,
                         results_file: str, decision_state: dict) -> tuple:
    """Check if model exists in results and prompt for retest decision.
    
    Args:
        model_name: Name of the model to check
        tested_count: Number of models already tested
        total_count: Total number of models to test
        results_file: Path to results file (can be None)
        decision_state: Shared state dict for YES_ALL/NO_ALL decisions
        
    Returns:
        Tuple of (should_test: bool, should_stop: bool)
    """
    # Check if we have a shared decision state (YES_ALL or NO_ALL)
    if decision_state.get('skip_all', False):
        return False, False
    if decision_state.get('test_all', False):
        return True, False
    
    # Check if model exists in results file (returns False if results_file is None)
    if not model_exists_in_results(results_file, model_name):
        return True, False
    
    # Model exists - prompt user
    decision = prompt_retest_decision(model_name, tested_count, total_count)
    
    if should_skip_model(decision, tested_count, total_count):
        if should_stop_testing(decision):
            return False, True
        return False, False
    
    # YES or YES_ALL
    if decision == RetestDecision.YES_ALL:
        decision_state['test_all'] = True
    elif decision == RetestDecision.NO_ALL:
        decision_state['skip_all'] = True
    
    return True, False


def _run_benchmark_workflow_impl(config: OllamaConfig, args):
    """Implementation of benchmark workflow."""
    from i18n import get_text

    print(get_text("app_title") + " (Context & VRAM Analyzer)\n")
    print(get_text("scanning_models"))

    # Create API client using factory (auto-detects local vs remote based on --ssh-host)
    ollama_client = ApiClientFactory.create_client(
        base_url=config.base_url,
        headers=config.get_headers(),
        timeout=config.timeout,
        ssh_host=getattr(args, 'ssh_host', None),
        ssh_user=getattr(args, 'ssh_user', None),
        ssh_port=getattr(args, 'ssh_port', 22),
        ssh_key=getattr(args, 'ssh_key', None)
    )

    # Fetch models (capabilities are already extracted from model_info)
    models = ollama_client.get_models()

    if not models:
        return

    # Initialize capabilities fetcher and add discovered models to cache
    from api.capabilities_fetcher import CapabilitiesFetcher
    capabilities_fetcher = CapabilitiesFetcher()
    
    # Check if we have a fresh model cache
    use_cache = capabilities_fetcher.is_cache_fresh()
    
    if use_cache:
        print(get_text("cache_using"))
    else:
        print(get_text("cache_fetching"))
    
    # Add all discovered models to the cache
    for m in models:
        model_name = m["name"]
        
        # Try to get metadata from cache first
        cached = capabilities_fetcher.get_model_from_cache(model_name)
        
        if cached and use_cache:
            # Use cached data
            cached_size_gb = cached.get('size_gb')
            # If cached size_gb is "N/A", try to get it from /api/tags response (m["size"])
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
            caps = cached.get('capabilities', {'vision': False, 'tools': False, 'thinking': False, 'audio': False})
            
            # Update size_gb if it was calculated from bytes
            if isinstance(m["size_gb"], (int, float)):
                m["size_gb"] = round(m["size_gb"], 1)
        else:
            # Fetch from API
            # Add size_gb from model size (may not be in /api/tags response)
            size_bytes = m.get("size", 0)
            m["size_gb"] = round(size_bytes / (1024**3), 1) if size_bytes > 0 else "N/A"
            
            # Add params from model details (may not be in /api/tags response)
            details = m.get("details", {})
            param_size = details.get("parameter_size", "N/A") if details else "N/A"
            m["params"] = param_size
            
            # Add quantization format from model details
            # Ollama API uses 'quantization_level' (e.g., 'Q4_K_M')
            quant = (details.get("quantization_level") or details.get("quantization_format") or "N/A") if details else "N/A"
            m["quant"] = quant
            
            # Add architecture from model info
            try:
                model_info = ollama_client.get_model_info(m["name"])
                architecture = model_info.get('model_info', {}).get('general.architecture', 'N/A')
                m["architecture"] = architecture if architecture else 'N/A'
            except Exception:
                m["architecture"] = 'N/A'
            
            # Add max_ctx and capabilities from model info
            try:
                model_info = ollama_client.get_model_info(m["name"])
                max_ctx = 131072  # default
                caps = {'vision': False, 'tools': False, 'thinking': False, 'audio': False}
                if model_info:
                    # Check for top-level 'capabilities' key (new Ollama API format)
                    top_level_caps = model_info.get("capabilities", [])
                    if isinstance(top_level_caps, list) and len(top_level_caps) > 0:
                        caps_list_str = ' '.join(top_level_caps).lower()
                        caps = {
                            'vision': 'vision' in caps_list_str or 'image' in caps_list_str,
                            'tools': 'tools' in caps_list_str or 'function' in caps_list_str,
                            'thinking': 'thinking' in caps_list_str or 'reason' in caps_list_str,
                            'audio': 'audio' in caps_list_str or 'whisper' in caps_list_str
                        }
                    else:
                        # Fallback: extract from model_info dict
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
                
                # Add metadata to cache
                capabilities_fetcher.add_model_metadata(model_name, model_info)
            except Exception:
                m["max_ctx"] = 131072  # default on error
                caps = {'vision': False, 'tools': False, 'thinking': False, 'audio': False}
        
        m["vision"] = "✅" if caps.get("vision", False) else "❌"
        m["tools"] = "✅" if caps.get("tools", False) else "❌"
        m["thinking"] = "✅" if caps.get("thinking", False) else "❌"
        
        # Add to capabilities fetcher cache
        capabilities_fetcher.add_model_from_api(m["name"], caps)

    # Apply capabilities filter if specified
    if args.capabilities:
        print(get_text("filter_applied", capabilities=args.capabilities))
        filtered = []
        for m in models:
            keep = True
            if 'v' in args.capabilities and m["vision"] != "✅":
                keep = False
            if 'T' in args.capabilities and m["tools"] != "✅":
                keep = False
            if 't' in args.capabilities and m["thinking"] != "✅":
                keep = False
            if keep:
                filtered.append(m)
        models = filtered
        if not models:
            print(get_text("no_models_match_filter"))
            return

    test_models = []

    if args.models:
        target_names = [name.strip() for name in args.models.split(',')]
        test_models = [m for m in models if m['name'] in target_names]

        found_names = [m['name'] for m in test_models]
        not_found = [name for name in target_names if name not in found_names]
        if not_found:
            print(get_text("no_models_found", models=', '.join(not_found)))
        if not test_models:
            print(get_text("no_test_models"))
            return
    else:
        # Print model list
        print_model_list(models)

        # Interactive model selection with curses
        # curses.wrapper temporarily replaces SIGINT handler, so re-register after
        try:
            test_models = curses.wrapper(lambda stdscr: interactive_model_select(stdscr, models, single_select=False))
        except curses.error:
            # Fallback to numeric input if curses fails (e.g., terminal not supported)
            print("\n⚠️  Interactive curses mode not available, using numeric input...")
            selected_idx = input(get_text("select_models") + "\n")
        except ValueError as e:
            # Fallback for format errors (e.g., size_gb is string instead of float)
            print(f"\n⚠️  Data format error in curses mode: {e}, using numeric input...")
            selected_idx = input(get_text("select_models") + "\n")
        except Exception as e:
            # Fallback to old numeric input if curses fails (e.g., Windows console)
            print(f"\n⚠️  Curses error: {e}, using numeric input...")
            selected_idx = input(get_text("select_models") + "\n")
            if selected_idx.strip().lower() == 'all':
                test_models = models
            else:
                try:
                    indices = [int(x.strip()) for x in selected_idx.split(",")]
                    test_models = [models[i] for i in indices]
                except (ValueError, IndexError):
                    print(get_text("invalid_input"))
                    return
        # Re-register SIGINT handler after curses.wrapper (curses may reset it)
        signal.signal(signal.SIGINT, signal_handler)

    # Save capabilities cache and model metadata cache immediately after all model data is collected
    capabilities_fetcher.save_cache()
    capabilities_fetcher.save_model_cache()

    model_names_for_cmd = ",".join([m["name"] for m in test_models])
    # Use sys.executable for portability and absolute path to script
    script_name = os.path.basename(sys.argv[0])
    cmd_str = f"sudo {sys.executable} {script_name} --models {model_names_for_cmd}"
    if args.capabilities:
        cmd_str += f" --capabilities {args.capabilities}"

    print("\n" + "="*60)
    print(get_text("repeated_run_header"))
    print(f"   {cmd_str}")
    print("="*60 + "\n")

    # Get context sizes
    context_sizes = get_context_sizes(args)

    # Handle prompt-related flags
    if args.list_chains:
        prompt_loader = PromptLoader(args.prompts_file)
        chains = prompt_loader.get_chains()
        print("Available chains:")
        for c in chains:
            print(f"  - {c['name']} ({c['id']})")
        return

    if args.list_independent:
        prompt_loader = PromptLoader(args.prompts_file)
        modes = prompt_loader.get_all_independent_modes()
        for mode in modes:
            prompts = prompt_loader.get_independent_prompts(mode)
            print(f"\n=== {mode.upper()} ===")
            for p in prompts:
                print(f"  - {p['name']} ({p['id']})")
        return

    # Create prompt loader by default (for saving prompts in results)
    prompts_file = args.prompts_file or config.prompts_file
    prompt_loader = PromptLoader(prompts_file)
    
    # Log prompts configuration
    logger.info(f"📝 Loaded prompts from: {prompts_file}")
    if prompt_loader.data.get('independent'):
        logger.info(f"📝 Available independent modes: {', '.join(prompt_loader.get_all_independent_modes())}")
    if prompt_loader.data.get('chains'):
        chains = prompt_loader.get_chains()
        logger.info(f"📝 Available chains: {', '.join(chain.get('name') for chain in chains)}")

    # Expert-Evaluator setup
    expert_evaluator = None
    expert_model_name = None
    
    expert_prompts_valid = validate_expert_prompts()
    logger.info("[Expert] validate_expert_prompts() returned %s", expert_prompts_valid)
    
    if expert_prompts_valid:
        enable_expert = prompt_user(get_text("ask_enable_expert"))
        logger.info("[Expert] User answered enable_expert=%s", enable_expert)
        
        if enable_expert:
            logger.info("[Expert] User enabled expert, starting model selection...")
            try:
                logger.info("[Expert] Calling curses.wrapper with %d models available", len(models))
                expert_model_name = curses.wrapper(
                    lambda stdcr: select_expert_model(stdcr, models)
                )
                logger.info("[Expert] Model selection returned: expert_model_name=%r", expert_model_name)
                if expert_model_name:
                    from benchmark.expert_evaluator import ExpertEvaluator
                    logger.info("[Expert] Creating ExpertEvaluator with model=%s", expert_model_name)
                    expert_evaluator = ExpertEvaluator(ollama_client, expert_model_name)
                    print(f"✅ Expert evaluator initialized with model: {expert_model_name}")
                    logger.info("[Expert] ExpertEvaluator created successfully")
                else:
                    print("⚠️  No expert model selected.")
                    logger.info("[Expert] No expert model selected by user (returned None)")
            except Exception as e:
                print(f"⚠️  Warning: Expert model selection failed: {e}")
                expert_model_name = None
                expert_evaluator = None
                logger.exception("[Expert] Exception during expert model selection")
    else:
        logger.warning("[Expert] validate_expert_prompts() returned False, skipping expert setup")

    # Initialize benchmark runner
    logger.info("[Expert] Before BenchmarkRunner init: expert_evaluator=%s", expert_evaluator is not None)
    logger.info("[num_predict] Using num_predict=%d (use --num-predict -1 for unlimited)", getattr(args, 'num_predict', 12000))
    temperature_test_values = get_temperature_test_values(args)
    logger.info("[temperature] Using temperature values: %s", temperature_test_values)
    benchmark_runner = BenchmarkRunner(
        ollama_client=ollama_client,
        context_sizes=context_sizes,
        num_runs=args.num_runs,
        restart_method=args.restart_method,
        no_restart=args.no_restart,
        disable_thinking=not args.no_thinking,  # no_thinking=True → disable_thinking=False (user wants thinking)
        prompt_loader=prompt_loader,
        temperature_test_values=temperature_test_values,
        expert_evaluator=expert_evaluator,
        num_predict=getattr(args, 'num_predict', 12000),
        independent_top=getattr(args, 'independent_top', None),
        user_context_sizes=getattr(args, 'context_sizes', None)
    )
    logger.info("[Expert] After BenchmarkRunner init: runner.expert_evaluator=%s",
                benchmark_runner.expert_evaluator is not None)

    selected_chain = None
    if args.chain:
        selected_chain = prompt_loader.get_chain_by_id(args.chain)
        if not selected_chain:
            print(f"Chain not found: {args.chain}")
            return

    used_prompts_config = build_used_prompts_config(prompt_loader, args, benchmark_runner)
    run_config = build_run_config(prompt_loader, args, benchmark_runner, test_models)

    # Display test parameters in JSON format
    import json
    model_names = [m["name"] for m in test_models]
    test_params = benchmark_runner.get_test_params(model_names, expert_model_name)
    test_params.update(run_config)
    print("\n" + "="*60)
    print(get_text("test_parameters_header", default="Test Parameters"))
    print("="*60)
    print(json.dumps(test_params, indent=2, ensure_ascii=False))
    print("="*60 + "\n")

    # Run benchmarks for each model
    from benchmark.result import BenchmarkResult
    all_results: list[BenchmarkResult] = []
    
    # Initialize retest decision state
    decision_state: dict = {}
    
    # Determine output file - prompt user if not specified via --output
    # Track whether we should save results
    should_save_results = False
    if args.output:
        results_file = args.output
        should_save_results = True
    else:
        # Ask to save results first
        if prompt_user(get_text("ask_save_results")):
            results_file = prompt_output_filename(DEFAULT_OUTPUT_FILE)
            should_save_results = True
        else:
            # User declined to save - set to None to indicate no file
            results_file = None
            should_save_results = False

    if getattr(args, 'output_format', None) == 'csv':
        save_mode = SAVE_MODE_DISABLED
    else:
        save_mode = prepare_results_output_file(results_file, run_config, used_prompts_config)
    if save_mode == "abort":
        return

    expert_results_manager = ExpertResultsManager()
    expert_results_manager.start_session(
        tested_model=model_names,
        expert_model=expert_model_name,
        run_config=run_config,
    )

    def handle_completed_result(benchmark_result: BenchmarkResult):
        if not benchmark_result:
            return
        all_results.append(benchmark_result)
        benchmark_runner.run_expert_evaluation_for_model(benchmark_result.model.name)
        expert_results_manager.append_model_result(benchmark_result)
        persist_model_result(
            save_mode,
            results_file,
            benchmark_result,
            all_results,
            used_prompts_config,
            run_config,
        )

    def run_model_benchmark(model_info: ModelInfo, runner_method, *method_args):
        try:
            return runner_method(model_info, *method_args)
        except BaseException:
            benchmark_runner._unload_tested_model(model_info.name)
            raise

    # Run based on mode
    if args.independent:
        # Run ALL independent prompts for each model
        for i, m in enumerate(test_models):
            # Check if model exists in results and prompt for retest decision
            should_test, should_stop = _check_model_retest(
                m['name'], i, len(test_models), results_file, decision_state
            )
            if should_stop:
                break
            if not should_test:
                continue
            
            model_info = make_model_info(m)
            benchmark_result, error = run_model_benchmark(
                model_info, benchmark_runner.run_all_independent_prompts
            )
            handle_completed_result(benchmark_result)
            if error:
                continue

    elif args.chains:
        # NEW: Run ALL chains for each model
        for i, m in enumerate(test_models):
            # Check if model exists in results and prompt for retest decision
            should_test, should_stop = _check_model_retest(
                m['name'], i, len(test_models), results_file, decision_state
            )
            if should_stop:
                break
            if not should_test:
                continue
            
            model_info = make_model_info(m)
            benchmark_result, error = run_model_benchmark(
                model_info, benchmark_runner.run_all_chains
            )
            handle_completed_result(benchmark_result)
            if error:
                continue

    elif args.chain:
        # Run specific chain for each model
        chain = selected_chain
        for i, m in enumerate(test_models):
            # Check if model exists in results and prompt for retest decision
            should_test, should_stop = _check_model_retest(
                m['name'], i, len(test_models), results_file, decision_state
            )
            if should_stop:
                break
            if not should_test:
                continue
            
            model_info = make_model_info(m)
            benchmark_result, error = run_model_benchmark(
                model_info, benchmark_runner.run_chain, chain
            )
            handle_completed_result(benchmark_result)
            if error:
                continue

    else:
        # Default: run ALL independent prompts
        if prompt_loader and prompt_loader.data.get('independent'):
            print(get_text("using_independent_prompts_default"))
            for i, m in enumerate(test_models):
                # Check if model exists in results and prompt for retest decision
                should_test, should_stop = _check_model_retest(
                    m['name'], i, len(test_models), results_file, decision_state
                )
                if should_stop:
                    break
                if not should_test:
                    continue
                
                model_info = make_model_info(m)
                benchmark_result, error = run_model_benchmark(
                    model_info, benchmark_runner.run_all_independent_prompts
                )
                handle_completed_result(benchmark_result)
                if error:
                    continue
        else:
            logger.warning("⚠️  No independent prompts found in prompts.jsonc, using default benchmark prompt")
            for i, m in enumerate(test_models):
                # Check if model exists in results and prompt for retest decision
                should_test, should_stop = _check_model_retest(
                    m['name'], i, len(test_models), results_file, decision_state
                )
                if should_stop:
                    break
                if not should_test:
                    continue
                
                model_info = make_model_info(m)
                benchmark_result, error = run_model_benchmark(
                    model_info, benchmark_runner.run_for_model
                )
                handle_completed_result(benchmark_result)
                if error:
                    continue

    # Print results
    print_results_table(all_results)

    # Print expert-guided recommendations grouped by mode with top 3 models per mode
    if all_results:
        print_expert_recommendations(all_results, test_models)

    # CSV export remains a final export. JSON results are persisted per model above.
    if hasattr(args, 'output_format') and args.output_format == 'csv':
        save_results(
            all_results,
            results_file,
            args.output_format,
            prompts_config=used_prompts_config,
            run_config=run_config,
            confirm_overwrite=False,
        )


def _post_benchmark_workflow(results_file: str, all_results: list, args):
    """Post-benchmark workflow: ask about AI analysis and model selection.
    
    Args:
        results_file: Path to results file (can be None)
        all_results: List of BenchmarkResult objects
        args: Parsed arguments
    """
    from i18n import get_text
    from export.ai_analyzer import AIAnalyzer, analyze_results_interactive
    
    # Only ask about AI analysis if we have results and a file to work with
    if not all_results or not results_file:
        return
    
    # Ask about AI analysis
    if prompt_user(get_text("ask_ai_analysis")):
        try:
            analyzer = AIAnalyzer()
            analyze_results_interactive(
                analyzer, all_results, 
                getattr(args, 'language', 'en'),
                getattr(args, 'restart_method', 'manual'),
                getattr(args, 'no_restart', False)
            )
        except Exception as e:
            print(f"⚠️  AI analysis failed: {e}")


def signal_handler(signum, frame):
    """Handle SIGINT signal gracefully."""
    print("\n" + get_text("benchmark_interrupted"))
    sys.exit(0)


def _main_impl():
    """Main implementation with language setup."""
    from i18n import set_language
    from cli import setup_logging
    
    args = parse_args()
    
    # Setup logging
    setup_logging(getattr(args, 'verbose', 0))
    
    # Set language
    lang = getattr(args, 'language', 'en')
    set_language(lang)
    
    # Get configuration
    config = get_ollama_config(args)
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run benchmark workflow
    run_benchmark_workflow(config, args)


def main():
    """Main entry point."""
    try:
        _main_impl()
    except KeyboardInterrupt:
        print("\n" + get_text("benchmark_interrupted"))
        sys.exit(0)


if __name__ == "__main__":
    main()
