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
from prompts.loader import PromptLoader

# Setup logging
logger = logging.getLogger('roo_bench')


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
            test_models = curses.wrapper(lambda stdscr: interactive_model_select(stdscr, models))
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
    logger.info("[num_predict] Using num_predict=%d (use --num-predict -1 for unlimited)", getattr(args, 'num_predict', 8192))
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
        num_predict=getattr(args, 'num_predict', 8192)
    )
    logger.info("[Expert] After BenchmarkRunner init: runner.expert_evaluator=%s",
                benchmark_runner.expert_evaluator is not None)

    # Run benchmarks for each model
    from benchmark.result import BenchmarkResult
    all_results: list[BenchmarkResult] = []

    # Run based on mode
    if args.independent:
        # Run ALL independent prompts for each model
        for m in test_models:
            moe_val = m.get('moe')
            moe_dict = moe_val if isinstance(moe_val, dict) else None
            model_info = ModelInfo(
                name=m['name'],
                size_gb=float(m['size_gb']) if m.get('size_gb') and m.get('size_gb') != 'N/A' else 0.0,
                params=m.get('params', 'N/A'),
                quant=m.get('quant', 'N/A'),
                architecture=m.get('architecture', 'N/A'),
                max_ctx=m.get('max_ctx', 131072),
                moe=moe_dict,
                vision=Capability.VISION if m.get('vision') == '✅' else Capability.TOOLS,
                tools=Capability.TOOLS if m.get('tools') == '✅' else Capability.THINKING,
                thinking=Capability.THINKING if m.get('thinking') == '✅' else Capability.AUDIO,
            )
            benchmark_result, error = benchmark_runner.run_all_independent_prompts(model_info)
            if benchmark_result:
                all_results.append(benchmark_result)
            if error:
                continue

    elif args.chains:
        # NEW: Run ALL chains for each model
        for m in test_models:
            moe_val = m.get('moe')
            moe_dict = moe_val if isinstance(moe_val, dict) else None
            model_info = ModelInfo(
                name=m['name'],
                size_gb=float(m['size_gb']) if m.get('size_gb') and m.get('size_gb') != 'N/A' else 0.0,
                params=m.get('params', 'N/A'),
                quant=m.get('quant', 'N/A'),
                architecture=m.get('architecture', 'N/A'),
                max_ctx=m.get('max_ctx', 131072),
                moe=moe_dict,
                vision=Capability.VISION if m.get('vision') == '✅' else Capability.TOOLS,
                tools=Capability.TOOLS if m.get('tools') == '✅' else Capability.THINKING,
                thinking=Capability.THINKING if m.get('thinking') == '✅' else Capability.AUDIO,
            )
            benchmark_result, error = benchmark_runner.run_all_chains(model_info)
            if benchmark_result:
                all_results.append(benchmark_result)
            if error:
                continue

    elif args.chain:
        # Run specific chain for each model
        chain = prompt_loader.get_chain_by_id(args.chain)
        if not chain:
            print(f"Chain not found: {args.chain}")
            return
        for m in test_models:
            moe_val = m.get('moe')
            moe_dict = moe_val if isinstance(moe_val, dict) else None
            model_info = ModelInfo(
                name=m['name'],
                size_gb=float(m['size_gb']) if m.get('size_gb') and m.get('size_gb') != 'N/A' else 0.0,
                params=m.get('params', 'N/A'),
                quant=m.get('quant', 'N/A'),
                architecture=m.get('architecture', 'N/A'),
                max_ctx=m.get('max_ctx', 131072),
                moe=moe_dict,
                vision=Capability.VISION if m.get('vision') == '✅' else Capability.TOOLS,
                tools=Capability.TOOLS if m.get('tools') == '✅' else Capability.THINKING,
                thinking=Capability.THINKING if m.get('thinking') == '✅' else Capability.AUDIO,
            )
            benchmark_result, error = benchmark_runner.run_chain(model_info, chain)
            if benchmark_result:
                all_results.append(benchmark_result)
            if error:
                continue

    else:
        # Default: run ALL independent prompts
        if prompt_loader and prompt_loader.data.get('independent'):
            print(get_text("using_independent_prompts_default"))
            for m in test_models:
                # Helper: convert moe value to dict or None
                moe_val = m.get('moe')
                moe_dict = moe_val if isinstance(moe_val, dict) else None
                
                model_info = ModelInfo(
                    name=m['name'],
                    size_gb=float(m['size_gb']) if m.get('size_gb') and m.get('size_gb') != 'N/A' else 0.0,
                    params=m.get('params', 'N/A'),
                    quant=m.get('quant', 'N/A'),
                    architecture=m.get('architecture', 'N/A'),
                    max_ctx=m.get('max_ctx', 131072),
                    moe=moe_dict,
                    vision=Capability.VISION if m.get('vision') == '✅' else Capability.TOOLS,
                    tools=Capability.TOOLS if m.get('tools') == '✅' else Capability.THINKING,
                    thinking=Capability.THINKING if m.get('thinking') == '✅' else Capability.AUDIO,
                )
                benchmark_result, error = benchmark_runner.run_all_independent_prompts(model_info)
                if benchmark_result:
                    all_results.append(benchmark_result)
                if error:
                    continue
        else:
            logger.warning("⚠️  No independent prompts found in prompts.jsonc, using default benchmark prompt")
            for m in test_models:
                moe_val = m.get('moe')
                moe_dict = moe_val if isinstance(moe_val, dict) else None
                model_info = ModelInfo(
                    name=m['name'],
                    size_gb=float(m['size_gb']) if m.get('size_gb') and m.get('size_gb') != 'N/A' else 0.0,
                    params=m.get('params', 'N/A'),
                    quant=m.get('quant', 'N/A'),
                    architecture=m.get('architecture', 'N/A'),
                    max_ctx=m.get('max_ctx', 131072),
                    moe=moe_dict,
                    vision=Capability.VISION if m.get('vision') == '✅' else Capability.TOOLS,
                    tools=Capability.TOOLS if m.get('tools') == '✅' else Capability.THINKING,
                    thinking=Capability.THINKING if m.get('thinking') == '✅' else Capability.AUDIO,
                )
                benchmark_result, error = benchmark_runner.run_for_model(model_info)
                if benchmark_result:
                    all_results.append(benchmark_result)
                if error:
                    continue

    # Run expert evaluation if enabled
    logger.info("[Expert] Before run_expert_evaluation: expert_evaluator=%s all_results_count=%d",
                expert_evaluator is not None, len(all_results))
    if expert_evaluator and all_results:
        logger.info("[Expert] Calling benchmark_runner.run_expert_evaluation()")
        benchmark_runner.run_expert_evaluation()
    else:
        logger.warning("[Expert] Skipping: expert_evaluator=%s all_results=%d",
                       expert_evaluator is not None, len(all_results))

    # Print results
    print_results_table(all_results)

    # Print recommendations grouped by mode with top 3 results per mode
    if all_results:
        print("\n" + "="*60)
        print(get_text("recommendations_header"))
        print("="*60)
        
        # Group all results by mode based on context size
        mode_groups = {
            get_text('architect_mode'): [],
            get_text('code_mode'): [],
            get_text('debug_mode'): [],
        }
        
        # Build model size lookup
        model_size_map = {}
        for m in test_models:
            model_size_map[m['name']] = m.get('size_gb', 0)
        
        for result in all_results:
            model_name = result.model_name
            model_size = model_size_map.get(model_name, 0)
            for run in result.results:
                ctx_val = run.ctx
                run_with_model = {
                    'ctx': ctx_val,
                    'avg_tps': run.avg_tps,
                    'model_name': model_name,
                    'model_size_gb': model_size,
                }
                if ctx_val >= 65536:
                    mode_groups[get_text('architect_mode')].append(run_with_model)
                elif ctx_val >= 16384:
                    mode_groups[get_text('code_mode')].append(run_with_model)
                else:
                    mode_groups[get_text('debug_mode')].append(run_with_model)
        
        # Sort each group by mode-specific criteria and pick top 3 results per mode
        modes_with_results = []
        for mode_title, recs in mode_groups.items():
            if recs:
                # Sort by mode-specific criteria
                if 'Architect' in mode_title:
                    # Architect: sort by context size descending, then by model size ascending (smaller models preferred)
                    recs.sort(key=lambda x: (-x.get('ctx', 0), x.get('model_size_gb', 0)))
                elif 'Code' in mode_title:
                    # Code: sort by TPS descending (faster = better)
                    recs.sort(key=lambda x: x.get('avg_tps', 0), reverse=True)
                else:
                    # Debug: sort by balance (combination of context and TPS)
                    # Score = ctx/1000 + avg_tps*100 - balances context size with speed
                    recs.sort(key=lambda x: x.get('ctx', 0)/1000 + x.get('avg_tps', 0)*100, reverse=True)
                
                # Get all unique results (unique by model_name + ctx combination), sorted by mode criteria
                seen_keys = set()
                unique_results = []
                for rec in recs:
                    model_name = rec.get('model_name', '')
                    ctx_val = rec.get('ctx', 0)
                    key = f"{model_name}_{ctx_val}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        unique_results.append(rec)
                modes_with_results.append((mode_title, unique_results))
        
        # Sort modes: Architect first, then Code, then Debug
        mode_order = [get_text('architect_mode'), get_text('code_mode'), get_text('debug_mode')]
        modes_with_results.sort(key=lambda x: mode_order.index(x[0]) if x[0] in mode_order else 999)
        
        # Display all modes with their top 3 results
        for mode_title, recs in modes_with_results:
            print(f"\n  {mode_title}")
            for j, rec in enumerate(recs[:3], 1):
                ctx_str = f"{rec['ctx'] // 1024}K" if rec['ctx'] >= 1024 else str(rec['ctx'])
                model_name = rec.get('model_name', 'unknown')
                
                # Get model info from test_models if available
                params = 'N/A'
                size_gb = 'N/A'
                for m in test_models:
                    if m['name'] == model_name:
                        params = m.get('params', 'N/A')
                        size_gb = m.get('size_gb', 'N/A')
                        break
                
                # Mark the recommended (first/best) option
                if j == 1:
                    print(f"  ★ {model_name} ({params}, {size_gb} GB)")
                    print(f"    {get_text('variant', i=j, ctx=ctx_str, tps=rec.get('avg_tps', 0))}")
                else:
                    print(f"    {model_name} ({params}, {size_gb} GB)")
                    print(f"      {get_text('variant', i=j, ctx=ctx_str, tps=rec.get('avg_tps', 0))}")

    # Save results to file (if --output flag specified)
    if hasattr(args, 'output') and hasattr(args, 'output_format'):
        save_results(
            all_results,
            args.output,
            args.output_format,
            prompts_config=prompt_loader.data if prompt_loader else None
        )
    
    # Post-benchmark interactive workflow (save + AI analysis)
    _post_benchmark_workflow(args, all_results, test_models, config)


def _post_benchmark_workflow(args, all_results: list[BenchmarkResult], test_models, config):
    """Handle post-benchmark save and analysis prompts.
    
    Args:
        args: Parsed command-line arguments
        all_results: List of BenchmarkResult objects
        test_models: List of model objects
        config: OllamaConfig object
    """
    from i18n import get_text
    from export.ai_analyzer import (
        save_results_interactive,
        analyze_results_interactive,
        AIAnalyzer,
        prompt_user
    )
    
    # Skip if --no-interactive flag is set
    if getattr(args, 'no_interactive', False):
        return
    
    # Only prompt if no output file was specified
    if not hasattr(args, 'output') or not args.output:
        # Step 1: Ask to save results
        saved_file = save_results_interactive(
            all_results,
            config.base_url, config.get_headers()
        )
        
        # Step 2: Ask for AI analysis
        if prompt_user(get_text("ask_analyze_ai")):
            analyzer = AIAnalyzer(
                base_url=config.base_url,
                headers=config.get_headers(),
                timeout=config.timeout
            )
            # Get restart params from args (same as benchmark)
            from api.factory import ApiClientFactory
            ollama_client = ApiClientFactory.create_client(
                base_url=config.base_url,
                headers=config.get_headers(),
                timeout=config.timeout,
                ssh_host=getattr(args, 'ssh_host', None),
                ssh_user=getattr(args, 'ssh_user', None),
                ssh_port=getattr(args, 'ssh_port', 22),
                ssh_key=getattr(args, 'ssh_key', None)
            )
            analyze_results_interactive(
                analyzer, all_results, args.lang,
                restart_method=getattr(args, 'restart_method', 'manual'),
                no_restart=False,
                ssh_client=ollama_client.ssh_client,
                ollama_client=ollama_client  # Pass API client for unload_model
            )


def main():
    """Main entry point function."""
    try:
        _main_impl()
    except KeyboardInterrupt:
        from i18n import get_text
        print("\n" + get_text("benchmark_interrupted"))


def _main_impl():
    """Main implementation separated for clean exception handling."""
    from i18n import get_text
    
    # Parse arguments
    args = parse_args()

    # Setup logging based on verbose level
    from cli import setup_logging
    setup_logging(getattr(args, 'verbose', 0))

    # Set language
    set_language(args.lang)

    # Handle --update-cache flag (update full model metadata cache)
    if getattr(args, 'update_cache', False):
        from api.capabilities_fetcher import CapabilitiesFetcher
        from api.factory import ApiClientFactory
        
        config = get_ollama_config(args)
        print(get_text("cache_update_start"))
        
        ollama_client = ApiClientFactory.create_client(
            base_url=config.base_url,
            headers=config.get_headers(),
            timeout=config.timeout
        )
        
        fetcher = CapabilitiesFetcher()
        models = ollama_client.get_models()
        
        if not models:
            print(get_text("no_models_found", models=""))
            return
        
        updated = 0
        for m in models:
            model_name = m["name"]
            try:
                model_info = ollama_client.get_model_info(model_name)
                # Add size from /api/tags response (it's not in /api/show)
                if m.get("size", 0) > 0:
                    model_info["size"] = m["size"]
                fetcher.add_model_metadata(model_name, model_info)
                
                # Extract capabilities from model_info
                # API returns capabilities as a list of strings (e.g., ['vision', 'tools'])
                caps_list = model_info.get('capabilities', [])
                if isinstance(caps_list, list) and len(caps_list) > 0:
                    caps_str = ' '.join(caps_list).lower()
                    capabilities_dict = {
                        'vision': 'vision' in caps_str or 'image' in caps_str,
                        'tools': 'tools' in caps_str or 'function' in caps_str,
                        'thinking': 'thinking' in caps_str or 'reason' in caps_str,
                        'audio': 'audio' in caps_str or 'whisper' in caps_str
                    }
                else:
                    capabilities_dict = {}
                
                fetcher.add_model_from_api(model_name, capabilities_dict)
                updated += 1
            except Exception as e:
                print(f"  ⚠️  Error updating {model_name}: {e}")
        
        fetcher.save_cache()
        fetcher.save_model_cache()
        print(get_text("cache_update_complete", count=updated))
        return

    # Handle --analyze-file flag (analyze saved benchmark results)
    analyze_file = getattr(args, 'analyze_file', None)
    if analyze_file:
        config = get_ollama_config(args)
        
        from export.ai_analyzer import AIAnalyzer
        from api.factory import ApiClientFactory
        
        # Create API client for unload_model support
        ollama_client = ApiClientFactory.create_client(
            base_url=config.base_url,
            headers=config.get_headers(),
            timeout=config.timeout,
            ssh_host=getattr(args, 'ssh_host', None),
            ssh_user=getattr(args, 'ssh_user', None),
            ssh_port=getattr(args, 'ssh_port', 22),
            ssh_key=getattr(args, 'ssh_key', None)
        )
        
        analyzer = AIAnalyzer(
            base_url=config.base_url,
            headers=config.get_headers(),
            timeout=config.timeout
        )
        
        # Get model name for analysis (default to first available or prompt)
        model_name = getattr(args, 'analysis_model', None)
        if not model_name:
            # Try to get from args or use default
            model_name = "qwen3.6:35b"  # default model for analysis
        
        success = analyzer.analyze_from_file(
            file_path=analyze_file,
            model_name=model_name,
            target_lang=args.lang,
            ollama_client=ollama_client,
            stream=not getattr(args, 'no_stream', False)  # stream=True по умолчанию
        )
        
        sys.exit(0 if success else 1)

    # Initialize Ollama configuration
    config = get_ollama_config(args)

    # Run benchmark workflow
    run_benchmark_workflow(config, args)


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    from i18n import get_text
    print("\n\n⚠️  Перервано користувачем (Ctrl+C)")
    print(get_text("stopping_tests", model_name="benchmark"))
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()
