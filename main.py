"""Main entry point for roo_bench - orchestration of all modules."""

import sys
import os
import signal
import curses
from i18n import set_language
from config import OllamaConfig
from cli import parse_args, get_context_sizes
from constants import CONTEXT_SIZES, DEFAULT_OLLAMA_URL
from api.factory import ApiClientFactory
from system.restart_manager import restart_ollama, RestartMethod
from system.gpu_monitor import check_gpu_available, get_vram_usage
from benchmark.runner import BenchmarkRunner
from benchmark.results import calculate_statistics, format_result_row, format_recommendations
from ui.curses_selector import interactive_model_select
from ui.output_formatter import print_model_list, print_results_table
from export.result_saver import save_results, ResultSaver


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
    
    # Add all discovered models to the cache
    for m in models:
        # Add size_gb from model size (may not be in /api/tags response)
        size_bytes = m.get("size", 0)
        m["size_gb"] = round(size_bytes / (1024**3), 1) if size_bytes > 0 else "N/A"
        
        # Add params from model details (may not be in /api/tags response)
        details = m.get("details", {})
        param_size = details.get("parameter_size", "N/A") if details else "N/A"
        m["params"] = param_size
        
        # Add quantization format from model details
        quant = (details.get("quantization_format") or "N/A") if details else "N/A"
        m["quant"] = quant
        
        # Add max_ctx and capabilities from model info
        try:
            model_info = ollama_client.get_model_info(m["name"])
            max_ctx = 131072  # default
            caps = {'vision': False, 'tools': False, 'thinking': False}
            if model_info:
                # Check for top-level 'capabilities' key (new Ollama API format)
                top_level_caps = model_info.get("capabilities", [])
                if isinstance(top_level_caps, list) and len(top_level_caps) > 0:
                    caps_list_str = ' '.join(top_level_caps).lower()
                    caps = {
                        'vision': 'vision' in caps_list_str or 'image' in caps_list_str,
                        'tools': 'tools' in caps_list_str or 'function' in caps_list_str,
                        'thinking': 'thinking' in caps_list_str or 'reason' in caps_list_str
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
        except Exception:
            m["max_ctx"] = 131072  # default on error
            caps = {'vision': False, 'tools': False, 'thinking': False}
        
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

    # Save capabilities cache immediately after all model data is collected
    capabilities_fetcher.save_cache()

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

    # Initialize benchmark runner
    benchmark_runner = BenchmarkRunner(
        ollama_client=ollama_client,
        context_sizes=context_sizes,
        num_runs=args.num_runs,
        restart_method=args.restart_method,
        no_restart=args.no_restart
    )

    # Run benchmarks for each model
    all_results = {}

    for m in test_models:
        model_name, results, error = benchmark_runner.run_for_model(m)
        all_results[model_name] = results

        if error:
            continue

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
        
        for model_name, runs in all_results.items():
            model_size = model_size_map.get(model_name, 0)
            for run in runs:
                ctx_val = run['ctx']
                run_with_model = dict(run)
                run_with_model['model_name'] = model_name
                run_with_model['model_size_gb'] = model_size
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
            [m['name'] for m in test_models],
            test_models
        )
    
    # Post-benchmark interactive workflow (save + AI analysis)
    _post_benchmark_workflow(args, all_results, test_models, config)


def _post_benchmark_workflow(args, all_results, test_models, config):
    """Handle post-benchmark save and analysis prompts.
    
    Args:
        args: Parsed command-line arguments
        all_results: Dictionary of results per model
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
            all_results, test_models,
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
                analyzer, all_results, test_models, args.lang,
                restart_method=getattr(args, 'restart_method', 'manual'),
                no_restart=False,
                ssh_client=ollama_client.ssh_client
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
    # Parse arguments
    args = parse_args()

    # Setup logging based on verbose level
    from cli import setup_logging
    setup_logging(getattr(args, 'verbose', 0))

    # Set language
    set_language(args.lang)

    # Handle --update-cache flag
    if getattr(args, 'update_cache', False):
        from api.capabilities_fetcher import CapabilitiesFetcher
        from api.factory import ApiClientFactory
        
        config = get_ollama_config(args)
        print("Updating capabilities cache...")
        
        ollama_client = ApiClientFactory.create_client(
            base_url=config.base_url,
            headers=config.get_headers(),
            timeout=config.timeout
        )
        
        fetcher = CapabilitiesFetcher()
        models = ollama_client.get_models()
        
        if not models:
            print("No models found!")
            return
        
        for m in models:
            caps = m.get("capabilities", {})
            fetcher.add_model_from_api(m["name"], caps)
        
        fetcher.save_cache()
        print("Cache update complete!")
        return

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
