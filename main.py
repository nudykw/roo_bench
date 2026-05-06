"""Main entry point for roo_bench - orchestration of all modules."""

import sys
import os
import curses
from i18n import set_language
from config import OllamaConfig
from cli import parse_args, get_context_sizes
from constants import CONTEXT_SIZES, DEFAULT_OLLAMA_URL
from api.ollama_client import OllamaClient
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

    print(get_text("app_title") + " (Context & VRAM Analyzer)\n")
    print(get_text("scanning_models"))

    # Initialize Ollama client
    ollama_client = OllamaClient(
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
        caps = m.get("capabilities", {})
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
        no_restart=args.no_restart,
        ssh_host=getattr(args, 'ssh_host', None),
        ssh_user=getattr(args, 'ssh_user', None),
        ssh_port=getattr(args, 'ssh_port', 22),
        ssh_key=getattr(args, 'ssh_key', None)
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

    # Save results to file
    if hasattr(args, 'output') and hasattr(args, 'output_format'):
        save_results(
            all_results,
            args.output,
            args.output_format,
            [m['name'] for m in test_models],
            test_models
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
        from api.ollama_client import OllamaClient
        from config import OllamaConfig
        
        config = get_ollama_config(args)
        print("Updating capabilities cache...")
        
        ollama_client = OllamaClient(
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


if __name__ == "__main__":
    main()
