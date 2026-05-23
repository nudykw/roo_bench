"""Benchmark workflow orchestration for roo_bench."""

import curses
import json
import logging
import os
import signal
import sys
from argparse import Namespace
from typing import Any, Callable, Optional, cast

from benchmark.result import BenchmarkResult, ModelInfo
from benchmark.runner import BenchmarkRunner
from config import OllamaConfig
from export.expert_results import ExpertResultsManager
from i18n import get_text
from prompts.loader import PromptLoader

from main_helpers import (
    DEFAULT_OUTPUT_FILE,
    SAVE_MODE_DISABLED,
    build_run_config,
    build_used_prompts_config,
    collect_model_data,
    make_model_info,
    persist_model_result,
    prepare_results_output_file,
)
logger = logging.getLogger('roo_bench')


def signal_handler(signum: int, frame: Any) -> None:
    """Handle SIGINT signal gracefully."""
    from i18n import get_text
    print("\n" + get_text("benchmark_interrupted"))
    sys.exit(0)


def _run_benchmark_workflow_impl(config: OllamaConfig, args: Namespace) -> None:
    """Implementation of benchmark workflow."""
    from api.capabilities_fetcher import CapabilitiesFetcher
    from api.factory import ApiClientFactory
    from cli import get_context_sizes, get_temperature_test_values
    from main import validate_expert_prompts
    from ui.curses_selector import interactive_model_select, select_expert_model
    from ui.output_formatter import print_model_list, print_results_table

    print(get_text("app_title") + " (Context & VRAM Analyzer)\n")
    print(get_text("scanning_models"))

    # Create API client using factory
    ollama_client = ApiClientFactory.create_client(
        base_url=config.base_url,
        headers=config.get_headers(),
        timeout=config.timeout,
        backend_type=config.backend_type,
        ssh_host=cast(Optional[str], getattr(args, 'ssh_host', None)),
        ssh_user=cast(Optional[str], getattr(args, 'ssh_user', None)),
        ssh_port=getattr(args, 'ssh_port', 22),
        ssh_key=cast(Optional[str], getattr(args, 'ssh_key', None))
    )

    # Fetch models
    models = ollama_client.get_models()
    if not models:
        return

    # Initialize capabilities fetcher
    capabilities_fetcher = CapabilitiesFetcher()
    use_cache = capabilities_fetcher.is_cache_fresh()
    print(get_text("cache_using") if use_cache else get_text("cache_fetching"))

    # Add all discovered models to the cache
    for m in models:
        model_name = m["name"]
        cached = capabilities_fetcher.get_model_from_cache(model_name)
        collect_model_data(m, cached, use_cache, ollama_client, capabilities_fetcher)

    capabilities_fetcher.save_cache()
    capabilities_fetcher.save_model_cache()

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

    # Select test models
    test_models: list[dict[str, Any]] = []
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
        print_model_list(models)
        try:
            test_models = curses.wrapper(
                lambda stdscr: interactive_model_select(stdscr, models, single_select=False)
            )
        except curses.error:
            print("\n⚠️  Interactive curses mode not available, using numeric input...")
            selected_idx = input(get_text("select_models") + "\n")
        except ValueError as e:
            print(f"\n⚠️  Data format error in curses mode: {e}, using numeric input...")
            selected_idx = input(get_text("select_models") + "\n")
        except Exception as e:
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
        signal.signal(signal.SIGINT, signal_handler)

    # Display repeat command
    model_names_for_cmd = ",".join([m["name"] for m in test_models])
    script_name = os.path.basename(sys.argv[0])
    cmd_str = f"sudo {sys.executable} {script_name} --models {model_names_for_cmd}"
    if args.capabilities:
        cmd_str += f" --capabilities {args.capabilities}"
    print("\n" + "=" * 60)
    print(get_text("repeated_run_header"))
    print(f"   {cmd_str}")
    print("=" * 60 + "\n")

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

    # Create prompt loader - pass None to use default resolution logic (.md priority)
    prompts_file = args.prompts_file
    prompt_loader = PromptLoader(prompts_file)
    logger.info("📝 Loaded prompts from: %s", prompts_file)
    if prompt_loader.data.get('independent'):
        logger.info(
            "📝 Available independent modes: %s",
            ', '.join(prompt_loader.get_all_independent_modes())
        )
    if prompt_loader.data.get('chains'):
        chains = prompt_loader.get_chains()
        logger.info(
            "📝 Available chains: %s",
            ', '.join(str(chain.get('name', '')) for chain in chains)
        )

    # Expert-Evaluator setup
    expert_evaluator = None
    expert_model_name = None
    expert_prompts_valid = validate_expert_prompts()
    logger.info("[Expert] validate_expert_prompts() returned %s", expert_prompts_valid)

    if expert_prompts_valid:
        try:
            try:
                enable_expert = input(f"{get_text('ask_enable_expert')} (y/n): ").strip().lower()
            except EOFError:
                enable_expert = "n"
        except EOFError:
            enable_expert = "n"
        enable_expert = enable_expert in ['y', 'yes']
        logger.info("[Expert] User answered enable_expert=%s", enable_expert)

        if enable_expert:
            try:
                expert_model_name = curses.wrapper(
                    lambda stdcr: select_expert_model(stdcr, models)
                )
                logger.info(
                    "[Expert] Model selection returned: expert_model_name=%r",
                    expert_model_name
                )
                if expert_model_name:
                    from benchmark.expert_evaluator import ExpertEvaluator
                    analysis_prompt_file = getattr(args, 'analysis_prompt_file', None)
                    expert_evaluator = ExpertEvaluator(ollama_client, expert_model_name, analysis_prompt_file)
                    print(f"✅ Expert evaluator initialized with model: {expert_model_name}")
                    logger.info("[Expert] ExpertEvaluator created successfully")
                else:
                    print("⚠️  No expert model selected.")
            except Exception as e:
                print(f"⚠️  Warning: Expert model selection failed: {e}")
                expert_model_name = None
                expert_evaluator = None
                logger.exception("[Expert] Exception during expert model selection")
    else:
        logger.warning(
            "[Expert] validate_expert_prompts() returned False, skipping expert setup"
        )

    # Initialize benchmark runner
    logger.info(
        "[Expert] Before BenchmarkRunner init: expert_evaluator=%s",
        expert_evaluator is not None
    )
    logger.info(
        "[num_predict] Using num_predict=%d (use --num-predict -1 for unlimited)",
        getattr(args, 'num_predict', 12000)
    )
    temperature_test_values = get_temperature_test_values(args)
    logger.info("[temperature] Using temperature values: %s", temperature_test_values)

    benchmark_runner = BenchmarkRunner(
        ollama_client=ollama_client,
        context_sizes=[],
        num_runs=args.num_runs,
        restart_method=args.restart_method,
        no_restart=args.no_restart,
        disable_thinking=not args.no_thinking,
        prompt_loader=prompt_loader,
        temperature_test_values=temperature_test_values,
        expert_evaluator=expert_evaluator,
        num_predict=getattr(args, 'num_predict', 12000),
        independent_top=getattr(args, 'independent_top', None),
        user_context_sizes=context_sizes,
    )
    logger.info(
        "[Expert] After BenchmarkRunner init: runner.expert_evaluator=%s",
        benchmark_runner.expert_evaluator is not None
    )

    selected_chain = None
    if args.chain:
        selected_chain = prompt_loader.get_chain_by_id(args.chain)
        if not selected_chain:
            print(f"Chain not found: {args.chain}")
            return

    used_prompts_config = build_used_prompts_config(
        prompt_loader, args, benchmark_runner
    )
    run_config = build_run_config(
        prompt_loader, args, benchmark_runner, test_models
    )

    # Display test parameters
    model_names = [m["name"] for m in test_models]
    test_params = benchmark_runner.get_test_params(model_names, expert_model_name)
    # Merge run_config excluding context_sizes to preserve user-specified values
    run_config_copy = {k: v for k, v in run_config.items() if k != 'context_sizes'}
    test_params.update(run_config_copy)
    print("\n" + "=" * 60)
    print(get_text("test_parameters_header", default="Test Parameters"))
    print("=" * 60)
    print(json.dumps(test_params, indent=2, ensure_ascii=False))
    print("=" * 60 + "\n")

    # Run benchmarks
    all_results: list[BenchmarkResult] = []
    decision_state: dict[str, bool] = {}

    if args.output:
        results_file = args.output
    else:
        try:
            enable_save = input(f"{get_text('ask_save_results')} (y/n): ").strip().lower()
        except EOFError:
            enable_save = "n"
        if enable_save in ['y', 'yes']:
            from main import prompt_output_filename
            results_file = prompt_output_filename(DEFAULT_OUTPUT_FILE)
        else:
            results_file = None

    if getattr(args, 'output_format', None) == 'csv':
        save_mode = SAVE_MODE_DISABLED
    else:
        save_mode = prepare_results_output_file(
            results_file, run_config,
            used_prompts_config if used_prompts_config is not None else {}
        )
    if save_mode == "abort":
        return

    expert_results_manager = ExpertResultsManager()
    expert_model_for_session = (
        expert_model_name if expert_model_name is not None else ""
    )
    expert_results_manager.start_session(
        tested_model=model_names,
        expert_model=expert_model_for_session,
        run_config=run_config,
    )

    def handle_completed_result(benchmark_result: BenchmarkResult) -> None:
        if not benchmark_result:
            return
        all_results.append(benchmark_result)
        benchmark_runner.run_expert_evaluation_for_model(benchmark_result.model.name)
        expert_results_manager.append_model_result(benchmark_result)
        persist_model_result(
            save_mode, results_file, benchmark_result, all_results,
            used_prompts_config if used_prompts_config is not None else {},
            run_config,
        )

    def run_model_benchmark(
        model_info: ModelInfo,
        runner_method: Callable[..., Any],
        *method_args: Any,
    ) -> Any:
        try:
            return runner_method(model_info, *method_args)
        except BaseException:
            benchmark_runner._unload_tested_model(model_info.name)
            raise

    # Validation: --chain and --all are incompatible
    if getattr(args, 'chain', None) and getattr(args, 'all', False):
        print(get_text("error_chain_all_conflict"))
        return

    # Helper function to run tests for all models
    def _run_tests_for_models(runner_method, *method_args):
        """Run tests for all models with the given runner method."""
        for i, m in enumerate(test_models):
            from main_helpers import _check_model_retest
            should_test, should_stop = _check_model_retest(
                m['name'], i, len(test_models), results_file, decision_state
            )
            if should_stop:
                break
            if not should_test:
                continue
            model_info = make_model_info(m)
            benchmark_result, error = run_model_benchmark(
                model_info, runner_method, *method_args
            )
            handle_completed_result(benchmark_result)
            if error:
                continue

    # Run based on mode
    if args.all:
        # Run independent tests first
        _run_tests_for_models(benchmark_runner.run_all_independent_prompts)
        # Then run all chains
        _run_tests_for_models(benchmark_runner.run_all_chains)

    elif args.independent:
        _run_tests_for_models(benchmark_runner.run_all_independent_prompts)

    elif args.chains:
        _run_tests_for_models(benchmark_runner.run_all_chains)

    elif args.chain:
        _run_tests_for_models(benchmark_runner.run_chain, selected_chain)

    else:
        if prompt_loader and prompt_loader.data.get('independent'):
            actual_file = prompt_loader.file_path
            print(f"{get_text('using_independent_prompts_default')}: {actual_file}")
            for i, m in enumerate(test_models):
                from main_helpers import _check_model_retest
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
            logger.warning(
                "⚠️  No independent prompts found in %s, "
                "using default benchmark prompt",
                prompt_loader.file_path if prompt_loader else "unknown"
            )
            for i, m in enumerate(test_models):
                from main_helpers import _check_model_retest
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

    if all_results:
        from main_recommendations import print_expert_recommendations
        print_expert_recommendations(all_results, test_models)

    if hasattr(args, 'output_format') and args.output_format == 'csv':
        from export.result_saver import save_results
        save_results(
            all_results, results_file, args.output_format,
            prompts_config=used_prompts_config,
            run_config=run_config, confirm_overwrite=False,
        )
