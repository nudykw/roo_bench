"""Core benchmark execution orchestration."""

import logging
from api.base_client import BaseApiClient
from system.restart_manager import restart_ollama, RestartMethod
from benchmark.results import calculate_statistics
from i18n import get_text
from prompts.loader import PromptLoader

# Setup logging
logger = logging.getLogger('roo_bench.benchmark')


class BenchmarkRunner:
    """Orchestrates benchmark execution for models."""

    def __init__(self, ollama_client: BaseApiClient, context_sizes: list, num_runs: int = 3,
                 restart_method: str = 'manual', no_restart: bool = False, disable_thinking: bool = True,
                 prompt_loader=None, temperature_test_values: list = None):
        """Initialize benchmark runner.

        Args:
            ollama_client: BaseApiClient instance (LocalApiClient or RemoteApiClient)
            context_sizes: List of context sizes to test
            num_runs: Number of runs per configuration
            restart_method: Ollama restart method
            no_restart: If True, restart is not performed
            disable_thinking: If True, disables thinking mode to prevent reasoning loops
            prompt_loader: PromptLoader instance for loading prompts
            temperature_test_values: List of temperature values for testing (default: None)
        """
        self.ollama_client = ollama_client
        self.context_sizes = context_sizes
        self.num_runs = num_runs
        self.no_restart = no_restart
        self.disable_thinking = disable_thinking
        self.ssh_client = ollama_client.ssh_client  # Get SSH client from the API client
        
        # Map string method to RestartMethod enum
        method_map = {
            'systemctl': RestartMethod.SYSTEMCTL,
            'docker': RestartMethod.DOCKER,
            'kill_start': RestartMethod.KILL_START,
            'manual': RestartMethod.MANUAL,
            'ssh': RestartMethod.SSH
        }
        self.restart_method = method_map.get(restart_method, RestartMethod.MANUAL)
        
        self.prompt_loader = prompt_loader
        self.temperature_test_values = temperature_test_values or [0.0, 0.66, 1.0]

    def filter_contexts(self, max_ctx: int) -> list:
        """Filter context sizes based on model's maximum context.
    
        Args:
            max_ctx: Model's maximum context size
    
        Returns:
            list: Filtered list of valid context sizes (minimum 64K)
        """
        # Filter contexts: take only those >= 64K and <= maximum
        MIN_CONTEXT = 32 768  # 32K minimum
        valid_contexts = [c for c in self.context_sizes if MIN_CONTEXT <= c <= max_ctx]
    
        # If the model supports some "non-standard" max context (e.g. 128000),
        # which is not in our list, but it's >= 64K and <= 256K, we can add it for testing
        if max_ctx not in valid_contexts and MIN_CONTEXT <= max_ctx <= 262144:
            valid_contexts.append(max_ctx)
            valid_contexts.sort()
    
        return valid_contexts

    def run_independent_prompts(self, model: dict, mode: str, temperature: float = None) -> tuple:
        """Run benchmarks using independent prompts for a specific mode.
        
        Args:
            model: Model dictionary
            mode: One of 'architect', 'code', 'debug'
            temperature: Temperature value to use (None for default)

        Returns:
            tuple: (model_name, results, error, prompts_used)
        """
        model_name = model["name"]
        max_ctx = model["max_ctx"]
        prompts = self.prompt_loader.get_independent_prompts(mode)
        results = []
        prompts_used = []
        
        # Display model name at the start
        print(f"\n🤖 Testing model: {model_name}")
        logger.info(f"🤖 Testing model: {model_name}")
        
        # Restart Ollama before testing
        restart_ollama(
            self.restart_method,
            self.no_restart,
            ssh_client=self.ssh_client
        )
        
        # Filter contexts (from 4K to MaxModelContext)
        valid_contexts = self.filter_contexts(max_ctx)
        
        # Get default temperature if not specified
        if temperature is None:
            temps = [1.0, 0.7, 0.3]  # High, medium, low creativity
        else:
            temps = [temperature]
        
        # NEW ORDER: Contexts → Temperatures → Prompts
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
                    
                    avg_tps, vram, tps_list, error, _, used_temp = self.ollama_client.run_generation(
                        model_name,
                        ctx,  # Use current context size
                        self.num_runs,
                        self.disable_thinking,
                        temperature=temp,
                        prompt=prompt,
                        prompt_metadata=prompt_metadata
                    )
                    
                    if error:
                        print(f"         Error running prompt {prompt_id}: {error}")
                        continue
                    
                    # Log response
                    if tps_list and tps_list[0].get('response'):
                        response = tps_list[0]['response']
                        # Log first part of response
                        logger.info(f"   🤖 Model response (first 500 chars): {response[:500]}...")
                    
                    # Show metrics
                    duration = tps_list[0].get('total_duration', 0) / 1e9
                    prompt_tokens = tps_list[0].get('prompt_eval_count', 0)
                    response_tokens = tps_list[0].get('eval_count', 0)
                    print(f"            Duration: {duration:.2f}s | Prompt: {prompt_tokens} | Response: {response_tokens}")
                    
                    # Calculate std_dev
                    if len(tps_list) > 1:
                        mean = avg_tps
                        variance = sum((run['tps'] - mean) ** 2 for run in tps_list) / len(tps_list)
                        std_dev = variance ** 0.5
                    else:
                        std_dev = 0.0
                    
                    results.append({
                        'prompt_id': prompt_id,
                        'prompt_name': prompt_name,
                        'ctx': ctx,  # Add context size
                        'temperature': used_temp,
                        'duration_sec': tps_list[0].get('total_duration', 0) / 1e9 if tps_list else 0,
                        'prompt_tokens': tps_list[0].get('prompt_eval_count', 0) if tps_list else 0,
                        'response_tokens': tps_list[0].get('eval_count', 0) if tps_list else 0,
                        'avg_tps': avg_tps,
                        'min_tps': min(run['tps'] for run in tps_list),
                        'max_tps': max(run['tps'] for run in tps_list),
                        'std_dev': std_dev,
                        'vram': vram,
                        'tps_list': tps_list
                    })
                    prompts_used.append(prompt_metadata)
        
        # Unload model from VRAM after all tests
        print(f"\n   🧹 Unloading model '{model_name}' from VRAM...")
        try:
            self.ollama_client.unload_model(model_name)
        except Exception as e:
            print(f"   ⚠️  Warning: Could not unload model: {e}")
        
        return model_name, results, None, prompts_used
    
    def run_chain(self, model: dict, chain: dict, temperature: float = None) -> tuple:
        """Run benchmarks using a prompt chain (Architect -> Code -> Debug).
        
        Args:
            model: Model dictionary
            chain: Chain dictionary from prompt loader
            temperature: Temperature value to use (None for default)

        Returns:
            tuple: (model_name, results, error, chain_responses)
        """
        model_name = model["name"]
        max_ctx = model["max_ctx"]
        chain_id = chain.get('id', 'unknown')
        chain_name = chain.get('name', 'Unknown Chain')
        
        # Display model name at the start
        print(f"\n🤖 Testing model: {model_name}")
        logger.info(f"🤖 Testing model: {model_name}")
        
        print(f"   Running chain: {chain_name} ({chain_id})")
        
        # Restart Ollama before testing
        restart_ollama(
            self.restart_method,
            self.no_restart,
            ssh_client=self.ssh_client
        )
        
        # Build chain context
        chain_prompts = self.prompt_loader.build_chain_context(chain)
        
        # Filter contexts (from 4K to MaxModelContext)
        valid_contexts = self.filter_contexts(max_ctx)
        
        # Get default temperature if not specified
        if temperature is None:
            temps = [1.0, 0.7, 0.3]  # High, medium, low creativity
        else:
            temps = [temperature]
        
        chain_responses = {}
        results = []
        
        # NEW ORDER: Contexts → Temperatures → Chain modes
        for ctx in valid_contexts:
            logger.info(f"   📏 Context size: {ctx}")
            print(f"   📏 Context: {ctx // 1024}K" if ctx >= 1024 else f"   📏 Context: {ctx}")
            
            for temp in temps:
                logger.info(f"   🌡️  Temperature: {temp:.2f}")
                print(f"      🌡️  Temperature: {temp:.2f}")
                
                # Run each mode in sequence
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
                    logger.debug(f"📝 Chain prompt [{mode}]: {prompt}")
                    
                    avg_tps, vram, tps_list, error, _, used_temp = self.ollama_client.run_generation(
                        model_name,
                        ctx,  # Use current context size
                        self.num_runs,
                        self.disable_thinking,
                        temperature=temp,
                        prompt=prompt,
                        prompt_metadata=prompt_metadata
                    )
                    
                    if error:
                        print(f"         Error in chain [{mode}]: {error}")
                        break
                    
                    # Store response for next mode in chain
                    response_length = sum(run.get('response_length', 0) for run in tps_list) if hasattr(tps_list[0], 'response_length') else len(tps_list) * 100
                    
                    chain_responses[mode] = f"[Response from {mode} mode - {response_length} tokens]"
                    
                    # Calculate std_dev
                    if len(tps_list) > 1:
                        mean = avg_tps
                        variance = sum((run['tps'] - mean) ** 2 for run in tps_list) / len(tps_list)
                        std_dev = variance ** 0.5
                    else:
                        std_dev = 0.0
                    
                    results.append({
                        'mode': mode,
                        'prompt_id': prompt_id,
                        'prompt_name': prompt_name,
                        'ctx': ctx,  # Add context size
                        'temperature': used_temp,
                        'duration_sec': tps_list[0].get('total_duration', 0) / 1e9 if tps_list else 0,
                        'prompt_tokens': tps_list[0].get('prompt_eval_count', 0) if tps_list else 0,
                        'response_tokens': tps_list[0].get('eval_count', 0) if tps_list else 0,
                        'avg_tps': avg_tps,
                        'min_tps': min(run['tps'] for run in tps_list),
                        'max_tps': max(run['tps'] for run in tps_list),
                        'std_dev': std_dev,
                        'vram': vram,
                        'tps_list': tps_list
                    })
        
        # Unload model from VRAM after all tests
        print(f"\n   🧹 Unloading model '{model_name}' from VRAM...")
        try:
            self.ollama_client.unload_model(model_name)
        except Exception as e:
            print(f"   ⚠️  Warning: Could not unload model: {e}")
        
        return model_name, results, None, chain_responses

    def run_for_model(self, model: dict, temperature: float = None) -> tuple:
        """Run benchmarks for a single model across valid contexts.

        Args:
            model: Model dictionary with name, max_ctx, etc.
            temperature: Temperature value to use (None for default)

        Returns:
            tuple: (model_name, results, error)
                model_name: Model name
                results: List of result dictionaries
                error: Error message if any
        """
        model_name = model["name"]
        max_ctx = model["max_ctx"]
        max_ctx_str = f"{max_ctx // 1024}K" if max_ctx >= 1024 else str(max_ctx)

        print("\n============================================================\n")
        print(f"🤖 Тестируется модель: {model_name}")
        print(get_text("testing_model", model_name=model_name))
        # Format size_gb: if it's a number, format to 1 decimal; if "N/A", keep as string
        size_gb_val = model['size_gb']
        if isinstance(size_gb_val, (int, float)):
            size_gb_str = f"{size_gb_val:.1f}"
        else:
            size_gb_str = str(size_gb_val)
        print(get_text("model_size", size_gb=size_gb_str, max_ctx_str=max_ctx_str))

        results = []
        valid_contexts = self.filter_contexts(max_ctx)

        if not valid_contexts:
            print(get_text("skipping_no_contexts"))
            return model_name, results, None

        for ctx in valid_contexts:
            restart_ollama(
                self.restart_method,
                self.no_restart,
                ssh_client=self.ssh_client
            )

            # Display model's default num_ctx (informational only)
            # Actual n_ctx will be verified during generation via /api/ps
            try:
                current_num_ctx = self.ollama_client.get_current_num_ctx(model_name)
                print(get_text("current_num_ctx", num_ctx=current_num_ctx))
                if current_num_ctx != ctx:
                    print(get_text("ctx_info_only", actual=current_num_ctx, expected=ctx))
            except Exception as e:
                print(f"   ⚠️  Error getting model settings: {e}")

            print(get_text("warming_up", ctx=ctx))
            
            # Get default temperature if not specified
            if temperature is None:
                default_temp = self.ollama_client.get_default_temperature(model_name)
                temps = [default_temp, default_temp * 2/3, default_temp * 1/3]
            else:
                temps = [temperature]
            
            # Run for each temperature
            for temp in temps:
                logger.info(f"   🌡️  Running with temperature: {temp:.2f}")
                print(f"   🌡️  Temperature: {temp:.2f}")
                print(f"   📦 Модель: {model_name}")
                
                avg_tps, vram, tps_list, error_msg, _, used_temp = self.ollama_client.run_generation(
                    model_name, ctx, self.num_runs, self.disable_thinking, temperature=temp
                )

                if error_msg:
                    print(get_text("benchmark_failed", error_msg=error_msg))
                    print(get_text("stopping_tests", model_name=model_name))
                    break

                # Show results of each run
                print(get_text("benchmark_runs_header"))
                for run in tps_list:
                    run_num = run['run']
                    tps = run['tps']
                    vram_run = run['vram']
                    if vram_run:
                        vram_str = f"{vram_run / 1024 / 1024:.1f} MiB"
                    else:
                        vram_str = "N/A"
                    print(f"   Run {run_num}: {tps:.2f} TPS (VRAM: {vram_str})")
                    # Show metrics
                    duration = run.get('total_duration', 0) / 1e9
                    prompt_tokens = run.get('prompt_eval_count', 0)
                    response_tokens = run.get('eval_count', 0)
                    print(f"      Duration: {duration:.2f}s | Prompt tokens: {prompt_tokens} | Response tokens: {response_tokens}")
                
                # Check n_ctx AFTER generation (to avoid blocking Ctrl+C)
                try:
                    actual_ctx = self.ollama_client.get_actual_num_ctx(model_name)
                    if actual_ctx > 0:
                        print(get_text("actual_n_ctx_after_gen", ctx=actual_ctx, expected=ctx))
                        if actual_ctx == ctx:
                            print(get_text("ctx_verified_after_gen", actual=actual_ctx, expected=ctx))
                        else:
                            print(get_text("ctx_mismatch_after_gen", actual=actual_ctx, expected=ctx))
                except Exception:
                    pass

                # Show summary
                print(get_text("benchmark_summary"))
                print(f"   Average: {avg_tps:.2f} TPS")
                print(f"   Min: {min(r['tps'] for r in tps_list):.2f} TPS")
                print(f"   Max: {max(r['tps'] for r in tps_list):.2f} TPS")

                # Calculate std dev
                mean = avg_tps
                variance = sum((r['tps'] - mean) ** 2 for r in tps_list) / len(tps_list)
                std_dev = variance ** 0.5
                print(f"   Std Dev: {std_dev:.2f} TPS")

                # Save averaged results
                results.append({
                    "ctx": ctx,
                    "temperature": used_temp,
                    "duration_sec": tps_list[0].get('total_duration', 0) / 1e9 if tps_list else 0,
                    "prompt_tokens": tps_list[0].get('prompt_eval_count', 0) if tps_list else 0,
                    "response_tokens": tps_list[0].get('eval_count', 0) if tps_list else 0,
                    "avg_tps": avg_tps,
                    "min_tps": min(r['tps'] for r in tps_list),
                    "max_tps": max(r['tps'] for r in tps_list),
                    "std_dev": std_dev,
                    "vram": vram
                })

        # Unload model from VRAM after all context tests are done
        print(f"\n   🧹 Unloading model '{model_name}' from VRAM...")
        try:
            self.ollama_client.unload_model(model_name)
        except Exception as e:
            print(f"   ⚠️  Warning: Could not unload model: {e}")

        return model_name, results, None
    
    def run_all_independent_prompts(self, model: dict, temperature: float = None) -> tuple:
        """Run ALL independent prompts for a model across contexts and temperatures.
        
        Execution order:
            Model → Contexts → Temperatures → ALL prompts (architect → code → debug)
        
        Args:
            model: Model dictionary with name, max_ctx, etc.
            temperature: Temperature value to use (None for default)
        
        Returns:
            tuple: (model_name, results, error, prompts_used)
        """
        model_name = model["name"]
        max_ctx = model["max_ctx"]
        results = []
        prompts_used = []
        
        # Display model name
        print(f"\n🤖 Testing model: {model_name}")
        logger.info(f"🤖 Testing model: {model_name}")
        
        # Restart Ollama once for the model
        restart_ollama(
            self.restart_method,
            self.no_restart,
            ssh_client=self.ssh_client
        )
        
        # Filter contexts
        valid_contexts = self.filter_contexts(max_ctx)
        
        # Get temperatures
        if temperature is None:
            temps = [1.0, 0.7, 0.3]
        else:
            temps = [temperature]
        
        # Get ALL prompts in order
        all_prompts = self.prompt_loader.get_all_independent_prompts_ordered()
        
        # NEW ORDER: Contexts → Temperatures → ALL prompts sequentially
        for ctx in valid_contexts:
            logger.info(f"   📏 Context size: {ctx}")
            print(f"   📏 Context: {ctx // 1024}K" if ctx >= 1024 else f"   📏 Context: {ctx}")
            
            for temp in temps:
                logger.info(f"   🌡️  Temperature: {temp:.2f}")
                print(f"      🌡️  Temperature: {temp:.2f}")
                
                # Run ALL prompts in order (architect → code → debug)
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
                    
                    avg_tps, vram, tps_list, error, _, used_temp = self.ollama_client.run_generation(
                        model_name,
                        ctx,
                        self.num_runs,
                        self.disable_thinking,
                        temperature=temp,
                        prompt=prompt,
                        prompt_metadata=prompt_metadata
                    )
                    
                    if error:
                        print(f"         Error: {error}")
                        continue
                    
                    # Calculate std_dev
                    if len(tps_list) > 1:
                        mean = avg_tps
                        variance = sum((run['tps'] - mean) ** 2 for run in tps_list) / len(tps_list)
                        std_dev = variance ** 0.5
                    else:
                        std_dev = 0.0
                    
                    results.append({
                        'mode': mode,
                        'prompt_id': prompt_id,
                        'prompt_name': prompt_name,
                        'ctx': ctx,
                        'temperature': used_temp,
                        'duration_sec': tps_list[0].get('total_duration', 0) / 1e9 if tps_list else 0,
                        'prompt_tokens': tps_list[0].get('prompt_eval_count', 0) if tps_list else 0,
                        'response_tokens': tps_list[0].get('eval_count', 0) if tps_list else 0,
                        'avg_tps': avg_tps,
                        'min_tps': min(run['tps'] for run in tps_list),
                        'max_tps': max(run['tps'] for run in tps_list),
                        'std_dev': std_dev,
                        'vram': vram,
                        'tps_list': tps_list
                    })
                    prompts_used.append(prompt_metadata)
        
        # Unload model once after all tests
        print(f"\n   🧹 Unloading model '{model_name}' from VRAM...")
        try:
            self.ollama_client.unload_model(model_name)
        except Exception as e:
            print(f"   ⚠️  Warning: Could not unload model: {e}")
        
        return model_name, results, None, prompts_used
    
    def run_all_chains(self, model: dict, temperature: float = None) -> tuple:
        """Run ALL chains for a model.
        
        Execution order:
            Model → Chains (sequentially) → Contexts → Temperatures → Modes
        
        Args:
            model: Model dictionary with name, max_ctx, etc.
            temperature: Temperature value to use (None for default)
        
        Returns:
            tuple: (model_name, results, error)
        """
        model_name = model["name"]
        max_ctx = model["max_ctx"]
        all_results = []
        
        # Display model name
        print(f"\n🤖 Testing model: {model_name}")
        logger.info(f"🤖 Testing model: {model_name}")
        
        # Restart Ollama once for the model
        restart_ollama(
            self.restart_method,
            self.no_restart,
            ssh_client=self.ssh_client
        )
        
        # Filter contexts
        valid_contexts = self.filter_contexts(max_ctx)
        
        # Get temperatures
        if temperature is None:
            temps = [1.0, 0.7, 0.3]
        else:
            temps = [temperature]
        
        # Get all chains
        chains = self.prompt_loader.get_chains()
        
        # NEW ORDER: Chains → Contexts → Temperatures → Modes (with context flow)
        for chain in chains:
            chain_id = chain.get('id', 'unknown')
            chain_name = chain.get('name', 'Unknown Chain')
            
            print(f"\n   🔗 Running chain: {chain_name} ({chain_id})")
            logger.info(f"🔗 Running chain: {chain_name} ({chain_id})")
            
            # Build chain context
            chain_prompts = self.prompt_loader.build_chain_context(chain)
            
            for ctx in valid_contexts:
                logger.info(f"   📏 Context size: {ctx}")
                print(f"      📏 Context: {ctx // 1024}K" if ctx >= 1024 else f"      📏 Context: {ctx}")
                
                for temp in temps:
                    logger.info(f"   🌡️  Temperature: {temp:.2f}")
                    print(f"         🌡️  Temperature: {temp:.2f}")
                    
                    chain_responses = {}
                    
                    # Run modes in sequence with context flow
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
                        
                        avg_tps, vram, tps_list, error, _, used_temp = self.ollama_client.run_generation(
                            model_name,
                            ctx,
                            self.num_runs,
                            self.disable_thinking,
                            temperature=temp,
                            prompt=prompt,
                            prompt_metadata=prompt_metadata
                        )
                        
                        if error:
                            print(f"            Error: {error}")
                            break
                        
                        # Store response for context flow
                        response_length = sum(run.get('response_length', 0) for run in tps_list) if tps_list and hasattr(tps_list[0], 'get') else len(tps_list) * 100
                        chain_responses[mode] = f"[Response from {mode} mode - {response_length} tokens]"
                        
                        # Calculate std_dev
                        if len(tps_list) > 1:
                            mean = avg_tps
                            variance = sum((run['tps'] - mean) ** 2 for run in tps_list) / len(tps_list)
                            std_dev = variance ** 0.5
                        else:
                            std_dev = 0.0
                        
                        all_results.append({
                            'chain_id': chain_id,
                            'chain_name': chain_name,
                            'mode': mode,
                            'prompt_id': prompt_id,
                            'prompt_name': prompt_name,
                            'ctx': ctx,
                            'temperature': used_temp,
                            'duration_sec': tps_list[0].get('total_duration', 0) / 1e9 if tps_list else 0,
                            'prompt_tokens': tps_list[0].get('prompt_eval_count', 0) if tps_list else 0,
                            'response_tokens': tps_list[0].get('eval_count', 0) if tps_list else 0,
                            'avg_tps': avg_tps,
                            'min_tps': min(run['tps'] for run in tps_list),
                            'max_tps': max(run['tps'] for run in tps_list),
                            'std_dev': std_dev,
                            'vram': vram,
                            'tps_list': tps_list
                        })
        
        # Unload model once after all chains
        print(f"\n   🧹 Unloading model '{model_name}' from VRAM...")
        try:
            self.ollama_client.unload_model(model_name)
        except Exception as e:
            print(f"   ⚠️  Warning: Could not unload model: {e}")
        
        return model_name, all_results, None
