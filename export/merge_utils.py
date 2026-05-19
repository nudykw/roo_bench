"""Utilities for merging benchmark results."""

import json
import os
import tempfile

from typing import Any

from benchmark.result import BenchmarkMetrics, BenchmarkResult, ModelInfo


def compute_metric_key(model_name: str, metric: BenchmarkMetrics) -> str:
    """Compute unique key for a metric within a model.

    Args:
        model_name: Name of the model.
        metric: BenchmarkMetrics instance.

    Returns:
    Unique key string: {model_name}|{ctx}|{temperature}|{prompt_id}|{chain_id}|{mode}
    """
    return (
        f"{model_name}|{metric.ctx}|{metric.temperature}|"
        f"{metric.prompt_id or ''}|{metric.chain_id or ''}|{metric.mode or ''}"
    )


def atomic_write_json(file_path: str, data: dict[str, Any]) -> None:
    """Atomically write JSON data to a file."""
    directory = os.path.dirname(os.path.abspath(file_path)) or "."
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(file_path)}.",
        suffix=".tmp",
        dir=directory,
        text=True,
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, file_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def load_results_file(file_path: str) -> tuple[list[BenchmarkResult], dict[str, Any]]:
    """Load existing results from JSON file.

    Args:
        file_path: Path to the results JSON file.

    Returns:
        Tuple of (results_list, prompts_config).
        Returns ([], {}) if file doesn't exist or is invalid.
    """
    if not os.path.exists(file_path):
        return [], {}

    try:
        with open(file_path, encoding='utf-8') as f:
            data = json.load(f)

        results = []
        for item in data.get('results', []):
            model_info = ModelInfo(**item['model'])
            metrics_list = [BenchmarkMetrics(**m) for m in item.get('results', [])]
            results.append(BenchmarkResult(model=model_info, results=metrics_list))

        prompts_config = data.get('prompts_config', {})
        return results, prompts_config

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        print(f"Warning: Could not load results file: {e}")
        return [], {}


def save_results_file(file_path: str, results: list[BenchmarkResult], prompts_config: dict[str, Any] | None = None,
                      run_config: dict[str, Any] | None = None) -> None:
    """Save results to JSON file.

    Args:
        file_path: Path to the results JSON file.
        results: List of BenchmarkResult objects.
        prompts_config: Optional prompts configuration.
    """
    export_data = {
        'run_config': run_config or {},
        'prompts_config': prompts_config or {},
        'results': [r.to_dict() for r in results]
    }

    atomic_write_json(file_path, export_data)


def merge_single_result(existing_results: list[BenchmarkResult], new_result: BenchmarkResult) -> list[BenchmarkResult]:
    """Merge a single BenchmarkResult into existing results.

    Args:
        existing_results: List of existing BenchmarkResult objects.
        new_result: New BenchmarkResult to merge.

    Returns:
        Updated list of BenchmarkResult objects.
    """
    model_name = new_result.model.name
    existing_model = next((r for r in existing_results if r.model.name == model_name), None)

    if not existing_model:
        existing_results.append(new_result)
        return existing_results

    for new_metric in new_result.results:
        key = compute_metric_key(model_name, new_metric)

        existing_index = next(
            (i for i, m in enumerate(existing_model.results)
             if compute_metric_key(model_name, m) == key),
            None
        )

        if existing_index is not None:
            existing_model.results[existing_index] = new_metric
        else:
            existing_model.results.append(new_metric)

    return existing_results


def merge_results(existing_file: str, new_result: BenchmarkResult, prompts_config: dict[str, Any] | None = None,
                  run_config: dict[str, Any] | None = None) -> None:
    """Merge new result into existing file.

    Args:
        existing_file: Path to the existing results file.
        new_result: New BenchmarkResult to merge.
        prompts_config: Optional prompts configuration to save.
    """
    existing_results, existing_prompts = load_results_file(existing_file)
    existing_run_config = load_run_config(existing_file)

    if prompts_config is None:
        prompts_config = existing_prompts
    if run_config is None:
        run_config = existing_run_config

    updated_results = merge_single_result(existing_results, new_result)
    save_results_file(existing_file, updated_results, prompts_config, run_config)


def load_run_config(file_path: str) -> dict[str, Any]:
    """Load run_config from a results file."""
    if not file_path or not os.path.exists(file_path):
        return {}

    try:
        with open(file_path, encoding='utf-8') as f:
            data = json.load(f)
        return data.get('run_config', {}) if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError, TypeError):
        return {}


def normalize_run_config_for_merge(config: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields that decide merge compatibility."""
    config = config or {}
    return {
        'used_prompt_ids': sorted(config.get('used_prompt_ids') or []),
        'used_chain_ids': sorted(config.get('used_chain_ids') or []),
        'context_sizes': sorted(config.get('context_sizes') or []),
        'temperature_test_values': sorted(config.get('temperature_test_values') or []),
    }


def run_configs_match(existing_config: dict[str, Any], new_config: dict[str, Any]) -> bool:
    """Return True when the fields required for merge compatibility match."""
    return normalize_run_config_for_merge(existing_config) == normalize_run_config_for_merge(new_config)


def check_prompts_match(file_path: str, new_prompts_config: dict[str, Any]) -> bool:
    """Check if prompts in file match new prompts.

    Args:
        file_path: Path to the results file.
        new_prompts_config: New prompts configuration to compare.

    Returns:
        True if prompts match, False otherwise.
    """
    _, existing_prompts = load_results_file(file_path)

    if not existing_prompts or not new_prompts_config:
        return True

    existing_independent = existing_prompts.get('independent', {})
    new_independent = new_prompts_config.get('independent', {})

    if set(existing_independent.keys()) != set(new_independent.keys()):
        return False

    for mode in existing_independent:
        existing_ids = {p['id'] for p in existing_independent.get(mode, [])}
        new_ids = {p['id'] for p in new_independent.get(mode, [])}
        if existing_ids != new_ids:
            return False

    return True


def model_exists_in_results(file_path: str, model_name: str) -> bool:
    """Check if a model already exists in the results file.

    Args:
        file_path: Path to the results JSON file. Can be None.
        model_name: Name of the model to check.

    Returns:
        True if model exists in results, False otherwise.
        Returns False if file_path is None or file doesn't exist.
    """
    if not file_path:
        return False
    existing_results, _ = load_results_file(file_path)
    return any(r.model.name == model_name for r in existing_results)
