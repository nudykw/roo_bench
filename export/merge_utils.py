"""Utilities for merging benchmark results."""

import json
import os
from typing import Optional, Tuple
from benchmark.result import BenchmarkResult, BenchmarkMetrics, ModelInfo


def compute_metric_key(model_name: str, metric: BenchmarkMetrics) -> str:
    """Compute unique key for a metric within a model.

    Args:
        model_name: Name of the model.
        metric: BenchmarkMetrics instance.

    Returns:
        Unique key string: {model_name}|{ctx}|{temperature}|{prompt_id}
    """
    return f"{model_name}|{metric.ctx}|{metric.temperature}|{metric.prompt_id or ''}"


def load_results_file(file_path: str) -> Tuple[list, dict]:
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
        with open(file_path, 'r', encoding='utf-8') as f:
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


def save_results_file(file_path: str, results: list, prompts_config: dict = None) -> None:
    """Save results to JSON file.

    Args:
        file_path: Path to the results JSON file.
        results: List of BenchmarkResult objects.
        prompts_config: Optional prompts configuration.
    """
    export_data = {
        'prompts_config': prompts_config or {},
        'results': [r.to_dict() for r in results]
    }

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)


def merge_single_result(existing_results: list, new_result: BenchmarkResult) -> list:
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

        existing_metric = next(
            (m for m in existing_model.results
             if compute_metric_key(model_name, m) == key),
            None
        )

        if existing_metric:
            existing_metric.response = new_metric.response
            existing_metric.expert_score = new_metric.expert_score
        else:
            existing_model.results.append(new_metric)

    return existing_results


def merge_results(existing_file: str, new_result: BenchmarkResult, prompts_config: dict = None) -> None:
    """Merge new result into existing file.

    Args:
        existing_file: Path to the existing results file.
        new_result: New BenchmarkResult to merge.
        prompts_config: Optional prompts configuration to save.
    """
    existing_results, existing_prompts = load_results_file(existing_file)

    if prompts_config is None:
        prompts_config = existing_prompts

    updated_results = merge_single_result(existing_results, new_result)
    save_results_file(existing_file, updated_results, prompts_config)


def check_prompts_match(file_path: str, new_prompts_config: dict) -> bool:
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