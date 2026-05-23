"""Tests for incremental JSON persistence helpers."""

import os
import tempfile
import unittest
from types import SimpleNamespace

from benchmark.expert_evaluator import ExpertEvaluator
from benchmark.result import BenchmarkMetrics, BenchmarkResult, ModelInfo
from export.merge_utils import (
    load_results_file,
    merge_results,
    run_configs_match,
    save_results_file,
)
from main_helpers import build_used_prompts_config
from main_recommendations import _build_mode_recommendations


class FakePromptLoader:
    def __init__(self):
        self.data = {
            'independent': {'architect': [], 'code': [], 'debug': []},
            'chains': [
                {
                    'id': 'chain_a',
                    'name': 'Chain A',
                    'prompts': {
                        'architect': {'id': 'a1', 'name': 'A1', 'prompt': 'plan'},
                    },
                },
                {
                    'id': 'chain_b',
                    'name': 'Chain B',
                    'prompts': {
                        'debug': {'id': 'b1', 'name': 'B1', 'prompt': 'debug'},
                    },
                },
            ],
        }

    def get_chain_by_id(self, chain_id):
        return next((c for c in self.data['chains'] if c['id'] == chain_id), None)

    def get_chains(self):
        return self.data['chains']


class FakeRunner:
    def get_used_independent_prompts(self):
        return [
            {'mode': 'architect', 'id': 'arch_1', 'name': 'Arch 1', 'prompt': 'plan'},
            {'mode': 'code', 'id': 'code_1', 'name': 'Code 1', 'prompt': 'code'},
        ]


class TestPromptConfigFiltering(unittest.TestCase):
    def test_independent_config_contains_only_independent(self):
        config = build_used_prompts_config(
            FakePromptLoader(),
            SimpleNamespace(chain=None, chains=False, independent=True),
            FakeRunner(),
        )

        self.assertIn('independent', config)
        self.assertNotIn('chains', config)
        self.assertEqual(config['independent']['architect'][0]['id'], 'arch_1')
        self.assertEqual(config['independent']['code'][0]['id'], 'code_1')

    def test_single_chain_config_contains_only_selected_chain(self):
        config = build_used_prompts_config(
            FakePromptLoader(),
            SimpleNamespace(chain='chain_b', chains=False, independent=False),
            FakeRunner(),
        )

        self.assertNotIn('independent', config)
        self.assertEqual([c['id'] for c in config['chains']], ['chain_b'])


class TestMergeUtils(unittest.TestCase):
    def test_run_config_match_uses_only_required_fields(self):
        existing = {
            'used_prompt_ids': ['p2', 'p1'],
            'used_chain_ids': [],
            'context_sizes': [65536, 32768],
            'temperature_test_values': [1.0, 0.0],
            'num_runs': 1,
        }
        new = {
            'used_prompt_ids': ['p1', 'p2'],
            'used_chain_ids': [],
            'context_sizes': [32768, 65536],
            'temperature_test_values': [0.0, 1.0],
            'num_runs': 99,
        }

        self.assertTrue(run_configs_match(existing, new))

    def test_merge_replaces_existing_metric_entirely(self):
        model = ModelInfo(name='model-a', size_gb=1.0)
        old_result = BenchmarkResult(
            model=model,
            results=[
                BenchmarkMetrics(
                    ctx=32768,
                    temperature=0.0,
                    prompt_id='p1',
                    mode='architect',
                    avg_tps=1.0,
                    min_tps=1.0,
                    max_tps=1.0,
                    std_dev=0.0,
                    response='old',
                    expert_score=10.0,
                )
            ],
        )
        new_result = BenchmarkResult(
            model=model,
            results=[
                BenchmarkMetrics(
                    ctx=32768,
                    temperature=0.0,
                    prompt_id='p1',
                    mode='architect',
                    avg_tps=9.0,
                    min_tps=8.0,
                    max_tps=10.0,
                    std_dev=0.5,
                    response='new',
                    expert_score=90.0,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'benchmark_results.json')
            save_results_file(path, [old_result], run_config={
                'used_prompt_ids': ['p1'],
                'used_chain_ids': [],
                'context_sizes': [32768],
                'temperature_test_values': [0.0],
            })
            merge_results(path, new_result)
            loaded, _ = load_results_file(path)

        metric = loaded[0].results[0]
        self.assertEqual(metric.response, 'new')
        self.assertEqual(metric.avg_tps, 9.0)
        self.assertEqual(metric.expert_score, 90.0)


class TestExpertRecommendations(unittest.TestCase):
    def test_recommendations_rank_by_expert_score_per_mode(self):
        fast_low_quality = BenchmarkResult(
            model=ModelInfo(name='fast-low', size_gb=1.0),
            results=[
                BenchmarkMetrics(
                    ctx=32768,
                    temperature=0.0,
                    prompt_id='code_1',
                    mode='code',
                    avg_tps=200.0,
                    min_tps=190.0,
                    max_tps=210.0,
                    std_dev=1.0,
                    expert_score=40.0,
                )
            ],
        )
        slower_high_quality = BenchmarkResult(
            model=ModelInfo(name='smart-slower', size_gb=2.0),
            results=[
                BenchmarkMetrics(
                    ctx=32768,
                    temperature=0.66,
                    prompt_id='code_1',
                    mode='code',
                    avg_tps=20.0,
                    min_tps=18.0,
                    max_tps=22.0,
                    std_dev=1.0,
                    expert_score=90.0,
                )
            ],
        )

        recs = _build_mode_recommendations(
            [fast_low_quality, slower_high_quality],
            [
                {'name': 'fast-low', 'params': '1B', 'quant': 'Q4', 'size_gb': 1.0},
                {'name': 'smart-slower', 'params': '2B', 'quant': 'Q4', 'size_gb': 2.0},
            ],
        )

        self.assertEqual(recs['code'][0]['model_name'], 'smart-slower')

    def test_expert_score_parser_uses_100_point_scale(self):
        self.assertEqual(ExpertEvaluator._parse_score("Score: 87"), 87)
        self.assertEqual(ExpertEvaluator._parse_score("100"), 100)
        self.assertEqual(ExpertEvaluator._parse_score("0"), 0)
        self.assertEqual(ExpertEvaluator._parse_score("Score: 150"), 50)


if __name__ == '__main__':
    unittest.main()
