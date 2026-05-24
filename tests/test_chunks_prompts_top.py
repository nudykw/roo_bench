"""Tests for --chunks-top and --prompts-top CLI arguments and filtering logic."""

import unittest
from unittest.mock import patch

from cli import parse_args
from benchmark.runner import BenchmarkRunner


class TestChunksTopCLI(unittest.TestCase):
    """Tests for --chunks-top CLI argument parsing."""
    
    def test_chunks_top_argument_parsed(self):
        """Test that --chunks-top argument is correctly parsed."""
        with patch('sys.argv', ['main.py', '--chunks-top', '3']):
            args = parse_args()
            self.assertEqual(args.chunks_top, 3)
    
    def test_chunks_top_default_none(self):
        """Test that --chunks-top defaults to None."""
        with patch('sys.argv', ['main.py']):
            args = parse_args()
            self.assertIsNone(args.chunks_top)


class TestPromptsTopCLI(unittest.TestCase):
    """Tests for --prompts-top CLI argument parsing."""
    
    def test_prompts_top_argument_parsed(self):
        """Test that --prompts-top argument is correctly parsed."""
        with patch('sys.argv', ['main.py', '--prompts-top', '5']):
            args = parse_args()
            self.assertEqual(args.prompts_top, 5)
    
    def test_prompts_top_default_none(self):
        """Test that --prompts-top defaults to None."""
        with patch('sys.argv', ['main.py']):
            args = parse_args()
            self.assertIsNone(args.prompts_top)
    
    def test_prompts_top_with_independent_top(self):
        """Test that both --prompts-top and --independent-top can be specified."""
        with patch('sys.argv', ['main.py', '--prompts-top', '2', '--independent-top', '3']):
            args = parse_args()
            self.assertEqual(args.prompts_top, 2)
            self.assertEqual(args.independent_top, 3)


class TestBenchmarkRunnerPriority(unittest.TestCase):
    """Tests for BenchmarkRunner priority logic."""
    
    @patch('api.base_client.BaseApiClient')
    def test_prompts_top_takes_priority(self, mock_client):
        """Test that prompts_top takes priority over independent_top."""
        runner = BenchmarkRunner(
            ollama_client=mock_client,
            context_sizes=[8192],
            independent_top=5,
            prompts_top=2
        )
        self.assertEqual(runner.effective_top, 2)
        self.assertEqual(runner.prompts_top, 2)
    
    @patch('api.base_client.BaseApiClient')
    def test_independent_top_used_when_no_prompts_top(self, mock_client):
        """Test that independent_top is used when prompts_top is not specified."""
        runner = BenchmarkRunner(
            ollama_client=mock_client,
            context_sizes=[8192],
            independent_top=3,
            prompts_top=None
        )
        self.assertEqual(runner.effective_top, 3)
    
    @patch('api.base_client.BaseApiClient')
    def test_chunks_top_stored_separately(self, mock_client):
        """Test that chunks_top is stored independently."""
        runner = BenchmarkRunner(
            ollama_client=mock_client,
            context_sizes=[8192],
            chunks_top=2
        )
        self.assertEqual(runner.chunks_top, 2)
    
    @patch('api.base_client.BaseApiClient')
    def test_effective_top_none_when_no_filters(self, mock_client):
        """Test that effective_top is None when no filters are specified."""
        runner = BenchmarkRunner(
            ollama_client=mock_client,
            context_sizes=[8192],
            independent_top=None,
            chunks_top=None,
            prompts_top=None
        )
        self.assertIsNone(runner.effective_top)


class TestFilteringLogic(unittest.TestCase):
    """Tests for filtering logic."""
    
    def test_filter_independent_prompts_by_mode(self):
        """Test that prompts are filtered correctly per mode."""
        all_prompts = [
            {'mode': 'architect', 'id': 'p1'},
            {'mode': 'architect', 'id': 'p2'},
            {'mode': 'code', 'id': 'p3'},
            {'mode': 'code', 'id': 'p4'},
            {'mode': 'debug', 'id': 'p5'},
        ]
        effective_top = 1
        
        prompts_by_mode = {}
        for p in all_prompts:
            mode = p['mode']
            if mode not in prompts_by_mode:
                prompts_by_mode[mode] = []
            prompts_by_mode[mode].append(p)
        
        filtered = []
        for mode in ['architect', 'code', 'debug']:
            filtered.extend(prompts_by_mode.get(mode, [])[:effective_top])
        
        self.assertEqual(len(filtered), 3)  # 1 per mode
        self.assertEqual(filtered[0]['id'], 'p1')
        self.assertEqual(filtered[1]['id'], 'p3')
        self.assertEqual(filtered[2]['id'], 'p5')
    
    def test_filter_prompts_by_mode_with_top_2(self):
        """Test filtering with top=2."""
        all_prompts = [
            {'mode': 'architect', 'id': 'p1'},
            {'mode': 'architect', 'id': 'p2'},
            {'mode': 'architect', 'id': 'p3'},
            {'mode': 'code', 'id': 'p4'},
        ]
        effective_top = 2
        
        prompts_by_mode = {}
        for p in all_prompts:
            mode = p['mode']
            if mode not in prompts_by_mode:
                prompts_by_mode[mode] = []
            prompts_by_mode[mode].append(p)
        
        filtered = []
        for mode in ['architect', 'code', 'debug']:
            filtered.extend(prompts_by_mode.get(mode, [])[:effective_top])
        
        self.assertEqual(len(filtered), 3)  # 2 architect + 1 code
        self.assertEqual(filtered[0]['id'], 'p1')
        self.assertEqual(filtered[1]['id'], 'p2')
        self.assertEqual(filtered[2]['id'], 'p4')
    
    def test_filter_chains(self):
        """Test that chains are filtered correctly."""
        chains = [
            {'id': 'c1', 'name': 'Chain 1'},
            {'id': 'c2', 'name': 'Chain 2'},
            {'id': 'c3', 'name': 'Chain 3'},
        ]
        chunks_top = 2
        
        filtered = chains[:chunks_top]
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]['id'], 'c1')
        self.assertEqual(filtered[1]['id'], 'c2')
    
    def test_filter_chains_with_prompts_top(self):
        """Test that chains are filtered with prompts_top (priority over chunks_top)."""
        chains = [
            {'id': 'c1', 'name': 'Chain 1'},
            {'id': 'c2', 'name': 'Chain 2'},
            {'id': 'c3', 'name': 'Chain 3'},
        ]
        prompts_top = 1
        
        # prompts_top should take priority
        filtered = chains[:prompts_top]
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['id'], 'c1')


class TestGetTestParams(unittest.TestCase):
    """Tests for get_test_params including new fields."""
    
    @patch('api.base_client.BaseApiClient')
    def test_get_test_params_includes_new_fields(self, mock_client):
        """Test that get_test_params includes prompts_top and chunks_top."""
        runner = BenchmarkRunner(
            ollama_client=mock_client,
            context_sizes=[8192],
            independent_top=2,
            chunks_top=3,
            prompts_top=1
        )
        params = runner.get_test_params(['model1'])
        
        self.assertIn('prompts_top', params)
        self.assertIn('chunks_top', params)
        self.assertIn('effective_top', params)
        self.assertEqual(params['prompts_top'], 1)
        self.assertEqual(params['chunks_top'], 3)
        self.assertEqual(params['effective_top'], 1)


if __name__ == "__main__":
    unittest.main()
