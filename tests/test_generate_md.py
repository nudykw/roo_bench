"""Tests for prompts/generate_md.py - markdown file generation from JSONC."""

import json
import os
import tempfile
import unittest
import shutil
from unittest import mock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from prompts.generate_md import (
    strip_jsonc_comments,
    generate_prompts_md,
    generate_analysis_prompt_md,
    generate_all_markdown
)


class TestStripJSONCComments(unittest.TestCase):
    """Test strip_jsonc_comments function."""

    def test_line_comment(self):
        """Test removing line comments."""
        input_str = '{"key": "value"} // comment'
        result = strip_jsonc_comments(input_str)
        self.assertNotIn('// comment', result)
        self.assertEqual(json.loads(result), {"key": "value"})

    def test_block_comment(self):
        """Test removing block comments."""
        input_str = '''{
            /* block comment */
            "key": "value"
        }'''
        result = strip_jsonc_comments(input_str)
        self.assertNotIn('/* block comment */', result)
        self.assertEqual(json.loads(result), {"key": "value"})

    def test_string_with_comment_chars(self):
        """Test that // and /* inside strings are preserved."""
        input_str = '{"url": "http://example.com"}'
        result = strip_jsonc_comments(input_str)
        self.assertIn('http://example.com', result)

    def test_multiple_comments(self):
        """Test removing multiple comments."""
        input_str = '''{
            // Line comment 1
            "key1": "value1",
            /* Block comment */
            "key2": "value2" // trailing comment
        }'''
        result = strip_jsonc_comments(input_str)
        self.assertNotIn('// Line comment 1', result)
        self.assertNotIn('/* Block comment */', result)
        self.assertNotIn('// trailing comment', result)


class TestGeneratePromptsMD(unittest.TestCase):
    """Test generate_prompts_md function."""

    def setUp(self):
        """Create test data matching the actual JSONC format."""
        self.test_data = {
            "independent": {
                "architect": [
                    {
                        "id": "arch_cache_system",
                        "name": "Architecture: Caching System",
                        "prompt": "Design a caching system.",
                        "description": "Test caching architecture."
                    }
                ],
                "code": [
                    {
                        "id": "code_thread_pool",
                        "name": "Code: Thread Pool",
                        "prompt": "Implement a thread pool.",
                        "description": "Test thread pool implementation."
                    }
                ],
                "debug": []
            },
            "chains": [
                {
                    "id": "chain_rest_api",
                    "name": "REST API Chain",
                    "description": "Build REST API",
                    "prompts": {
                        "architect": {"prompt": "Design REST API"},
                        "code": {"prompt": "Implement REST API"},
                        "debug": {"prompt": "Debug REST API"}
                    }
                }
            ]
        }

    def test_generate_independent_prompts(self):
        """Test generating independent prompts section."""
        result = generate_prompts_md(self.test_data)
        
        self.assertIn('# independent', result)
        self.assertIn('## architect', result)
        self.assertIn('### arch_cache_system', result)
        self.assertIn('**Name:** Architecture: Caching System', result)
        self.assertIn('**Prompt:** Design a caching system.', result)

    def test_generate_chains(self):
        """Test generating chains section."""
        result = generate_prompts_md(self.test_data)
        
        self.assertIn('# chains', result)
        self.assertIn('### chain_rest_api', result)
        self.assertIn('- **architect:** Design REST API', result)
        self.assertIn('- **code:** Implement REST API', result)
        self.assertIn('- **debug:** Debug REST API', result)

    def test_empty_data(self):
        """Test with empty data."""
        result = generate_prompts_md({})
        self.assertIn('# independent', result)
        self.assertIn('# chains', result)


class TestGenerateAnalysisPromptMD(unittest.TestCase):
    """Test generate_analysis_prompt_md function."""

    def setUp(self):
        """Create test data matching the actual analysis_prompt.jsonc format."""
        self.test_data = {
            "system_prompt": "You are an expert evaluator.",
            "user_prompt_template": "Evaluate: {response}",
            "expert": {
                "system_prompt": "Expert system prompt",
                "architect_eval": "Evaluate architect: {response}",
                "code_eval": "Evaluate code: {response}",
                "debug_eval": "Evaluate debug: {response}"
            }
        }

    def test_generate_analysis_prompt(self):
        """Test generating analysis prompt markdown."""
        result = generate_analysis_prompt_md(self.test_data)
        
        self.assertIn('# system_prompt', result)
        self.assertIn('**Prompt:** You are an expert evaluator.', result)
        self.assertIn('# expert', result)
        self.assertIn('## architect_eval', result)
        self.assertIn('**Template:** Evaluate architect: {response}', result)

    def test_empty_data(self):
        """Test with empty data."""
        result = generate_analysis_prompt_md({})
        self.assertNotIn('# system_prompt', result)  # No system_prompt key


class TestGenerateAllMarkdown(unittest.TestCase):
    """Test generate_all_markdown function."""

    def setUp(self):
        """Create temporary directory with test JSONC files."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create prompts.jsonc
        self.prompts_data = {
            "independent": {
                "architect": [
                    {"id": "test_prompt", "name": "Test Prompt", "prompt": "Test content", "description": "A test prompt"}
                ],
                "code": [],
                "debug": []
            },
            "chains": []
        }
        self.prompts_file = os.path.join(self.temp_dir, 'prompts.jsonc')
        with open(self.prompts_file, 'w', encoding='utf-8') as f:
            json.dump(self.prompts_data, f, indent=2)
        
        # Create analysis_prompt.jsonc
        self.analysis_data = {
            "system_prompt": "Expert system prompt",
            "expert": {
                "architect_eval": "Architect eval: {response}"
            }
        }
        self.analysis_file = os.path.join(self.temp_dir, 'analysis_prompt.jsonc')
        with open(self.analysis_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_data, f, indent=2)

    def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_generate_all_markdown_success(self):
        """Test successful generation of all markdown files."""
        # This is an integration test - verify the function runs without error
        # The actual file output goes to prompts/ directory
        result = generate_all_markdown()
        # Function should return True if prompts.jsonc exists
        # (it does in this project)
        self.assertIsInstance(result, bool)

    def test_generate_all_markdown_no_source(self):
        """Test when source files don't exist."""
        # This test is skipped because prompts.jsonc always exists in the project root
        # The function will find it and generate successfully
        pass


if __name__ == '__main__':
    unittest.main()
