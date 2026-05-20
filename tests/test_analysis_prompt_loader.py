"""Tests for prompts/analysis_prompt_loader.py - markdown and jsonc analysis prompt loading."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from prompts.analysis_prompt_loader import AnalysisPromptLoader


class TestAnalysisPromptLoaderMarkdown(unittest.TestCase):
    """Test AnalysisPromptLoader with markdown files."""

    def setUp(self):
        """Create a temporary markdown analysis prompt file."""
        self.md_content = """# analysis_prompt

## architect_eval
system_prompt: You are an expert architect evaluator.
architect_eval: Evaluate this architectural response on 0-100 scale.
Response:
{response}

Score (0-100 only):

## code_eval
system_prompt: You are an expert code evaluator.
code_eval: Evaluate this code response on 0-100 scale.
Response:
{response}

Score (0-100 only):

## debug_eval
system_prompt: You are an expert debug evaluator.
debug_eval: Evaluate this debug response on 0-100 scale.
Response:
{response}

Score (0-100 only):
"""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False, encoding='utf-8'
        )
        self.temp_file.write(self.md_content)
        self.temp_file.close()

    def tearDown(self):
        """Remove temporary file."""
        os.unlink(self.temp_file.name)

    def test_parse_markdown_sections(self):
        """Test parsing sections from markdown."""
        loader = AnalysisPromptLoader(self.temp_file.name)
        data = loader._parse_markdown(self.md_content)

        # Data uses flat dot-notation keys
        self.assertIn('architect_eval.system_prompt', data)
        self.assertIn('code_eval.system_prompt', data)
        self.assertIn('debug_eval.system_prompt', data)

    def test_parse_markdown_keys(self):
        """Test parsing keys within sections."""
        loader = AnalysisPromptLoader(self.temp_file.name)
        data = loader._parse_markdown(self.md_content)

        self.assertIn('architect_eval.system_prompt', data)
        self.assertEqual(
            data['architect_eval.system_prompt'],
            'You are an expert architect evaluator.'
        )

    def test_file_path_property(self):
        """Test that file_path property returns correct path."""
        loader = AnalysisPromptLoader(self.temp_file.name)
        self.assertEqual(loader.file_path, self.temp_file.name)


class TestAnalysisPromptLoaderJSONC(unittest.TestCase):
    """Test AnalysisPromptLoader with JSONC files."""

    def setUp(self):
        """Create a temporary JSONC analysis prompt file."""
        self.jsonc_content = '''{
            // Expert evaluation prompts
            "expert": {
                "system_prompt": "You are an expert evaluator.",
                "architect_eval": "Evaluate architect response: {response}",
                "code_eval": "Evaluate code response: {response}",
                "debug_eval": "Evaluate debug response: {response}"
            }
        }'''
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonc', delete=False, encoding='utf-8'
        )
        self.temp_file.write(self.jsonc_content)
        self.temp_file.close()

    def tearDown(self):
        """Remove temporary file."""
        os.unlink(self.temp_file.name)

    def test_parse_jsonc(self):
        """Test parsing JSONC file."""
        loader = AnalysisPromptLoader(self.temp_file.name)
        data = loader.load()

        self.assertIn('expert', data)
        self.assertIn('architect_eval', data['expert'])

    def test_strip_comments(self):
        """Test comment stripping from JSONC."""
        loader = AnalysisPromptLoader()
        cleaned = loader._strip_comments('{"key": "value"} // comment')
        self.assertEqual(json.loads(cleaned), {"key": "value"})


class TestAnalysisPromptLoaderGetPrompt(unittest.TestCase):
    """Test get_prompt method."""

    def setUp(self):
        """Create test data."""
        self.jsonc_content = '''{
            "expert": {
                "system_prompt": "System prompt",
                "architect_eval": "Architect eval: {response}"
            }
        }'''
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonc', delete=False, encoding='utf-8'
        )
        self.temp_file.write(self.jsonc_content)
        self.temp_file.close()

    def tearDown(self):
        """Remove temporary file."""
        os.unlink(self.temp_file.name)

    def test_get_prompt_with_key(self):
        """Test getting prompt with section and key."""
        loader = AnalysisPromptLoader(self.temp_file.name)
        result = loader.get_prompt('expert', 'system_prompt')
        self.assertEqual(result, 'System prompt')

    def test_get_prompt_with_section(self):
        """Test getting entire section."""
        loader = AnalysisPromptLoader(self.temp_file.name)
        result = loader.get_prompt('expert')
        self.assertIsInstance(result, dict)
        self.assertIn('system_prompt', result)

    def test_get_prompt_not_found(self):
        """Test getting non-existent prompt."""
        loader = AnalysisPromptLoader(self.temp_file.name)
        result = loader.get_prompt('nonexistent', 'key')
        self.assertIsNone(result)


class TestAnalysisPromptLoaderFileResolution(unittest.TestCase):
    """Test file resolution priority."""

    def test_resolve_custom_file(self):
        """Test that custom file path is used."""
        custom_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonc', delete=False, encoding='utf-8'
        )
        custom_file.write('{"expert": {}}')
        custom_file.close()

        try:
            loader = AnalysisPromptLoader(custom_file.name)
            self.assertEqual(loader.prompts_file, custom_file.name)
        finally:
            os.unlink(custom_file.name)

    def test_default_jsonc_fallback(self):
        """Test that .jsonc is used when no .md exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonc_path = os.path.join(tmpdir, 'analysis_prompt.jsonc')
            with open(jsonc_path, 'w', encoding='utf-8') as f:
                f.write('{"expert": {}}')

            with patch.object(AnalysisPromptLoader, 'DEFAULT_PROMPTS_FILE', '/nonexistent/prompts.md'), \
                 patch.object(AnalysisPromptLoader, 'DEFAULT_JSONC_FALLBACK', jsonc_path):
                loader = AnalysisPromptLoader()
                self.assertEqual(loader.prompts_file, jsonc_path)


class TestAnalysisPromptLoaderGetSections(unittest.TestCase):
    """Test get_all_sections method."""

    def setUp(self):
        """Create test data."""
        self.jsonc_content = '''{
            "expert": {
                "system_prompt": "System prompt",
                "architect_eval": "Architect eval"
            },
            "analyze": {
                "prompt": "Analyze results"
            }
        }'''
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonc', delete=False, encoding='utf-8'
        )
        self.temp_file.write(self.jsonc_content)
        self.temp_file.close()

    def tearDown(self):
        """Remove temporary file."""
        os.unlink(self.temp_file.name)

    def test_get_all_sections(self):
        """Test getting all section names."""
        loader = AnalysisPromptLoader(self.temp_file.name)
        sections = loader.get_all_sections()
        
        self.assertIn('expert', sections)
        self.assertIn('analyze', sections)
        self.assertEqual(len(sections), 2)


if __name__ == '__main__':
    unittest.main()
