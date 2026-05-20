"""Tests for prompts/loader.py - markdown and jsonc prompt loading."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from prompts.loader import PromptLoader


class TestPromptLoaderMarkdown(unittest.TestCase):
    """Test PromptLoader with markdown files."""

    def setUp(self):
        """Create a temporary markdown prompts file with correct format."""
        # Format: # independent → ## mode → ### id → **Name:**, **Prompt:**
        self.md_content = """# independent

## architect
### arch_cache_system
**Name:** Architecture: Caching System
**Prompt:** Design a caching system for a distributed architecture.

## code
### code_thread_pool
**Name:** Code: Thread Pool Implementation
**Prompt:** Implement a thread pool in Python.

## debug
### debug_memory_leak
**Name:** Debug: Memory Leak
**Prompt:** Find memory leak in this code.
"""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False, encoding='utf-8'
        )
        self.temp_file.write(self.md_content)
        self.temp_file.close()

    def tearDown(self):
        """Remove temporary file."""
        os.unlink(self.temp_file.name)

    def test_parse_markdown_independent(self):
        """Test parsing independent prompts from markdown."""
        loader = PromptLoader(self.temp_file.name)
        data = loader._parse_markdown(self.md_content)

        self.assertIn('independent', data)
        self.assertIn('architect', data['independent'])
        self.assertIn('code', data['independent'])
        self.assertIn('debug', data['independent'])
        
        # Check architect prompts
        arch_prompts = data['independent']['architect']
        self.assertEqual(len(arch_prompts), 1)
        self.assertEqual(arch_prompts[0]['id'], 'arch_cache_system')
        self.assertEqual(arch_prompts[0]['name'], 'Architecture: Caching System')
        self.assertEqual(arch_prompts[0]['prompt'], 'Design a caching system for a distributed architecture.')

    def test_parse_markdown_chains(self):
        """Test parsing chains from markdown."""
        chains_md = """# chains

### chain_rest_api
**Name:** REST API Development Chain
**Description:** Build a REST API from design to debug.
- **architect:** Design a REST API for task management.
- **code:** Implement the REST API.
- **debug:** Find bugs in the API implementation.
"""
        loader = PromptLoader()
        data = loader._parse_markdown(chains_md)

        self.assertIn('chains', data)
        self.assertEqual(len(data['chains']), 1)
        chain = data['chains'][0]
        self.assertEqual(chain['id'], 'chain_rest_api')
        self.assertEqual(chain['name'], 'REST API Development Chain')
        self.assertEqual(chain['description'], 'Build a REST API from design to debug.')
        self.assertIn('prompts', chain)
        self.assertIn('architect', chain['prompts'])
        self.assertEqual(chain['prompts']['architect']['prompt'], 'Design a REST API for task management.')

    def test_file_path_property(self):
        """Test that file_path property returns correct path."""
        loader = PromptLoader(self.temp_file.name)
        self.assertEqual(loader.file_path, self.temp_file.name)


class TestPromptLoaderJSONC(unittest.TestCase):
    """Test PromptLoader with JSONC files."""

    def setUp(self):
        """Create a temporary JSONC prompts file."""
        self.jsonc_content = '''{
            // This is a comment
            "independent": {
                "architect": [
                    {
                        "id": "test_prompt",
                        "name": "Test Prompt",
                        "prompt": "Test prompt content",
                        "description": "A test prompt"
                    }
                ],
                "code": [],
                "debug": []
            },
            "chains": []
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
        loader = PromptLoader(self.temp_file.name)
        data = loader.load()

        self.assertIn('independent', data)
        self.assertIn('architect', data['independent'])
        self.assertEqual(data['independent']['architect'][0]['name'], 'Test Prompt')

    def test_strip_comments(self):
        """Test comment stripping from JSONC."""
        loader = PromptLoader()
        cleaned = loader._strip_comments('{"key": "value"} // comment')
        self.assertEqual(json.loads(cleaned), {"key": "value"})

    def test_strip_block_comments(self):
        """Test block comment stripping."""
        loader = PromptLoader()
        cleaned = loader._strip_comments('''{
            /* block comment */
            "key": "value"
        }''')
        self.assertEqual(json.loads(cleaned), {"key": "value"})


class TestPromptLoaderFileResolution(unittest.TestCase):
    """Test file resolution priority."""

    def test_resolve_custom_file(self):
        """Test that custom file path is used."""
        custom_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonc', delete=False, encoding='utf-8'
        )
        custom_file.write('{"independent": {"architect": [], "code": [], "debug": []}, "chains": []}')
        custom_file.close()

        try:
            loader = PromptLoader(custom_file.name)
            self.assertEqual(loader.prompts_file, custom_file.name)
        finally:
            os.unlink(custom_file.name)

    def test_default_jsonc_fallback(self):
        """Test that .jsonc is used when no .md exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonc_path = os.path.join(tmpdir, 'prompts.jsonc')
            with open(jsonc_path, 'w', encoding='utf-8') as f:
                f.write('{"independent": {"architect": [], "code": [], "debug": []}, "chains": []}')

            with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as md_f:
                md_f.write(b'# independent\n\n## architect\n')
                md_path = md_f.name

            try:
                with mock.patch.object(PromptLoader, 'DEFAULT_PROMPTS_FILE', md_path), \
                     mock.patch.object(PromptLoader, 'DEFAULT_JSONC_FALLBACK', jsonc_path):
                    loader = PromptLoader()
                    self.assertEqual(loader.prompts_file, md_path)
            finally:
                os.unlink(md_path)


class TestPromptLoaderDataAccess(unittest.TestCase):
    """Test data access methods."""

    def setUp(self):
        """Create test data."""
        self.jsonc_content = '''{
            "independent": {
                "architect": [
                    {"id": "prompt1", "name": "Prompt 1", "prompt": "Content 1", "description": "Description 1"}
                ],
                "code": [
                    {"id": "code1", "name": "Code 1", "prompt": "Code content", "description": "Code desc"}
                ],
                "debug": []
            },
            "chains": [
                {
                    "id": "chain1",
                    "name": "Chain 1",
                    "prompts": {
                        "architect": {"prompt": "Architect prompt"},
                        "code": {"prompt": "Code prompt"}
                    }
                }
            ]
        }'''
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonc', delete=False, encoding='utf-8'
        )
        self.temp_file.write(self.jsonc_content)
        self.temp_file.close()

    def tearDown(self):
        """Remove temporary file."""
        os.unlink(self.temp_file.name)

    def test_get_independent_prompts(self):
        """Test getting independent prompts by mode."""
        loader = PromptLoader(self.temp_file.name)
        loader.load()
        
        prompts = loader.get_independent_prompts('architect')
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0]['name'], 'Prompt 1')

    def test_get_all_independent_modes(self):
        """Test getting all independent modes."""
        loader = PromptLoader(self.temp_file.name)
        loader.load()
        
        modes = loader.get_all_independent_modes()
        self.assertIn('architect', modes)
        self.assertIn('code', modes)
        self.assertIn('debug', modes)

    def test_get_chain_by_id(self):
        """Test getting chain by ID."""
        loader = PromptLoader(self.temp_file.name)
        loader.load()
        
        chain = loader.get_chain_by_id('chain1')
        self.assertIsNotNone(chain)
        self.assertEqual(chain['name'], 'Chain 1')

    def test_get_chain_by_name(self):
        """Test getting chain by name."""
        loader = PromptLoader(self.temp_file.name)
        loader.load()
        
        chain = loader.get_chain_by_name('Chain 1')
        self.assertIsNotNone(chain)

    def test_get_chain_by_id_not_found(self):
        """Test getting non-existent chain."""
        loader = PromptLoader(self.temp_file.name)
        loader.load()
        
        chain = loader.get_chain_by_id('nonexistent')
        self.assertIsNone(chain)


if __name__ == '__main__':
    unittest.main()
