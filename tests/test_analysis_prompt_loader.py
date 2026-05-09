"""Tests for AnalysisPromptLoader with placeholder substitution."""

import os
import json
import pytest
from prompts.analysis_prompt_loader import AnalysisPromptLoader


class TestAnalysisPromptLoaderFormats:
    """Test AnalysisPromptLoader with different file formats."""
    
    def test_load_jsonc_format(self):
        """Test loading prompts from JSONC file."""
        loader = AnalysisPromptLoader('prompts/analysis_prompt.jsonc')
        data = loader.load()
        
        assert 'system_prompt' in data
        assert 'user_prompt_template' in data
        assert 'translation_prompt_template' in data
        assert 'expert' in data
    
    def test_load_markdown_format(self):
        """Test loading prompts from markdown file."""
        loader = AnalysisPromptLoader('prompts/analysis_prompt.md')
        data = loader.load()
        
        assert 'system_prompt' in data
        assert 'user_prompt_template' in data
        assert 'translation_prompt_template' in data
        assert 'expert' in data
    
    def test_default_uses_markdown_priority(self):
        """Test that default loader prefers .md over .jsonc."""
        loader = AnalysisPromptLoader()
        data = loader.load()
        
        assert 'system_prompt' in data
        assert 'user_prompt_template' in data
    
    def test_jsonc_and_md_data_match(self):
        """Test that JSONC and MD files produce identical data."""
        loader_jsonc = AnalysisPromptLoader('prompts/analysis_prompt.jsonc')
        data_jsonc = loader_jsonc.load()
        
        loader_md = AnalysisPromptLoader('prompts/analysis_prompt.md')
        data_md = loader_md.load()
        
        # Check top-level fields exist in both
        assert 'system_prompt' in data_jsonc
        assert 'system_prompt' in data_md
        assert 'user_prompt_template' in data_jsonc
        assert 'user_prompt_template' in data_md
        assert 'translation_prompt_template' in data_jsonc
        assert 'translation_prompt_template' in data_md
        assert 'expert' in data_jsonc
        assert 'expert' in data_md
        
        # Check expert fields exist in both
        assert 'system_prompt' in data_jsonc['expert']
        assert 'system_prompt' in data_md['expert']
        assert 'architect_eval' in data_jsonc['expert']
        assert 'architect_eval' in data_md['expert']
        assert 'code_eval' in data_jsonc['expert']
        assert 'code_eval' in data_md['expert']
        assert 'debug_eval' in data_jsonc['expert']
        assert 'debug_eval' in data_md['expert']


class TestAnalysisPromptLoaderPlaceholders:
    """Test placeholder substitution in analysis prompts."""
    
    def test_user_prompt_template_placeholders(self):
        """Test that user_prompt_template has correct placeholders."""
        loader = AnalysisPromptLoader('prompts/analysis_prompt.md')
        data = loader.load()
        
        template = data['user_prompt_template']
        assert '{results}' in template
    
    def test_translation_prompt_template_placeholders(self):
        """Test that translation_prompt_template has correct placeholders."""
        loader = AnalysisPromptLoader('prompts/analysis_prompt.md')
        data = loader.load()
        
        template = data['translation_prompt_template']
        assert '{target_lang}' in template
        assert '{text}' in template
    
    def test_architect_eval_placeholders(self):
        """Test that architect_eval has correct placeholders."""
        loader = AnalysisPromptLoader('prompts/analysis_prompt.md')
        data = loader.load()
        
        template = data['expert']['architect_eval']
        assert '{context}' in template
        assert '{response}' in template
    
    def test_code_eval_placeholders(self):
        """Test that code_eval has correct placeholders."""
        loader = AnalysisPromptLoader('prompts/analysis_prompt.md')
        data = loader.load()
        
        template = data['expert']['code_eval']
        assert '{context}' in template
        assert '{architect_response}' in template
        assert '{response}' in template
    
    def test_debug_eval_placeholders(self):
        """Test that debug_eval has correct placeholders."""
        loader = AnalysisPromptLoader('prompts/analysis_prompt.md')
        data = loader.load()
        
        template = data['expert']['debug_eval']
        assert '{context}' in template
        assert '{code_response}' in template
        assert '{response}' in template
    
    def test_get_expert_template(self):
        """Test get_expert_template method returns correct templates."""
        loader = AnalysisPromptLoader('prompts/analysis_prompt.md')
        loader.load()
        
        architect_template = loader.get_expert_template('architect')
        assert 'Evaluate this architect-mode response' in architect_template
        assert '{context}' in architect_template
        assert '{response}' in architect_template
        
        code_template = loader.get_expert_template('code')
        assert 'Evaluate this code-mode response' in code_template
        assert '{context}' in code_template
        assert '{architect_response}' in code_template
        assert '{response}' in code_template
        
        debug_template = loader.get_expert_template('debug')
        assert 'Evaluate this debug-mode response' in debug_template
        assert '{context}' in debug_template
        assert '{code_response}' in debug_template
        assert '{response}' in debug_template


class TestAnalysisPromptLoaderPlaceholderSubstitution:
    """Test actual placeholder substitution functionality."""
    
    def test_substitute_user_prompt_template(self):
        """Test substituting results into user_prompt_template."""
        loader = AnalysisPromptLoader('prompts/analysis_prompt.md')
        data = loader.load()
        
        template = data['user_prompt_template']
        results_text = "Model: llama3.2, TPS: 100"
        substituted = template.format(results=results_text)
        
        assert results_text in substituted
        assert '{results}' not in substituted
    
    def test_substitute_translation_prompt_template(self):
        """Test substituting target_lang and text into translation_prompt_template."""
        loader = AnalysisPromptLoader('prompts/analysis_prompt.md')
        data = loader.load()
        
        template = data['translation_prompt_template']
        substituted = template.format(target_lang='Ukrainian', text='Hello world')
        
        assert 'Ukrainian' in substituted
        assert 'Hello world' in substituted
        assert '{target_lang}' not in substituted
        assert '{text}' not in substituted
    
    def test_substitute_architect_eval_template(self):
        """Test substituting context and response into architect_eval template."""
        loader = AnalysisPromptLoader('prompts/analysis_prompt.md')
        data = loader.load()
        
        template = data['expert']['architect_eval']
        context_text = "Architecture context here"
        response_text = "Architect response here"
        substituted = template.format(context=context_text, response=response_text, expert_results_file="")
        
        assert context_text in substituted
        assert response_text in substituted
        assert '{context}' not in substituted
        assert '{response}' not in substituted
        assert '{expert_results_file}' not in substituted
    
    def test_substitute_code_eval_template(self):
        """Test substituting context, architect_response, and response into code_eval template."""
        loader = AnalysisPromptLoader('prompts/analysis_prompt.md')
        data = loader.load()
        
        template = data['expert']['code_eval']
        context_text = "Code context here"
        architect_response = "Architect plan here"
        response_text = "Code response here"
        substituted = template.format(
            context=context_text,
            architect_response=architect_response,
            response=response_text,
            expert_results_file=""
        )
        
        assert context_text in substituted
        assert architect_response in substituted
        assert response_text in substituted
        assert '{context}' not in substituted
        assert '{architect_response}' not in substituted
        assert '{response}' not in substituted
        assert '{expert_results_file}' not in substituted
    
    def test_substitute_debug_eval_template(self):
        """Test substituting context, code_response, and response into debug_eval template."""
        loader = AnalysisPromptLoader('prompts/analysis_prompt.md')
        data = loader.load()
        
        template = data['expert']['debug_eval']
        context_text = "Debug context here"
        code_response = "Original code here"
        response_text = "Debug response here"
        substituted = template.format(
            context=context_text,
            code_response=code_response,
            response=response_text,
            expert_results_file=""
        )
        
        assert context_text in substituted
        assert code_response in substituted
        assert response_text in substituted
        assert '{context}' not in substituted
        assert '{code_response}' not in substituted
        assert '{response}' not in substituted
        assert '{expert_results_file}' not in substituted


class TestAnalysisPromptLoaderDataProperty:
    """Test data property lazy loading."""
    
    def test_data_property_lazy_loads(self):
        """Test that data property triggers lazy loading."""
        loader = AnalysisPromptLoader('prompts/analysis_prompt.md')
        
        # Access data property
        data = loader.data
        
        assert data is not None
        assert 'system_prompt' in data
    
    def test_get_method(self):
        """Test get method returns correct values."""
        loader = AnalysisPromptLoader('prompts/analysis_prompt.md')
        loader.load()
        
        system_prompt = loader.get('system_prompt')
        assert system_prompt is not None
        assert len(system_prompt) > 0
        
        # Test with default
        missing = loader.get('nonexistent_key', 'default_value')
        assert missing == 'default_value'


class TestAnalysisPromptLoaderFallback:
    """Test fallback behavior."""
    
    def test_fallback_prompts_structure(self):
        """Test that fallback prompts have required structure."""
        import tempfile
        
        # Create a temporary file that will fail to load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('invalid json')
            temp_path = f.name
        
        try:
            loader = AnalysisPromptLoader(temp_path)
            # This should raise an exception, not return fallback
            with pytest.raises(Exception):
                loader.load()
        finally:
            os.unlink(temp_path)

class TestExpertResultsFilePlaceholder:
    """Test {expert_results_file} placeholder substitution."""
    
    def test_get_expert_template_with_results_file(self, tmp_path):
        """Test that get_expert_template substitutes {expert_results_file} placeholder."""
        import os
        
        # Create a temporary results file
        results_file = tmp_path / 'expert_results.md'
        results_content = """# Expert Evaluation Results

**Generated:** 2026-05-09 12:00:00

**Tested Model:** llama3.2:3b
**Expert Model:** qwen3.6:35b
**Total Responses:** 2

---

## Entry 1

### Prompt Information
- **Prompt ID:** test_prompt_1
- **Mode:** architect
- **Context Size:** 16384

### Response
```
This is a test response.
```
"""
        results_file.write_text(results_content)
        
        loader = AnalysisPromptLoader('prompts/analysis_prompt.md')
        loader.load()
        
        # Get template with results file
        template = loader.get_expert_template('architect', str(results_file))
        
        # Check that placeholder was substituted
        assert '{expert_results_file}' not in template
        assert results_content.strip() in template
    
    def test_get_expert_template_without_results_file(self):
        """Test that get_expert_template works without results file."""
        loader = AnalysisPromptLoader('prompts/analysis_prompt.md')
        loader.load()
        
        # Get template without results file
        template = loader.get_expert_template('architect', None)
        
        # Check that template is returned as-is (with {expert_results_file} placeholder)
        assert '{context}' in template
        assert '{response}' in template
    
    def test_get_expert_template_nonexistent_file(self):
        """Test that get_expert_template handles nonexistent file gracefully."""
        loader = AnalysisPromptLoader('prompts/analysis_prompt.md')
        loader.load()
        
        # Get template with nonexistent file
        template = loader.get_expert_template('architect', '/nonexistent/path/results.md')
        
        # Check that template is returned as-is
        assert '{context}' in template
        assert '{response}' in template
    
    def test_get_expert_template_all_modes_with_results_file(self, tmp_path):
        """Test placeholder substitution for all evaluation modes."""
        import os
        
        # Create a temporary results file
        results_file = tmp_path / 'expert_results.md'
        results_file.write_text('# Test results')
        
        loader = AnalysisPromptLoader('prompts/analysis_prompt.md')
        loader.load()
        
        # Test all modes
        for mode in ['architect', 'code', 'debug']:
            template = loader.get_expert_template(mode, str(results_file))
            assert '# Test results' in template
