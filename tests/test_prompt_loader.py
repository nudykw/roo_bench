"""Tests for PromptLoader class."""

import os
import json
import pytest
from prompts.loader import PromptLoader


class TestPromptLoaderPriority:
    """Tests for file priority resolution."""
    
    def test_md_file_priority(self, tmp_path):
        """Test that .md file has priority over .jsonc."""
        # Create both files
        md_file = tmp_path / "prompts.md"
        jsonc_file = tmp_path / "prompts.jsonc"
        
        md_file.write_text("""# independent
## architect
[{"id": "test", "name": "Test", "prompt": "test"}]

# chains
[]
""")
        jsonc_file.write_text('{"independent": {}, "chains": []}')
        
        loader = PromptLoader(prompts_file=str(md_file))
        assert loader.prompts_file == str(md_file)
    
    def test_custom_file_used(self, tmp_path):
        """Test that custom file is used when specified."""
        custom_file = tmp_path / "custom.jsonc"
        custom_file.write_text('{"independent": {}, "chains": []}')
        
        loader = PromptLoader(prompts_file=str(custom_file))
        assert loader.prompts_file == str(custom_file)


class TestPromptLoaderMarkdown:
    """Tests for markdown file parsing."""
    
    def test_parse_markdown_independent(self, tmp_path):
        """Test parsing independent prompts from markdown."""
        md_file = tmp_path / "prompts.md"
        md_file.write_text("""# independent
## architect
[{"id": "arch_test", "name": "Architect Test", "prompt": "Design something"}]

# chains
[]
""")
        loader = PromptLoader(prompts_file=str(md_file))
        data = loader.load()
        
        assert "independent" in data
        assert "architect" in data["independent"]
        assert len(data["independent"]["architect"]) == 1
        assert data["independent"]["architect"][0]["id"] == "arch_test"
    
    def test_parse_markdown_chains(self, tmp_path):
        """Test parsing chains from markdown."""
        md_file = tmp_path / "prompts.md"
        md_file.write_text("""# independent
## architect
[]

# chains
[{"id": "chain_test", "name": "Chain Test", "description": "Test chain"}]
""")
        loader = PromptLoader(prompts_file=str(md_file))
        data = loader.load()
        
        assert "chains" in data
        assert len(data["chains"]) == 1
        assert data["chains"][0]["id"] == "chain_test"


class TestPromptLoaderJsonc:
    """Tests for JSONC file parsing."""
    
    def test_parse_jsonc_file(self, tmp_path):
        """Test parsing JSONC file."""
        jsonc_file = tmp_path / "prompts.jsonc"
        jsonc_file.write_text("""{
  // This is a comment
  "independent": {
    "architect": [{"id": "test", "name": "Test", "prompt": "test"}]
  },
  "chains": []
}""")
        
        loader = PromptLoader(prompts_file=str(jsonc_file))
        data = loader.load()
        
        assert "independent" in data
        assert data["independent"]["architect"][0]["id"] == "test"


class TestPromptLoaderValidation:
    """Tests for structure validation."""
    
    def test_validate_missing_independent(self, tmp_path):
        """Test validation fails when independent is missing."""
        jsonc_file = tmp_path / "prompts.jsonc"
        jsonc_file.write_text('{"chains": []}')
        
        loader = PromptLoader(prompts_file=str(jsonc_file))
        with pytest.raises(ValueError, match="Missing required field: independent"):
            loader.load()
    
    def test_validate_missing_chains(self, tmp_path):
        """Test validation fails when chains is missing."""
        jsonc_file = tmp_path / "prompts.jsonc"
        jsonc_file.write_text('{"independent": {}}')
        
        loader = PromptLoader(prompts_file=str(jsonc_file))
        with pytest.raises(ValueError, match="Missing required field: chains"):
            loader.load()


class TestPromptLoaderMethods:
    """Tests for PromptLoader methods."""
    
    def test_get_independent_prompts(self, tmp_path):
        """Test getting independent prompts for a mode."""
        md_file = tmp_path / "prompts.md"
        md_file.write_text("""# independent
## architect
[{"id": "arch1", "name": "Test 1", "prompt": "p1"}]
## code
[{"id": "code1", "name": "Test 2", "prompt": "p2"}]

# chains
[]
""")
        loader = PromptLoader(prompts_file=str(md_file))
        loader.load()
        
        architect_prompts = loader.get_independent_prompts("architect")
        assert len(architect_prompts) == 1
        assert architect_prompts[0]["id"] == "arch1"
        
        code_prompts = loader.get_independent_prompts("code")
        assert len(code_prompts) == 1
        assert code_prompts[0]["id"] == "code1"
    
    def test_get_chains(self, tmp_path):
        """Test getting chains."""
        md_file = tmp_path / "prompts.md"
        md_file.write_text("""# independent
## architect
[]

# chains
[{"id": "chain1", "name": "Chain 1"}]
""")
        loader = PromptLoader(prompts_file=str(md_file))
        loader.load()
        
        chains = loader.get_chains()
        assert len(chains) == 1
        assert chains[0]["id"] == "chain1"
    
    def test_get_chain_by_id(self, tmp_path):
        """Test getting chain by ID."""
        md_file = tmp_path / "prompts.md"
        md_file.write_text("""# independent
## architect
[]

# chains
[{"id": "chain1", "name": "Chain 1"}]
""")
        loader = PromptLoader(prompts_file=str(md_file))
        loader.load()
        
        chain = loader.get_chain_by_id("chain1")
        assert chain is not None
        assert chain["name"] == "Chain 1"
        
        chain = loader.get_chain_by_id("nonexistent")
        assert chain is None


class TestPromptLoaderEquivalence:
    """Tests for equivalence between .md and .jsonc formats."""
    
    def test_md_jsonc_equivalence(self, tmp_path):
        """Test that .md and .jsonc produce equivalent data."""
        md_file = tmp_path / "prompts.md"
        jsonc_file = tmp_path / "prompts.jsonc"
        
        # Create identical content in both formats
        md_content = """# independent
## architect
[{"id": "test", "name": "Test", "prompt": "test prompt"}]

# chains
[{"id": "chain1", "name": "Chain", "description": "desc", "prompts": {}}]
"""
        jsonc_content = '{"independent": {"architect": [{"id": "test", "name": "Test", "prompt": "test prompt"}]}, "chains": [{"id": "chain1", "name": "Chain", "description": "desc", "prompts": {}}]}'
        
        md_file.write_text(md_content)
        jsonc_file.write_text(jsonc_content)
        
        loader_md = PromptLoader(prompts_file=str(md_file))
        loader_jsonc = PromptLoader(prompts_file=str(jsonc_file))
        
        data_md = loader_md.load()
        data_jsonc = loader_jsonc.load()
        
        assert data_md == data_jsonc