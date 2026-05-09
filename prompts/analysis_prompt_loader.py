"""Loader for analysis_prompt files (.md, .jsonc, .json)."""

import json
import os
import re
import logging
from typing import Dict, Any, Optional


logger = logging.getLogger('roo_bench')


class AnalysisPromptLoader:
    """Load and manage analysis prompts from file.
    
    Supports multiple formats with priority:
    1. .md file (default: analysis_prompt.md)
    2. .jsonc file (fallback: analysis_prompt.jsonc)
    3. Custom file specified via prompts_file parameter
    """
    
    DEFAULT_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), 'analysis_prompt.md')
    DEFAULT_JSONC_FALLBACK = os.path.join(os.path.dirname(__file__), 'analysis_prompt.jsonc')
    
    def __init__(self, prompts_file: Optional[str] = None):
        """Initialize analysis prompt loader.
        
        Args:
            prompts_file: Path to prompts file (.md, .jsonc, or .json). If None, uses default.
        """
        self._custom_prompts_file = prompts_file
        self.prompts_file = self._resolve_prompts_file(prompts_file)
        self._data: Optional[Dict] = None
    
    def _resolve_prompts_file(self, prompts_file: Optional[str]) -> str:
        """Resolve the prompts file path with priority: .md > .jsonc > custom."""
        if prompts_file:
            return prompts_file
        
        md_path = self.DEFAULT_PROMPTS_FILE
        if os.path.exists(md_path):
            return md_path
        
        jsonc_path = self.DEFAULT_JSONC_FALLBACK
        if os.path.exists(jsonc_path):
            return jsonc_path
        
        return md_path
    
    def _strip_comments(self, jsonc: str) -> str:
        """Remove comments from JSONC content."""
        result = []
        i = 0
        in_string = False
        escape = False
        
        while i < len(jsonc):
            if in_string and escape:
                result.append(jsonc[i])
                escape = False
                i += 1
                continue
            
            if escape:
                escape = False
                result.append(jsonc[i])
                i += 1
                continue
            
            if jsonc[i] == '\\':
                escape = True
                result.append(jsonc[i])
                i += 1
                continue
            
            if in_string and jsonc[i] == '"':
                in_string = not in_string
                result.append(jsonc[i])
                i += 1
                continue
            
            if not in_string:
                if jsonc[i:i+2] == '//':
                    while i < len(jsonc) and jsonc[i] != '\n':
                        i += 1
                    continue
                if jsonc[i:i+2] == '/*':
                    end_pos = jsonc.find('*/', i + 2)
                    if end_pos == -1:
                        i = len(jsonc)
                    else:
                        i = end_pos + 2
                    continue
            
            result.append(jsonc[i])
            i += 1
        
        return ''.join(result)
    
    def _parse_markdown_analysis_prompt(self, md_content: str) -> Dict[str, Any]:
        """Parse analysis_prompt.md format with nested sections.
        
        Supports:
        - # section (top-level)
        - ## subsection (nested under section)
        """
        result = {}
        lines = md_content.split('\n')
        current_section = None
        current_subsection = None
        current_content = []
        in_html_comment = False
        
        for line in lines:
            # Track HTML comment state
            if '<!--' in line:
                in_html_comment = True
            if '-->' in line:
                in_html_comment = False
                continue
            
            # Skip content inside HTML comments
            if in_html_comment:
                continue
            
            # Check for top-level section (# heading)
            section_match = re.match(r'^#\s+(\w+)$', line)
            if section_match:
                # Save previous content
                self._save_md_content(result, current_section, current_subsection, current_content)
                
                current_section = section_match.group(1)
                current_subsection = None
                current_content = []
                result[current_section] = {}
                continue
            
            # Check for subsection (## heading)
            sub_match = re.match(r'^##\s+(\w+)$', line)
            if sub_match:
                # Save previous content
                self._save_md_content(result, current_section, current_subsection, current_content)
                
                current_subsection = sub_match.group(1)
                current_content = []
                continue
            
            # Add content
            current_content.append(line)
        
        # Save last content
        self._save_md_content(result, current_section, current_subsection, current_content)
        
        return result
    
    def _save_md_content(self, result: Dict, section: Optional[str], 
                         subsection: Optional[str], content: List[str]) -> None:
        """Save markdown content to result structure."""
        if not content or not section:
            return
        
        content_str = '\n'.join(content).strip()
        if not content_str:
            return
        
        if subsection:
            # Nested content under subsection
            if subsection not in result[section]:
                result[section][subsection] = content_str
        else:
            # Top-level content
            result[section] = content_str
    
    def load(self) -> Dict[str, Any]:
        """Load prompts from file (.md, .jsonc, or .json)."""
        ext = os.path.splitext(self.prompts_file)[1].lower()
        
        with open(self.prompts_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if ext == '.md':
            self._data = self._parse_markdown_analysis_prompt(content)
        else:
            json_content = self._strip_comments(content)
            self._data = json.loads(json_content)
        
        return self._data
    
    @property
    def data(self) -> Dict[str, Any]:
        """Lazy-load and return prompts data."""
        if self._data is None:
            self.load()
        return self._data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a specific key from the prompts data."""
        return self.data.get(key, default)
    
    def get_expert_template(self, mode: str, expert_results_file: Optional[str] = None) -> str:
        """Get expert evaluation template for a specific mode.
        
        Args:
            mode: Evaluation mode ('architect', 'code', 'debug').
            expert_results_file: Path to the expert results file for placeholder substitution.
                                 If provided, {expert_results_file} placeholder will be replaced
                                 with the file contents.
        
        Returns:
            Template string with placeholders substituted if file path provided.
        """
        expert = self.data.get('expert', {})
        if isinstance(expert, dict):
            templates = {
                'architect': expert.get('architect_eval', ''),
                'code': expert.get('code_eval', ''),
                'debug': expert.get('debug_eval', ''),
            }
            template = templates.get(mode, templates.get('architect', ''))
            
            # Substitute {expert_results_file} placeholder if file path provided
            if expert_results_file and os.path.exists(expert_results_file):
                try:
                    with open(expert_results_file, 'r', encoding='utf-8') as f:
                        file_contents = f.read()
                    template = template.replace('{expert_results_file}', file_contents)
                except Exception as e:
                    logger.warning(f"Failed to read expert results file: {e}")
            
            return template
        return ''