"""Loader for analysis prompts from .md or .jsonc files."""

from __future__ import annotations

import json
import os
from typing import Any


class AnalysisPromptLoader:
    """Load and manage analysis prompts from .md or .jsonc files."""

    DEFAULT_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), 'analysis_prompt.md')
    DEFAULT_JSONC_FALLBACK = os.path.join(os.path.dirname(__file__), 'analysis_prompt.jsonc')

    def __init__(self, prompts_file: str | None = None):
        """Initialize loader with optional custom file path.
        
        Args:
            prompts_file: Optional path to custom prompts file.
                         If None, uses default .md or falls back to .jsonc.
        """
        self._custom_prompts_file = prompts_file
        self.prompts_file = self._resolve_file(prompts_file)
        self._data: dict[str, Any] | None = None

    def _resolve_file(self, prompts_file: str | None) -> str:
        """Resolve with priority: custom > .md > .jsonc.
        
        Args:
            prompts_file: Optional custom file path.
            
        Returns:
            Path to the resolved prompts file.
        """
        if prompts_file:
            return prompts_file
        
        md_path = self.DEFAULT_PROMPTS_FILE
        if os.path.exists(md_path):
            return md_path
        
        jsonc_path = self.DEFAULT_JSONC_FALLBACK
        if os.path.exists(jsonc_path):
            return jsonc_path
        
        return md_path

    @property
    def file_path(self) -> str:
        """Return the path to the currently loaded analysis prompts file."""
        return self.prompts_file or "NOT SET"

    @staticmethod
    def _strip_comments(jsonc: str) -> str:
        """Remove // and /* */ comments from JSONC content.
        
        Args:
            jsonc: JSONC string content.
            
        Returns:
            Clean JSON string.
        """
        result = []
        i = 0
        in_string = False
        escape_next = False
        
        while i < len(jsonc):
            char = jsonc[i]
            
            if escape_next:
                result.append(char)
                escape_next = False
                i += 1
                continue
            
            if char == '\\':
                result.append(char)
                escape_next = True
                i += 1
                continue
            
            if char == '"':
                in_string = not in_string
                result.append(char)
                i += 1
                continue
            
            if not in_string and i + 1 < len(jsonc):
                if jsonc[i:i+2] == '//':
                    # Line comment - skip until end of line
                    while i < len(jsonc) and jsonc[i] != '\n':
                        i += 1
                    continue
                elif jsonc[i:i+2] == '/*':
                    # Block comment - skip until */
                    i += 2
                    while i + 1 < len(jsonc) and jsonc[i:i+2] != '*/':
                        i += 1
                    i += 2  # skip */
                    continue
            
            result.append(char)
            i += 1
        
        return ''.join(result)

    def _parse_markdown(self, md_content: str) -> dict[str, Any]:
        """Parse markdown analysis prompt file to extract sections.
        
        Expected format:
        ```
        # analysis_prompt
        
        ## section_name
        key: value
        
        ## another_section
        key: value
        ```
        
        Args:
            md_content: Markdown string content.
            
        Returns:
            Dictionary with parsed sections.
        """
        data: dict[str, Any] = {}
        current_section = None
        current_key = None
        current_value: list[str] = []
        
        for line in md_content.split('\n'):
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#') and not stripped.startswith('##'):
                continue
            
            # Section header (## header)
            if stripped.startswith('## '):
                # Save previous section
                if current_section:
                    if current_key:
                        final_value = '\n'.join(current_value).strip() if current_value else ''
                        data[f'{current_section}.{current_key}'] = final_value
                    data[current_section] = data.get(current_section, {})
                
                current_section = stripped[3:].strip()
                current_key = None
                current_value = []
                continue
            
            # Key: value pair
            if ':' in stripped and not stripped.startswith('-'):
                # Save previous key-value
                if current_section and current_key:
                    final_value = '\n'.join(current_value).strip() if current_value else ''
                    if isinstance(data.get(current_section), dict):
                        data[current_section][current_key] = final_value
                    else:
                        data[f'{current_section}.{current_key}'] = final_value
                
                parts = stripped.split(':', 1)
                current_key = parts[0].strip()
                value = parts[1].strip()
                current_value = [value] if value else []
                continue
            
            # List item or continuation
            if stripped.startswith('- '):
                if current_value:
                    current_value.append(stripped[2:])
                elif current_key:
                    current_value = [stripped[2:]]
                continue
            
            # Continuation line
            if current_value:
                current_value.append(stripped)
        
        # Save last section
        if current_section and current_key:
            final_value = '\n'.join(current_value).strip() if current_value else ''
            if isinstance(data.get(current_section), dict):
                data[current_section][current_key] = final_value
            else:
                data[f'{current_section}.{current_key}'] = final_value
        
        return data

    def load(self) -> dict[str, Any]:
        """Load analysis prompts from file (.md or .jsonc).
        
        Returns:
            Dictionary with loaded data.
        """
        if self._data is not None:
            return self._data
        
        if not os.path.exists(self.prompts_file):
            raise FileNotFoundError(
                f"Analysis prompts file not found: {self.prompts_file}"
            )
        
        with open(self.prompts_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if self.prompts_file.endswith('.md'):
            self._data = self._parse_markdown(content)
        else:
            json_content = self._strip_comments(content)
            self._data = json.loads(json_content)
        
        return self._data

    @property
    def data(self) -> dict[str, Any]:
        """Return loaded data, loading if necessary."""
        if self._data is None:
            return self.load()
        return self._data

    def get_prompt(self, section: str, key: str | None = None) -> str | None:
        """Get a specific prompt value.
        
        Args:
            section: Section name (e.g., 'architect_eval').
            key: Optional key within section (e.g., 'system_prompt').
                If None, returns the entire section.
                
        Returns:
            Prompt string or section dict, or None if not found.
        """
        data = self.data
        
        if key:
            # Try nested access: section.key
            nested_key = f'{section}.{key}'
            if nested_key in data:
                return data[nested_key]
            # Try dict access: section[key]
            if isinstance(data.get(section), dict):
                return data[section].get(key)
            return None
        
        # Return entire section
        return data.get(section)

    def get_all_sections(self) -> list[str]:
        """Get all section names.
        
        Returns:
            List of section names.
        """
        data = self.data
        
        # If data is flat (dot-notation keys)
        if any('.' in k for k in data):
            sections = set()
            for key in data:
                section = key.split('.')[0]
                sections.add(section)
            return sorted(sections)
        
        # If data is nested (dict of sections)
        return sorted([k for k in data if isinstance(data[k], (dict, str))])
