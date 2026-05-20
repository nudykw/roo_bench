"""JSONC (JSON with Comments) and Markdown loader for benchmark prompts."""

import json
import os
import re
from typing import Any


class PromptLoader:
    """Load and manage benchmark prompts from .md or .jsonc files."""
    
    DEFAULT_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), 'prompts.md')
    DEFAULT_JSONC_FALLBACK = os.path.join(os.path.dirname(__file__), '..', 'prompts.jsonc')
    
    def __init__(self, prompts_file: str | None = None):
        """Initialize prompt loader.
        
        Args:
            prompts_file: Path to prompts file (.md, .jsonc, or .json). If None, uses default.
        """
        self._custom_prompts_file = prompts_file
        self.prompts_file = self._resolve_prompts_file(prompts_file)
        self._data: dict[str, Any] | None = None
    
    def _resolve_prompts_file(self, prompts_file: str | None) -> str:
        """Resolve the prompts file path with priority: .md > .jsonc > custom.
        
        Args:
            prompts_file: Custom prompts file path or None
            
        Returns:
            str: Resolved file path
        """
        # If custom file specified, use it directly
        if prompts_file:
            return prompts_file
        
        # Check for .md file first (priority)
        md_path = self.DEFAULT_PROMPTS_FILE
        if os.path.exists(md_path):
            return md_path
        
        # Fallback to .jsonc file
        jsonc_path = self.DEFAULT_JSONC_FALLBACK
        if os.path.exists(jsonc_path):
            return jsonc_path
        
        # Return default (will raise FileNotFoundError on load)
        return md_path
    
    @property
    def file_path(self) -> str:
        """Return the path to the currently loaded prompts file."""
        return self.prompts_file or "NOT SET"
    
    def _strip_comments(self, jsonc: str) -> str:
        """Remove comments from JSONC content.
        
        Handles:
        - Single-line comments: // ...
        - Multi-line comments: /* ... */
        - Preserves strings (doesn't strip // or /* inside quotes)
        
        Args:
            jsonc: JSONC string with comments
            
        Returns:
            str: Clean JSON string without comments
        """
        result = []
        i = 0
        in_string = False
        escape = False
        
        while i < len(jsonc):
            # Handle escape sequences in strings
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
            
            # Toggle string mode
            if jsonc[i] == '"':
                in_string = not in_string
                result.append(jsonc[i])
                i += 1
                continue
            
            # Skip comments only outside strings
            if not in_string:
                # Single-line comment
                if jsonc[i:i+2] == '//':
                    # Skip until end of line
                    while i < len(jsonc) and jsonc[i] != '\n':
                        i += 1
                    continue
                # Multi-line comment
                if jsonc[i:i+2] == '/*':
                    end_pos = jsonc.find('*/', i + 2)
                    if end_pos == -1:
                        # Unclosed comment, skip to end
                        i = len(jsonc)
                    else:
                        i = end_pos + 2
                    continue
            
            result.append(jsonc[i])
            i += 1
        
        return ''.join(result)
    
    def _parse_markdown(self, md_content: str) -> dict[str, Any]:
        """Parse markdown file to extract prompt sections.
        
        Supports:
        1. Named prompt format: ### id headers with **Name:** and **Prompt:** fields
        2. Chain format: ### chain_id with **Name:**, **Description:**, **Prompts:** fields
        
        Args:
            md_content: Markdown content
            
        Returns:
            dict: Parsed structure with 'independent' and 'chains' sections
        """
        result = {}
        lines = md_content.split('\n')
        current_section = None
        current_subsection = None
        current_prompt_id = None
        in_html_comment = False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Track HTML comment state
            if '<!--' in line:
                in_html_comment = True
            if '-->' in line:
                in_html_comment = False
                i += 1
                continue
            
            if in_html_comment:
                i += 1
                continue
            
            # Top-level sections
            if line.startswith('# chains') and not line.startswith('##'):
                current_section = 'chains'
                result['chains'] = []
                i += 1
                continue
            
            if line.startswith('# independent') and not line.startswith('##'):
                current_section = 'independent'
                result['independent'] = {}
                i += 1
                continue
            
            # Mode sections (## architect, ## code, ## debug)
            mode_match = re.match(r'^## (architect|code|debug)', line)
            if mode_match and current_section == 'independent':
                current_subsection = mode_match.group(1)
                if current_subsection not in result['independent']:
                    result['independent'][current_subsection] = []
                i += 1
                continue
            
            # Prompt entries (### id)
            prompt_match = re.match(r'^### (.+)', line)
            if prompt_match:
                current_prompt_id = prompt_match.group(1).strip()
                
                if current_section == 'independent' and current_subsection:
                    prompt_data = {'id': current_prompt_id}
                    result['independent'][current_subsection].append(prompt_data)
                elif current_section == 'chains':
                    chain_entry = {'id': current_prompt_id}
                    if 'chains' not in result:
                        result['chains'] = []
                    result['chains'].append(chain_entry)
                
                i += 1
                continue
            
            # Parse **Name:** field
            name_match = re.match(r'^\*\*Name:\*\*\s*(.+)', line)
            if name_match and current_prompt_id:
                if current_section == 'independent' and current_subsection:
                    result['independent'][current_subsection][-1]['name'] = name_match.group(1).strip()
                elif current_section == 'chains':
                    result['chains'][-1]['name'] = name_match.group(1).strip()
                i += 1
                continue
            
            # Parse **Description:** field (chains only)
            desc_match = re.match(r'^\*\*Description:\*\*\s*(.+)', line)
            if desc_match and current_section == 'chains':
                result['chains'][-1]['description'] = desc_match.group(1).strip()
                i += 1
                continue
            
            # Parse **Prompt:** field
            prompt_match = re.match(r'^\*\*Prompt:\*\*\s*(.+)', line)
            if prompt_match and current_prompt_id:
                if current_section == 'independent' and current_subsection:
                    result['independent'][current_subsection][-1]['prompt'] = prompt_match.group(1).strip()
                elif current_section == 'chains':
                    result['chains'][-1]['prompt'] = prompt_match.group(1).strip()
                i += 1
                continue
            
            # Parse chain mode prompts (- **architect:**, - **code:**, - **debug:**)
            chain_mode_match = re.match(r'^- \*\*(architect|code|debug):\*\*\s*(.*)', line)
            if chain_mode_match and current_section == 'chains':
                mode = chain_mode_match.group(1)
                mode_prompt = chain_mode_match.group(2).strip()
                
                if 'prompts' not in result['chains'][-1]:
                    result['chains'][-1]['prompts'] = {}
                result['chains'][-1]['prompts'][mode] = {'prompt': mode_prompt}
                i += 1
                continue
            
            i += 1
        
        return result
    
    def load(self) -> dict[str, Any]:
        """Load prompts from file (.md or .jsonc).
        
        Returns:
            dict: Parsed prompts configuration
            
        Raises:
            FileNotFoundError: If prompts file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        if not self.prompts_file:
            raise FileNotFoundError("No prompts file specified")
        
        with open(self.prompts_file, encoding='utf-8') as f:
            content = f.read()
        
        if self.prompts_file.endswith('.md'):
            self._data = self._parse_markdown(content)
        else:
            # JSONC parsing
            json_content = self._strip_comments(content)
            self._data = json.loads(json_content)
        
        return self._data
    
    @property
    def data(self) -> dict[str, Any]:
        """Lazy-load and return prompts data."""
        if self._data is None:
            self.load()
        return self._data if self._data is not None else {}
    
    def get_independent_prompts(self, mode: str) -> list[dict[str, str]]:
        """Get independent prompts for a specific mode.
        
        Args:
            mode: One of 'architect', 'code', 'debug'
            
        Returns:
            list: List of prompt dictionaries with 'id', 'name', 'prompt' keys
        """
        data = self.data
        return data.get('independent', {}).get(mode, [])  # type: ignore[no-any-return]
    
    def get_all_independent_modes(self) -> list[str]:
        """Get list of all available independent modes.
        
        Returns:
            list: List of mode names
        """
        return list(self.data.get('independent', {}).keys())
    
    def get_all_independent_prompts_ordered(self) -> list[dict[str, Any]]:
        """Get all independent prompts in order: architect → code → debug.
        
        Returns:
            list: List of prompt dictionaries with 'mode', 'id', 'name', 'prompt' keys
        """
        all_prompts: list[dict[str, Any]] = []
        for mode in ['architect', 'code', 'debug']:
            prompts = self.get_independent_prompts(mode)
            for prompt in prompts:
                prompt_with_mode = dict(prompt)
                prompt_with_mode['mode'] = mode  # Add mode to each prompt
                all_prompts.append(prompt_with_mode)
        
        return all_prompts
    
    def get_chains(self) -> list[dict[str, Any]]:
        """Get all prompt chains.
        
        Returns:
            list: List of chain dictionaries
        """
        return self.data.get('chains', [])
    
    def get_chain_by_id(self, chain_id: str) -> dict[str, Any] | None:
        """Get a specific chain by ID.
        
        Args:
            chain_id: Chain identifier (e.g., 'chain_rest_api')
            
        Returns:
            dict or None: Chain dictionary if found, None otherwise
        """
        for chain in self.get_chains():
            if chain.get('id') == chain_id:
                return chain
        return None
    
    def get_chain_by_name(self, chain_name: str) -> dict[str, Any] | None:
        """Get a specific chain by name.
        
        Args:
            chain_name: Chain name (e.g., 'REST API Server')
            
        Returns:
            dict or None: Chain dictionary if found, None otherwise
        """
        for chain in self.get_chains():
            if chain.get('name') == chain_name:
                return chain
        return None
    
    def build_chain_context(self, chain: dict[str, Any],
                           architect_response: str | None = None,
                           code_response: str | None = None) -> dict[str, dict[str, str]]:
        """Build final prompts for a chain with context substitution.
        
        Args:
            chain: Chain dictionary from config
            architect_response: Response from architect mode (replaces [ARCHITECT_PLAN])
            code_response: Response from code mode (replaces [CODE_FROM_CODE])
            
        Returns:
            dict: Dictionary with mode keys and final prompt strings
        """
        prompts = chain.get('prompts', {})
        result: dict[str, dict[str, str]] = {}
        
        for mode in ['architect', 'code', 'debug']:
            if mode in prompts:
                prompt_data = prompts[mode]
                prompt_text = prompt_data.get('prompt', '')
                
                # Substitute context placeholders
                if mode == 'code' and architect_response:
                    prompt_text = prompt_text.replace('[ARCHITECT_PLAN]', architect_response)
                elif mode == 'debug' and code_response:
                    prompt_text = prompt_text.replace('[CODE_FROM_CODE]', code_response)
                
                result[mode] = {
                    'id': prompt_data.get('id', f'{mode}_{chain.get("id", "unknown")}'),
                    'name': prompt_data.get('name', mode),
                    'prompt': prompt_text
                }
        
        return result
