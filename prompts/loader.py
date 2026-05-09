"""JSONC (JSON with Comments) loader for benchmark prompts."""

import json
import re
import os
from typing import Dict, List, Any, Optional


class PromptLoader:
    """Load and manage benchmark prompts from file."""
    
    DEFAULT_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), 'prompts.md')
    DEFAULT_JSONC_FALLBACK = os.path.join(os.path.dirname(__file__), 'prompts.jsonc')
    
    def __init__(self, prompts_file: Optional[str] = None):
        """Initialize prompt loader.
        
        Args:
            prompts_file: Path to prompts file (.md, .jsonc, or .json). If None, uses default.
        """
        self._custom_prompts_file = prompts_file
        self.prompts_file = self._resolve_prompts_file(prompts_file)
        self._data: Optional[Dict] = None
    
    def _resolve_prompts_file(self, prompts_file: Optional[str]) -> str:
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
    
    def _parse_markdown(self, md_content: str) -> Dict[str, Any]:
        """Parse markdown file to extract prompt sections.
        
        Supports two formats:
        1. JSON array format: Content starts with '['
        2. Named prompt format: ### id headers with **Name:** and **Prompt:** fields
        
        Args:
            md_content: Markdown content
            
        Returns:
            dict: Parsed structure with sections
        """
        result = {}
        lines = md_content.split('\n')
        current_section = None
        current_subsection = None
        current_prompt_id = None
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
            
            # Check for main section (# heading)
            main_match = re.match(r'^#\s+(\w+)$', line)
            if main_match:
                # Save previous content
                self._save_prompt_content(
                    result, current_section, current_subsection, 
                    current_prompt_id, current_content
                )
                
                current_section = main_match.group(1)
                current_subsection = None
                current_prompt_id = None
                current_content = []
                result[current_section] = {}
                continue
            
            # Check for subsection (## heading)
            sub_match = re.match(r'^##\s+(\w+)$', line)
            if sub_match:
                # Save previous content
                self._save_prompt_content(
                    result, current_section, current_subsection,
                    current_prompt_id, current_content
                )
                
                current_subsection = sub_match.group(1)
                current_prompt_id = None
                current_content = []
                # Ensure section is a dict for subsections
                if not isinstance(result[current_section], dict):
                    result[current_section] = {}
                continue
            
            # Check for prompt ID (### heading)
            prompt_match = re.match(r'^###\s+(\w+)$', line)
            if prompt_match:
                # Save previous content
                self._save_prompt_content(
                    result, current_section, current_subsection,
                    current_prompt_id, current_content
                )
                
                current_prompt_id = prompt_match.group(1)
                current_content = []
                continue
            
            # Add content
            current_content.append(line)
        
        # Save last content
        self._save_prompt_content(
            result, current_section, current_subsection,
            current_prompt_id, current_content
        )
        
        return result
    
    def _save_prompt_content(self, result: Dict, section: Optional[str], 
                             subsection: Optional[str], prompt_id: Optional[str],
                             content: List[str]) -> None:
        """Save prompt content to result structure.
        
        Args:
            result: Result dictionary to update
            section: Current section name
            subsection: Current subsection name
            prompt_id: Current prompt ID (if parsing named prompts)
            content: Content lines to save
        """
        if not content or not section:
            return
        
        content_str = '\n'.join(content).strip()
        if not content_str:
            return
        
        # Check if content is JSON array format
        if content_str.startswith('['):
            try:
                parsed = json.loads(content_str)
                if subsection:
                    result[section][subsection] = parsed
                else:
                    result[section] = parsed
                return
            except json.JSONDecodeError:
                pass
        
        # Parse named prompt format
        if prompt_id and subsection:
            prompt_data = self._parse_named_prompt(content_str, prompt_id)
            if prompt_data:
                if subsection not in result[section]:
                    result[section][subsection] = []
                result[section][subsection].append(prompt_data)
    
    def _parse_named_prompt(self, content: str, prompt_id: str) -> Optional[Dict[str, str]]:
        """Parse a named prompt from markdown content.
        
        Format:
        **Name:** Prompt name
        **Prompt:** Prompt content
        
        Args:
            content: Content string
            prompt_id: The prompt ID from the ### heading
            
        Returns:
            dict with 'id', 'name', 'prompt' keys or None
        """
        name_match = re.search(r'\*\*Name:\*\*\s*(.+)', content)
        prompt_match = re.search(r'\*\*Prompt:\*\*\s*(.+)', content, re.DOTALL)
        
        if name_match and prompt_match:
            return {
                'id': prompt_id,
                'name': name_match.group(1).strip(),
                'prompt': prompt_match.group(1).strip()
            }
        return None
    
    def _load_and_validate(self, file_path: str, expected_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Load and validate prompts file.
        
        Args:
            file_path: Path to the file to load
            expected_structure: Expected structure definition (not used currently, reserved for future)
            
        Returns:
            dict: Loaded and validated data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is invalid
            ValueError: If structure validation fails
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if ext == '.md':
            data = self._parse_markdown(content)
        else:
            json_content = self._strip_comments(content)
            data = json.loads(json_content)
        
        self._validate_structure(data)
        return data
    
    def _validate_structure(self, data: Dict[str, Any]) -> None:
        """Validate the structure of loaded prompts data.
        
        Supports two formats:
        1. analysis_prompt format (has system_prompt at top level)
        2. prompts.jsonc format (has independent and chains)
        
        Args:
            data: Parsed prompts data
            
        Raises:
            ValueError: If required fields are missing
        """
        # Check if this is analysis_prompt format (has system_prompt at top level)
        if 'system_prompt' in data:
            # Analysis prompt format
            required_fields = ['system_prompt', 'user_prompt_template', 'translation_prompt_template', 'expert']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            if 'expert' in data:
                expert_fields = ['system_prompt', 'architect_eval', 'code_eval', 'debug_eval']
                for field in expert_fields:
                    if field not in data['expert']:
                        raise ValueError(f"Missing required field in expert: {field}")
        else:
            # Standard prompts.jsonc format - check for independent and chains
            if 'independent' not in data:
                raise ValueError("Missing required field: independent")
            if 'chains' not in data:
                raise ValueError("Missing required field: chains")
    
    def _get_file_extension(self) -> str:
        """Get the file extension of the prompts file."""
        _, ext = os.path.splitext(self.prompts_file)
        return ext.lower()
    
    def load(self) -> Dict[str, Any]:
        """Load prompts from file (.md, .jsonc, or .json).
        
        Returns:
            dict: Parsed prompts configuration
            
        Raises:
            FileNotFoundError: If prompts file doesn't exist
            json.JSONDecodeError: If JSON is invalid (after comment removal)
            ValueError: If structure validation fails
        """
        ext = self._get_file_extension()
        
        with open(self.prompts_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if ext == '.md':
            # Parse markdown
            self._data = self._parse_markdown(content)
        else:
            # Handle .jsonc and .json
            json_content = self._strip_comments(content)
            self._data = json.loads(json_content)
        
        # Validate structure
        self._validate_structure(self._data)
        
        return self._data
    
    @property
    def data(self) -> Dict[str, Any]:
        """Lazy-load and return prompts data."""
        if self._data is None:
            self.load()
        return self._data
    
    def get_independent_prompts(self, mode: str) -> List[Dict[str, str]]:
        """Get independent prompts for a specific mode.
        
        Args:
            mode: One of 'architect', 'code', 'debug'
            
        Returns:
            list: List of prompt dictionaries with 'id', 'name', 'prompt' keys
        """
        data = self.data
        return data.get('independent', {}).get(mode, [])
    
    def get_all_independent_modes(self) -> List[str]:
        """Get list of all available independent modes.
        
        Returns:
            list: List of mode names
        """
        return list(self.data.get('independent', {}).keys())
    
    def get_all_independent_prompts_ordered(self) -> List[Dict[str, Any]]:
        """Get all independent prompts in order: architect → code → debug.
        
        Returns:
            list: List of prompt dictionaries with 'mode', 'id', 'name', 'prompt' keys
        """
        all_prompts = []
        for mode in ['architect', 'code', 'debug']:
            prompts = self.get_independent_prompts(mode)
            for prompt in prompts:
                prompt_with_mode = prompt.copy()
                prompt_with_mode['mode'] = mode  # Add mode to each prompt
                all_prompts.append(prompt_with_mode)
        return all_prompts
    
    def get_chains(self) -> List[Dict[str, Any]]:
        """Get all prompt chains.
        
        Returns:
            list: List of chain dictionaries
        """
        return self.data.get('chains', [])
    
    def get_chain_by_id(self, chain_id: str) -> Optional[Dict[str, Any]]:
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
    
    def get_chain_by_name(self, chain_name: str) -> Optional[Dict[str, Any]]:
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
    
    def build_chain_context(self, chain: Dict[str, Any],
                           architect_response: str = None,
                           code_response: str = None) -> Dict[str, str]:
        """Build final prompts for a chain with context substitution.
        
        Args:
            chain: Chain dictionary from config
            architect_response: Response from architect mode (replaces [ARCHITECT_PLAN])
            code_response: Response from code mode (replaces [CODE_FROM_CODE])
            
        Returns:
            dict: Dictionary with mode keys and final prompt strings
        """
        prompts = chain.get('prompts', {})
        result = {}
        
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
