"""JSONC (JSON with Comments) loader for benchmark prompts."""

import json
import re
import os
from typing import Dict, List, Any, Optional


class PromptLoader:
    """Load and manage benchmark prompts from JSONC file."""
    
    DEFAULT_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), '..', 'prompts.jsonc')
    
    def __init__(self, prompts_file: Optional[str] = None):
        """Initialize prompt loader.
        
        Args:
            prompts_file: Path to prompts.jsonc file. If None, uses default.
        """
        self.prompts_file = prompts_file or self.DEFAULT_PROMPTS_FILE
        self._data: Optional[Dict] = None
    
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
    
    def load(self) -> Dict[str, Any]:
        """Load prompts from JSONC file.
        
        Returns:
            dict: Parsed prompts configuration
            
        Raises:
            FileNotFoundError: If prompts file doesn't exist
            json.JSONDecodeError: If JSON is invalid (after comment removal)
        """
        with open(self.prompts_file, 'r', encoding='utf-8') as f:
            jsonc_content = f.read()
        
        # Strip comments
        json_content = self._strip_comments(jsonc_content)
        
        # Parse JSON
        self._data = json.loads(json_content)
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
