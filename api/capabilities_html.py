"""HTML parsing methods for CapabilitiesFetcher."""

import re
from typing import Any

import requests
from bs4 import BeautifulSoup

from api.capabilities_defaults import COLOR_PATTERN, SIZE_PATTERN


class CapabilitiesHtml:
    """Mixin class for HTML parsing methods."""

    MODEL_CAPABILITIES: dict[str, Any]
    CACHE_VERSION: int

    def _is_size_tag(self, text: str) -> bool:
        """Check if text is a model size indicator (e.g., '1.5b', '70b', '7b')."""
        if not text:
            return False
        # Remove any quantization info
        text = text.split('-')[0].strip()
        return bool(SIZE_PATTERN.match(text)) or bool(COLOR_PATTERN.match(text))

    def get_capabilities_from_html(self, base_name: str) -> tuple[str, str, str] | None:
        """Get capabilities by parsing the model page HTML.
        
        Returns:
            tuple: (vision, tools, thinking) - capability statuses, or None if not found
        """
        url = f"https://ollama.com/library/{base_name}"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Primary method: Find capabilities in the div with size tags
        # Capabilities are in the same div.flex-wrap as size tags, but as colored badges
        # Size tags have bg-[#color] classes, capabilities don't
        flex_wraps = soup.find_all('div', class_=lambda c: c and 'flex-wrap' in c)
        
        for div in flex_wraps:
            spans = div.find_all('span')
            if len(spans) < 2:
                continue
            
            # Check if this div contains size indicators
            has_size = False
            for span in spans:
                text = span.get_text(strip=True).lower()
                if self._is_size_tag(text):
                    has_size = True
                    break
            
            if not has_size:
                continue
            
            # This is the capabilities+sizes div
            # Capabilities are plain text spans (vision, tools, thinking)
            # Sizes are spans with bg-[*] color classes
            found_vision = False
            found_tools = False
            found_thinking = False
            
            for span in spans:
                text = span.get_text(strip=True).lower()
                span_classes = span.get('class')
                classes = ' '.join(span_classes) if span_classes else ''
                
                # Skip size tags (they have bg- color classes)
                if 'bg-' in classes or self._is_size_tag(text):
                    continue
                
                # Skip if this looks like a version number
                if re.match(r'^\d+(\.\d+)*$', text):
                    continue
                
                # Check for capability keywords
                if text == 'vision' or 'multimodal' in text:
                    found_vision = True
                elif text == 'tools' or 'tool use' in text:
                    found_tools = True
                elif text in ('thinking', 'reasoning'):
                    found_thinking = True
            
            if found_vision or found_tools or found_thinking:
                return (
                    "\u2705" if found_vision else "\u274c",
                    "\u2705" if found_tools else "\u274c",
                    "\u2705" if found_thinking else "\u274c"
                )
        
        # Fallback: check meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            content = meta_desc.get('content')
            if isinstance(content, str):
                desc = content.lower()
            else:
                desc = ''
            vision = "\u2705" if any(w in desc for w in ['vision', 'visual', 'multimodal', 'image']) else "\u274c"
            tools = "\u2705" if any(w in desc for w in ['tool', 'function', 'api']) else "\u274c"
            thinking = "\u2705" if any(w in desc for w in ['reasoning', 'think', 'chain of thought']) else "\u274c"
            if vision != "\u274c" or tools != "\u274c" or thinking != "\u274c":
                return vision, tools, thinking
        
        return None  # Signal that HTML parsing didn't find anything definitive
