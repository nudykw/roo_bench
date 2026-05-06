"""Model capabilities fetching from ollama.com."""

import requests
import urllib.parse
from bs4 import BeautifulSoup


class CapabilitiesFetcher:
    """Fetches model capabilities (vision, tools, thinking) from Ollama API and HTML."""

    def get_capabilities(self, model_name: str) -> tuple:
        """Get model capabilities from Ollama API or HTML parsing.

        Args:
            model_name: Model name (e.g., "llama3.2" or "dev-qwen2")

        Returns:
            tuple: (vision, tools, thinking) - capability statuses
        """
        base_name = model_name.split(':')[0]
        if base_name.startswith('dev-'):
            base_name = base_name[4:]

        # Attempt 1: Use Ollama API
        try:
            api_url = f"https://ollama.com/api/library/{base_name}"
            response = requests.get(api_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)

            if response.status_code == 200:
                data = response.json()
                model_data = data.get("model", {})
                
                # Get model_info dict (contains keys like "vision_model.vision", "tool.function.tool_use", etc.)
                model_info = model_data.get("model_info", {})
                if not model_info and isinstance(model_data, dict):
                    model_info = model_data.get("model_info", {})

                # Check for direct capabilities object first
                direct_caps = model_data.get("capabilities", {})
                if direct_caps:
                    vision = "✅" if direct_caps.get("vision") else "❌"
                    tools = "✅" if direct_caps.get("tools") else "❌"
                    thinking = "✅" if direct_caps.get("thinking") else "❌"
                    if vision != "❌" or tools != "❌" or thinking != "❌":
                        return vision, tools, thinking

                # Parse model_info keys to detect capabilities
                # Vision: keys containing "vision" (e.g., "vision_model.vision", "clip.vision")
                # Tools: keys containing "tool" or "function" (e.g., "tool.function.tool_use")
                # Thinking: keys containing "reasoning" or "thinking" (e.g., "reasoning.thinking")
                vision_keys = [k for k in model_info.keys() if 'vision' in k.lower() or 'multimodal' in k.lower()]
                tools_keys = [k for k in model_info.keys() if 'tool' in k.lower() or 'function' in k.lower()]
                thinking_keys = [k for k in model_info.keys() if 'reasoning' in k.lower() or 'thinking' in k.lower()]

                vision = "✅" if vision_keys else "❌"
                tools = "✅" if tools_keys else "❌"
                thinking = "✅" if thinking_keys else "❌"

                if vision != "❌" or tools != "❌" or thinking != "❌":
                    return vision, tools, thinking

            # If API returned an error or doesn't contain the needed information, try HTML parsing
            if response.status_code != 200:
                # Try search via search API
                search_url = f"https://ollama.com/api/search?q={urllib.parse.quote(base_name)}"
                search_resp = requests.get(search_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                if search_resp.status_code == 200:
                    search_data = search_resp.json()
                    for item in search_data.get("results", []):
                        if item.get("name") == base_name:
                            caps = item.get("capabilities", {})
                            vision = "✅" if caps.get("vision") else "❌"
                            tools = "✅" if caps.get("tools") else "❌"
                            thinking = "✅" if caps.get("thinking") else "❌"
                            return vision, tools, thinking
                    return "❓", "❓", "❌"

        except requests.exceptions.Timeout:
            print(f"⚠️  Timeout requesting Ollama API for {base_name}, using HTML parsing")
        except requests.exceptions.ConnectionError:
            print(f"⚠️  ConnectionError requesting Ollama API for {base_name}, using HTML parsing")
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code in [404, 500]:
                print(f"⚠️  HTTP {e.response.status_code} from Ollama API for {base_name}, using HTML parsing")
            else:
                raise
        except Exception as e:
            print(f"⚠️  Error requesting Ollama API for {base_name}: {e}, using HTML parsing")

        # Fallback: HTML parsing with more reliable selectors
        try:
            url = f"https://ollama.com/library/{base_name}"
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)

            if response.status_code != 200:
                # Try search via HTML
                search_url = f"https://ollama.com/search?q={urllib.parse.quote(base_name)}"
                search_resp = requests.get(search_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                if search_resp.status_code == 200:
                    soup = BeautifulSoup(search_resp.text, 'html.parser')
                    link = soup.find('a', href=lambda href: href and '/library/' in href)
                    if link:
                        url = "https://ollama.com" + link['href']
                        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Method 1: Search via data-attributes or specific classes
                capabilities_section = soup.find('div', class_=lambda c: c and 'capabilities' in c.lower())
                if capabilities_section:
                    caps_text = capabilities_section.get_text().lower()
                    vision = "✅" if "vision" in caps_text else "❌"
                    tools = "✅" if "tools" in caps_text or "tool use" in caps_text else "❌"
                    thinking = "✅" if "thinking" in caps_text or "reasoning" in caps_text else "❌"
                    return vision, tools, thinking

                # Method 2: Search via description/summary
                description = soup.find('div', class_='description') or soup.find('div', class_='summary')
                if description:
                    desc_text = description.get_text().lower()
                    vision = "✅" if "vision" in desc_text or "multimodal" in desc_text else "❌"
                    tools = "✅" if "tools" in desc_text or "tool use" in desc_text else "❌"
                    thinking = "✅" if "thinking" in desc_text or "reasoning" in desc_text else "❌"
                    return vision, tools, thinking

                # Method 3: Search via JSON-LD (if available)
                json_ld = soup.find('script', type='application/ld+json')
                if json_ld:
                    try:
                        import json
                        ld_data = json.loads(json_ld.string)
                        if isinstance(ld_data, dict):
                            capabilities = ld_data.get('capability', {})
                            if isinstance(capabilities, dict):
                                vision = "✅" if capabilities.get('vision') else "❌"
                                tools = "✅" if capabilities.get('tools') else "❌"
                                thinking = "✅" if capabilities.get('thinking') else "❌"
                                return vision, tools, thinking
                    except json.JSONDecodeError:
                        pass

                # Method 4: Search via alt-text of images (capabilities icons)
                vision_icon = soup.find('img', alt=lambda alt: alt and 'vision' in alt.lower())
                tools_icon = soup.find('img', alt=lambda alt: alt and 'tools' in alt.lower())
                thinking_icon = soup.find('img', alt=lambda alt: alt and ('thinking' in alt.lower() or 'reasoning' in alt.lower()))

                vision = "✅" if vision_icon else "❌"
                tools = "✅" if tools_icon else "❌"
                thinking = "✅" if thinking_icon else "❌"

                if vision != "❓" or tools != "❓" or thinking != "❓":
                    return vision, tools, thinking

                # Method 5: Final fallback - search in page text
                page_text = soup.get_text().lower()
                vision = "✅" if "vision" in page_text else "❌"
                tools = "✅" if "tools" in page_text or "tool use" in page_text else "❌"
                thinking = "✅" if "thinking" in page_text or "reasoning" in page_text or "deepseek" in base_name.lower() else "❌"

                return vision, tools, thinking

        except requests.exceptions.Timeout:
            print(f"⚠️  Timeout during HTML parsing for {base_name}")
        except requests.exceptions.ConnectionError:
            print(f"⚠️  ConnectionError during HTML parsing for {base_name}")
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code in [404, 500]:
                print(f"⚠️  HTTP {e.response.status_code} during HTML parsing for {base_name}")
            else:
                raise
        except Exception as e:
            print(f"⚠️  Error during HTML parsing for {base_name}: {e}")

        return "❓", "❓", "❓"
