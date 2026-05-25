"""Generate Markdown prompt files from JSONC configuration files.

Usage:
    python roo_bench.py --generate-md
    
This command reads prompts.jsonc and analysis_prompt.jsonc,
then generates prompts/prompts.md and prompts/analysis_prompt.md.
The generated files are independent and can be edited manually.
"""

import json
import os
from typing import Any


def strip_jsonc_comments(jsonc: str) -> str:
    """Remove // and /* */ comments from JSONC content.
    
    Preserves strings - doesn't strip // or /* inside quotes.
    
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
                while i < len(jsonc) and jsonc[i] != '\n':
                    i += 1
                continue
            # Multi-line comment
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


def generate_prompts_md(data: dict[str, Any]) -> str:
    """Convert prompts.jsonc data to Markdown format.
    
    Args:
        data: Parsed prompts.jsonc dictionary
        
    Returns:
        str: Markdown-formatted string
    """
    lines = []
    
    # Header comment
    lines.append("<!--")
    lines.append("  Prompts for Roo Bench benchmark system")
    lines.append("  Contains independent prompts and chains for different modes")
    lines.append("  Generated from prompts.jsonc: python roo_bench.py --generate-md")
    lines.append("-->")
    lines.append("")
    
    # Independent prompts section
    lines.append("# independent")
    lines.append("<!--")
    lines.append("  Independent prompts - each runs without context from other modes")
    lines.append("-->")
    lines.append("")
    
    independent = data.get("independent", {})
    
    # Process each mode
    mode_comments = {
        "architect": "Architect mode prompts - test model's ability to design systems",
        "code": "Code mode prompts - test model's ability to implement code",
        "debug": "Debug mode prompts - test model's ability to find and fix bugs",
    }
    
    for mode, comment in mode_comments.items():
        prompts = independent.get(mode, [])
        if prompts:
            lines.append("<!--")
            lines.append(f"  {comment}")
            lines.append("-->")
            lines.append(f"## {mode}")
            lines.append("")
            
            for prompt in prompts:
                prompt_id = prompt.get("id", "")
                name = prompt.get("name", "")
                prompt_text = prompt.get("prompt", "")
                
                lines.append(f"### {prompt_id}")
                lines.append(f"**Name:** {name}")
                lines.append(f"**Prompt:** {prompt_text}")
                lines.append("")
    
    # Chains section
    lines.append("<!--")
    lines.append("  Chain prompts - context flows between modes")
    lines.append("-->")
    lines.append("# chains")
    lines.append("")
    
    chains = data.get("chains", [])
    
    for chain in chains:
        chain_id = chain.get("id", "")
        chain_name = chain.get("name", "")
        chain_desc = chain.get("description", "")
        chain_prompts = chain.get("prompts", {})
        
        lines.append(f"### {chain_id}")
        lines.append(f"**Name:** {chain_name}")
        lines.append(f"**Description:** {chain_desc}")
        lines.append("**Prompts:**")
        
        for mode in ["architect", "code", "debug"]:
            if mode in chain_prompts:
                mode_prompt = chain_prompts[mode]
                mode_text = mode_prompt.get("prompt", "")
                lines.append(f"- **{mode}:** {mode_text}")
        
        lines.append("")
    
    return "\n".join(lines)


def generate_analysis_prompt_md(data: dict[str, Any]) -> str:
    """Convert analysis_prompt.jsonc data to Markdown format.
    
    Args:
        data: Parsed analysis_prompt.jsonc dictionary
        
    Returns:
        str: Markdown-formatted string
    """
    lines = []
    
    # Header
    lines.append("<!--")
    lines.append("  Analysis prompts for Roo Bench expert evaluation")
    lines.append("  Contains system prompts, user templates, and expert evaluation criteria")
    lines.append("  Generated from analysis_prompt.jsonc: python roo_bench.py --generate-md")
    lines.append("-->")
    lines.append("")
    
    # System prompt
    system_prompt = data.get("system_prompt", "")
    if system_prompt:
        lines.append("# system_prompt")
        lines.append(f"**Prompt:** {system_prompt}")
        lines.append("")
    
    # User prompt template
    user_template = data.get("user_prompt_template", "")
    if user_template:
        lines.append("# user_prompt_template")
        lines.append(f"**Template:** {user_template}")
        lines.append("")
    
    # Translation prompt template
    translation_template = data.get("translation_prompt_template", "")
    if translation_template:
        lines.append("# translation_prompt_template")
        lines.append(f"**Template:** {translation_template}")
        lines.append("")
    
    # Expert evaluation prompts
    expert = data.get("expert", {})
    if expert:
        lines.append("# expert")
        lines.append("")
        
        expert_system = expert.get("system_prompt", "")
        if expert_system:
            lines.append("## system_prompt")
            lines.append(f"**Prompt:** {expert_system}")
            lines.append("")
        
        for key in ["architect_eval", "code_eval", "debug_eval"]:
            template = expert.get(key, "")
            if template:
                lines.append(f"## {key}")
                lines.append(f"**Template:** {template}")
                lines.append("")
    
    return "\n".join(lines)


def generate_all_markdown() -> bool:
    """Generate all Markdown prompt files from JSONC sources.
    
    Returns:
        bool: True if generation was successful, False otherwise
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Generate prompts.md from prompts.jsonc
    prompts_jsonc_path = os.path.join(base_dir, "prompts", "prompts.jsonc")
    prompts_md_path = os.path.join(base_dir, "prompts", "prompts.md")
    
    if not os.path.exists(prompts_jsonc_path):
        print(f"Error: {prompts_jsonc_path} not found")
        return False
    
    with open(prompts_jsonc_path, encoding="utf-8") as f:
        jsonc_content = f.read()
    
    json_content = strip_jsonc_comments(jsonc_content)
    data = json.loads(json_content)
    md_content = generate_prompts_md(data)
    
    os.makedirs(os.path.dirname(prompts_md_path), exist_ok=True)
    with open(prompts_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"Generated: {prompts_md_path}")
    
    # Generate analysis_prompt.md from analysis_prompt.jsonc
    analysis_jsonc_path = os.path.join(base_dir, "prompts", "analysis_prompt.jsonc")
    analysis_md_path = os.path.join(base_dir, "prompts", "analysis_prompt.md")
    
    if not os.path.exists(analysis_jsonc_path):
        print(f"Warning: {analysis_jsonc_path} not found, skipping analysis_prompt.md")
    else:
        with open(analysis_jsonc_path, encoding="utf-8") as f:
            analysis_jsonc_content = f.read()
        
        analysis_json_content = strip_jsonc_comments(analysis_jsonc_content)
        analysis_data = json.loads(analysis_json_content)
        analysis_md_content = generate_analysis_prompt_md(analysis_data)
        
        with open(analysis_md_path, "w", encoding="utf-8") as f:
            f.write(analysis_md_content)
        
        print(f"Generated: {analysis_md_path}")
    
    print("\nMarkdown files generated successfully.")
    print("You can now edit them independently, or regenerate with --generate-md")
    return True
