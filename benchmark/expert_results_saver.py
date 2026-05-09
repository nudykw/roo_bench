"""Save expert evaluation results to a markdown file."""

import os
from datetime import datetime
from typing import List, Optional
from benchmark.expert_evaluator_types import ExpertEvaluationEntry


class ExpertResultsSaver:
    """Saves expert evaluation results to a markdown file.
    
    The file is overwritten on each save (typically at program start).
    Format: markdown with prompt, response, and metadata.
    """
    
    DEFAULT_OUTPUT_FILE = 'export/expert_results.md'
    
    def __init__(self, output_file: Optional[str] = None):
        """Initialize the saver.
        
        Args:
            output_file: Path to output file. If None, uses default.
        """
        self.output_file = output_file or self.DEFAULT_OUTPUT_FILE
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self) -> None:
        """Create output directory if it doesn't exist."""
        dir_path = os.path.dirname(self.output_file)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    
    def save(self, entries: List[ExpertEvaluationEntry], 
             model_name: str, expert_model: str) -> str:
        """Save expert evaluation results to markdown file.
        
        Args:
            entries: List of evaluation entries with responses.
            model_name: Name of the tested model.
            expert_model: Name of the expert model used.
            
        Returns:
            Path to the saved file.
        """
        self._ensure_directory_exists()
        
        lines = []
        lines.append(f"# Expert Evaluation Results\n")
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"\n**Tested Model:** {model_name}\n")
        lines.append(f"\n**Expert Model:** {expert_model}\n")
        lines.append(f"\n**Total Responses:** {len(entries)}\n")
        lines.append(f"\n---\n")
        
        for i, entry in enumerate(entries, 1):
            lines.append(f"\n## Entry {i}\n")
            lines.append(f"\n### Prompt Information\n")
            lines.append(f"\n- **Prompt ID:** `{entry.prompt_id}`\n")
            lines.append(f"\n- **Prompt Name:** {entry.prompt_name}\n")
            lines.append(f"\n- **Mode:** {entry.mode or 'default'}\n")
            lines.append(f"\n- **Context Size:** {entry.ctx}\n")
            lines.append(f"\n- **Temperature:** {entry.temperature}\n")
            lines.append(f"\n- **Average TPS:** {entry.avg_tps:.2f}\n")
            lines.append(f"\n- **Model:** {entry.model_name}\n")
            lines.append(f"\n### Response\n")
            lines.append(f"\n```\n{entry.response}\n```\n")
            lines.append(f"\n---\n")
        
        content = ''.join(lines)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return self.output_file
    
    def clear(self) -> None:
        """Clear the output file by overwriting with empty content."""
        self._ensure_directory_exists()
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write('')