"""Expert evaluation results management for expert_results.md file."""

from datetime import datetime
from typing import Optional, List
from benchmark.result import BenchmarkMetrics


class ExpertResultsManager:
    """Manages expert evaluation results in Markdown format."""

    def __init__(self, output_file: str = "export/expert_results.md"):
        """Initialize expert results manager.

        Args:
            output_file: Path to the expert results file.
        """
        self.output_file = output_file
        self.entries: List[dict] = []
        self.tested_model: Optional[str] = None
        self.expert_model: Optional[str] = None
        self.generated_at: str = ""

    def start_session(self, tested_model: str, expert_model: str) -> None:
        """Start a new expert evaluation session.

        Args:
            tested_model: Name of the model being tested.
            expert_model: Name of the expert model used for evaluation.
        """
        self.tested_model = tested_model
        self.expert_model = expert_model
        self.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.entries = []

    def add_entry(self, metrics: BenchmarkMetrics) -> None:
        """Add a new entry to the results.

        Args:
            metrics: BenchmarkMetrics containing response and expert_score.
        """
        entry = {
            'prompt_id': metrics.prompt_id or 'unknown',
            'prompt_name': metrics.prompt_name or 'Unknown',
            'mode': metrics.mode or 'unknown',
            'ctx': metrics.ctx,
            'temperature': metrics.temperature,
            'avg_tps': metrics.avg_tps,
            'model': self.tested_model or 'unknown',
            'response': metrics.response or '',
            'expert_score': metrics.expert_score,
        }
        self.entries.append(entry)

    def _format_entry(self, entry: dict, index: int) -> str:
        """Format a single entry as Markdown.

        Args:
            entry: Entry dictionary with prompt info and results.
            index: Entry number.

        Returns:
            Formatted Markdown string.
        """
        lines = [
            f"## Entry {index}",
            "",
            "### Prompt Information",
            "",
            f"- **Prompt ID:** `{entry['prompt_id']}`",
            f"- **Prompt Name:** {entry['prompt_name']}",
            f"- **Mode:** {entry['mode']}",
            f"- **Context Size:** {entry['ctx']}",
            f"- **Temperature:** {entry['temperature']}",
            f"- **Average TPS:** {entry['avg_tps']:.2f}",
            f"- **Model:** {entry['model']}",
            "",
            "### Response",
            "",
            "```",
            entry['response'],
            "```",
            "",
        ]

        if entry['expert_score'] is not None:
            lines.extend([
                f"**Expert Score:** {entry['expert_score']:.1f}",
                "",
            ])

        lines.append("---")
        lines.append("")

        return "\n".join(lines)

    def save(self) -> None:
        """Save all entries to the expert results file."""
        if not self.output_file:
            return

        lines = [
            "# Expert Evaluation Results",
            "",
            f"**Generated:** {self.generated_at}",
            "",
            f"**Tested Model:** {self.tested_model or 'unknown'}",
            "",
            f"**Expert Model:** {self.expert_model or 'none'}",
            "",
            f"**Total Responses:** {len(self.entries)}",
            "",
            "---",
            "",
        ]

        for i, entry in enumerate(self.entries, 1):
            lines.append(self._format_entry(entry, i))

        content = "\n".join(lines)

        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(content)

    def append_entry(self, metrics: BenchmarkMetrics) -> None:
        """Add entry and save immediately (for incremental writes).

        Args:
            metrics: BenchmarkMetrics containing response and expert_score.
        """
        self.add_entry(metrics)
        self.save()

    def get_entry_count(self) -> int:
        """Get the number of entries.

        Returns:
            Number of saved entries.
        """
        return len(self.entries)