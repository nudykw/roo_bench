"""Expert evaluation results management for expert_results.md file."""

import os
from datetime import datetime
from typing import Any

from benchmark.result import BenchmarkMetrics, BenchmarkResult


class ExpertResultsManager:
    """Manages expert evaluation results in Markdown format."""

    def __init__(self, output_file: str = "export/expert_results.md"):
        """Initialize expert results manager.

        Args:
            output_file: Path to the expert results file.
        """
        self.output_file = output_file
        self.entries: list[dict[str, Any]] = []
        self.tested_model: str | None = None
        self.expert_model: str | None = None
        self.generated_at: str = ""
        self.run_config: dict[str, Any] = {}

    def start_session(self, tested_model: str | list[str] | None = None, expert_model: str | None = None,
                      run_config: dict[str, Any] | None = None) -> None:
        """Start a new expert evaluation session.

        Args:
            tested_model: Name of the model being tested.
            expert_model: Name of the expert model used for evaluation.
        """
        if isinstance(tested_model, list):
            self.tested_model = ", ".join(tested_model)
        else:
            self.tested_model = tested_model
        self.expert_model = expert_model
        self.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.entries = []
        self.run_config = run_config or {}
        self.save()

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

    def append_model_result(self, model_result: BenchmarkResult) -> None:
        """Add all response-bearing metrics for a model and save immediately."""
        previous_model = self.tested_model
        self.tested_model = model_result.model.name
        for metrics in model_result.results:
            if metrics.response:
                self.add_entry(metrics)
        self.tested_model = previous_model
        self.save()

    def _format_entry(self, entry: dict[str, Any], index: int) -> str:
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
                f"**Expert Score:** {int(entry['expert_score'])}/100",
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
        ]

        if self.run_config:
            lines.extend([
                "## Run Config",
                "",
                f"- **Context Sizes:** {', '.join(str(x) for x in self.run_config.get('context_sizes', []))}",
                f"- **Temperatures:** {', '.join(str(x) for x in self.run_config.get('temperature_test_values', []))}",
                f"- **Prompt IDs:** {', '.join(self.run_config.get('used_prompt_ids', [])) or 'none'}",
                f"- **Chain IDs:** {', '.join(self.run_config.get('used_chain_ids', [])) or 'none'}",
                "",
            ])

        lines.extend([
            "---",
            "",
        ])

        for i, entry in enumerate(self.entries, 1):
            lines.append(self._format_entry(entry, i))

        content = "\n".join(lines)

        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

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
