"""Expert evaluation results management for expert_results.md file."""

import os
import re
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

    @staticmethod
    def _entry_key(entry: dict[str, Any]) -> str:
        """Compute unique key for an entry.

        Args:
            entry: Entry dictionary.

        Returns:
            Unique key string: {model}|{ctx}|{temperature}|{prompt_id}
        """
        return (
            f"{entry.get('model', '')}|{entry.get('ctx', '')}|"
            f"{entry.get('temperature', '')}|{entry.get('prompt_id', '')}"
        )

    def _load_existing_entries(self) -> list[dict[str, Any]]:
        """Load existing entries from expert_results.md file.

        Parses the markdown file to extract entry data.

        Returns:
            List of entry dictionaries.
        """
        if not os.path.exists(self.output_file):
            return []

        entries: list[dict[str, Any]] = []
        current_entry: dict[str, Any] = {}

        try:
            with open(self.output_file, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()

                    # Parse Prompt ID
                    if line.startswith('- **Prompt ID:** `'):
                        match = re.search(r'\*\*Prompt ID:\*\* `([^`]+)', line)
                        if match:
                            current_entry['prompt_id'] = match.group(1)

                    # Parse Prompt Name
                    elif line.startswith('- **Prompt Name:**'):
                        match = re.search(r'\*\*Prompt Name:\*\* (.+)', line)
                        if match:
                            current_entry['prompt_name'] = match.group(1).strip()

                    # Parse Mode
                    elif line.startswith('- **Mode:**'):
                        match = re.search(r'\*\*Mode:\*\* (.+)', line)
                        if match:
                            current_entry['mode'] = match.group(1).strip()

                    # Parse Context Size
                    elif line.startswith('- **Context Size:**'):
                        match = re.search(r'\*\*Context Size:\*\* (\d+)', line)
                        if match:
                            current_entry['ctx'] = int(match.group(1))

                    # Parse Temperature
                    elif line.startswith('- **Temperature:**'):
                        match = re.search(r'\*\*Temperature:\*\* ([\d.]+)', line)
                        if match:
                            current_entry['temperature'] = float(match.group(1))

                    # Parse Average TPS
                    elif line.startswith('- **Average TPS:**'):
                        match = re.search(r'\*\*Average TPS:\*\* ([\d.]+)', line)
                        if match:
                            current_entry['avg_tps'] = float(match.group(1))

                    # Parse Model
                    elif line.startswith('- **Model:**'):
                        match = re.search(r'\*\*Model:\*\* (.+)', line)
                        if match:
                            current_entry['model'] = match.group(1).strip()

                    # Parse Expert Score
                    elif line.startswith('**Expert Score:**'):
                        match = re.search(r'\*\*Expert Score:\*\* (\d+)', line)
                        if match:
                            current_entry['expert_score'] = int(match.group(1))

                    # End of entry (separator)
                    elif line == '---' and current_entry:
                        if current_entry.get('prompt_id') and current_entry.get('model'):
                            entries.append(current_entry)
                        current_entry = {}

        except (OSError, ValueError) as e:
            print(f"Warning: Could not load existing entries: {e}")
            return []

        return entries

    def _merge_entries(self, new_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge new entries with existing ones.

        New entries are added only if their key doesn't exist in existing entries.

        Args:
            new_entries: List of new entry dictionaries.

        Returns:
            Merged list of entry dictionaries.
        """
        existing = self._load_existing_entries()
        existing_keys = {self._entry_key(e) for e in existing}

        for new_entry in new_entries:
            key = self._entry_key(new_entry)
            if key not in existing_keys:
                existing.append(new_entry)
                existing_keys.add(key)

        return existing

    def start_session(self, tested_model: str | list[str] | None = None, expert_model: str | None = None,
                      run_config: dict[str, Any] | None = None,
                      merge_mode: str = "overwrite") -> None:
        """Start a new expert evaluation session.

        Args:
            tested_model: Name of the model being tested.
            expert_model: Name of the expert model used for evaluation.
            merge_mode: How to handle existing file - 'overwrite' or 'merge'.
        """
        if isinstance(tested_model, list):
            self.tested_model = ", ".join(tested_model)
        else:
            self.tested_model = tested_model
        self.expert_model = expert_model
        self.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if merge_mode == "merge" and os.path.exists(self.output_file):
            self.entries = self._load_existing_entries()
        else:
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
            entry.get('response', ''),
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
