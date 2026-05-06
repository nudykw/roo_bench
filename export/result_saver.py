"""Results export to JSON and CSV files."""

import json
import csv
from datetime import datetime
from i18n import get_text, _current_language


class ResultSaver:
    """Handles saving benchmark results to files."""

    def __init__(self, output_file: str, output_format: str):
        """Initialize result saver.

        Args:
            output_file: Path to output file
            output_format: Output format ('json' or 'csv')
        """
        self.output_file = output_file
        self.output_format = output_format

    def save(self, results: dict, model_names: list, test_models: list):
        """Save results to file.

        Args:
            results: Dictionary of results per model
            model_names: List of tested model names
            test_models: List of model objects
        """
        if not self.output_file or not self.output_format:
            return

        # Prepare data for export
        export_data = self._prepare_export_data(results, model_names, test_models)

        # Save based on format
        if self.output_format == 'json':
            self._save_json(export_data)
        elif self.output_format == 'csv':
            self._save_csv(export_data)

    def _prepare_export_data(self, results: dict, model_names: list, test_models: list) -> list:
        """Prepare data for export.

        Args:
            results: Dictionary of results per model
            model_names: List of tested model names
            test_models: List of model objects

        Returns:
            list: List of export data dictionaries
        """
        export_data = []

        for model_name in model_names:
            if model_name not in results:
                continue

            model_obj = next((m for m in test_models if m['name'] == model_name), None)
            if not model_obj:
                continue

            model_info = {
                'model_name': model_name,
                'params': model_obj.get('params', 'N/A'),
                'quant': model_obj.get('quant', 'N/A'),
                'size_gb': model_obj.get('size_gb', 0),
                'max_ctx': model_obj.get('max_ctx', 0),
                'vision': model_obj.get('vision', '❓'),
                'tools': model_obj.get('tools', '❓'),
                'thinking': model_obj.get('thinking', '❌'),
                'language': _current_language if '_current_language' in globals() else "en",
                'timestamp': datetime.now().isoformat()
            }

            # Add results for each context
            for run in results.get(model_name, []):
                run_data = {
                    'model_name': model_name,
                    'ctx': run['ctx'],
                    'ctx_str': f"{run['ctx'] // 1024}K" if run['ctx'] >= 1024 else str(run['ctx']),
                    'avg_tps': round(run['avg_tps'], 2),
                    'min_tps': round(run['min_tps'], 2),
                    'max_tps': round(run['max_tps'], 2),
                    'std_dev': round(run['std_dev'], 2),
                    'vram': run['vram'] if run['vram'] else None,
                    'vram_str': f"{run['vram'] / 1024 / 1024:.1f} MiB" if run['vram'] else None,
                    **model_info
                }
                export_data.append(run_data)

        return export_data

    def _save_json(self, export_data: list):
        """Save results to JSON file.

        Args:
            export_data: List of result dictionaries
        """
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(get_text("output_json", output_file=self.output_file))
        except Exception as e:
            print(get_text("error_unknown", error_details=f"JSON export failed: {e}"))

    def _save_csv(self, export_data: list):
        """Save results to CSV file.

        Args:
            export_data: List of result dictionaries
        """
        try:
            fieldnames = ['model_name', 'ctx', 'ctx_str', 'avg_tps', 'min_tps', 'max_tps',
                         'std_dev', 'vram', 'vram_str', 'params', 'quant', 'size_gb',
                         'max_ctx', 'vision', 'tools', 'thinking', 'language', 'timestamp']

            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(export_data)
            print(get_text("output_csv", output_file=self.output_file))
        except Exception as e:
            print(get_text("error_unknown", error_details=f"CSV export failed: {e}"))


def save_results(results: dict, output_file: str, output_format: str,
                 model_names: list, test_models: list):
    """Convenience function to save results.

    Args:
        results: Dictionary of results per model
        output_file: Path to output file
        output_format: Output format ('json' or 'csv')
        model_names: List of tested model names
        test_models: List of model objects
    """
    saver = ResultSaver(output_file=output_file, output_format=output_format)
    saver.save(results, model_names, test_models)
