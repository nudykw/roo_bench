"""Results export to JSON and CSV files."""

import json
import csv
import os
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

    def save(self, results: dict, model_names: list, test_models: list,
             prompts_config: dict = None):
        """Save results to file.

        Args:
            results: Dictionary of results per model
            model_names: List of tested model names
            test_models: List of model objects
            prompts_config: Optional prompts configuration to include in export
        """
        if not self.output_file or not self.output_format:
            return

        # Prepare data for export
        export_data = self._prepare_export_data(results, model_names, test_models, prompts_config)

        # Save based on format
        if self.output_format == 'json':
            self._save_json(export_data)
        elif self.output_format == 'csv':
            self._save_csv(export_data)

    def _prepare_export_data(self, results: dict, model_names: list, test_models: list,
                             prompts_config: dict = None) -> dict:
        """Prepare data for export.

        Args:
            results: Dictionary of results per model
            model_names: List of tested model names
            test_models: List of model objects
            prompts_config: Optional prompts configuration to include in export

        Returns:
            dict: Export data with prompts_config at root level
        """
        # Include prompts configuration if provided
        prompts_section = None
        if prompts_config:
            prompts_section = {
                'independent': {
                    mode: [{'id': p.get('id'), 'name': p.get('name'), 'prompt': p.get('prompt')}
                           for p in prompts]
                    for mode, prompts in prompts_config.get('independent', {}).items()
                },
                'chains': [
                    {
                        'id': chain.get('id'),
                        'name': chain.get('name'),
                        'prompts': {
                            mode: {
                                'id': p.get('id'),
                                'name': p.get('name'),
                                'prompt': p.get('prompt')
                            }
                            for mode, p in chain.get('prompts', {}).items()
                        }
                    }
                    for chain in prompts_config.get('chains', [])
                ]
            }

        # Prepare results list
        results_list = []

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
                'timestamp': datetime.now().isoformat(),
            }

            # Add results for each context
            for run in results.get(model_name, []):
                run_data = {
                    'ctx': run['ctx'],
                    'ctx_str': f"{run['ctx'] // 1024}K" if run['ctx'] >= 1024 else str(run['ctx']),
                    'avg_tps': round(run['avg_tps'], 2),
                    'min_tps': round(run['min_tps'], 2),
                    'max_tps': round(run['max_tps'], 2),
                    'std_dev': round(run['std_dev'], 2),
                    'vram': run['vram'] if run['vram'] else None,
                    'vram_str': f"{run['vram'] / 1024 / 1024:.1f} MiB" if run['vram'] else None,
                    'prompt_id': run.get('prompt_id', ''),
                    'duration_sec': run.get('duration_sec', 0),
                    'prompt_tokens': run.get('prompt_tokens', 0),
                    'response_tokens': run.get('response_tokens', 0),
                    'temperature': run.get('temperature', 0),
                    **model_info
                }
                results_list.append(run_data)

        # Return new structure with prompts_config at root level
        return {
            'prompts_config': prompts_section,
            'results': results_list
        }

    def _save_json(self, export_data: dict):
        """Save results to JSON file.

        Args:
            export_data: List of result dictionaries
        """
        try:
            # Check if file exists and ask for confirmation
            if os.path.exists(self.output_file):
                file_size = os.path.getsize(self.output_file)
                print(get_text("output_file_exists",
                              output_file=self.output_file,
                              file_size=f"{file_size / 1024:.1f} KB"))
                while True:
                    try:
                        response = input(get_text("ask_overwrite") + " (y/n): ").strip().lower()
                        if response in ('y', 'yes', 'так', 'т', 'да', 'д'):
                            break
                        elif response in ('n', 'no', 'н', 'ні', 'не', 'нет', 'н'):
                            print(get_text("save_cancelled"))
                            return
                    except (EOFError, KeyboardInterrupt):
                        print(get_text("save_cancelled"))
                        return

            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(get_text("output_json", output_file=self.output_file))
        except Exception as e:
            print(get_text("error_unknown", error_details=f"JSON export failed: {e}"))

    def _save_csv(self, export_data: dict):
        """Save results to CSV file.

        Args:
            export_data: Dictionary with prompts_config and results
        """
        try:
            # Check if file exists and ask for confirmation
            if os.path.exists(self.output_file):
                file_size = os.path.getsize(self.output_file)
                print(get_text("output_file_exists",
                              output_file=self.output_file,
                              file_size=f"{file_size / 1024:.1f} KB"))
                while True:
                    try:
                        response = input(get_text("ask_overwrite") + " (y/n): ").strip().lower()
                        if response in ('y', 'yes', 'так', 'т', 'да', 'д'):
                            break
                        elif response in ('n', 'no', 'н', 'ні', 'не', 'нет', 'н'):
                            print(get_text("save_cancelled"))
                            return
                    except (EOFError, KeyboardInterrupt):
                        print(get_text("save_cancelled"))
                        return

            # Extract results list from export_data
            results_list = export_data.get('results', [])

            # Updated fieldnames with new metrics
            fieldnames = ['model_name', 'ctx', 'ctx_str', 'avg_tps', 'min_tps', 'max_tps',
                         'std_dev', 'vram', 'vram_str', 'params', 'quant', 'size_gb',
                         'max_ctx', 'vision', 'tools', 'thinking', 'language', 'timestamp',
                         'prompt_id', 'duration_sec', 'prompt_tokens', 'response_tokens', 'temperature']

            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(results_list)
            print(get_text("output_csv", output_file=self.output_file))
        except Exception as e:
            print(get_text("error_unknown", error_details=f"CSV export failed: {e}"))


def load_results_from_file(file_path: str) -> tuple:
    """Load benchmark results from a saved JSON or CSV file.

    Args:
        file_path: Path to the saved results file

    Returns:
        tuple: (all_results dict, test_models list) compatible with AIAnalyzer
    """
    import csv
    
    if not os.path.exists(file_path):
        print(get_text("analyze_file_not_found", file_path=file_path))
        return None, None
    
    ext = os.path.splitext(file_path)[1].lower()
    all_results = {}
    test_models = []
    model_data_map = {}
    
    try:
        if ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Handle new structure with prompts_config at root level
            if isinstance(data, dict) and 'results' in data:
                # New structure: {'prompts_config': ..., 'results': [...]}
                # Flatten results to group by model_name
                for result in data.get('results', []):
                    model_name = result.get('model_name', 'unknown')
                    
                    # Build model info (first occurrence creates the model entry)
                    if model_name not in model_data_map:
                        model_data_map[model_name] = {
                            'name': model_name,
                            'params': result.get('params', 'N/A'),
                            'quant': result.get('quant', 'N/A'),
                            'size_gb': float(result.get('size_gb', 0)) if result.get('size_gb', '0') != 'N/A' else 'N/A',
                            'max_ctx': int(result.get('max_ctx', 131072)) if result.get('max_ctx', '0') != 'N/A' else 131072,
                            'vision': result.get('vision', '❓'),
                            'tools': result.get('tools', '❓'),
                            'thinking': result.get('thinking', '❌'),
                        }
                    
                    # Build results per model
                    if model_name not in all_results:
                        all_results[model_name] = []
                    
                    all_results[model_name].append({
                        'ctx': result.get('ctx', 0),
                        'ctx_str': result.get('ctx_str', f"{result.get('ctx', 0) // 1024}K"),
                        'avg_tps': result.get('avg_tps', 0),
                        'min_tps': result.get('min_tps', 0),
                        'max_tps': result.get('max_tps', 0),
                        'std_dev': result.get('std_dev', 0),
                        'vram': result.get('vram'),
                        'vram_str': result.get('vram_str'),
                        'prompt_id': result.get('prompt_id', ''),
                        'duration_sec': result.get('duration_sec', 0),
                        'prompt_tokens': result.get('prompt_tokens', 0),
                        'response_tokens': result.get('response_tokens', 0),
                        'temperature': result.get('temperature', 0),
                    })
        elif ext == '.csv':
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                data = list(reader)
        else:
            print(get_text("analyze_file_unknown_format", ext=ext))
            return None, None
        
        if not data:
            print(get_text("analyze_file_empty"))
            return None, None
        
        test_models = list(model_data_map.values())
        return all_results, test_models
        
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(get_text("analyze_file_parse_error", error=str(e)))
        return None, None


def save_results(results: dict, output_file: str, output_format: str,
                 model_names: list, test_models: list,
                 prompts_config: dict = None):
    """Convenience function to save results.

    Args:
        results: Dictionary of results per model
        output_file: Path to output file
        output_format: Output format ('json' or 'csv')
        model_names: List of tested model names
        test_models: List of model objects
        prompts_config: Optional prompts configuration to include in export
    """
    saver = ResultSaver(output_file=output_file, output_format=output_format)
    saver.save(results, model_names, test_models, prompts_config)
