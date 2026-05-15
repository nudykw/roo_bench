"""Results export to JSON and CSV files."""

import json
import csv
import os
from datetime import datetime
from typing import List, Optional, Any
from i18n import get_text, _current_language
from benchmark.result import BenchmarkResult, ModelInfo, BenchmarkMetrics
from export.merge_utils import atomic_write_json


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

    def save(self, results: List[BenchmarkResult],
             prompts_config: Optional[dict] = None,
             run_config: Optional[dict] = None,
             confirm_overwrite: bool = True):
        """Save results to file.

        Args:
            results: List of BenchmarkResult objects
            prompts_config: Optional prompts configuration to include in export
        """
        if not self.output_file or not self.output_format:
            return

        # Prepare data for export
        export_data = self._prepare_export_data(results, prompts_config, run_config)

        # Save based on format
        if self.output_format == 'json':
            self._save_json(export_data, confirm_overwrite=confirm_overwrite)
        elif self.output_format == 'csv':
            self._save_csv(export_data)

    def _prepare_export_data(self, results: List[BenchmarkResult],
                             prompts_config: Optional[dict] = None,
                             run_config: Optional[dict] = None) -> dict:
        """Prepare data for export.

        Args:
            results: List of BenchmarkResult objects
            prompts_config: Optional prompts configuration to include in export

        Returns:
            dict: Export data with prompts_config at root level
        """
        # Include prompts configuration if provided
        prompts_section = None
        if prompts_config:
            prompts_section = {}
            if 'independent' in prompts_config:
                prompts_section['independent'] = {
                    mode: [{'id': p.get('id'), 'name': p.get('name'), 'prompt': p.get('prompt')}
                           for p in prompts]
                    for mode, prompts in prompts_config.get('independent', {}).items()
                }
            if 'chains' in prompts_config:
                prompts_section['chains'] = [
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

        # Prepare results list - nested structure (one entry = one model)
        results_list = []
        for result in results:
            result_dict = {
                'model': result.model.model_dump(),
                'results': [m.model_dump() for m in result.results],
            }
            results_list.append(result_dict)

        # Return new structure with prompts_config at root level
        return {
            'run_config': run_config or {},
            'prompts_config': prompts_section,
            'results': results_list
        }

    def _save_json(self, export_data: dict, confirm_overwrite: bool = True):
        """Save results to JSON file.

        Args:
            export_data: List of result dictionaries
        """
        try:
            # Check if file exists and ask for confirmation
            if confirm_overwrite and os.path.exists(self.output_file):
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

            atomic_write_json(self.output_file, export_data)
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

            # Flatten nested structure to flat CSV rows
            csv_rows = []
            for result_entry in results_list:
                model_data = result_entry.get('model', {})
                metrics_list = result_entry.get('results', [])
                
                for metric in metrics_list:
                    row = {
                        'model_name': model_data.get('name', ''),
                        'params': model_data.get('params', 'N/A'),
                        'quant': model_data.get('quant', 'N/A'),
                        'size_gb': model_data.get('size_gb', ''),
                        'max_ctx': model_data.get('max_ctx', ''),
                        'vision': model_data.get('vision', ''),
                        'tools': model_data.get('tools', ''),
                        'thinking': model_data.get('thinking', ''),
                        'audio': model_data.get('audio', ''),
                        'architecture': model_data.get('architecture', ''),
                        'ctx': metric.get('ctx', ''),
                        'ctx_str': metric.get('ctx_str', ''),
                        'avg_tps': metric.get('avg_tps', ''),
                        'min_tps': metric.get('min_tps', ''),
                        'max_tps': metric.get('max_tps', ''),
                        'std_dev': metric.get('std_dev', ''),
                        'vram': metric.get('vram', ''),
                        'vram_str': metric.get('vram_str', ''),
                        'prompt_id': metric.get('prompt_id', ''),
                        'prompt_name': metric.get('prompt_name', ''),
                        'duration_sec': metric.get('duration_sec', ''),
                        'prompt_tokens': metric.get('prompt_tokens', ''),
                        'response_tokens': metric.get('response_tokens', ''),
                        'temperature': metric.get('temperature', ''),
                        'mode': metric.get('mode', ''),
                        'chain_id': metric.get('chain_id', ''),
                        'chain_name': metric.get('chain_name', ''),
                        'expert_score': metric.get('expert_score', ''),
                    }
                    csv_rows.append(row)

            # CSV fieldnames
            fieldnames = ['model_name', 'params', 'quant', 'size_gb', 'max_ctx',
                         'vision', 'tools', 'thinking', 'audio', 'architecture',
                         'ctx', 'ctx_str', 'avg_tps', 'min_tps', 'max_tps',
                         'std_dev', 'vram', 'vram_str',
                         'prompt_id', 'prompt_name', 'duration_sec',
                         'prompt_tokens', 'response_tokens', 'temperature',
                         'mode', 'chain_id', 'chain_name', 'expert_score']

            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(csv_rows)
            print(get_text("output_csv", output_file=self.output_file))
        except Exception as e:
            print(get_text("error_unknown", error_details=f"CSV export failed: {e}"))


def load_results_from_file(file_path: str) -> tuple:
    """Load benchmark results from a saved JSON or CSV file.

    Args:
        file_path: Path to the saved results file

    Returns:
        tuple: (all_results List[BenchmarkResult], prompts_config dict) compatible with AIAnalyzer
    """
    import csv
    
    if not os.path.exists(file_path):
        print(get_text("analyze_file_not_found", file_path=file_path))
        return None, None
    
    ext = os.path.splitext(file_path)[1].lower()
    all_results: List[BenchmarkResult] = []
    prompts_config = None
    
    try:
        if ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract prompts_config if present
            if isinstance(data, dict):
                prompts_section = data.get('prompts_config')
                if prompts_section:
                    prompts_config = {
                        'independent': prompts_section.get('independent', {}),
                        'chains': prompts_section.get('chains', [])
                    }
                
                # Load new structure: {'prompts_config': ..., 'results': [...]}
                for result_data in data.get('results', []):
                    model_data = result_data.get('model', {})
                    metrics_list = result_data.get('results', [])
                    
                    # Create ModelInfo
                    model_info = ModelInfo(
                        name=model_data.get('name', 'unknown'),
                        size_gb=float(model_data.get('size_gb', 0)) if model_data.get('size_gb') and model_data.get('size_gb') != 'N/A' else 0.0,
                        params=model_data.get('params', 'N/A'),
                        quant=model_data.get('quant', 'N/A'),
                        architecture=model_data.get('architecture', 'N/A'),
                        max_ctx=int(model_data.get('max_ctx', 131072)) if model_data.get('max_ctx', '0') != 'N/A' else 131072,
                        moe=model_data.get('moe'),
                    )
                    
                    metrics = [BenchmarkMetrics(**m) for m in metrics_list]
                    
                    # Create BenchmarkResult
                    benchmark_result = BenchmarkResult(
                        model=model_info,
                        results=metrics,
                    )
                    all_results.append(benchmark_result)
        
        elif ext == '.csv':
            # CSV loading - create simple BenchmarkResult from flat CSV data
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if rows:
                # Group by model_name
                model_groups = {}
                for row in rows:
                    model_name = row.get('model_name', 'unknown')
                    if model_name not in model_groups:
                        size_gb_str = row.get('size_gb', '0')
                        max_ctx_str = row.get('max_ctx', '131072')
                        model_groups[model_name] = {
                            'model': {
                                'name': model_name,
                                'params': row.get('params', 'N/A'),
                                'quant': row.get('quant', 'N/A'),
                                'size_gb': float(size_gb_str) if size_gb_str and size_gb_str != 'N/A' else 0.0,
                                'max_ctx': int(max_ctx_str) if max_ctx_str and max_ctx_str != 'N/A' else 131072,
                            },
                            'results': []
                        }
                    
                    # Helper to parse values with proper None handling
                    def parse_float(val, default=0.0):
                        if not val or val.strip() == '':
                            return default
                        try:
                            return float(val)
                        except ValueError:
                            return default
                    
                    def parse_int(val, default=0):
                        if not val or val.strip() == '':
                            return default
                        try:
                            return int(val)
                        except ValueError:
                            return default
                    
                    def parse_optional_int(val):
                        if not val or val.strip() == '':
                            return None
                        try:
                            return int(val)
                        except ValueError:
                            return None
                    
                    model_groups[model_name]['results'].append({
                        'ctx': parse_int(row.get('ctx', '0')),
                        'temperature': parse_float(row.get('temperature', '0')),
                        'avg_tps': parse_float(row.get('avg_tps', '0')),
                        'min_tps': parse_float(row.get('min_tps', '0')),
                        'max_tps': parse_float(row.get('max_tps', '0')),
                        'std_dev': parse_float(row.get('std_dev', '0')),
                        'vram': parse_optional_int(row.get('vram')),
                        'duration_sec': parse_float(row.get('duration_sec', '0')),
                        'prompt_tokens': parse_int(row.get('prompt_tokens', '0')),
                        'response_tokens': parse_int(row.get('response_tokens', '0')),
                        'prompt_id': row.get('prompt_id', ''),
                    })
                
                # Create BenchmarkResult for each model
                for model_name, data in model_groups.items():
                    model_info = ModelInfo(**data['model'])
                    metrics = [BenchmarkMetrics(**m) for m in data['results']]
                    all_results.append(BenchmarkResult(model=model_info, results=metrics))
        else:
            print(get_text("analyze_file_unknown_format", ext=ext))
            return None, None
        
        if not all_results:
            print(get_text("analyze_file_empty"))
            return None, None
        
        return all_results, prompts_config
        
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(get_text("analyze_file_parse_error", error=str(e)))
        return None, None


def save_results(results: List[BenchmarkResult], output_file: str, output_format: str,
                 prompts_config: Optional[dict] = None,
                 run_config: Optional[dict] = None,
                 confirm_overwrite: bool = True):
    """Convenience function to save results.

    Args:
        results: List of BenchmarkResult objects
        output_file: Path to output file
        output_format: Output format ('json' or 'csv')
        prompts_config: Optional prompts configuration to include in export
    """
    saver = ResultSaver(output_file=output_file, output_format=output_format)
    saver.save(results, prompts_config, run_config, confirm_overwrite)
