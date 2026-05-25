"""Integration tests for benchmark result saving, merging, and expert results management.

These tests verify:
1. Response field is NOT saved to benchmark_results.json
2. Chains prompts ARE saved when using --all
3. Expert results file path is derived from output file
4. Merge mode works for both JSON and expert results files
5. Error handling works when merge file is corrupted

NOTE: These tests are marked with @pytest.mark.integration and are excluded
from default test runs. To run them manually:
    .venv/bin/python -m pytest tests/test_integration.py -v
    OR
    .venv/bin/python -m pytest tests/ -m integration -v
"""

import os
import tempfile
import unittest

import pytest

# Default model for integration tests - must be available in local Ollama
TEST_MODEL = "nomic-embed-text:latest"

# Default context size for tests
TEST_CONTEXT_SIZE = 8192

# Default chain for tests
TEST_CHAIN = "chain_rest_api"


class FakePromptLoader:
    """Fake prompt loader for testing."""
    data = {
        'independent': {'architect': [], 'code': [], 'debug': []},
        'chains': [
            {'id': 'chain_a', 'name': 'Chain A', 'prompts': {}},
        ],
    }

    def get_chains(self):
        return self.data['chains']

    def get_chain_by_id(self, chain_id):
        return next((c for c in self.data['chains'] if c['id'] == chain_id), None)


class FakeRunner:
    """Fake runner for testing."""

    def get_used_independent_prompts(self):
        return [{'mode': 'architect', 'id': 'p1', 'name': 'P1', 'prompt': 'test'}]


@pytest.mark.integration
class TestResponseNotInJson(unittest.TestCase):
    """Test that response field is NOT saved to benchmark_results.json."""

    def test_response_not_in_to_dict(self):
        """Verify response field is excluded from to_dict() output."""
        from benchmark.result import BenchmarkMetrics

        metrics = BenchmarkMetrics(
            ctx=TEST_CONTEXT_SIZE,
            temperature=0.0,
            avg_tps=10.0,
            min_tps=5.0,
            max_tps=15.0,
            std_dev=1.0,
            response="test response",
        )

        d = metrics.to_dict()
        self.assertNotIn("response", d, "response field should NOT be in to_dict()")

    def test_response_still_in_benchmark_metrics(self):
        """Verify response field still exists in BenchmarkMetrics class."""
        from benchmark.result import BenchmarkMetrics

        metrics = BenchmarkMetrics(
            ctx=TEST_CONTEXT_SIZE,
            temperature=0.0,
            avg_tps=10.0,
            min_tps=5.0,
            max_tps=15.0,
            std_dev=1.0,
            response="test response",
        )

        self.assertEqual(metrics.response, "test response")


@pytest.mark.integration
class TestChainsPromptsConfig(unittest.TestCase):
    """Test that chains prompts ARE saved when using --all."""

    def test_chains_saved_with_all_mode(self):
        """Verify both chains and independent prompts are saved with --all."""
        from types import SimpleNamespace

        from main_helpers import build_used_prompts_config

        config = build_used_prompts_config(
            FakePromptLoader(),
            SimpleNamespace(chain=None, chains=False, independent=False, all=True),
            FakeRunner(),
        )

        self.assertIsNotNone(config)
        self.assertIn("chains", config)
        self.assertIn("independent", config)

    def test_chains_saved_with_chains_flag(self):
        """Verify only chains are saved when using --chains."""
        from types import SimpleNamespace

        from main_helpers import build_used_prompts_config

        config = build_used_prompts_config(
            FakePromptLoader(),
            SimpleNamespace(chain=None, chains=True, independent=False, all=False),
            FakeRunner(),
        )

        self.assertIsNotNone(config)
        self.assertIn("chains", config)
        self.assertNotIn("independent", config)

    def test_single_chain_saved(self):
        """Verify single chain is saved when using --chain."""
        from types import SimpleNamespace

        from main_helpers import build_used_prompts_config

        loader = FakePromptLoader()
        loader.data = {
            'chains': [
                {'id': 'chain_rest_api', 'name': 'REST API', 'prompts': {}},
            ],
        }

        config = build_used_prompts_config(
            loader,
            SimpleNamespace(chain='chain_rest_api', chains=False, independent=False, all=False),
            FakeRunner(),
        )

        self.assertIsNotNone(config)
        self.assertIn("chains", config)
        self.assertEqual(len(config["chains"]), 1)
        self.assertEqual(config["chains"][0]["id"], "chain_rest_api")


@pytest.mark.integration
class TestExpertResultsFilePath(unittest.TestCase):
    """Test that expert results file path is derived from output file."""

    def test_expert_file_derived_from_json_output(self):
        """Verify expert results file path is derived from JSON output name."""
        json_output = "/tmp/test_expert_derived.json"
        expected_expert = "/tmp/test_expert_derived.md"

        if json_output:
            base_name = os.path.splitext(json_output)[0]
            derived_expert = base_name + ".md"
        else:
            derived_expert = "export/expert_results.md"

        self.assertEqual(derived_expert, expected_expert)

    def test_expert_file_default_when_no_output(self):
        """Verify default expert file path when no output specified."""
        json_output = None

        if json_output:
            base_name = os.path.splitext(json_output)[0]
            derived_expert = base_name + ".md"
        else:
            derived_expert = "export/expert_results.md"

        self.assertEqual(derived_expert, "export/expert_results.md")

    def test_expert_file_in_same_directory(self):
        """Verify expert file is created in same directory as JSON output."""
        json_output = "/tmp/subdir/my_results.json"
        expected_expert = "/tmp/subdir/my_results.md"

        base_name = os.path.splitext(json_output)[0]
        derived_expert = base_name + ".md"

        self.assertEqual(derived_expert, expected_expert)


@pytest.mark.integration
class TestExpertResultsMerge(unittest.TestCase):
    """Test expert results file merge functionality."""

    def test_expert_results_manager_load_existing(self):
        """Verify ExpertResultsManager can load existing entries."""
        from export.expert_results import ExpertResultsManager

        with tempfile.TemporaryDirectory() as tmpdir:
            expert_file = os.path.join(tmpdir, "test_merge.md")

            test_content = """# Expert Evaluation Results

**Generated:** 2026-05-25 10:00:00

**Tested Model:** test-model

**Expert Model:** none

**Total Responses:** 2

---

## Entry 1

### Prompt Information

- **Prompt ID:** `test_prompt_1`
- **Prompt Name:** Test Prompt 1
- **Mode:** architect
- **Context Size:** 8192
- **Temperature:** 0.0
- **Average TPS:** 50.0
- **Model:** test-model

### Response

```
Test response 1
```

**Expert Score:** 80

---

## Entry 2

### Prompt Information

- **Prompt ID:** `test_prompt_2`
- **Prompt Name:** Test Prompt 2
- **Mode:** code
- **Context Size:** 8192
- **Temperature:** 0.66
- **Average TPS:** 45.0
- **Model:** test-model

### Response

```
Test response 2
```

**Expert Score:** 75

---
"""
            with open(expert_file, "w", encoding="utf-8") as f:
                f.write(test_content)

            manager = ExpertResultsManager(output_file=expert_file)
            entries = manager._load_existing_entries()

            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0]["prompt_id"], "test_prompt_1")
            self.assertEqual(entries[0]["mode"], "architect")
            self.assertEqual(entries[0]["ctx"], 8192)
            self.assertEqual(entries[0]["temperature"], 0.0)
            self.assertEqual(entries[0]["avg_tps"], 50.0)
            self.assertEqual(entries[0]["model"], "test-model")
            self.assertEqual(entries[0]["expert_score"], 80)

            self.assertEqual(entries[1]["prompt_id"], "test_prompt_2")
            self.assertEqual(entries[1]["mode"], "code")
            self.assertEqual(entries[1]["expert_score"], 75)

    def test_expert_results_merge_adds_new_entries(self):
        """Verify merge adds new entries without duplicating existing ones."""
        from export.expert_results import ExpertResultsManager

        with tempfile.TemporaryDirectory() as tmpdir:
            expert_file = os.path.join(tmpdir, "test_merge_entries.md")

            initial_content = """# Expert Evaluation Results

**Generated:** 2026-05-25 10:00:00

**Tested Model:** test-model

**Expert Model:** none

**Total Responses:** 1

---

## Entry 1

### Prompt Information

- **Prompt ID:** `existing_prompt`
- **Prompt Name:** Existing Prompt
- **Mode:** architect
- **Context Size:** 8192
- **Temperature:** 0.0
- **Average TPS:** 50.0
- **Model:** test-model

### Response

```
Existing response
```

**Expert Score:** 80

---
"""
            with open(expert_file, "w", encoding="utf-8") as f:
                f.write(initial_content)

            manager = ExpertResultsManager(output_file=expert_file)
            manager.start_session(
                tested_model="test-model",
                expert_model=None,
                merge_mode="merge",
            )

            self.assertEqual(manager.get_entry_count(), 1)

            from benchmark.result import BenchmarkMetrics

            new_metrics = BenchmarkMetrics(
                ctx=8192,
                temperature=0.66,
                avg_tps=45.0,
                min_tps=40.0,
                max_tps=50.0,
                std_dev=2.0,
                prompt_id="new_prompt",
                prompt_name="New Prompt",
                mode="code",
                response="New response",
                expert_score=90.0,
            )
            manager.tested_model = "test-model"
            manager.add_entry(new_metrics)
            manager.save()

            self.assertEqual(manager.get_entry_count(), 2)

            manager2 = ExpertResultsManager(output_file=expert_file)
            manager2.start_session(
                tested_model="test-model",
                expert_model=None,
                merge_mode="merge",
            )

            self.assertEqual(manager2.get_entry_count(), 2)

    def test_expert_results_entry_key(self):
        """Verify entry key generation is correct."""
        from export.expert_results import ExpertResultsManager

        entry1 = {
            "model": "test-model",
            "ctx": 8192,
            "temperature": 0.0,
            "prompt_id": "prompt_1",
        }
        entry2 = {
            "model": "test-model",
            "ctx": 8192,
            "temperature": 0.0,
            "prompt_id": "prompt_1",
        }
        entry3 = {
            "model": "test-model",
            "ctx": 8192,
            "temperature": 0.66,
            "prompt_id": "prompt_1",
        }

        key1 = ExpertResultsManager._entry_key(entry1)
        key2 = ExpertResultsManager._entry_key(entry2)
        key3 = ExpertResultsManager._entry_key(entry3)

        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)


@pytest.mark.integration
class TestMergeErrorHandling(unittest.TestCase):
    """Test error handling during merge operations."""

    def test_merge_handles_corrupted_json(self):
        """Verify merge handles corrupted JSON file gracefully (creates new file)."""
        from benchmark.result import BenchmarkMetrics, BenchmarkResult, ModelInfo
        from export.merge_utils import load_results_file, merge_results

        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt_file = os.path.join(tmpdir, "corrupt.json")

            with open(corrupt_file, "w", encoding="utf-8") as f:
                f.write("invalid json{{{")

            model = ModelInfo(name="test-model", size_gb=1.0)
            new_result = BenchmarkResult(
                model=model,
                results=[
                    BenchmarkMetrics(
                        ctx=8192,
                        temperature=0.0,
                        avg_tps=10.0,
                        min_tps=5.0,
                        max_tps=15.0,
                        std_dev=1.0,
                    )
                ],
            )

            # Should handle gracefully - the corrupted file is treated as empty
            # and the new result is saved
            merge_results(corrupt_file, new_result)

            # Verify the file was created/updated with the new result
            loaded, _ = load_results_file(corrupt_file)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].model.name, "test-model")

    def test_merge_handles_empty_file(self):
        """Verify merge handles empty JSON file gracefully (creates new file)."""
        from benchmark.result import BenchmarkMetrics, BenchmarkResult, ModelInfo
        from export.merge_utils import load_results_file, merge_results

        with tempfile.TemporaryDirectory() as tmpdir:
            empty_file = os.path.join(tmpdir, "empty.json")

            with open(empty_file, "w", encoding="utf-8") as f:
                f.write("")

            model = ModelInfo(name="test-model", size_gb=1.0)
            new_result = BenchmarkResult(
                model=model,
                results=[
                    BenchmarkMetrics(
                        ctx=8192,
                        temperature=0.0,
                        avg_tps=10.0,
                        min_tps=5.0,
                        max_tps=15.0,
                        std_dev=1.0,
                    )
                ],
            )

            # Should handle gracefully - empty file is treated as having no results
            merge_results(empty_file, new_result)

            # Verify the file was created with the new result
            loaded, _ = load_results_file(empty_file)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].model.name, "test-model")

    def test_persist_model_result_handles_disabled_mode(self):
        """Verify persist_model_result handles disabled mode gracefully."""
        from benchmark.result import BenchmarkMetrics, BenchmarkResult, ModelInfo
        from main_helpers import persist_model_result

        model = ModelInfo(name="test-model", size_gb=1.0)
        benchmark_result = BenchmarkResult(
            model=model,
            results=[
                BenchmarkMetrics(
                    ctx=8192,
                    temperature=0.0,
                    avg_tps=10.0,
                    min_tps=5.0,
                    max_tps=15.0,
                    std_dev=1.0,
                )
            ],
        )

        # Should not raise any exception
        persist_model_result(
            save_mode="disabled",
            results_file="/tmp/test.json",
            benchmark_result=benchmark_result,
            all_results=[],
            prompts_config={},
            run_config={},
        )


@pytest.mark.integration
class TestBenchmarkMetricsToDict(unittest.TestCase):
    """Test BenchmarkMetrics.to_dict() output."""

    def test_to_dict_excludes_response(self):
        """Verify to_dict() does not include response field."""
        from benchmark.result import BenchmarkMetrics

        metrics = BenchmarkMetrics(
            ctx=8192,
            temperature=0.0,
            avg_tps=10.0,
            min_tps=5.0,
            max_tps=15.0,
            std_dev=1.0,
            prompt_id="test_prompt",
            prompt_name="Test Prompt",
            response="This should not appear in dict",
            expert_score=80.0,
        )

        d = metrics.to_dict()
        self.assertNotIn("response", d)
        self.assertEqual(d["ctx"], 8192)
        self.assertEqual(d["temperature"], 0.0)
        self.assertEqual(d["prompt_id"], "test_prompt")
        self.assertEqual(d["expert_score"], 80.0)

    def test_to_dict_includes_all_other_fields(self):
        """Verify to_dict() includes all expected fields."""
        from benchmark.result import BenchmarkMetrics

        metrics = BenchmarkMetrics(
            ctx=8192,
            temperature=0.66,
            avg_tps=100.0,
            min_tps=90.0,
            max_tps=110.0,
            std_dev=5.0,
            vram=5000000000,
            prompt_id="test",
            prompt_name="Test",
            duration_sec=10.5,
            prompt_tokens=100,
            response_tokens=200,
            mode="architect",
            chain_id="test_chain",
            chain_name="Test Chain",
            expert_score=85.0,
            avg_cpu_percent=50.0,
            max_cpu_percent=80.0,
            avg_ram_percent=60.0,
            max_ram_percent=90.0,
            avg_vram_percent=70.0,
            max_vram_percent=95.0,
        )

        d = metrics.to_dict()

        expected_fields = [
            "ctx", "ctx_str", "temperature", "avg_tps", "min_tps", "max_tps",
            "std_dev", "vram", "vram_str", "prompt_id", "prompt_name",
            "duration_sec", "prompt_tokens", "response_tokens", "mode",
            "chain_id", "chain_name", "expert_score",
            "avg_cpu_percent", "max_cpu_percent",
            "avg_ram_percent", "max_ram_percent",
            "avg_vram_percent", "max_vram_percent",
        ]

        for field in expected_fields:
            self.assertIn(field, d, f"Field '{field}' should be in to_dict() output")


if __name__ == "__main__":
    unittest.main()
