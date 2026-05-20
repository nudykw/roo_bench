"""Expert model-based response evaluation module."""

import logging

import requests

from api.base_client import BaseApiClient
from benchmark.expert_evaluator_types import ExpertEvaluationEntry
from prompts.analysis_prompt_loader import AnalysisPromptLoader

logger = logging.getLogger('roo_bench.benchmark')


class ExpertEvaluator:
    """Evaluates benchmark responses using an expert LLM model."""

    def __init__(self, ollama_client: BaseApiClient, expert_model_name: str,
                 analysis_prompt_file: str | None = None):
        """Initialize expert evaluator.

        Args:
            ollama_client: API client for Ollama communication.
            expert_model_name: Name of the expert model to use.
            analysis_prompt_file: Optional path to analysis prompts file.
                                 If None, uses default .md or .jsonc.
        """
        self.ollama_client = ollama_client
        self.expert_model_name = expert_model_name
        self._prompt_loader = AnalysisPromptLoader(analysis_prompt_file)
        self.prompts = self._load_prompts()
        logger.info("[Expert] Loaded analysis prompts from: %s", self._prompt_loader.file_path)

    def _load_prompts(self) -> dict:
        """Load expert evaluation prompts from AnalysisPromptLoader.

        Returns:
            Dict with prompt templates. Returns defaults if file not found.
        """
        try:
            data = self._prompt_loader.data
            logger.debug("[Expert] Prompts loaded, keys: %s", list(data.keys()))
            return data
        except FileNotFoundError:
            logger.warning("Prompts file not found, using default templates")
            return self._default_prompts()
        except Exception as e:
            logger.warning(f"Error loading prompts: {e}, using defaults")
            return self._default_prompts()

    def evaluate_batch(self, entries: list[ExpertEvaluationEntry]) -> None:
        """Evaluate all responses in batch mode and assign scores directly.

        Uses sequential API calls with the expert model. Each response is
        evaluated independently with a temperature of 0.1 for consistency.
        Scores are written directly to entry.metrics_ref.expert_score.

        Args:
            entries: List of evaluation entries with responses.
        """
        logger.debug("[Expert] evaluate_batch: %d entries to evaluate", len(entries))
        sorted_entries = sorted(entries, key=lambda e: (e.ctx, e.temperature, e.mode or '', e.prompt_id or ''))
        current_group = None

        for i, entry in enumerate(sorted_entries):
            logger.debug(
                "[Expert] Entry[%d]: model=%s prompt_id=%r mode=%r response_len=%d metrics_ref=%s",
                i, entry.model_name, entry.prompt_id, entry.mode,
                len(entry.response), "set" if entry.metrics_ref is not None else "NONE"
            )

            group_key = (entry.ctx, entry.temperature)
            if group_key != current_group:
                current_group = group_key
                ctx_str = f"{entry.ctx // 1024}K" if entry.ctx >= 1024 else str(entry.ctx)
                print(f"\n   Context: {ctx_str} | Temperature: {entry.temperature:.2f}")

            score = self._evaluate_single(entry)

            if entry.metrics_ref is not None:
                entry.metrics_ref.expert_score = score
                logger.debug("[Expert] Wrote score %.1f to metrics_ref for entry[%d]", score, i)
            else:
                logger.debug("[Expert] metrics_ref is None for entry[%d], score not written", i)

            progress = (i + 1) / len(entries) * 100
            prompt_label = entry.prompt_id or entry.prompt_name or "default"
            mode_label = entry.mode or "default"
            print(
                f"      Expert evaluation: {i + 1}/{len(entries)} ({progress:.0f}%) "
                f"[{mode_label}:{prompt_label}] - Score: {score:.1f}/100"
            )

    def _evaluate_single(self, entry: ExpertEvaluationEntry) -> float:
        """Evaluate a single response.

        Args:
            entry: Evaluation entry containing the response.

        Returns:
            Float score between 0 and 100.
        """
        context = (
            f"ctx={entry.ctx}, temp={entry.temperature}, "
            f"mode={entry.mode or 'default'}, prompt={entry.prompt_id}"
        )

        prompt_template = self._get_prompt_template(entry.mode)

        template_vars = {
            'context': context,
            'response': entry.response[:4000]
        }

        if entry.chain_context:
            template_vars.update(entry.chain_context)
        else:
            for key in ['architect_response', 'code_response']:
                template_vars[key] = 'N/A (independent prompt)'

        try:
            eval_prompt = prompt_template.format(**template_vars)
        except KeyError:
            eval_prompt = prompt_template.format(
                context=context,
                response=entry.response[:4000]
            )

        score = self._call_expert_api(eval_prompt)
        return score

    def _get_prompt_template(self, mode: str | None) -> str:
        """Get appropriate prompt template based on mode.

        Args:
            mode: Evaluation mode (architect, code, debug, or None).

        Returns:
            Prompt template string.
        """
        templates = {
            'architect': self.prompts.get('expert', {}).get('architect_eval', ''),
            'code': self.prompts.get('expert', {}).get('code_eval', ''),
            'debug': self.prompts.get('expert', {}).get('debug_eval', ''),
        }
        return templates.get(mode, templates.get('architect', ''))

    def _call_expert_api(self, prompt: str) -> float:
        """Call Ollama API for evaluation.

        Uses temperature=0.1 for consistent scoring.
        Parses integer response from 0-100 range.

        Args:
            prompt: Evaluation prompt to send.

        Returns:
            Float score between 0 and 100.
        """
        payload = {
            "model": self.expert_model_name,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.1,
            },
        }

        logger.debug("[Expert] _call_expert_api: model=%s prompt_len=%d", self.expert_model_name, len(prompt))
        logger.debug("[Expert] Prompt (first 300 chars): %s", prompt[:300])
        try:
            response = requests.post(
                f"{self.ollama_client.base_url}/api/generate",
                json=payload,
                headers=self.ollama_client.headers,
                timeout=120
            )

            logger.debug("[Expert] API response status: %d", response.status_code)
            if response.status_code != 200:
                logger.warning(f"Expert evaluation failed: {response.text}")
                return 50.0

            result = response.json()
            response_text = result.get("response", "5")
            logger.debug("[Expert] Raw response text: %r", response_text)

            score = self._parse_score(response_text)
            logger.debug("[Expert] Parsed score: %.1f", score)
            return score

        except Exception as e:
            logger.warning(f"Expert API error: {e}")
            return 50.0

    @staticmethod
    def _parse_score(response_text: str) -> float:
        """Extract numeric score from model response.

        Args:
            response_text: Raw text response from expert model.

        Returns:
            Float score clamped to 0-100 range.
        """
        import re
        match = re.search(r'\b(100|[1-9]?[0-9])\b', response_text.strip())
        if match:
            score = int(match.group(1))
            return float(min(100, max(0, score)))
        return 50.0
