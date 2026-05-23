"""Diagnostic test for ExpertEvaluator.

Sends a real large-size request to Ollama using the same code path as the
expert evaluation pipeline. Prints a detailed analysis of every API response field.

Usage:
    python test_expert_evaluator.py
    python test_expert_evaluator.py --ollama-url http://aorus-cachyos-server:11434
    python test_expert_evaluator.py --ollama-url http://aorus-cachyos-server:11434 --model gemma4:latest
"""

import logging
import sys

# ── Full debug logging identical to production ─────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_expert_evaluator")

import requests  # noqa: E402  (after logging setup so urllib3 debug is visible)

from api.local_client import LocalApiClient
from benchmark.expert_evaluator import ExpertEvaluator
from benchmark.expert_evaluator_types import ExpertEvaluationEntry
from benchmark.result import BenchmarkMetrics
from tests.test_data import LARGE_RESPONSE, parse_cli_args

# ── CLI argument parsing (mirrors main.py pattern) ─────────────────────────────
OLLAMA_URL, MODEL = parse_cli_args()


# ── Helpers ────────────────────────────────────────────────────────────────────
def header(title: str) -> None:
    print(f"\n{'═' * 64}")
    print(f"  {title}")
    print(f"{'═' * 64}")


# ── Phase 1: Inspect prompt building (no network call) ─────────────────────────
def phase1_inspect_prompt() -> tuple:
    """Build and log the evaluation prompt exactly as ExpertEvaluator._evaluate_single does."""
    header("PHASE 1 — Prompt building (no API call)")

    client    = LocalApiClient(base_url=OLLAMA_URL, timeout=120)
    evaluator = ExpertEvaluator(ollama_client=client, expert_model_name=MODEL)

    print(f"  client.base_url          : {client.base_url}")
    print(f"  client.headers           : {client.headers!r}")
    print(f"  evaluator.expert_model   : {evaluator.expert_model_name}")
    print(f"  prompts top-level keys   : {list(evaluator.prompts.keys())}")
    print(f"  prompts['expert'] keys   : {list(evaluator.prompts.get('expert', {}).keys())}")

    entry = ExpertEvaluationEntry(
        model_name  = "llama3.2:3b",
        ctx         = 32768,
        temperature = 0.7,
        prompt_id   = "architect_001",
        prompt_name = "System Design: Multi-tenant SaaS Auth",
        mode        = "architect",
        chain_id    = None,
        chain_name  = None,
        response    = LARGE_RESPONSE,
        avg_tps     = 42.5,
        metrics_ref = None,
        chain_context = {},
    )

    print(f"\n  entry.mode               : {entry.mode!r}")
    print(f"  entry.response length    : {len(entry.response)} chars")
    print(f"  entry.response[:4000] len: {min(len(entry.response), 4000)} chars")

    # ── Replicate _evaluate_single logic verbatim ──────────────────────────────
    context = (
        f"ctx={entry.ctx}, temp={entry.temperature}, "
        f"mode={entry.mode or 'default'}, prompt={entry.prompt_id}"
    )
    template = evaluator._get_prompt_template(entry.mode)
    template_vars = {
        "context":            context,
        "response":           entry.response[:4000],
        "architect_response": "N/A (independent prompt)",
        "code_response":      "N/A (independent prompt)",
    }
    if entry.chain_context:
        template_vars.update(entry.chain_context)

    try:
        eval_prompt = template.format(**template_vars)
    except KeyError as exc:
        print(f"  ⚠️  KeyError filling template: {exc} — falling back to minimal format")
        eval_prompt = template.format(context=context, response=entry.response[:4000])

    print(f"\n  Template for mode='architect':\n  {template!r}")
    print(f"\n  Final prompt length      : {len(eval_prompt)} chars")
    print("\n  ── Prompt first 600 chars ──")
    print(eval_prompt[:600])
    print("  ── Prompt last 200 chars ──")
    print(eval_prompt[-200:])

    return evaluator, entry, eval_prompt


# ── Phase 2: Raw requests.post — maximum visibility into API response ───────────
def phase2_raw_api_call(eval_prompt: str) -> dict | None:
    """Send the same payload ExpertEvaluator._call_expert_api would send,
    but intercept every byte of the response for analysis."""
    header("PHASE 2 — Raw requests.post (identical payload to _call_expert_api)")

    # Mirrors _call_expert_api payload exactly: no num_predict limit, model stops at EOS
    payload = {
        "model":  MODEL,
        "prompt": eval_prompt,
        "stream": False,
        "think":  False,
        "options": {
            "temperature": 0.1,
        },
    }
    headers_used = {}  # LocalApiClient.headers == {}

    print(f"  URL                      : {OLLAMA_URL}/api/generate")
    print(f"  payload['model']         : {payload['model']!r}")
    print(f"  payload['stream']        : {payload['stream']}")
    print(f"  payload['think']         : {payload['think']}")
    print(f"  payload['options']       : {payload['options']}")
    print(f"  prompt length            : {len(payload['prompt'])} chars")
    print(f"  headers sent             : {headers_used!r}")
    print("\n  ⏳ Sending request (timeout=120s)…")

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            headers=headers_used,
            timeout=120,
        )

        print("\n  ── HTTP Response ──")
        print(f"  status_code              : {response.status_code}")
        print(f"  Content-Type             : {response.headers.get('content-type', 'N/A')}")
        print(f"  content-length header    : {response.headers.get('content-length', '<not set>')}")
        print(f"  actual body size         : {len(response.content)} bytes")
        print(f"\n  raw body (first 1 000 bytes): {response.content[:1000]!r}")

        if response.status_code != 200:
            print(f"\n  ❌ Non-200 status — full body: {response.text[:500]!r}")
            return None

        try:
            data = response.json()
        except Exception as exc:
            print(f"\n  ❌ JSON parse failed: {exc}")
            print(f"  raw text: {response.text[:500]!r}")
            return None

        print("\n  ── Parsed JSON fields ──")
        print(f"  keys present             : {list(data.keys())}")

        response_field = data.get("response", "<<MISSING>>")
        print(f"\n  'response' type          : {type(response_field).__name__}")
        print(f"  'response' value (repr)  : {response_field!r}")
        print(f"  'response' length        : {len(response_field) if isinstance(response_field, str) else 'N/A'}")

        print(f"\n  'done'                   : {data.get('done')}")
        print(f"  'done_reason'            : {data.get('done_reason')!r}")
        print(f"  'model'                  : {data.get('model')!r}")
        print(f"  'eval_count'             : {data.get('eval_count')}")
        print(f"  'prompt_eval_count'      : {data.get('prompt_eval_count')}")
        td = data.get("total_duration", 0)
        ld = data.get("load_duration", 0)
        pd = data.get("prompt_eval_duration", 0)
        ed = data.get("eval_duration", 0)
        print(f"  'total_duration'         : {td / 1e9:.3f}s")
        print(f"  'load_duration'          : {ld / 1e9:.3f}s")
        print(f"  'prompt_eval_duration'   : {pd / 1e9:.3f}s")
        print(f"  'eval_duration'          : {ed / 1e9:.3f}s")

        # ── Parse score exactly as ExpertEvaluator._parse_score ────────────────
        parsed_score = ExpertEvaluator._parse_score(response_field if isinstance(response_field, str) else "50")
        print("\n  ── Score parsing ──")
        print(f"  ExpertEvaluator._parse_score({response_field!r}) → {parsed_score}")
        if not isinstance(response_field, str) or response_field.strip() == "":
            print("  ⚠️  WARNING: 'response' field is empty/missing — _parse_score will return default 50!")

        return data

    except requests.exceptions.Timeout:
        print("\n  ❌ TIMEOUT after 120 s")
    except requests.exceptions.ConnectionError as exc:
        print(f"\n  ❌ CONNECTION ERROR: {exc}")
    except Exception as exc:
        print(f"\n  ❌ UNEXPECTED ERROR: {exc}")
        import traceback
        traceback.print_exc()

    return None


# ── Phase 3: Full evaluate_batch() path with a real BenchmarkMetrics ref ───────
def phase3_evaluate_batch(evaluator: ExpertEvaluator, entry: ExpertEvaluationEntry) -> float | None:
    """Run the actual evaluate_batch() code that production uses."""
    header("PHASE 3 — evaluate_batch() with metrics_ref (production path)")

    metrics = BenchmarkMetrics(
        ctx         = entry.ctx,
        temperature = entry.temperature,
        avg_tps     = entry.avg_tps,
        min_tps     = 38.0,
        max_tps     = 47.0,
        std_dev     = 2.0,
        mode        = entry.mode,
        prompt_id   = entry.prompt_id,
        prompt_name = entry.prompt_name,
    )
    entry.metrics_ref = metrics

    print(f"  metrics.expert_score     : {metrics.expert_score}  (before)")
    print(f"  entry.metrics_ref is None: {entry.metrics_ref is None}")
    print("  Calling evaluator.evaluate_batch([entry])…")

    evaluator.evaluate_batch([entry])

    print(f"\n  metrics.expert_score     : {metrics.expert_score}  (after)")
    print(f"  type(expert_score)       : {type(metrics.expert_score).__name__}")
    return metrics.expert_score


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"OLLAMA_URL       : {OLLAMA_URL}")
    print(f"MODEL            : {MODEL}")
    print(f"LARGE_RESPONSE   : {len(LARGE_RESPONSE)} chars")

    try:
        evaluator, entry, eval_prompt = phase1_inspect_prompt()
        raw_data                      = phase2_raw_api_call(eval_prompt)
        final_score                   = phase3_evaluate_batch(evaluator, entry)

        header("SUMMARY")
        print(f"  eval_prompt length       : {len(eval_prompt)} chars")
        if raw_data:
            resp_field = raw_data.get("response", "<<MISSING>>")
            print(f"  raw 'response' field     : {resp_field!r}")
            print(f"  raw eval_count           : {raw_data.get('eval_count')}")
        else:
            print("  raw API call             : FAILED (see PHASE 2 output)")
        print(f"  evaluate_batch score     : {final_score}")

        if raw_data and raw_data.get("response", ""):
            print("\n  ✅ API returned a non-empty 'response' — score parsing should succeed.")
        elif raw_data:
            print()
            print("  ⚠️  DIAGNOSIS: The API returned an EMPTY 'response' field.")
            print("     _parse_score('') finds no match → returns default 50.0.")
            print("     Root cause candidates:")
            print("     1. num_predict too low — model cannot emit even one token.")
            print("     2. Model is a thinking model and 'think: false' is not propagated.")
            print("     3. Prompt is too long and model truncates before the score token.")
            print("     4. Ollama version bug with non-streaming + very short num_predict.")

    except Exception as exc:
        print(f"\nFATAL: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
