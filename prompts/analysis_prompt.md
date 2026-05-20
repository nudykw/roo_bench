<!--
  Analysis prompts for Roo Bench expert evaluation
  Contains system prompts, user templates, and expert evaluation criteria
  Generated from analysis_prompt.jsonc: python roo_bench.py --generate-md
-->

# system_prompt
**Prompt:** You are a performance analysis assistant specialized in evaluating LLM benchmark results for Roo Code workflow optimization.

# user_prompt_template
**Template:** You are analyzing benchmark results to recommend the best local LLM model for each Roo Code workflow mode.

## Benchmark Parameters Explained

**Performance Metrics:**
- avg_tps: Average tokens/sec generated (higher = faster). Primary speed indicator.
- min_tps / max_tps: Worst and best-case speed. A wide gap indicates inconsistent performance under load.
- std_dev: TPS standard deviation — lower means more stable and predictable generation.
- vram: VRAM consumed (MB). Models that exceed GPU VRAM fall back to RAM, causing severe TPS slowdown.
- ctx: Context window size tested (e.g. 8K / 16K / 32K / 64K / 128K tokens).
- expert_score: AI-assigned quality score (0–100) evaluating response relevance, technical accuracy, and completeness.
- duration_sec: Total wall-clock time for the generation run.

**Model Characteristics:**
- params: Model size (e.g. 7B, 14B, 32B). Larger models are generally more capable but slower.
- size_gb: Disk/VRAM footprint. Determines if the model fits in GPU VRAM (fast) or spills to RAM (slow).
- architecture: Dense (all parameters active per token) vs MoE (sparse — only a subset of parameters active per token).
  - MoE advantage: High capability at lower runtime cost — good for throughput-critical tasks.
  - Dense advantage: More consistent and factual output, lower hallucination risk — good for precision tasks.
- temperature used during benchmark: 0.0 = fully deterministic, 1.0 = highly creative/varied.

## Mode-Specific Requirements

### 🏗️ Architect Mode — System Design & Planning
**Priority: Quality of analysis and depth of reasoning over speed**
- ctx: CRITICAL — the more the better (32K+ strongly preferred). Architect must analyze entire codebases and project structures.
- avg_tps: Secondary — slower generation is acceptable if quality is high. Users can wait minutes for a well-thought-out architecture.
- temperature: 0.5–0.8 recommended — allows creative exploration of solutions, trade-off analysis, and alternative approaches.
- expert_score: HIGH importance — architect responses must be coherent, comprehensive, and strategically sound.
- architecture: MoE or Dense are both acceptable; prefer higher parameter count for deeper reasoning capacity.
- std_dev: Less critical — some TPS variance is acceptable for this mode.

### 💻 Code Mode — Code Generation & Implementation
**Priority: Speed + strict correctness of code output**
- avg_tps: CRITICAL — fast code generation directly reduces developer wait time. Target >20 TPS.
- ctx: Medium (8K–32K) — sufficient for file-level context, function bodies, and class definitions.
- temperature: 0.0–0.3 REQUIRED — strict execution of the architect's plan with no improvisation or creative deviation.
- expert_score: HIGH importance — generated code must be syntactically correct and faithfully follow the architect plan.
- architecture: MoE PREFERRED — higher throughput with acceptable accuracy for structured, deterministic code output.
- std_dev: Should be low — consistent generation speed is important for a smooth developer experience.
- vram: Monitor carefully — VRAM overflow causes dramatic TPS drops and unpredictable behavior.

### 🐛 Debug Mode — Bug Detection & Root Cause Analysis
**Priority: Factual accuracy + maximum sustained speed**
- avg_tps: CRITICAL — fast iteration is essential for tight debugging cycles. Target maximum possible TPS.
- ctx: HIGH (32K+) — must simultaneously hold full stack traces, file contents, error messages, and surrounding code context.
- temperature: 0.0–0.1 REQUIRED — ironclad factual analysis only. Hallucinated stack traces, incorrect variable names, or fabricated error codes are completely unacceptable.
- expert_score: HIGHEST importance — debug responses must be factually accurate. An incorrect fix introduces regressions and wastes debugging time.
- architecture: Dense PREFERRED — lower hallucination rate at very low temperature settings compared to MoE.
- std_dev: Low preferred — predictable and stable timing is important for iterative, rapid debugging sessions.
- min_tps: Examine carefully — if min_tps drops far below avg_tps, the model is unstable and unreliable under sustained load.

## Task

Based on the benchmark results below, provide recommendations for each of the three Roo Code modes (Architect, Code, Debug).
For each mode provide: recommended model name, recommended ctx size, recommended temperature setting, and a concise justification referencing the specific metrics that support your choice.

{results}

# translation_prompt_template
**Template:** Translate the following text to {target_lang}. Output ONLY the translated text:

{text}

# expert

## system_prompt
**Prompt:** You are an expert LLM evaluator. Your task is to assess the quality of model responses on a scale of 0-100. Be strict but fair. Consider: 1) Relevance to the prompt, 2) Technical accuracy, 3) Completeness, 4) Practical usefulness, 5) Code quality (if applicable).

## architect_eval
**Template:** Evaluate this architect-mode response on a scale of 0-100.

Context: {context}

Response:
{response}

Score (0-100 only):

## code_eval
**Template:** Evaluate this code-mode response on a scale of 0-100.

Context: {context}

Architect Plan (reference):
{architect_response}

Response:
{response}

Score (0-100 only):

## debug_eval
**Template:** Evaluate this debug-mode response on a scale of 0-100.

Context: {context}

Original Code (reference):
{code_response}

Response:
{response}

Score (0-100 only):
