<!--
  System prompt: Sets the expert model's role and evaluation guidelines
  Used as the system message for all expert evaluations
-->
# system_prompt
You are a performance analysis assistant specialized in evaluating LLM benchmark results for Roo Code workflow optimization.

<!--
  User prompt template: Instructions for analyzing benchmark results
  Placeholder: {results} = formatted benchmark results summary (model name, params, size, TPS values, VRAM, ctx sizes)
-->
# user_prompt_template
You are analyzing benchmark results to recommend the best local LLM model for each Roo Code workflow mode.

Benchmark Results:
{results}

Please provide recommendations for optimizing each mode (Architect, Code, Debug).

<!--
  Translation prompt: Used to translate analysis to user's language
  Placeholder: {target_lang} = 'en' or 'ua', {text} = analysis text
-->
# translation_prompt_template
Translate the following text to {target_lang}. Output ONLY the translated text:

{text}

<!--
  Expert evaluation prompts for different evaluation aspects
-->
# expert
<!--
    System prompt: Defines expert evaluator role and scoring criteria
-->
## system_prompt
You are an expert LLM evaluator. Your task is to assess the quality of model responses on a scale of 0-10. Be strict but fair. Consider: 1) Relevance to the prompt, 2) Technical accuracy, 3) Completeness, 4) Practical usefulness, 5) Code quality (if applicable).
<!--
    Architect-mode evaluation template
    Evaluates system design quality, architecture decisions, and completeness
-->
## architect_eval
Evaluate this architect-mode response on a scale of 0-10.

Context: {context}

Response:
{response}

Score (0-10 only):
<!--
    Code-mode evaluation template
    Evaluates code correctness, completeness, style, and best practices
    If chain_context is provided, evaluates how well the code follows the architect plan
-->
## code_eval
Evaluate this code-mode response on a scale of 0-10.

Context: {context}

Architect Plan (reference):
{architect_response}

Response:
{response}

Score (0-10 only):
<!--
    Debug-mode evaluation template
    Evaluates bug detection accuracy, fix quality, and explanation clarity
    If chain_context is provided, evaluates if the fix addresses the actual implementation issues
-->
## debug_eval
Evaluate this debug-mode response on a scale of 0-10.

Context: {context}

Original Code (reference):
{code_response}

Response:
{response}

Score (0-10 only):