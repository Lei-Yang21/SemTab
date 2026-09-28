# config

Experiment configurations for the ranking stage. Each subfolder is one experiment group. Existing sweeps use the matching `Job/job_ranking_<group>.sh` script one level up; `heuristic/` provides direct launch commands in its README. Files are plain text, one `KEY:value` per line, `#` for comments.

## Groups

- `heuristic/` — compares quality transforms (`1 / quality` and `quality`) with the same weights and no LLM. See its README for adding another formula.
- `methods/` — compares the selection methods on the Valid split: `limited_slm`, `full_slm`, `slm_context`, plus `config_noslm.txt` (pure heuristic, `*_USE_SLM:False` and `CTA_FROM_SELECTION:False`) as the no-LLM floor. Contains the **single baseline** `config_slm_context.txt` (slm_context, Qwen2.5-3B, `candidate_lora_fp16` input, margin 0.10, zero-shot). All other groups vary exactly one factor relative to this baseline, so the duplicate baseline configs were removed from the other groups.
- `context_margin/` — sweep of the `LLM_CONTEXT_MARGIN` gate threshold.
- `models/` — compares base LLMs (Qwen2.5, Llama 3.x, Gemma, GLM-4).
- `prompts/` — compares prompting strategies (one-shot, few-shot, CoT, self-consistency).
- `inputs/` — same ranking on candidate sets from different retrieval variants.
- `finetuning/` — LoRA-finetuned ranking models (3B and 7B, for both `slm_context` and `limited_slm`) vs. the base model (baselines = `methods/config_slm_context.txt` and `methods/config_limited_slm.txt`).
- `cta_nosel/` — same three methods with `CTA_FROM_SELECTION:False` (CTA from the full candidate set instead of the CEA selection), on both the base Valid split and the Training split (`cta_nosel/other_dataset/`).
- `cta_cpa_noslm/` — same three methods with the SLM disabled on CTA and CPA (`CTA_USE_SLM:False`, `CPA_USE_SLM:False`), CEA unchanged; `nosel/` repeats it with `CTA_FROM_SELECTION:False`.
- `cta_margin/` — sweep of the CTA tie-break parameters `CTA_MARGIN` and `CTA_TOPK`, method fixed to slm_context.
- `other_dataset/` — the main comparison replayed on the 2024R1 **Training** split, to check the conclusions are not specific to Valid.
- `tablellama/` — the table-specialised `osunlp/TableLlama` as ranking model, with the pipeline's own prompts and with prompts rewritten in the TableLlama instruction format.

Each group has its own README with the per-config details.

## Key parameters

- `INPUT_FOLDER` / `PREPROCESS_FOLDER` / `OUTPUT_FOLDER` — candidate CSVs, preprocessing files, result destination.
- `METHOD` — `limited_slm` | `full_slm` | `slm_context`; `MODEL_NAME` — HuggingFace model id; `ADAPTER_PATH` — optional LoRA adapter.
- `TASKS` — which of `cea,cta,cpa` to produce.
- `SCORING_METHOD` — scorer class registered for the run; `heuristic` by default.
- `CEA_FEATURES` / `CEA_WEIGHTS` — heuristic components and their corresponding weights (`string,quality,type` and `0.5,0.2,0.3` by default).
- `CEA_QUALITY_METHOD` — quality transform registered in `scoring.QUALITY_METHODS`; `inverse` by default, or `identity`.
- `LLM_GATE` / `LLM_CONTEXT_MARGIN` / `LLM_CONTEXT_MAX_ROWS` — when and with how much table context the LLM is invoked (slm_context only).
- `CONTEXT_COT`, `CONTEXT_SELF_CONSISTENCY` + `SC_*` — CoT and self-consistency options; `SC_*` only take effect when self-consistency is enabled.
- `CONTEXT_PROMPT`, `CTA_PROMPT`, `CPA_PROMPT` — override the built-in prompt templates (used by `prompts/` and `tablellama/`).
- `LLM_VERIFY`, `LLM_TOPK` — debate/verify options (method `full_slm`).
- `CEA_*`, `CTA_*`, `CPA_*` — per-task SLM usage and tie-break margins; `CEA_CONTEXT_MARGIN` is inert while `CEA_CONTEXT_TIEBREAK:False`.
- `ENTITY_AS_URI`, `WRITE_HEADER`, `ROW_OFFSET` — SemTab output format.
