# config

Retrieval configurations, grouped one subfolder per experiment. Plain text, one `KEY:value` per line, `#` for comments. Unless stated otherwise every group reads the same input (`Valid/preprocessing_nollm`, the best preprocessing variant) and writes to its own candidate folder under `Valid/`, which is what `Evaluator/Evaluate_candidate.ipynb` then scores.

## Groups

- `config_test_preprocessing/` — **which preprocessing to build on.** Same retrieval setup (Qwen2.5-3B, 10 results/query), one config per preprocessing output: `nollm`, `basic`, `no_few_shot`, `qwen2`, `qwen3`, `llama`, `gemma`, `glm`. This is the group that established that no-LLM preprocessing wins. `config_candidate_tablellm.txt` is a leftover draft (empty `MODEL_NAME`, same folders as the llama config) and is not in the job list.
- `config_size/` — **model size and family**, everything else fixed (10 results/query): Qwen2.5 0.5B/1.5B/3B/7B, Llama-3.2 1B/3B, Gemma-2 9B, GLM-4 9B. Outputs `candidate_size_*_10`.
- `config_prompting/` — **prompting strategy** for the LLM generator, on Qwen2.5-3B: `zeroshot`, `oneshot`, `fewshot`, `cot`, `dumb` (deliberately minimal prompt) and `sc` (`LLM_SELF_CONSISTENCY:true`, the only non-deterministic retrieval run). Outputs `candidate_qwen25_3b_10_*`.
- `config_limit/` — **`SEARCH_LIMIT` sweep** (5, 20, 30, 50 results per query) on GLM-4 9B. Outputs `candidate_glm_9b_*_a`.
- `config_test_finetuning/` — **finetuned generators** vs. the non-finetuned `config_basic.txt`: `config_lora.txt` (`lora-fp16-adapter`), `config_lora_32.txt` (`lora-fp16-adapter-32`), `config_lora_32_30.txt` / `config_lora_32_50.txt` (same adapter, `SEARCH_LIMIT` 30 / 50), `config_lora_all.txt` (`lora-fp16-adapter-all`), `config_qlora.txt` (`qlora-adapter`, `LOAD_IN_4BIT:true`). `config_lora.txt` produces `candidate_lora_fp16`, the input of the ranking baseline and of the full run.
- `config_order/` — **generator order ablation**: `GENERATOR_ORDER:direct,fuzzy` with `USE_LLM:false`, i.e. the deterministic fuzzy variants replacing the LLM entirely, on the Valid and Training splits. Outputs `candidate_fuzzy_direct`.
- `config_other_dataset/` — **generalisation to other splits**: `config_basic_test.txt` and `config_lora_training.txt` on the 2024R1 Training split (the latter produces `candidate_lora_fp16_32`, the input of the ranking `other_dataset` group), and `config_glm_9b_10_2024R2.txt` on the 2024R2 data (`For_Thesis_finetune/`) used to build the finetuning datasets.

## Key parameters

- `INPUT_FOLDER` / `OUTPUT_FOLDER` — preprocessing output to read, candidate CSVs to write.
- `GENERATOR_ORDER` + `USE_<NAME>` — which registered generators run and in which order. Built-ins are `direct`, `llm` and `fuzzy`; custom generators follow the same convention.
- `<NAME>_QUALITY` — positive integer rank stored in the candidate CSV, independent of generator order. Defaults: direct 1, llm 2, fuzzy 3. With the default `1 / quality` scorer, smaller ranks score higher.
- `DUPLICATE_QUALITY` — quality retained for a repeated QID: `first` (default), `min` or `max`. `min` supports a later generator with a better rank; labels and row order retain the first occurrence.
- `SEARCH_LIMIT` / `<NAME>_SEARCH_LIMIT` — shared or per-generator results requested per query. `MAX_CANDIDATES_PER_CELL:0` means no cap after deduplication; with `min`/`max`, later generators still run to update the qualities of retained candidates after the cap is reached.
- `LANGUAGE`, `API_SLEEP`, `ENRICH_CANDIDATES` — API client and the enrichment pass (descriptions, aliases, P31/P279, sitelinks).
- `MODEL_NAME`, `ADAPTER_PATH`, `LOAD_IN_4BIT`, `PROMPT`, `LLM_MAX_SUGGESTIONS`, `LLM_MAX_NEW_TOKENS` — the LLM generator.
- `LLM_SELF_CONSISTENCY` + `SC_SAMPLES` / `SC_TEMPERATURE` / `SC_TOP_P` — self-consistency sampling; the `SC_*` keys are inert while it is false.

See the [retrieval README](../README.md#adding-generators) for registering
another strategy. Existing generator qualities can be changed directly with
`DIRECT_QUALITY`, `LLM_QUALITY` and `FUZZY_QUALITY`.
