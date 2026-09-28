# Preprocessing

First pipeline stage. Cleans the raw dataset tables and attaches the annotation targets, producing the input of the retrieval stage. Chained steps (each reads the previous step's output): noise removal → typo correction → target marking.

## Usage

```
python main_preprocessing.py config/config_preprocessing_nollm.txt
```

## Main files

- `main_preprocessing.py` — entry point; parses the config and runs the three steps in order.
- `noise.py` — deterministic cleanup: HTML tags, extra whitespace, quote/markdown artifacts.
- `typo_method.py` — typo prompts, few-shot examples, correction guards, per-table cache and GPU worker orchestration. Skipped entirely when `NEED_CORRECTTYPO:false`.
- `../common/generation.py` — shared model/tokenizer loading, batching, generation and token accounting, also used by retrieval, ranking and the direct baseline.
- `hasAnnotation.py` — reads `cea/cta/cpa_targets.csv` from `TARGET_FOLDER` and appends a `Metadata:` column marking which cells/columns each downstream task must annotate.
- `logger_preprocessing.py` — cost logging for the correction step: per-table GPU memory and prompt/generated token counts. Both files are written to the `log_preprocessing/` folder, created on the first write, and are named after the run's `OUTPUT_FOLDER` (`vram_log_preprocessing_<output folder>.csv`, `token_log_preprocessing_<output folder>.csv`), so successive variants do not mix into one file; setting `LOG_DIR` moves the folder, and `VRAM_LOG_FILE` / `TOKEN_LOG_FILE` override the full path. Token rows carry the per-table delta and the running total; with `NUM_GPUS` > 1 each worker writes its own rows, identified by `gpu` and `pid`, so sum the last row of each worker to get the run total.
- `config/` — one config per preprocessing variant. See its README.
- `Job/` — SLURM submission scripts (submit from this folder, the paths inside are relative to it):
  - `job_preprocessing.sh` — the eight Valid-split variants, sequentially.
  - `job_preprocessing_2024R2.sh` — single no-LLM run on the separate 2024R2 dataset (`For_Thesis_finetune/`), used to build the finetuning data.
  - `job_preprocessing_finetuning.sh` — no-LLM run on the Training split; note it still points at `config_preprocessing_nollm_training.txt`, which was renamed to `config_preprocessing_nollm_2024R2.txt`.

## Typo correction and shared generation

`build_typo_engine` calls the common backend factory and supplies the preprocessing
token logger. The preprocessing defaults remain float16 on GPU, float32 on CPU,
the model's `max_position_embeddings` context limit (fallback 1024), batches of
32 and 64 generated tokens. Configure `MODEL_DTYPE`, `MAX_CTX`, `LLM_BATCH_SIZE`
and `MAX_NEW_TOKENS` to override these settings. Common `ADAPTER_PATH`,
`LOAD_IN_4BIT` and `GENERATION_ENGINE` options are also supported.

`build_prompts` creates `GenerationPrompt` objects with the original few-shot
conversation and a plain-text fallback. The shared backend applies the tokenizer
template and generates the responses. Preprocessing retains the model's default
thinking behavior and its existing prompt text.

`clean_correction` keeps the correction guards in `typo_method.py`: strip
surrounding quotes, replace newlines with spaces, remove an added final period,
and keep the original cell if the correction is empty or exceeds three times
its length. Numeric and metadata columns remain untouched. Repeated text values
share the per-table correction cache.

`NUM_GPUS` still controls HuggingFace workers; each worker loads its own engine
through the common factory. A single GPU run can use `DEVICE_ID`. An alternative
registered backend, or `process_folder(config, engine=engine)`, uses one process.
`NEED_CORRECTTYPO:False` returns without loading any model. See
[shared generation](../common/README.md) for the backend interface.

## Result

Evaluation (`Evaluator/Evaluate_preprocessing.ipynb`) showed the **no-LLM** variant gives the best downstream retrieval: the correction models change more correct cell values than they fix. `config_preprocessing_nollm.txt` is therefore the variant used by the full run.
