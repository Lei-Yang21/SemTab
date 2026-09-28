# Baseline_direct

Baseline that bypasses the retrieval + ranking pipeline: the LLM is prompted with the table content and asked to output the Wikidata QID (CEA), column type (CTA) or property PID (CPA) directly, out of its parametric knowledge. Serves as the reference point to quantify what the structured pipeline adds.

## Usage

```
python main_baseline.py config_baseline.txt
```

## Main files

- `main_baseline.py` — entry point, same `KEY:value` config format as the pipeline stages.
- `baseline_direct.py` — builds the CEA/CTA/CPA prompts per target (table block, marked cell, headers), parses QIDs/PIDs from the model output with regexes, NIL when unidentifiable.
- `../common/generation.py` — shared generation backend (batched, greedy decoding, optional adapter / 4-bit loading).
- `output_writer_baseline.py` — writes the annotations in the SemTab submission format, identical to the ranking stage so the same evaluators apply.
- `vram_logger_baseline.py` — per-table GPU memory logging.
- `config_baseline.txt` / `job_baseline.sh` — configuration and SLURM job.

## Setting

`GENERATION_ENGINE` selects a registered backend (`huggingface` by default).
An initialized engine can also be supplied through
`baseline_folder(config, engine=engine)`. Prompt construction and direct QID/PID
parsing remain specific to this baseline. See [shared generation](../common/README.md)
for configuration and backend interfaces.

It reads the same `preprocessing_nollm` folder as the pipeline, so both see exactly the same tables and targets — only the annotation strategy differs. The model is deliberately the largest of the comparison (Qwen2.5-7B-Instruct): the baseline is given the advantage, and the pipeline still has to beat it with a 3B model.

Prompt-context sizes are capped by `MAX_ROWS_CONTEXT` (20 rows), `CTA_MAX_VALUES` (30 column values) and `CPA_MAX_PAIRS` (15 column pairs) to stay inside `MAX_CTX`. Results are collected in `Evaluator/Testing_data/baseline/`.
