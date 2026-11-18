# Pipeline_semtab

The main annotation pipeline. Tables go through three sequential stages, each with its own folder, configs and SLURM job scripts:

1. `Preprocessing/` : cleans the raw tables (noise removal, LLM typo correction) and marks the cells/columns targeted by the ground truth.
2. `Candidate_Retrieval/` : retrieves candidate Wikidata entities for each target cell (API search + optional LLM query expansion + optional fuzzy variants) and enriches them with descriptions, aliases and types.
3. `Ranking/` : selects the final entity per cell among the candidates and derives the CEA/CTA/CPA annotations.

- `Baseline_direct/` : comparison baseline that skips retrieval entirely and asks the LLM for the answers directly.

Each stage writes CSVs that the next stage reads, so stages can also be run and evaluated independently. Every stage takes a single `KEY:value` config file as argument and keeps its experiment configs in `config/` and its SLURM scripts in `Job/`.

## Full run

- `main_pipeline.py` : runs the three stages end-to-end. Takes the three config paths as arguments (defaults to the files in `config/`):

  ```
  python main_pipeline.py config/config_preprocessing.txt config/config_candidate.txt config/config_ranking.txt
  ```

- `config/` — chained reference configs: no-LLM preprocessing → LoRA-finetuned candidate generator → slm_context ranking (Qwen2.5-3B, margin 0.10). The ranking config has no adapter; reproducing the thesis's fine-tuned ranking results requires a config from `Ranking/config/finetuning/` with matching input and adapter paths. Each stage's `OUTPUT_FOLDER` is the next stage's `INPUT_FOLDER`; final annotations go to `results/full_run`.
- `job_full_pipeline.sh` — SLURM job; submit from this folder, which must contain (or symlink) `WikidataTables2024R1/` and the `lora-fp16-adapter/` used by the retrieval stage.

## Smoke run

Smaller end-to-end check with the same settings as the full run, on a 5-table subset. It still requires the base model, retrieval adapter and Wikidata access; runtime depends on the hardware and API:

- `make_smoke_subset.py` : builds `WikidataTables2024R1/DataSets/Valid_smoke/` deterministically: up to 5 tables with the fewest CEA targets, prioritising tables with CTA and CPA targets too, with the target files filtered accordingly.
- `config/smoke/` : same three configs as `config/` but pointing to the subset; outputs to `results/smoke_run`.
- `job_smoke_pipeline.sh` : 2h SLURM job; builds the subset if missing, then runs `main_pipeline.py` on it. Locally: `python make_smoke_subset.py && python main_pipeline.py config/smoke/config_preprocessing.txt config/smoke/config_candidate.txt config/smoke/config_ranking.txt`.

## Running a subset of the stages

`run_stages.py` is a sibling of `main_pipeline.py` that runs any subset of the
three stages, in pipeline order, from the same launchers and the same configs:

```
python run_stages.py ranking=config/config_ranking.txt
python run_stages.py preprocessing=config/config_preprocessing.txt candidate=config/config_candidate.txt
```

Useful when only the ranking stage has to be rerun on candidates that already
exist, or when a run needs preprocessing and retrieval but no ranking.

## Retrieval variants and their cost

`job_retrieval_cost_valid.sh` — SLURM job, submitted from this folder like `job_full_pipeline.sh` (same requirements: `WikidataTables2024R1/`, `lora-fp16-adapter/` and an existing `logs/` folder). It runs the no-LLM preprocessing of the Valid split once, then six retrieval variants on that same preprocessed folder, through `run_stages.py`, so a single working directory holds the dataset and every log:

- `config_prompting/` — `zeroshot` (reference), `fewshot`, `cot`, `sc`: the prompt is the only thing that changes.
- `config_size/config_glm_9b.txt` — same prompt as `zeroshot`, larger model.
- `config_test_finetuning/config_lora.txt` — the LoRA-finetuned generator of the full run.

They are ordered cheapest first, self-consistency last, since it samples `SC_SAMPLES` times per cell. Unlike the per-group jobs in each stage's `Job/`, this one records the cost of each variant: `log_preprocessing/` and `log_candidate_retrieval/` collect one VRAM CSV and one token CSV per `OUTPUT_FOLDER`, and the job ends by printing the total tokens of each variant side by side. Resubmitting with `sbatch --export=ALL,RESUME=1 job_retrieval_cost_valid.sh` skips the variants whose output folder already exists, which is how to finish the run if it hits the 48h wall clock.

## Experiments vs. full run

The `config/` folder here contains the chained reference example. The ablations and comparisons reported in the thesis live in each stage's own `config/` subfolders (`Preprocessing/config/`, `Candidate_Retrieval/config/`, `Ranking/config/`), each with its own README; their outputs are collected locally in `Evaluator/Testing_data/`, which is ignored by Git.
