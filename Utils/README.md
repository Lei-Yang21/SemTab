# Utils

Helper scripts and notebooks that support the pipeline but are not part of it: dataset preparation, finetuning-data construction, config generation and the analyses that fixed some pipeline constants.

## Dataset preparation

- `divide_dataset.py` : samples a reduced evaluation split from a full SemTab split: `N_SMALL` tables with at most `SMALL_MAX_CEA` CEA targets plus `N_RANDOM` tables drawn from the rest, then filters the target and ground-truth files accordingly. Config-driven, fixed seed.
- `config_divide_dataset.txt` : its config; current values sample 250 small (≤ 50 CEA rows) + 250 random tables from `Test` into `Small_Test`, seed 1.
- `job_divide_dataset.sh` : SLURM job for the above.

## Finetuning data

- `dataset_build.ipynb` : builds the JSON training set of the retrieval candidate generator (`candidate_gen_*.json`).
- `dataset_build_ranking.py` : builds the ranking training sets (`ft_slm_context_*.json`, `ft_slm_limited_*.json`) using the same `ScoringMethod`, `prepare_table`, `PromptBuilder` and `TableLabels` as inference. `WikidataClient` supplies labels by default.
- `finetune/config_finetune.txt` its config: which split to read, `FT_METHOD` (which of the two datasets to build), output folder, validation ratio and seed, plus the gate keys mirroring the ranking inference config so the training distribution matches inference.
- `job_create.sh` : SLURM job running `dataset_build_ranking.py` over the configs listed in it.

> `dataset_build_ranking.py` also builds the MammoTab 2025 sets, with `ALLOW_NIL` on so a mention with no entity becomes a NIL training target. Its config and job live in `Mammotab/`, outside the thesis material.

Both outputs land in `Finetuning/Dataset/finetune_datasets/`.

`build_datasets` accepts a config file path or a config dictionary. The scorer,
knowledge source, prompt builder and table loader can be supplied directly:

```python
from Utils.dataset_build_ranking import build_datasets

examples = build_datasets(config, scorer=scorer, knowledge=source,
                          prompts=prompt_builder, table_loader=loader)
```

The loader has the same contract as ranking: `(input_path, preprocess_path)`
returns a `TableContext` whose features and scores have not yet been prepared.
Folder traversal keeps the CSV naming convention. The scorer is created once
and called once per table; a scorer requiring an LLM can be constructed with
its selector, or selected through config with `llm=selector` supplied to
`build_datasets`. Omitted dependencies use the inference defaults.

Training now honors `SCORING_METHOD`, `CEA_FEATURES`, `CEA_WEIGHTS`,
`CEA_QUALITY_METHOD`, `CEA_LLM_TOPK` and the shared prompt options, including
`CONTEXT_COT`. Use the same config and prompt builder for inference and training.
`FT_METHOD` accepts `limited_slm`, its existing alias `slm_limited`, and
`slm_context`; output filenames retain the chosen method name.

The function returns the examples by method as well as writing the JSON files.
Gold filtering, exclusion of trivial shortlists, NIL sampling and the split by
table remain specific to dataset construction; it does not generate LLM answers.

## Analyses that feed the pipeline

- `weights_margin.ipynb` : grid search of the CEA scoring weights (similarity / quality / type coherence) and of the tie-break margin, replaying `Ranking/scoring.py` on `Evaluator/Testing_data/folder_for_ranking` against `cea_gt.csv`. This is where `DEFAULT_WEIGHTS = (0.5, 0.2, 0.3)` in `Ranking/cea.py` comes from.
- `generate_config.ipynb` : generates the experiment config files (sweeps) filled into the pipeline `config/` folders.
- `boundaries.ipynb` : decision-boundary figures (moons dataset, linear vs. kNN) for the ML background chapter of the thesis; unrelated to the pipeline.
- `wikidata_example.ipynb` : scratch notebook for exploring Wikidata API responses.

The environment/setup helper `env.sh` now lives in `Setup/`.
