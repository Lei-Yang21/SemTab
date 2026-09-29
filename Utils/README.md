# Utils

Helper scripts and notebooks that support the pipeline but are not part of it: dataset preparation, finetuning-data construction, config generation and the analyses that fixed some pipeline constants.

## Dataset preparation

- `divide_dataset.py` : samples a reduced evaluation split from a full SemTab split: `N_SMALL` tables with at most `SMALL_MAX_CEA` CEA targets plus `N_RANDOM` tables drawn from the rest, then filters the target and ground-truth files accordingly. Config-driven, fixed seed.
- `config_divide_dataset.txt` : its config; current values sample 250 small (≤ 50 CEA rows) + 250 random tables from `Test` into `Small_Test`, seed 1.
- `job_divide_dataset.sh` : SLURM job for the above.

## Finetuning data

- `dataset_build.ipynb` : builds the JSON training set of the retrieval candidate generator (`candidate_gen_*.json`).
- `dataset_build_ranking.py` : builds the ranking training sets (`ft_slm_context_*.json`, `ft_slm_limited_*.json`) by replaying the ranking prompts against the ground truth; uses the shared `WikidataClient` for labels/descriptions, with maxlag handling and exponential backoff. `build_datasets(config_path, knowledge=source)` accepts a replacement `KnowledgeSource`, such as a local dump reader. Config-driven.
- `finetune/config_finetune.txt` its config: which split to read, `FT_METHOD` (which of the two datasets to build), output folder, validation ratio and seed, plus the gate keys mirroring the ranking inference config so the training distribution matches inference.
- `job_create.sh` : SLURM job running `dataset_build_ranking.py` over the configs listed in it.

> `dataset_build_ranking.py` also builds the MammoTab 2025 sets, with `ALLOW_NIL` on so a mention with no entity becomes a NIL training target. Its config and job live in `Mammotab/`, outside the thesis material.

Both outputs land in `Finetuning/Dataset/finetune_datasets/`.

## Analyses that feed the pipeline

- `weights_margin.ipynb` : grid search of the CEA scoring weights (similarity / quality / type coherence) and of the tie-break margin, replaying `Ranking/scoring.py` on `Evaluator/Testing_data/folder_for_ranking` against `cea_gt.csv`. This is where `DEFAULT_WEIGHTS = (0.5, 0.2, 0.3)` in `Ranking/cea.py` comes from.
- `generate_config.ipynb` : generates the experiment config files (sweeps) filled into the pipeline `config/` folders.
- `boundaries.ipynb` : decision-boundary figures (moons dataset, linear vs. kNN) for the ML background chapter of the thesis; unrelated to the pipeline.
- `wikidata_example.ipynb` : scratch notebook for exploring Wikidata API responses.

The environment/setup helper `env.sh` now lives in `Setup/`.
