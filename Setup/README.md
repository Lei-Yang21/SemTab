# Setup

Everything needed to reproduce the execution environment of the experiments, and the construction of a reduced local Wikidata subset so they do not depend solely on the live Wikidata API.

## Main files

- `Configuration/` — snapshot of the cluster conda environment (`myenv`): conda/pip exports, minimal requirements and hardware info, with instructions to recreate it. **Start here.** See its README.
- `env.sh` — the SLURM job that produced that snapshot: it exports `environment.yml` / `environment_nobuild.yml`, a full `pip freeze`, a filtered `requirements_core.txt`, and a `system_info.txt` with node, GPU, driver, CUDA and torch versions, into a timestamped `env_snapshot_<jobid>/` folder. Rerun it to refresh `Configuration/` after changing the environment.
- `extract_wikidata.bash` — SLURM job that filters a full Wikidata N-Triples dump (`wikidata2024.nt`) down to the triples needed for SemTab: English labels (`rdfs:label`, `schema:name`), descriptions, and the relevant claims. Runs on the cluster's scratch storage (~70 GB RAM, up to 24 h) and writes the reduced dataset to `${GLOBALSCRATCH}/${USER}/reduced/`. Requires Java 21 for the sorting/filtering step.

## Note

No code in the repository reads the reduced dump: the pipeline defaults to the live Wikidata API through `Pipeline_semtab/common/knowledge.py`. To use this extraction, implement the `KnowledgeSource` interface for the dump and pass that source to retrieval and ranking through `knowledge=`; setting a path to `reduced/` alone does not enable it.
