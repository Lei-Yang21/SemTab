# Candidate_Retrieval

Second pipeline stage. For every target cell it builds a list of candidate Wikidata entities, later ranked by the Ranking stage. Output: one `<table>_candidates.csv` per table with the candidates and their enrichment data.

## Usage

```
python main_candidate.py config/config_test_finetuning/config_lora.txt
```

## Main files

- `main_candidate.py` — entry point; parses the config and processes the input folder.
- `candidate_retrieval.py` — orchestrator; reads the preprocessing metadata to know which cells to process, runs the generators, deduplicates and writes results.
- `generators.py` — extensible candidate generators, applied in `GENERATOR_ORDER`, each with its own configurable quality rank used later by the scorer. Defaults:
  - `DirectGenerator` (quality 1) — Wikidata full-text search on the cell value as-is.
  - `LLMGenerator` (quality 2) — the LLM proposes alternative surface forms/labels, each of which is then searched; optional self-consistency sampling (`LLM_SELF_CONSISTENCY`).
  - `FuzzyGenerator` (quality 3) — deterministic surface variants (quotes and parentheses stripped, leading article removed, `"Last, First"` reordered, split on `-` / `/`, first-word and first-two-word prefixes), each searched separately. Off by default.
- `enrichment.py` — fetches description, aliases, P31/P279 types and sitelink count for each candidate QID (features used by the ranking scorer).
- `../common/generation.py` — shared generation backend (chat formatting, batching, optional LoRA/4-bit loading and token accounting). Retrieval keeps a default context length of 2048.
- `wikidata_api.py` — rate-limited Wikidata API client with retry/backoff and caching.
- `logger_candidate_retrieval.py` — cost logging for the LLM generator: per-table GPU memory and prompt/generated token counts. Both files are written to the `log_candidate_retrieval/` folder, created on the first write, and are named after the run's `OUTPUT_FOLDER` (`vram_log_candidate_retrieval_<output folder>.csv`, `token_log_candidate_retrieval_<output folder>.csv`), so successive variants do not mix into one file; setting `LOG_DIR` moves the folder, and `VRAM_LOG_FILE` / `TOKEN_LOG_FILE` override the full path. Token rows carry the per-table delta and the running total. Nothing is logged when `USE_LLM` is off (no `MODEL_NAME`): the stage is then pure API search.
- `config/` — the experiment groups, one subfolder each. See its README.
- `Job/` — one SLURM script per experiment group; submit from this folder. Two of them (`job_limit_retrieval.sh`, `job_prompting_retrieval.sh`) are job arrays indexed by `SLURM_ARRAY_TASK_ID`, the others loop over the configs sequentially.

> The config paths hard-coded in most `Job/*.sh` scripts are the flat `config/config_*.txt` paths that predate the reorganisation of `config/` into per-group subfolders. Update the `CONFIGS=(...)` list (e.g. `config/config_size/config_glm_9b.txt`) before resubmitting them.

## Generator configuration

Each generator has independent activation, quality and search-limit settings:

```text
GENERATOR_ORDER:direct,llm,fuzzy
USE_DIRECT:True
USE_LLM:True
USE_FUZZY:True
DIRECT_QUALITY:1
LLM_QUALITY:2
FUZZY_QUALITY:3
SEARCH_LIMIT:10
FUZZY_SEARCH_LIMIT:5
DUPLICATE_QUALITY:first
```

Quality values are positive integer ranks, independent of execution order.
The default ranking formula `1 / quality` favors smaller values. Changing the
retrieval quality requires regenerating the candidate CSVs before ranking.
Existing configurations retain qualities 1, 2 and 3, with fuzzy disabled.

`DUPLICATE_QUALITY` controls the quality stored when generators return the same
QID: `first` keeps the first occurrence (default), `min` keeps the smallest rank,
and `max` keeps the largest. Use `min` with `1 / quality` to favor the best rank,
or `max` if the ranking transform rewards larger values, such as `identity`.
Labels and row order still come from the first occurrence.

`MAX_CANDIDATES_PER_CELL` limits the number of distinct entities in the output.
With `first`, retrieval stops once the cap is reached. With `min` or `max`, it
continues running the generators to update qualities for retained entities;
additional distinct entities are discarded. These policies can make more API
calls. The cap's candidate selection still depends on generator order.

## Adding generators

`SearchGenerator` handles searching the suggested terms. Subclasses implement
`suggestions(query, context_str)`. For a source that returns entities directly,
subclass `BaseGenerator` and implement `candidates(query, context_str)`, returning
an iterable of `(label, qid)` pairs.

For another search strategy, register a factory under its own name:

```python
from generators import SearchGenerator, GENERATOR_FACTORIES, search_options

class CustomGenerator(SearchGenerator):
    name = "custom"
    quality = 4

    def suggestions(self, query, context_str):
        return [query.lower()]

def build_custom(config, engine, search):
    return CustomGenerator(**search_options(config, "custom", search))

GENERATOR_FACTORIES["custom"] = (build_custom, True)
```

Place the implementation and registry entry in `generators.py`, or import your
extension module before calling `candidate_retrieval_folder`. Add `custom` to
`GENERATOR_ORDER`; `USE_CUSTOM`, `CUSTOM_QUALITY` and `CUSTOM_SEARCH_LIMIT` follow
the same convention as the built-in generators.

Changing the quality of existing generators requires only configuration. For
example, to give fuzzy candidates a higher quality score than LLM candidates
under `1 / quality`:

```text
GENERATOR_ORDER:direct,llm,fuzzy
USE_FUZZY:True
DIRECT_QUALITY:1
LLM_QUALITY:3
FUZZY_QUALITY:2
DUPLICATE_QUALITY:min
```

Swap the quality values to reverse that preference. No new generator is needed.

Registry factories receive `(config, engine, search)`; the boolean sets the
default enabled state. They may return `None` when unavailable, as the LLM
factory does without an engine. `SearchGenerator` and `build_generators` accept
an optional `search(query, language, limit)` function to replace Wikidata search.
An unrecognized generator name is warned about and skipped, as before.

`LLMGenerator` uses the same `GenerationEngine.generate` interface as ranking
and the direct baseline. Select a registered backend with `GENERATION_ENGINE`
(default `huggingface`), or pass an initialized engine to
`candidate_retrieval_folder(folder, config, engine=engine)`. See
[shared generation](../common/README.md) for the backend contract. The generator
retains its term parsing, suggestion limits and self-consistency vote.

## Result

The LoRA-finetuned Qwen2.5-3B generator (`config_test_finetuning/config_lora.txt`) on top of no-LLM preprocessing gives the best gold-entity coverage; its output folder `candidate_lora_fp16` is the input used by the ranking baseline and by the full run.
