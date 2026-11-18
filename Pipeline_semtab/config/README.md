# config (full run)

The three chained reference configs of the end-to-end run driven by `../main_pipeline.py`. They combine no-LLM preprocessing and adapted retrieval with an unadapted ranking model; they do not reproduce the highest fine-tuned ranking scores in the thesis.

| File | Stage | Setting |
|---|---|---|
| `config_preprocessing.txt` | Preprocessing | no LLM typo correction (`NEED_CORRECTTYPO:false`) — the variant that gave the best downstream retrieval |
| `config_candidate.txt` | Candidate Retrieval | direct search + LoRA-finetuned Qwen2.5-3B generator, 10 results per query |
| `config_ranking.txt` | Ranking | `slm_context`, Qwen2.5-3B-Instruct without a ranking adapter, `LLM_CONTEXT_MARGIN:0.10`, zero-shot |

The chaining is done through the folder keys: preprocessing `OUTPUT_FOLDER` = candidate `INPUT_FOLDER`, candidate `OUTPUT_FOLDER` = ranking `INPUT_FOLDER`, and the ranking `PREPROCESS_FOLDER` points back to the preprocessing output. Final annotations land in `results/full_run`.

- `smoke/` — same three configs pointing at the 5-table subset built by `../make_smoke_subset.py`; outputs to `results/smoke_run`. Requires the same base model and retrieval adapter as the full example.

Editing any folder key means editing it in the two configs that share it — the pipeline does not re-derive them.
