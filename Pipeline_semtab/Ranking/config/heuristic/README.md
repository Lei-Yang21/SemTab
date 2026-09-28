# Configurable heuristic

These configurations use the same inputs and no-LLM settings as
`../methods/config_noslm.txt`. Only the quality transform and output folder
differ between the two runs. Run them from `Pipeline_semtab/Ranking`:

```text
python main_ranking.py config/heuristic/config_inverse.txt
python main_ranking.py config/heuristic/config_identity.txt
```

The heuristic combines three components:

```text
score = 0.5 * string_similarity + 0.2 * quality_transform(quality) + 0.3 * type_coherence
```

| Configuration | Quality transform | Results |
| --- | --- | --- |
| `config_inverse.txt` | `1 / quality` (existing default) | `results/heuristic/inverse` |
| `config_identity.txt` | `quality` | `results/heuristic/identity` |

Quality is a retrieval rank: direct search uses 1, LLM suggestions 2 and fuzzy
search 3 by default. With positive weights, `inverse` favors smaller ranks;
`identity` favors larger ranks and changes the component's scale. It is an
experimental alternative, not an equivalent replacement. Nonpositive ranks
contribute zero with either transform. Candidates without a QID score zero.

Change `CEA_WEIGHTS` to adjust the contributions, or `CEA_FEATURES` to choose
and order them. For example, `CEA_FEATURES:string,quality` with
`CEA_WEIGHTS:0.8,0.2` removes type coherence. Weights are not normalized.
`INPUT_FOLDER`, `PREPROCESS_FOLDER` and `OUTPUT_FOLDER` can be changed for
another dataset; adjust `ROW_OFFSET` to its annotation format.

To introduce another quality formula, add a function in `scoring.py` and an
entry in `QUALITY_METHODS`. For example:

```python
def inverse_square_quality(quality):
    return 1.0 / quality ** 2

QUALITY_METHODS["inverse_square"] = inverse_square_quality
```

Then set `CEA_QUALITY_METHOD:inverse_square` in the experiment configuration.
The function receives a positive integer rank and must return a finite number.
For programmatic runs, the same registration can happen before `rank_folder`.
Unknown method names raise an error. No candidate-retrieval changes are needed.
