# Shared generation

Preprocessing, candidate retrieval, ranking and the direct baseline use `generation.py`.
`GenerationEngine` defines the backend interface; `HuggingFaceEngine` handles
model/tokenizer loading, LoRA adapters, optional 4-bit loading, chat formatting,
batching and token accounting. Model dependencies are imported only when the
HuggingFace engine is constructed.

## Backend contract

`generate(user_msgs, system_prompt="", **options)` returns a list of decoded
responses in input order. Options are `max_new_tokens`, `batch_size`,
`do_sample`, `temperature`, `top_p` and `num_return_sequences`. With multiple
return sequences, each input's responses stay together. Greedy decoding is the
default; temperature and top-p are passed to the model only during sampling.

Inputs may also be `GenerationPrompt(messages, text, template_options={})`
objects: `messages` contains an explicit chat conversation (including few-shot
turns), and `text` is its fallback for tokenizers without a chat template.
These prompts already contain their system instructions. HuggingFace applies
only their supplied template options. Preprocessing uses this form to preserve
its chat formatting and the model's default thinking behavior. A custom backend
used for preprocessing must support these objects as well as generation options.

`base_model()` returns a context manager. Its default is a no-op. The
HuggingFace backend temporarily disables the adapter inside this context;
ranking uses it for CTA and CPA because its adapter was trained on CEA.

HuggingFace configuration keys include `MODEL_NAME`, `ADAPTER_PATH`,
`LOAD_IN_4BIT`, `DEVICE_ID`, `MAX_CTX` and `LLM_BATCH_SIZE`. Retrieval defaults to
2048 context tokens; ranking and the direct baseline default to 8192. The
baseline passes its existing `BATCH_SIZE` explicitly. Each task retains its
own output-token limits. CPU loading uses float32; 4-bit loading requires CUDA.
The configured device is also used for quantization and adapters.

`MODEL_DTYPE` selects `auto`, `float16`, `bfloat16` or `float32` on GPU;
`auto` keeps the existing bf16/fp16 choice. Preprocessing defaults to `float16`
on GPU and `MAX_CTX:model`, which reads `max_position_embeddings` from the
loaded model (fallback 1024). It uses batches of 32 and 64 generated tokens by
default. `MAX_CTX` may instead be an integer, and `generate(..., max_ctx=...)`
can override the context limit for one call.

The backend accumulates `tokens_in` and `tokens_out` and reports each batch to
`token_callback(input_count, output_count)` when supplied. Preprocessing, retrieval and ranking
connect their existing token loggers through this callback. For multiple
sampled responses, input-token counts are multiplied by the number of sequences,
preserving the existing accounting convention.

## Replacing the backend

Register a factory before launching a stage:

```python
from Pipeline_semtab.common.generation import GenerationEngine, GENERATION_ENGINES

class CustomEngine(GenerationEngine):
    def __init__(self, config, token_callback=None):
        self.config = config
        self.token_callback = token_callback

    def generate(self, user_msgs, system_prompt="", **options):
        raise NotImplementedError

GENERATION_ENGINES["custom"] = CustomEngine
```

Implement `generate`, then set `GENERATION_ENGINE:custom` in the configuration.
The default is `huggingface`. Custom factories receive `(config,
token_callback=None)` and should invoke the callback to populate token logs.
They may also expose `tokens_in` and `tokens_out` for the baseline summary.

An initialized engine can be passed to
`candidate_retrieval_folder(folder, config, engine=engine)` or
`baseline_folder(config, engine=engine)`. For ranking, create
`LLMSelector(config, backend=engine)` and pass it to
`rank_folder(config, llm=selector)`. Preprocessing accepts
`typo_method.process_folder(config, engine=engine)`. An injected engine owns its initialization,
token counters and callback wiring.

Preprocessing keeps its few-shot prompts, correction guards, per-table cache
and worker orchestration in `Preprocessing/typo_method.py`. Each HuggingFace GPU
worker builds its own shared engine with its device ID and token callback.
Registered alternative backends and injected engines run in one process.

## Ranking components

- `Ranking/prompts.py`: `PromptBuilder` builds CEA, contextual CEA, CTA, CPA,
  debate and verification prompts, without loading a model.
- `Ranking/response_parser.py`: `ResponseParser` interprets candidate IDs, final
  answers, NIL and verification replies, without generating text.
- `Ranking/llm_selection.py`: `LLMSelector` connects the builder, parser and
  backend, chooses generation limits, and orchestrates voting and selection.

`LLMSelector(config, backend=engine, prompts=builder, parser=parser)` accepts each
component independently. Builders expose `entity`, `context`, `column_type`,
`property`, `debate` and `verify`. Parsers expose `match_id`, `final_id` and
`verified_id`. Subclass the supplied implementations to change a single task.
Existing prompt overrides and NIL, CoT and self-consistency settings still apply.

The old per-stage engines have been removed. Import the shared backend for text
generation and `LLMSelector` for ranking decisions.
