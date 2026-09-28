from collections import Counter
from Pipeline_semtab.common.generation import build_generation_engine
from logger_ranking import add_tokens
from prompts import PromptBuilder, DEFAULT_SIZES
from response_parser import ResponseParser

class LLMSelector:
    def __init__(self, config, device_id=None, max_ctx=None, backend=None, prompts=None, parser=None):
        config = dict(config)
        if device_id is not None:
            config["DEVICE_ID"] = device_id
        if max_ctx is not None:
            config["MAX_CTX"] = max_ctx
        self.config = config
        self.system_prompt = config.get("SYSTEM_PROMPT", "")
        self.backend = backend if backend is not None else build_generation_engine(config, add_tokens)
        self.prompt_builder = prompts if prompts is not None else PromptBuilder(config)
        self.parser = parser if parser is not None else ResponseParser(config)
        self.sizes = {k: int(config.get(k, v)) for k, v in DEFAULT_SIZES.items()}
        self.context_cot = str(config.get("CONTEXT_COT", "false")).strip().lower() == "true"
        self.sc_enabled = str(config.get("CONTEXT_SELF_CONSISTENCY", "false")).strip().lower() == "true"
        self.sc_samples = int(config.get("SC_SAMPLES", "5"))
        self.sc_temperature = float(config.get("SC_TEMPERATURE", "0.7"))
        self.sc_top_p = float(config.get("SC_TOP_P", "0.95"))

    def generate(self, user_msgs, system_prompt="", **options):
        return self.backend.generate(user_msgs, system_prompt=system_prompt, **options)

    def vote(self, prompt, candidate_ids, max_new_tokens):
        responses = self.generate([prompt], system_prompt=self.system_prompt,max_new_tokens=max_new_tokens, do_sample=True,temperature=self.sc_temperature, top_p=self.sc_top_p,num_return_sequences=self.sc_samples)
        votes = [v for v in (self.parser.final_id(r, candidate_ids) for r in responses) if v]
        if not votes:
            return None
        return Counter(votes).most_common(1)[0][0]

    def select_best_entity(self, mention, candidates, row_context="", col_header=""):
        prompt = self.prompt_builder.entity(mention, candidates, row_context, col_header)
        response = self.generate([prompt], system_prompt=self.system_prompt,max_new_tokens=self.sizes["MAX_NEW_TOKENS_CEA"])[0]
        return self.parser.match_id(response, [c["qid"] for c in candidates])

    def select_with_context(self, mention, candidates, table_text="",col_header="", col_type="", target_row=None,target_col=None, max_new_tokens=None,type_labels=None, enrich=False):
        prompt = self.prompt_builder.context(mention, candidates, table_text, col_header, col_type, target_row, target_col, type_labels=type_labels, enrich=enrich)
        if max_new_tokens is None:
            key = "MAX_NEW_TOKENS_CONTEXT_COT" if self.context_cot else "MAX_NEW_TOKENS_CONTEXT"
            max_new_tokens = self.sizes[key]
        cand_ids = [c["qid"] for c in candidates]
        if self.sc_enabled:
            return self.vote(prompt, cand_ids, max_new_tokens)
        response = self.generate([prompt], system_prompt=self.system_prompt,max_new_tokens=max_new_tokens)[0]
        parser = self.parser.final_id if self.context_cot else self.parser.match_id
        return parser(response, cand_ids)

    def select_best_type(self, candidates, col_values=None, col_header=""):
        prompt = self.prompt_builder.column_type(candidates, col_values, col_header)
        with self.backend.base_model():
            response = self.generate([prompt], system_prompt=self.system_prompt,max_new_tokens=self.sizes["MAX_NEW_TOKENS_CTA"])[0]
        return self.parser.match_id(response, [c["qid"] for c in candidates])

    def select_best_property(self, candidates, col_header="", sample_values=None, sub_type="", obj_type=""):
        prompt = self.prompt_builder.property(candidates, col_header, sample_values, sub_type, obj_type)
        with self.backend.base_model():
            response = self.generate([prompt], system_prompt=self.system_prompt,max_new_tokens=self.sizes["MAX_NEW_TOKENS_CPA"])[0]
        return self.parser.match_id(response, [c["pid"] for c in candidates])

    def debate_select(self, mention, candidates, row_context="", col_header=""):
        return self.debate_select_batch([{"mention": mention, "candidates": candidates, "row_context": row_context,"col_header": col_header}])[0]

    def debate_select_batch(self, items):
        if not items:
            return []
        prompts = [self.prompt_builder.debate(i["mention"], i["candidates"],i.get("row_context", ""), i.get("col_header", ""))for i in items]
        responses = self.generate(prompts, system_prompt=self.system_prompt, max_new_tokens=self.sizes["MAX_NEW_TOKENS_DEBATE"])
        return [(self.parser.match_id(resp, [c["qid"] for c in i["candidates"]]), resp)for i, resp in zip(items, responses)]

    def verify(self, mention, chosen, candidates, row_context="", col_header=""):
        return self.verify_batch([{"mention": mention, "chosen": chosen,"candidates": candidates, "row_context": row_context, "col_header": col_header}])[0]

    def verify_batch(self, items):
        if not items:
            return []
        prompts = [self.prompt_builder.verify(i["mention"], i["chosen"], i["candidates"],i.get("row_context", ""), i.get("col_header", ""))for i in items]
        responses = self.generate(prompts, system_prompt=self.system_prompt,max_new_tokens=self.sizes["MAX_NEW_TOKENS_VERIFY"])
        return [self.parser.verified_id(resp, [c["qid"] for c in it["candidates"]], it["chosen"])
                for it, resp in zip(items, responses)]

