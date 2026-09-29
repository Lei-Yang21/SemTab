import re
from Pipeline_semtab.common.knowledge import WikidataClient

class BaseGenerator:
    name = "base"
    quality = 0

    def candidates(self, query, context_str):
        raise NotImplementedError

class SearchGenerator(BaseGenerator):
    def __init__(self, language="en", limit=10, search=None):
        self.language = language
        self.limit = limit
        self.search = search if search is not None else WikidataClient().search_entities

    def suggestions(self, query, context_str):
        raise NotImplementedError

    def candidates(self, query, context_str):
        rows = []
        for term in self.suggestions(query, context_str):
            rows.extend(self.search(term, self.language, self.limit))
        return rows

class DirectGenerator(SearchGenerator):
    name = "direct"
    quality = 1

    def suggestions(self, query, context_str):
        return [query]

class LLMGenerator(SearchGenerator):
    name = "llm"
    quality = 2
    def __init__(self, engine, system_prompt, language="en", limit=10,max_suggestions=8, max_new_tokens=64,sc_enabled=False, sc_samples=5, sc_temperature=0.7, sc_top_p=0.95, search=None):
        super().__init__(language, limit, search)
        self.engine = engine
        self.system_prompt = system_prompt
        self.max_suggestions = max_suggestions
        self.max_new_tokens = max_new_tokens
        self.sc_enabled = sc_enabled
        self.sc_samples = sc_samples
        self.sc_temperature = sc_temperature
        self.sc_top_p = sc_top_p

    def parse_terms(self, raw):
        for marker in ("FINAL:", "Candidates:", "Answer:"):
            if marker in raw:
                raw = raw.split(marker)[-1]
                break
        raw = raw.replace("\n", " ")
        terms = []
        for term in raw.split(";"):
            term = re.sub(r"^\s*\d+[\.\)]\s*", "", term).strip()
            if term and term not in terms:
                terms.append(term)
        return terms

    def suggestions(self, query, context_str):
        user_msg = f"Cell value: {query}"
        if context_str:
            user_msg += f"\nContext: {context_str}"
        user_msg += "\nCandidates:"
        if not self.sc_enabled or self.sc_samples <= 1:
            raw = self.engine.generate([user_msg], self.system_prompt,max_new_tokens=self.max_new_tokens)[0]
            return self.parse_terms(raw)[: self.max_suggestions]
        samples = self.engine.generate([user_msg], self.system_prompt,max_new_tokens=self.max_new_tokens,do_sample=True, temperature=self.sc_temperature,top_p=self.sc_top_p, num_return_sequences=self.sc_samples)
        votes = {}
        for raw in samples:
            seen = set()
            for term in self.parse_terms(raw):
                key = term.casefold()
                if key in seen:
                    continue
                seen.add(key)
                if key in votes:
                    votes[key][0] += 1
                else:
                    votes[key] = [1, len(votes), term]
        ranked = sorted(votes.values(), key=lambda v: (-v[0], v[1]))
        return [v[2] for v in ranked][: self.max_suggestions]

class FuzzyGenerator(SearchGenerator):
    name = "fuzzy"
    quality = 3

    def suggestions(self, query, context_str):
        return generate_fuzzy_variants(query)

def generate_fuzzy_variants(query):
    variants = set()
    cleaned = re.sub(r'["\*]', "", query).strip()
    if cleaned and cleaned != query:
        variants.add(cleaned)

    no_paren = re.sub(r"\s*\(.*?\)\s*", " ", cleaned).strip()
    if no_paren and no_paren != cleaned:
        variants.add(no_paren)

    for p in re.findall(r"\(([^)]+)\)", cleaned):
        if p.strip():
            variants.add(p.strip())

    no_article = re.sub(r"^(the|a|an)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    if no_article and no_article != cleaned:
        variants.add(no_article)

    if "," in cleaned:
        parts = [p.strip() for p in cleaned.split(",", 1)]
        if len(parts) == 2 and all(parts):
            variants.add(f"{parts[1]} {parts[0]}")
            variants.add(parts[0])

    for sep in (" - ", " – ", " / "):
        if sep in cleaned:
            for part in cleaned.split(sep):
                part = part.strip()
                if part:
                    variants.add(part)

    words = cleaned.split()
    first_word = words[0] if words else ""
    if first_word and first_word.lower() not in {"the", "a", "an"} and first_word != cleaned:
        variants.add(first_word)

    if len(words) >= 3:
        variants.add(" ".join(words[:2]))

    variants.discard(query)
    return [v for v in variants if v]

def as_bool(config, key, default):
    value = config.get(key)
    if value is None:
        return default
    return str(value).strip().lower() == "true"

def search_options(config, name, search):
    return {"language": config.get("LANGUAGE", "en"),
            "limit": int(config.get(f"{name.upper()}_SEARCH_LIMIT", config.get("SEARCH_LIMIT", "10"))),
            "search": search}

def build_direct(config, engine, search):
    return DirectGenerator(**search_options(config, "direct", search))

def build_llm(config, engine, search):
    if engine is None:
        return None
    return LLMGenerator(engine, config.get("PROMPT", ""),
                        **search_options(config, "llm", search),
                        max_suggestions=int(config.get("LLM_MAX_SUGGESTIONS", "5")),
                        max_new_tokens=int(config.get("LLM_MAX_NEW_TOKENS", "1024")),
                        sc_enabled=as_bool(config, "LLM_SELF_CONSISTENCY", False),
                        sc_samples=int(config.get("SC_SAMPLES", "5")),
                        sc_temperature=float(config.get("SC_TEMPERATURE", "0.7")),
                        sc_top_p=float(config.get("SC_TOP_P", "0.95")))

def build_fuzzy(config, engine, search):
    return FuzzyGenerator(**search_options(config, "fuzzy", search))

GENERATOR_FACTORIES = {"direct": (build_direct, True), "llm": (build_llm, True), "fuzzy": (build_fuzzy, False)}

def build_generators(config, engine=None, search=None):
    if search is None:
        search = WikidataClient(sleep=config.get("API_SLEEP", "0.1")).search_entities
    order = [n.strip() for n in config.get("GENERATOR_ORDER", "direct,llm,fuzzy").split(",")]
    generators = []
    for name in order:
        if name not in GENERATOR_FACTORIES:
            print(f"Warning: unknown generator '{name}' ignored")
            continue
        factory, enabled = GENERATOR_FACTORIES[name]
        if as_bool(config, f"USE_{name.upper()}", enabled):
            gen = factory(config, engine, search)
            if gen is not None:
                gen.quality = int(config.get(f"{name.upper()}_QUALITY", gen.quality))
                if gen.quality <= 0:
                    raise ValueError(f"{name.upper()}_QUALITY must be a positive integer")
                generators.append(gen)
    return generators
