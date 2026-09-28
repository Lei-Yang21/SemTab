import string

DEFAULT_PROMPTS = {
    "CEA_PROMPT": (
        "Cell value: ${mention}\n${row_block}\n"
        "Candidate entities:\n${candidates}\n\n"
        "Return only the QID of the entity that best matches the cell "
        "value in this context."
    ),
    "CONTEXT_PROMPT": (
        "You are disambiguating one cell of a table against Wikidata.\n\n"
        "${table_block}Target cell (${location}) value: ${mention}\n"
        "${coltype_block}\n"
        "Candidate entities:\n${candidates}\n\n${instruction}"
    ),
    "CONTEXT_INSTRUCTION_ENRICH": (
        "The other columns of the marked row (>>...<<) describe the "
        "SAME entity (e.g. a place, date, category or related value). "
        "Compare those cells against each candidate's description, "
        "aliases and type, and pick the candidate they fit. Return "
        "ONLY the QID of the best match."
    ),
    "CONTEXT_INSTRUCTION_PLAIN": (
        "Using the whole table as context (the other columns and rows "
        "describe the same kind of thing), return only the QID of the "
        "entity that best matches the target cell."
    ),
    "CONTEXT_INSTRUCTION_COT": (
        "Reason step by step: weigh the target cell value, its column header, "
        "likely column type and the row context against each candidate's "
        "label, description and type. Briefly say why the close alternatives "
        "are eliminated, then finish with one single final line, exactly in "
        "the form:\nAnswer: <QID>"
    ),
    "NIL_CANDIDATE_LINE": (
        "- NIL: none of the above, the cell does not refer to any of these entities"
    ),
    "NIL_INSTRUCTION": (
        "Some cells refer to no Wikidata entity at all, and the retrieved "
        "candidates are then all wrong. Answer NIL when that is the case: an "
        "answer of NIL is correct for such a cell, and picking a plausible but "
        "wrong entity is not. Only give a QID when the candidate really is the "
        "entity the cell names."
    ),
    "CTA_PROMPT": (
        "${values_block}\n"
        "Candidate types:\n${candidates}\n\n"
        "Return only the QID of the type that best describes all the "
        "values in this column."
    ),
    "CPA_PROMPT": (
        "${subtype_block}${objtype_block}${values_block}\n"
        "Candidate properties (subject -> object):\n${candidates}\n\n"
        "Return only the PID of the property that best links the two columns."
    ),
    "DEBATE_PROMPT": (
        "Cell value: ${mention}\n${row_block}\n"
        "Candidate entities from Wikidata:\n${candidates}\n\n"
        "Select the best matching entity and give 3 short arguments.\n"
        "Output format:\nQID: <qid>\nArguments: <arguments>"
    ),
    "VERIFY_PROMPT": (
        "Cell value: ${mention}\n${row_block}"
        "Currently selected entity: ${chosen}\n\n"
        "All candidates:\n${candidates}\n\n"
        "Check the selection fits the cell value, column and row context. "
        "Revise if a better candidate exists, or answer NIL if none fits.\n"
        "Output format:\nWinning QID: <qid or NIL>"
    ),
}
DEFAULT_SIZES = {
    "MAX_NEW_TOKENS_CEA": 128,
    "MAX_NEW_TOKENS_CONTEXT": 512,
    "MAX_NEW_TOKENS_CONTEXT_COT": 512,
    "MAX_NEW_TOKENS_CTA": 128,
    "MAX_NEW_TOKENS_CPA": 128,
    "MAX_NEW_TOKENS_DEBATE": 256,
    "MAX_NEW_TOKENS_VERIFY": 256,
}

def render(template, values):
    return string.Template(template).safe_substitute(values)
def opt_line(label, value):
    return f"{label}{value}\n" if value else ""
def entity_lines(candidates):
    return "\n".join(f"- {c['qid']}: {c['label']} ({c.get('description', '')})" for c in candidates)
def property_lines(candidates):
    return "\n".join(f"- {c['pid']}: {c.get('label', '')}" for c in candidates)
def candidate_line(c, type_labels=None, enrich=False):
    line = f"{c['qid']}: {c['label']} ({c.get('description', '')})"
    if not enrich:
        return line
    aliases = [a for a in str(c.get("aliases", "") or "").split("|") if a][:5]
    if aliases:
        line += f" [also known as: {', '.join(aliases)}]"
    if type_labels:
        types, seen = [], set()
        for q in (c.get("P31") or []) + (c.get("P279") or []):
            lab = type_labels.get(q)
            if lab and lab not in seen:
                seen.add(lab)
                types.append(lab)
        if types:
            line += f" [type: {', '.join(types[:4])}]"
    return line
class PromptBuilder:
    def __init__(self, config):
        self.prompts = {k: config.get(k, v) for k, v in DEFAULT_PROMPTS.items()}
        self.allow_nil = str(config.get("ALLOW_NIL", "false")).strip().lower() == "true"
        self.context_cot = str(config.get("CONTEXT_COT", "false")).strip().lower() == "true"

    def nil_candidates(self, block):
        if not self.allow_nil:
            return block
        return block + "\n" + self.prompts["NIL_CANDIDATE_LINE"]

    def with_nil_instruction(self, prompt):
        if not self.allow_nil:
            return prompt
        return prompt + "\n\n" + self.prompts["NIL_INSTRUCTION"]

    def entity(self, mention, candidates, row_context="", col_header=""):
        values = {"mention": mention,"col_header": col_header,"row_context": row_context,"header_block": opt_line("Column header: ", col_header),"row_block": opt_line("Row context: ", row_context),"candidates": self.nil_candidates(entity_lines(candidates))}
        prompt = self.with_nil_instruction(render(self.prompts["CEA_PROMPT"], values))
        return prompt

    def context(self, mention, candidates, table_text="",col_header="", col_type="", target_row=None,target_col=None, type_labels=None, enrich=False):
        loc = []
        if target_row is not None:
            loc.append(f"row {target_row}")
        if target_col is not None:
            loc.append(f"column {target_col}")
        if self.context_cot:
            instruction = self.prompts["CONTEXT_INSTRUCTION_COT"]
        elif enrich:
            instruction = self.prompts["CONTEXT_INSTRUCTION_ENRICH"]
        else:
            instruction = self.prompts["CONTEXT_INSTRUCTION_PLAIN"]
        values = {"mention": mention,"table_text": table_text,"table_block": f"Table:\n{table_text}\n\n" if table_text else "", "location": ", ".join(loc),"col_header": col_header,"header_block": opt_line("Column header: ", col_header),"col_type": col_type,"coltype_block": opt_line("Likely column type: ", col_type),"candidates": self.nil_candidates("\n".join( "- " + candidate_line(c, type_labels, enrich) for c in candidates)),"instruction": instruction}
        prompt = self.with_nil_instruction(render(self.prompts["CONTEXT_PROMPT"], values))
        return prompt

    def column_type(self, candidates, col_values=None, col_header=""):
        values_str = ", ".join(str(v) for v in col_values[:10]) if col_values else ""
        values = {"col_header": col_header,"header_block": opt_line("Column header: ", col_header),"col_values": values_str,"values_block": opt_line("Column values: ", values_str) if col_values else "","candidates": entity_lines(candidates)}
        prompt = render(self.prompts["CTA_PROMPT"], values)
        return prompt

    def property(self, candidates, col_header="", sample_values=None, sub_type="", obj_type=""):
        values_str = ", ".join(str(v) for v in sample_values[:10]) if sample_values else ""
        values = {"col_header": col_header, "header_block": opt_line("Object column header: ", col_header), "subtype_block": opt_line("Subject column type: ", sub_type), "objtype_block": opt_line("Object column type: ", obj_type), "sample_values": values_str,"values_block": opt_line("Object column values: ", values_str) if sample_values else "","candidates": property_lines(candidates)}
        prompt = render(self.prompts["CPA_PROMPT"], values)
        return prompt

    def debate(self, mention, candidates, row_context="", col_header=""):
        values = {"mention": mention, "col_header": col_header, "row_context": row_context, "header_block": opt_line("Column header: ", col_header), "row_block": opt_line("Row context: ", row_context), "candidates": entity_lines(candidates)}
        return render(self.prompts["DEBATE_PROMPT"], values)

    def verify(self, mention, chosen, candidates, row_context="", col_header=""):
        values = {"mention": mention, "chosen": chosen, "col_header": col_header, "row_context": row_context,"header_block": opt_line("Column header: ", col_header), "row_block": opt_line("Row context: ", row_context), "candidates": entity_lines(candidates)}
        return render(self.prompts["VERIFY_PROMPT"], values)

