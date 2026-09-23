from collections import defaultdict
import re
import pandas as pd

from wikidata_api_ranking import get_entities

def build_type_pct(cta_result):
    out = {}
    for col_id, p31, p279 in cta_result:
        merged = dict(p31)
        for q, pct in p279.items():
            merged[q] = max(merged.get(q, 0.0), 0.5 * pct)
        out[col_id] = merged
    return out
# Function to build the percentages of cover for each type in the CTA results 
def cta_from_cea(cea_df):
    results = []

    for col_id, col_df in cea_df.groupby("columns"):
        cells = col_df.groupby(["row", "columns"])
        n_cells = cells.ngroups
        if n_cells == 0:
            continue

        p31_support = defaultdict(set)
        p279_support = defaultdict(set)

        for cell_key, c_df in cells:
            for prop, support in (("P31", p31_support), ("P279", p279_support)):
                types_cell = set()
                for cell in c_df[prop].dropna():
                    types_cell.update(str(cell).split("|"))
                for qid in types_cell:
                    support[qid].add(cell_key)

        p31_pct = {q: len(s) / n_cells for q, s in p31_support.items()}
        p279_pct = {q: len(s) / n_cells for q, s in p279_support.items()}

        p31_pct = dict(sorted(p31_pct.items(), key=lambda x: x[1], reverse=True))
        p279_pct = dict(sorted(p279_pct.items(), key=lambda x: x[1], reverse=True))

        results.append((col_id, p31_pct, p279_pct))

    return results
def cta_from_selection(cand_df, cea_choice):
    if not cea_choice:
        return None
    keys = {(int(r), int(c), qid) for (r, c), qid in cea_choice.items()}
    tuples = zip(cand_df["row"].astype(int), cand_df["columns"].astype(int), cand_df["QID"])
    mask = [t in keys for t in tuples]
    sel = cand_df[pd.Series(mask, index=cand_df.index)]
    if len(sel) == 0:
        return None
    return cta_from_cea(sel)
def choose_cta(col_result, llm=None, use_slm=True, language="en", col_values=None,col_header="", margin=0.3, topk=5):

    _, p31, p279 = col_result
    ranked = list((p31 or p279).items())
    if not ranked:
        return None
    top = ranked[0][0]
    if not (use_slm and llm) or len(ranked) == 1:
        return top
    if ranked[0][1] - ranked[1][1] < margin:
        top_ids = [q for q, _ in ranked[:topk]]
        info = get_entities(top_ids, language)
        cands = [{"qid": q,"label": info.get(q, {}).get("label", q),"description": info.get(q, {}).get("description", "")}for q in top_ids]
        choice = llm.select_best_type(cands, col_values, col_header)
        if choice:
            return choice
    return top
def infer_literal_type(values):
    #Safety-net CTA for a column without candidates (numeric -> Q11563).
    total = nums = 0
    for v in values:
        if v is None or str(v).strip() == "":
            continue
        total += 1
        if re.fullmatch(r"[-+]?\d[\d,]*\.?\d*", str(v).strip()):
            nums += 1
    if total and nums / total > 0.8:
        return "Q11563" 
    return None