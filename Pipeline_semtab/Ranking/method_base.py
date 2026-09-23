import re
from data_loader import (load_candidates, load_preprocess, group_by_cell,candidate_columns, tab_id_from_filename)
from cta import build_type_pct, cta_from_cea, cta_from_selection
from scoring_method import build_scorer
from wikidata_api_ranking import get_entities
import cpa as cpa_mod


class TableContext:

    def __init__(self, input_path, preprocess_path, config, llm, writer, scorer=None):
        self.tab_id = tab_id_from_filename(input_path)
        self.cand_df = load_candidates(input_path)
        self.data_df, self.cta_cols, self.cpa_pairs = load_preprocess(preprocess_path)
        self.n_rows = self.data_df.shape[0]

        self.cells = group_by_cell(self.cand_df)
        self.cand_cols = candidate_columns(self.cand_df)    
        self.cta_result = cta_from_cea(self.cand_df)        
        self.cta_by_col = {c: (c, p31, p279) for c, p31, p279 in self.cta_result}
        self.type_pct = build_type_pct(self.cta_result)     
        self.config = config
        self.llm = llm
        self.language = config.get("LANGUAGE", "en")
        self.writer = writer
        self.cea_choice = {}
        self.cta_choice = {}
        self.tasks = {t.strip().lower()for t in config.get("TASKS", "cea,cta,cpa").split(",") if t.strip()}
        self._type_label = {}
        self._final_label = {}
        self.scorer = scorer if scorer is not None else build_scorer(config, llm)
        self.scored_df = self.scorer.score(self.cand_df, self) if self.tasks & {"cea", "cpa"} else self.cand_df
        self.cells = group_by_cell(self.scored_df)

    def rebuild_cta_from_selection(self):
        res = cta_from_selection(self.cand_df, self.cea_choice)
        if res is not None:
            self.cta_result = res
            self.cta_by_col = {c: (c, p31, p279) for c, p31, p279 in res}
            self._type_label = {}

    def col_header(self, col):
        try:
            return str(self.data_df.columns[col])
        except Exception:
            return ""

    def row_context(self, row):
        try:
            return " | ".join(str(x) for x in self.data_df.iloc[row].tolist())
        except Exception:
            return ""

    def col_values(self, col):
        try:
            return self.data_df.iloc[:, col].tolist()
        except Exception:
            return []

    def row_terms(self, row, exclude_col=None, min_len=3):
        terms = set()
        try:
            vals = self.data_df.iloc[row].tolist()
        except Exception:
            return terms
        for ci, v in enumerate(vals):
            if ci == exclude_col:
                continue
            for tok in re.findall(r"[a-z0-9]+", str(v).lower()):
                if len(tok) >= min_len:
                    terms.add(tok)
        return terms

    def column_type_label(self, col):
        if col in self._type_label:
            return self._type_label[col]
        label = ""
        res = self.cta_by_col.get(col)
        if res:
            _, p31, p279 = res
            ranked = list((p31 or p279).items())
            if ranked:
                top = ranked[0][0]
                info = get_entities([top], self.language)
                label = info.get(top, {}).get("label", top)
        self._type_label[col] = label
        return label

    def final_type_label(self, col):
        # Label of the CTA decision for this column (LLM tie-break included);
        # falls back to the dominant candidate type when CTA was not run.
        qid = self.cta_choice.get(col)
        if not qid:
            return self.column_type_label(col)
        if qid not in self._final_label:
            info = get_entities([qid], self.language)
            self._final_label[qid] = info.get(qid, {}).get("label", qid)
        return self._final_label[qid]

    def table_text(self, target_row=None, target_col=None, max_rows=20):
        df = self.data_df
        ncols = df.shape[1]
        header = " | ".join(str(df.columns[c]) for c in range(ncols))
        rows = list(range(self.n_rows))
        if self.n_rows > max_rows and target_row is not None:
            half = max_rows // 2
            lo = max(0, target_row - half)
            rows = list(range(lo, min(self.n_rows, lo + max_rows)))
        lines = ["col_ids: " + " | ".join(str(c) for c in range(ncols)), header]
        for r in rows:
            cells = []
            for c in range(ncols):
                val = str(df.iat[r, c])
                if r == target_row and c == target_col:
                    val = f">>{val}<<"
                cells.append(val)
            lines.append(" | ".join(cells))
        return "\n".join(lines)


def run_cpa(ctx, use_slm):
    for sub_col, obj_col in ctx.cpa_pairs:
        obj_is_entity = obj_col in ctx.cand_cols
        pid = cpa_mod.resolve_pair(
            sub_col, obj_col, ctx.n_rows, ctx.cea_choice, ctx.data_df,
            obj_is_entity, llm=ctx.llm, use_slm=use_slm,
            col_header=ctx.col_header(obj_col),
            sub_type=ctx.final_type_label(sub_col),
            obj_type=ctx.final_type_label(obj_col) if obj_is_entity else "")
        ctx.writer.add_cpa(ctx.tab_id, sub_col, obj_col, pid)
