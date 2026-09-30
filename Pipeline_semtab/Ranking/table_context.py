import re
from data_loader import load_candidates, load_preprocess, group_by_cell, candidate_columns, tab_id_from_filename
from cta import cta_from_cea, build_type_pct

class TableContext:
    def __init__(self, tab_id, data_df, cand_df, cta_cols=None, cpa_pairs=None):
        self.tab_id = tab_id
        self.data_df = data_df
        self.cand_df = cand_df
        self.cta_cols = list(cta_cols) if cta_cols is not None else []
        self.cpa_pairs = list(cpa_pairs) if cpa_pairs is not None else []
        self.n_rows = data_df.shape[0]
        self.cells = {}
        self.cand_cols = set()
        self.cta_result = []
        self.cta_by_col = {}
        self.type_pct = {}
        self.scored_df = None

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

def load_table_context(input_path, preprocess_path):
    cand_df = load_candidates(input_path)
    data_df, cta_cols, cpa_pairs = load_preprocess(preprocess_path)
    return TableContext(tab_id_from_filename(input_path), data_df, cand_df, cta_cols, cpa_pairs)

def prepare_table(table, scorer=None):
    table.cand_cols = candidate_columns(table.cand_df)
    table.cta_result = cta_from_cea(table.cand_df)
    table.cta_by_col = {c: (c, p31, p279) for c, p31, p279 in table.cta_result}
    table.type_pct = build_type_pct(table.cta_result)
    table.cells = group_by_cell(table.cand_df)
    table.scored_df = scorer.score(table.cand_df, table) if scorer is not None else table.cand_df
    table.cells = group_by_cell(table.scored_df)
    return table
