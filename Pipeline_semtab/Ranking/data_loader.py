import os
import re
import ast
import pandas as pd

ENTITY_URI = "http://www.wikidata.org/entity/"
def parse_metadata(preprocess_df):
    s = str(preprocess_df.columns[-1]).strip()
    m = re.match(r"Metadata:CTA:(\[.*?\]),CPA:(\[.*\])$", s)
    if not m:
        raise ValueError(f"Invalid metadata format: {s}")
    cta = ast.literal_eval(m.group(1))
    cpa = ast.literal_eval(m.group(2))
    return cta, cpa
def load_preprocess(path):
    df = pd.read_csv(path)
    cta_cols, cpa_pairs = parse_metadata(df)
    data = df.iloc[:, :-1].reset_index(drop=True)
    return data, cta_cols, cpa_pairs
def split_pipe(val):
    if val is None or val == "":
        return []
    if isinstance(val, float) and pd.isna(val):
        return []
    return [x for x in str(val).split("|") if x]
def load_candidates(path):
    df = pd.read_csv(path)
    df["row"] = df["row"].astype(int)
    df["columns"] = df["columns"].astype(int)
    return df
def cand_dict(row):
    cand = {
        "qid": row["QID"] if pd.notna(row["QID"]) else None,
        "label": str(row["candidates"]) if pd.notna(row["candidates"]) else "",
        "mention": str(row["data"]) if pd.notna(row["data"]) else "",
        "quality": int(row["quality"]) if pd.notna(row["quality"]) else 99,
        "description": str(row["description"]) if pd.notna(row["description"]) else "",
        "aliases": str(row["aliases"]) if pd.notna(row["aliases"]) else "",
        "P31": split_pipe(row.get("P31")),
        "P279": split_pipe(row.get("P279")),
    }
    if "score" in row:
        cand["score"] = float(row["score"])
    return cand

def candidates_frame(cands):
    rows = [{"QID": c["qid"], "candidates": c["label"], "data": c["mention"],
             "quality": c["quality"], "description": c.get("description", ""),
             "aliases": c["aliases"], "P31": "|".join(c["P31"]),
             "P279": "|".join(c["P279"]), "row": 0, "columns": 0} for c in cands]
    return pd.DataFrame(rows)
def group_by_cell(cand_df):
    cells = {}
    for (r, c), g in cand_df.groupby(["row", "columns"]):
        cells[(int(r), int(c))] = [cand_dict(row) for _, row in g.iterrows()]
    return cells
def candidate_columns(cand_df):
    return set(cand_df["columns"].dropna().astype(int).unique().tolist())
def tab_id_from_filename(filename):
    return os.path.basename(filename).replace("_candidates", "").replace(".csv", "")
