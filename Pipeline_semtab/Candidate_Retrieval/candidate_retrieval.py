import os
import ast
import time
import pandas as pd

from enrichment import enrich_rows, ENRICHMENT_COLUMNS
from generators import build_generators
from logger_candidate_retrieval import log_vram, reset_peaks, log_tokens, reset_tokens, set_run_name, add_tokens
import wikidata_api

QUALITY_MERGES = {"first": lambda old, new: old, "min": min, "max": max}

def read_which_to_process(input_file):
    df = pd.read_csv(input_file)

    metadata_col = None
    for col in df.columns:
        if str(col).startswith("Metadata:"):
            metadata_col = col
            break
    if metadata_col is None:
        raise ValueError(f"No metadata column found in {input_file}")

    process = []
    for idx, value in enumerate(df[metadata_col]):
        if pd.notna(value):
            try:
                col_indices = ast.literal_eval(str(value))
                if isinstance(col_indices, list):
                    for col_idx in col_indices:
                        process.append((idx, col_idx))
            except (ValueError, SyntaxError):
                pass
    df_clean = df.drop(columns=[metadata_col])
    return df_clean, process
def build_context_string(df, idx, col_idx):
    row_values = [str(v) for c, v in enumerate(df.iloc[idx])if c != col_idx and pd.notna(v) and str(v).strip()]
    if row_values:
        return "row context: " + ", ".join(row_values[:5])
    return ""

def clean_query(raw_query):
    query = str(raw_query).strip()
    query = query.replace('"', '').replace("**", "").strip()
    if query.endswith(".") and not str(raw_query).strip().endswith(".."):
        query = query[:-1].strip()
    return query
def retrieve_for_cell(query, context_str, idx, col_idx, generators, max_candidates=0, duplicate_quality="first"):
    if duplicate_quality not in QUALITY_MERGES:
        raise ValueError(f"Unknown duplicate quality policy: {duplicate_quality}")
    merge_quality = QUALITY_MERGES[duplicate_quality]
    by_qid = {}
    rows = []
    for gen in generators:
        if duplicate_quality == "first" and max_candidates and len(rows) >= max_candidates:
            break
        for label, qid in gen.candidates(query, context_str):
            if not qid:
                continue
            if qid in by_qid:
                row = by_qid[qid]
                row["quality"] = merge_quality(row["quality"], gen.quality)
                continue
            if max_candidates and len(rows) >= max_candidates:
                continue
            row = {"data": query,"candidates": label,"QID": qid,"row": idx,"columns": col_idx,"quality": gen.quality}
            by_qid[qid] = row
            rows.append(row)
            if duplicate_quality == "first" and max_candidates and len(rows) >= max_candidates:
                break
    if not rows:
        rows.append({"data": query, "candidates": "", "QID": "","row": idx, "columns": col_idx, "quality": 0})
    return rows
def candidate_retrieval_file(input_file, generators, max_candidates=0, output_folder=None,enrich=True, language="en", emit_empty=False, duplicate_quality="first"):
    df, process = read_which_to_process(input_file)
    rows = []

    for idx, col_idx in process:
        raw_query = df.iloc[idx, col_idx]
        if pd.isna(raw_query) or str(raw_query).strip() == "":
            if emit_empty:
                rows.append({"data": "", "candidates": "", "QID": "","row": idx, "columns": col_idx, "quality": 0})
            continue
        query = clean_query(raw_query)
        context_str = build_context_string(df, idx, col_idx)
        rows.extend(retrieve_for_cell(query, context_str, idx, col_idx,generators, max_candidates, duplicate_quality))

    columns = ["data", "candidates", "QID", "row", "columns", "quality"]
    if enrich:
        enrich_rows(rows, language)
        columns += ENRICHMENT_COLUMNS
    df_out = pd.DataFrame(rows, columns=columns)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder,os.path.basename(input_file).replace(".csv", "_candidates.csv"))
    else:
        output_file = input_file.replace(".csv", "_candidates.csv")
    df_out.to_csv(output_file, index=False)
def candidate_retrieval_folder(folder, config, engine=None):
    wikidata_api.set_rate_limit(config.get("API_SLEEP", "0.05"))
    max_candidates = int(config.get("MAX_CANDIDATES_PER_CELL", "0"))
    duplicate_quality = config.get("DUPLICATE_QUALITY", "first").strip().lower()
    if duplicate_quality not in QUALITY_MERGES:
        raise ValueError(f"Unknown duplicate quality policy: {duplicate_quality}")
    output_folder = config.get("OUTPUT_FOLDER") or None
    set_run_name(output_folder or folder)

    enrich = str(config.get("ENRICH_CANDIDATES", "true")).strip().lower() == "true"
    language = config.get("LANGUAGE", "en")
    emit_empty = str(config.get("EMIT_EMPTY_CELLS", "false")).strip().lower() == "true"

    use_llm = str(config.get("USE_LLM", "true")).strip().lower() == "true"
    model_name = config.get("MODEL_NAME", "").strip()
    backend = config.get("GENERATION_ENGINE", "huggingface").strip().lower()

    if engine is None and use_llm and (backend != "huggingface" or model_name and model_name != "?"):
        from Pipeline_semtab.common.generation import build_generation_engine
        reset_peaks()
        reset_tokens()
        engine = build_generation_engine({"MAX_CTX": "2048", **config}, token_callback=add_tokens)
        log_vram("model_loaded")
    generators = build_generators(config, engine)

    files = [f for f in os.listdir(folder)if f.endswith(".csv") and not f.endswith("_candidates.csv")]
    total = len(files)
    start_time = time.time()
    for i, f in enumerate(files, 1):
       candidate_retrieval_file(os.path.join(folder, f), generators,max_candidates, output_folder,enrich=enrich, language=language,emit_empty=emit_empty, duplicate_quality=duplicate_quality)
       if engine is not None:
           log_tokens(f)

    if engine is not None:
        log_vram("retrieval_done")
        log_tokens("retrieval_done")
    print(f"Finished folder {folder}: {total} files in {time.time() - start_time:.2f}s")
