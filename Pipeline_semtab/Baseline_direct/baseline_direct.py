import os
import re
import ast
import time
import string
import pandas as pd

from Pipeline_semtab.common.generation import build_generation_engine
from output_writer_baseline import OutputWriter
from vram_logger_baseline import log_vram, reset_peaks

QID_RE = re.compile(r"\bQ\d+\b")
PID_RE = re.compile(r"\bP\d+\b")

DEFAULT_PROMPTS = {
    "CEA_PROMPT": (
        "${table_block}\n"
        "Target cell (row ${row}, column ${col}, marked >>...<<) value: ${mention}\n"
        "Column header: ${header}\n\n"
        "Give the Wikidata QID of the entity this cell refers to. "
        "Answer with only the QID (e.g. Q42), or NIL if you cannot identify it."
    ),
    "CTA_PROMPT": (
        "Column header: ${header}\n"
        "Column values: ${values}\n\n"
        "Give the QID of the most specific Wikidata class that is the type "
        "(value of P31) of all the entities in this column. "
        "Answer with only the QID (e.g. Q5), or NIL if you cannot identify it."
    ),
    "CPA_PROMPT": (
        "Subject column header: ${sub_header}\n"
        "Object column header: ${obj_header}\n"
        "Sample value pairs (subject -> object):\n${pairs}\n\n"
        "Give the PID of the Wikidata property that links the subject column "
        "to the object column. Answer with only the PID (e.g. P17), or NIL if "
        "you cannot identify it."
    ),
}
DEFAULT_SYSTEM_PROMPT = "You are an expert on the Wikidata knowledge graph. Answer with Wikidata identifiers only."

def render(template, values):
    return string.Template(template).safe_substitute(values)

def parse_metadata(df):
    s = str(df.columns[-1]).strip()
    m = re.match(r"Metadata:CTA:(\[.*?\]),CPA:(\[.*\])$", s)
    if not m:
        raise ValueError(f"Invalid metadata format: {s}")
    return ast.literal_eval(m.group(1)), ast.literal_eval(m.group(2))

def load_table(path):
    df = pd.read_csv(path)
    cta_cols, cpa_pairs = parse_metadata(df)
    meta_col = df.columns[-1]
    cea_targets = []
    for idx, value in enumerate(df[meta_col]):
        if pd.isna(value):
            continue
        try:
            col_indices = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            continue
        if isinstance(col_indices, list):
            for col_idx in col_indices:
                cea_targets.append((idx, int(col_idx)))
    data = df.iloc[:, :-1].reset_index(drop=True)
    return data, cea_targets, cta_cols, cpa_pairs

def cell(df, r, c, max_len=80):
    val = df.iat[r, c]
    s = "" if pd.isna(val) else str(val)
    return s[:max_len]

def table_text(df, target_row, target_col, max_rows=20):
    ncols = df.shape[1]
    n_rows = df.shape[0]
    rows = list(range(n_rows))
    if n_rows > max_rows:
        half = max_rows // 2
        lo = max(0, target_row - half)
        rows = list(range(lo, min(n_rows, lo + max_rows)))
    lines = ["col_ids: " + " | ".join(str(c) for c in range(ncols)), " | ".join(str(df.columns[c]) for c in range(ncols))]
    for r in rows:
        cells = []
        for c in range(ncols):
            val = cell(df, r, c)
            if r == target_row and c == target_col:
                val = f">>{val}<<"
            cells.append(val)
        lines.append(" | ".join(cells))
    return "\n".join(lines)

def col_header(df, col):
    try:
        return str(df.columns[col])
    except Exception:
        return ""

def first_id(text, pattern):
    m = pattern.search(text or "")
    return m.group(0) if m else None

def annotate_file(path, engine, writer, prompts, system_prompt, cfg, tasks):
    tab_id = os.path.splitext(os.path.basename(path))[0]
    df, cea_targets, cta_cols, cpa_pairs = load_table(path)
    max_rows = int(cfg.get("MAX_ROWS_CONTEXT", "20"))
    batch_size = int(cfg.get("BATCH_SIZE", "16"))
    max_new = int(cfg.get("MAX_NEW_TOKENS", "32"))

    if "cea" in tasks and cea_targets:
        msgs = []
        for r, c in cea_targets:
            msgs.append(render(prompts["CEA_PROMPT"], {"table_block": table_text(df, r, c, max_rows),"row": r, "col": c,"mention": cell(df, r, c),"header": col_header(df, c)}))
        answers = engine.generate(msgs, system_prompt, max_new_tokens=max_new, batch_size=batch_size)
        for (r, c), ans in zip(cea_targets, answers):
            qid = first_id(ans, QID_RE)
            if qid:
                writer.add_cea(tab_id, r, c, qid)

    if "cta" in tasks and cta_cols:
        max_vals = int(cfg.get("CTA_MAX_VALUES", "30"))
        msgs = []
        for c in cta_cols:
            values = [str(v) for v in df.iloc[:, c].dropna().unique().tolist()[:max_vals]]
            msgs.append(render(prompts["CTA_PROMPT"], {"header": col_header(df, c),"values": " | ".join(values)}))
        answers = engine.generate(msgs, system_prompt, max_new_tokens=max_new, batch_size=batch_size)
        for c, ans in zip(cta_cols, answers):
            qid = first_id(ans, QID_RE)
            if qid:
                writer.add_cta(tab_id, c, qid)

    if "cpa" in tasks and cpa_pairs:
        max_pairs = int(cfg.get("CPA_MAX_PAIRS", "15"))
        msgs = []
        for sub_col, obj_col in cpa_pairs:
            pairs = []
            for r in range(df.shape[0]):
                s, o = cell(df, r, sub_col), cell(df, r, obj_col)
                if s and o:
                    pairs.append(f"{s} -> {o}")
                if len(pairs) >= max_pairs:
                    break
            msgs.append(render(prompts["CPA_PROMPT"], {"sub_header": col_header(df, sub_col),"obj_header": col_header(df, obj_col),"pairs": "\n".join(pairs)}))
        answers = engine.generate(msgs, system_prompt, max_new_tokens=max_new, batch_size=batch_size)
        for (sub_col, obj_col), ans in zip(cpa_pairs, answers):
            pid = first_id(ans, PID_RE)
            if pid:
                writer.add_cpa(tab_id, sub_col, obj_col, pid)

def baseline_folder(config, engine=None):
    input_folder = config["INPUT_FOLDER"]
    output_folder = config.get("OUTPUT_FOLDER") or (input_folder + "_baseline")
    tasks = {t.strip().lower() for t in config.get("TASKS", "cea,cta,cpa").split(",") if t.strip()}
    as_uri = config.get("ENTITY_AS_URI", "True").lower() == "true"
    write_header = config.get("WRITE_HEADER", "False").lower() == "true"
    row_offset = int(config.get("ROW_OFFSET", "1"))

    prompts = {k: config.get(k, v) for k, v in DEFAULT_PROMPTS.items()}
    system_prompt = config.get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

    writer = OutputWriter(output_folder, as_uri=as_uri, write_header=write_header, row_offset=row_offset)

    if engine is None:
        reset_peaks()
        engine = build_generation_engine(config)
        log_vram("model_loaded")

    files = sorted(f for f in os.listdir(input_folder) if f.endswith(".csv") and not f.endswith("_candidates.csv"))
    start_time = time.time()
    for i, filename in enumerate(files, 1):
        print(f"Baseline [{i}/{len(files)}] {filename}")
        try:
            annotate_file(os.path.join(input_folder, filename), engine, writer, prompts, system_prompt, config, tasks)
        except Exception as e:
            print(f"ERROR on {filename}: {e}")

    log_vram("baseline_done")
    writer.flush()
    print(f"Tokens: {getattr(engine, 'tokens_in', 0)} in, {getattr(engine, 'tokens_out', 0)} out")
    print(f"Finished folder {input_folder}: {len(files)} files in {time.time() - start_time:.2f}s")
