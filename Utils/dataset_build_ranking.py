import os
import csv
import sys
import json
import random
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Pipeline_semtab" / "Ranking"))

from Pipeline_semtab.common.knowledge import WikidataClient
from main_ranking import load_config
from scoring_method import build_scorer
from cea import rank_cell, context_tiebreak
from table_context import load_table_context, prepare_table
from table_labels import TableLabels
from prompts import PromptBuilder
from method_slm_context import use_llm

def flag(cfg, key, default="True"):
    return cfg.get(key, default).lower() == "true"
def load_cea_gold(path, row_offset=1, allow_nil=False, nil_label="NIL"):
    gold = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for parts in csv.reader(f):
            if len(parts) < 4:
                continue
            tab, row, col, url = parts[0], parts[1], parts[2], parts[3]
            try:
                row, col = int(row), int(col)
            except ValueError:
                continue
            qid = str(url).strip().rstrip("/").split("/")[-1]
            if qid.startswith("Q"):
                gold[(tab, row - row_offset, col)] = qid
            elif allow_nil:
                gold[(tab, row - row_offset, col)] = nil_label
    return gold

def gold_rank(scored, gold):
    for i, (_, c) in enumerate(scored):
        if c["qid"] == gold:
            return i + 1
    return 0
def nil_settings(cfg):
    allow = flag(cfg, "ALLOW_NIL", "False")
    return (allow, cfg.get("NIL_LABEL", "NIL").strip() or "NIL",
            float(cfg.get("NIL_REVIEW_SCORE", "0.0")) if allow else 0.0)


def build_limited_slm_example(ctx, r, c, scored, gold, prompts, cfg, labels=None):
    margin = float(cfg.get("CEA_TIEBREAK_MARGIN", "0.05"))
    ctx_tiebreak = flag(cfg, "CEA_CONTEXT_TIEBREAK", "False")
    ctx_margin = float(cfg.get("CEA_CONTEXT_MARGIN", "0.10")) if ctx_tiebreak else 0.0
    allow_nil, nil_label, nil_review = nil_settings(cfg)
    is_nil = allow_nil and gold == nil_label
    if len(scored) <= 1:
        return None, "not_ambiguous"
    if ctx_margin > 0:
        if context_tiebreak(scored, ctx.row_terms(r, exclude_col=c), ctx_margin):
            return None, "resolved_by_tiebreak"
    gated = scored[0][0] - scored[1][0] < margin
    reviewed = nil_review > 0 and scored[0][0] < nil_review
    if not (gated or reviewed):
        return None, "not_gated"
    top = [cc for _, cc in scored[:int(cfg.get("CEA_LLM_TOPK", "5"))]]
    if not is_nil and gold not in {t["qid"] for t in top}:
        return None, "gold_absent"
    if len({t["qid"] for t in top}) < 2:
        return None, "trivial"
    user = prompts.entity(top[0]["mention"], top, ctx.row_context(r), ctx.col_header(c))
    return {"user": user, "n_candidates": len(top),
            "gold_rank": 0 if is_nil else gold_rank(scored, gold),
            "is_nil": is_nil,
            "margin": round(scored[0][0] - scored[1][0], 4)}, "kept_nil" if is_nil else "kept"
def build_context_example(ctx, r, c, scored, gold, prompts, cfg, labels=None):
    gate = cfg.get("LLM_GATE", "uncertain").lower()
    margin = float(cfg.get("LLM_CONTEXT_MARGIN", "0.10"))
    topk = int(cfg.get("LLM_TOPK", "10"))
    max_rows = int(cfg.get("LLM_CONTEXT_MAX_ROWS", "20"))
    enrich = flag(cfg, "LLM_ENRICH", "False")
    ctx_tiebreak = flag(cfg, "CEA_CONTEXT_TIEBREAK", "False")
    ctx_margin = float(cfg.get("CEA_CONTEXT_MARGIN", "0.10")) if ctx_tiebreak else 0.0
    allow_nil, nil_label, nil_review = nil_settings(cfg)
    is_nil = allow_nil and gold == nil_label
    if not scored:
        return None, "no_candidates"
    if ctx_margin > 0 and len(scored) > 1:
        if context_tiebreak(scored, ctx.row_terms(r, exclude_col=c), ctx_margin):
            return None, "resolved_by_tiebreak"
    reviewed = nil_review > 0 and scored[0][0] < nil_review
    if not (use_llm(gate, scored, margin) or reviewed):
        return None, "not_gated"
    shortlist = [cc for _, cc in scored[:topk]]
    if not is_nil and gold not in {s["qid"] for s in shortlist}:
        return None, "gold_absent"
    if len({s["qid"] for s in shortlist}) < 2:
        return None, "trivial"
    type_labels = labels.candidate_types(shortlist) if enrich else None
    user = prompts.context(shortlist[0]["mention"], shortlist,
                           table_text=ctx.table_text(r, c, max_rows), col_header=ctx.col_header(c),
                           col_type=labels.column_type(ctx, c), target_row=r, target_col=c,
                           type_labels=type_labels, enrich=enrich)
    return {"user": user, "n_candidates": len(shortlist),
            "gold_rank": 0 if is_nil else gold_rank(scored, gold),
            "is_nil": is_nil,
            "margin": round(scored[0][0] - scored[1][0], 4) if len(scored) > 1 else 1.0}, "kept_nil" if is_nil else "kept"

BUILDERS = {"limited_slm": build_limited_slm_example, "slm_limited": build_limited_slm_example, "slm_context": build_context_example}

def build_datasets(config_path, knowledge=None, scorer=None, prompts=None, llm=None, table_loader=load_table_context):
    cfg = dict(config_path) if isinstance(config_path, dict) else load_config(config_path)
    if knowledge is None:
        knowledge = WikidataClient(sleep=cfg.get("API_SLEEP", "0.1"))
    if scorer is None:
        scorer = build_scorer(cfg, llm)
    if prompts is None:
        prompts = PromptBuilder(cfg)
    input_folder = cfg["INPUT_FOLDER"]
    preprocess_folder = cfg["PREPROCESS_FOLDER"]
    gt_file = cfg["GT_FILE"]
    methods = [m.strip().lower() for m in cfg.get("FT_METHOD", "limited_slm,slm_context").split(",") if m.strip()]
    for m in methods:
        if m not in BUILDERS:
            raise ValueError(f"Unknown FT_METHOD: {m}")
    out_folder = cfg.get("OUT_FOLDER", "finetune_datasets")
    os.makedirs(out_folder, exist_ok=True)
    row_offset = int(cfg.get("ROW_OFFSET", "1"))
    val_ratio = float(cfg.get("VAL_RATIO", "0.1"))
    seed = int(cfg.get("SEED", "42"))
    system_prompt = cfg.get("SYSTEM_PROMPT", "")
    language = cfg.get("LANGUAGE", "en")

    allow_nil, nil_label, _ = nil_settings(cfg)
    nil_max_share = float(cfg.get("NIL_MAX_SHARE", "0.35"))
    gold = load_cea_gold(gt_file, row_offset, allow_nil, nil_label)
    n_nil = sum(1 for v in gold.values() if v == nil_label) if allow_nil else 0
    print(f"Read {len(gold)} CEA gold cells from {gt_file}"
          + (f" ({n_nil} of them NIL)" if allow_nil else ""))

    examples = {m: [] for m in methods}
    stats = {m: Counter() for m in methods}

    files = sorted(f for f in os.listdir(input_folder) if f.endswith(".csv"))
    for i, filename in enumerate(files, 1):
        input_path = os.path.join(input_folder, filename)
        preprocess_path = os.path.join(preprocess_folder, filename.replace("_candidates", ""))
        if not os.path.exists(preprocess_path):
            print(f"Preprocess file missing for {filename}, skipping.")
            continue
        ctx = prepare_table(table_loader(input_path, preprocess_path), scorer)
        labels = TableLabels(knowledge, language)
        for (r, c), cands in ctx.cells.items():
            g = gold.get((ctx.tab_id, r, c))
            if not g:
                for m in methods:
                    stats[m]["no_gold"] += 1
                continue
            scored = rank_cell(cands, ctx.type_pct.get(c, {}))
            for m in methods:
                ex, status = BUILDERS[m](ctx, r, c, scored, g, prompts, cfg, labels)
                stats[m][status] += 1
                if ex:
                    ex.update({"system": system_prompt, "completion": g, "gold_qid": g,"table": ctx.tab_id, "row": r, "col": c, "method": m})
                    examples[m].append(ex)
        if i % 50 == 0:
            kept = ", ".join(f"{m}: {len(examples[m])}" for m in methods)
            print(f"  {i}/{len(files)} tables ({kept})", flush=True)

    rng = random.Random(seed)
    for m in methods:
        exs = examples[m]
        print(f"\n[{m}] stats: {dict(stats[m])}")
        if not exs:
            print(f"[{m}] no examples, nothing written")
            continue
        if allow_nil and 0 < nil_max_share < 1:
            nils = [e for e in exs if e.get("is_nil")]
            rest = [e for e in exs if not e.get("is_nil")]
            cap = int(len(rest) * nil_max_share / (1 - nil_max_share))
            if len(nils) > cap:
                nils = rng.sample(nils, cap)
                print(f"[{m}] NIL examples capped at {nil_max_share:.0%} of the set: "f"{cap} kept")
            exs = sorted(rest + nils, key=lambda e: (e["table"], e["row"], e["col"]))
            examples[m] = exs
        if not exs:
            print(f"[{m}] no examples after NIL filtering, nothing written")
            continue
        n_nil_ex = sum(1 for e in exs if e.get("is_nil"))
        if allow_nil:
            print(f"[{m}] {n_nil_ex} NIL examples ({n_nil_ex / len(exs):.0%} of the set)")
        hard = sum(1 for e in exs if e["gold_rank"] != 1)
        print(f"[{m}] {len(exs)} examples, gold not rank-1: {hard} ({hard / len(exs):.0%})")
        tables = sorted({e["table"] for e in exs})
        rng.shuffle(tables)
        n_val = max(1, int(len(tables) * val_ratio)) if val_ratio > 0 and len(tables) > 1 else 0
        val_tables = set(tables[:n_val])
        train = [e for e in exs if e["table"] not in val_tables]
        val = [e for e in exs if e["table"] in val_tables]
        train_path = os.path.join(out_folder, f"ft_{m}_train.json")
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train, f, ensure_ascii=False, indent=2)
        print(f"[{m}] wrote {len(train)} train examples ({len(tables) - n_val} tables) -> {train_path}")
        if val:
            val_path = os.path.join(out_folder, f"ft_{m}_valid.json")
            with open(val_path, "w", encoding="utf-8") as f:
                json.dump(val, f, ensure_ascii=False, indent=2)
            print(f"[{m}] wrote {len(val)} valid examples ({n_val} tables) -> {val_path}")
    return examples

if __name__ == "__main__":
    build_datasets(sys.argv[1] if len(sys.argv) > 1 else "config_finetune.txt")
