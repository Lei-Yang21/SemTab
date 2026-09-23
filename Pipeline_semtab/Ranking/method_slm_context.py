import cea as cea_mod
from cta import choose_cta, infer_literal_type
from method_base import run_cpa
from wikidata_api_ranking import get_entities

def flag(cfg, key, default="True"):
    return cfg.get(key, default).lower() == "true"
def use_llm(gate, scored, margin):
    if len(scored) <= 1:
        return False
    if gate == "all":
        return True
    return scored[0][0] - scored[1][0] < margin
def get_type_labels(ctx, shortlist):
    qids = []
    for c in shortlist:
        qids += (c.get("P31") or []) + (c.get("P279") or [])
    if not qids:
        return {}
    try:
        info = get_entities(qids, ctx.language)
    except Exception:
        return {}
    return {q: v.get("label", "") for q, v in info.items()}
def annotate(ctx):
    cfg = ctx.config
    gate = cfg.get("LLM_GATE", "uncertain").lower()
    margin = float(cfg.get("LLM_CONTEXT_MARGIN", "0.10"))
    topk = int(cfg.get("LLM_TOPK", "10"))
    max_rows = int(cfg.get("LLM_CONTEXT_MAX_ROWS", "20"))
    enrich = flag(cfg, "LLM_ENRICH", "False")
    ctx_tiebreak = flag(cfg, "CEA_CONTEXT_TIEBREAK", "False")
    heuristic_margin = float(cfg.get("CEA_CONTEXT_MARGIN", "0.10")) if ctx_tiebreak else 0.0
    use_slm_cta = flag(cfg, "CTA_USE_SLM")
    use_slm_cpa = flag(cfg, "CPA_USE_SLM")
    cta_from_sel = flag(cfg, "CTA_FROM_SELECTION")
    cta_margin = float(cfg.get("CTA_MARGIN", "0.3"))
    cta_topk = int(cfg.get("CTA_TOPK", "5"))
    allow_nil = flag(cfg, "ALLOW_NIL", "False")
    nil_label = cfg.get("NIL_LABEL", "NIL").strip() or "NIL"
    nil_threshold = float(cfg.get("NIL_SCORE_THRESHOLD", "0.0"))
    nil_review = float(cfg.get("NIL_REVIEW_SCORE", "0.0"))
    calls = 0

    if "cea" in ctx.tasks or "cpa" in ctx.tasks:
        for (r, c), cands in ctx.cells.items():
            # Try to resolve with context tiebreak first, then LLM if needed
            scored = cea_mod.rank_cell(cands, ctx.type_pct.get(c, {}))
            if not scored:
                if allow_nil and "cea" in ctx.tasks:
                    ctx.writer.add_cea(ctx.tab_id, r, c, nil_label)
                continue
            qid = scored[0][1]["qid"]
            resolved = False
            # Not true at for the testing part 
            if heuristic_margin > 0 and len(scored) > 1:
                choice = cea_mod.context_tiebreak(scored, ctx.row_terms(r, exclude_col=c), heuristic_margin)
                if choice:
                    qid = choice
                    resolved = True
            needs_nil_review = allow_nil and nil_review > 0 and scored[0][0] < nil_review
            if not resolved and ctx.llm is not None and (use_llm(gate, scored, margin) or needs_nil_review):
                shortlist = [cc for _, cc in scored[:topk]]
                type_labels = get_type_labels(ctx, shortlist) if enrich else None
                choice = ctx.llm.select_with_context(shortlist[0]["mention"], shortlist,table_text=ctx.table_text(r, c, max_rows),col_header=ctx.col_header(c),col_type=ctx.column_type_label(c),target_row=r, target_col=c,type_labels=type_labels, enrich=enrich)
                calls += 1
                if choice:
                    qid = choice
                    resolved = True
            if allow_nil and not resolved and scored[0][0] < nil_threshold:
                qid = nil_label
            if qid:
                if qid != nil_label:
                    ctx.cea_choice[(r, c)] = qid
                if "cea" in ctx.tasks:
                    ctx.writer.add_cea(ctx.tab_id, r, c, qid)
     # Handle CTA
    if "cta" in ctx.tasks:
        if cta_from_sel:
            ctx.rebuild_cta_from_selection()
        for col in ctx.cta_cols:
            if col in ctx.cta_by_col:
                qid = choose_cta(ctx.cta_by_col[col], llm=ctx.llm, use_slm=use_slm_cta,language=ctx.language, col_values=ctx.col_values(col),col_header=ctx.col_header(col), margin=cta_margin, topk=cta_topk)
            else:
                qid = infer_literal_type(ctx.col_values(col))
            if qid:
                ctx.cta_choice[col] = qid
            ctx.writer.add_cta(ctx.tab_id, col, qid)
    # Handle CPA
    if "cpa" in ctx.tasks:
        run_cpa(ctx, use_slm_cpa)

    return calls
