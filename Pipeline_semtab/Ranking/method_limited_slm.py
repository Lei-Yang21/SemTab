import cea as cea_mod
from cta import choose_cta, infer_literal_type
from method_base import run_cpa

def flag(cfg, key, default="True"):
    return cfg.get(key, default).lower() == "true"
def annotate(ctx):
    cfg = ctx.config
    use_slm_cea = flag(cfg, "CEA_USE_SLM")
    use_slm_cta = flag(cfg, "CTA_USE_SLM")
    use_slm_cpa = flag(cfg, "CPA_USE_SLM")
    cta_from_sel = flag(cfg, "CTA_FROM_SELECTION")
    margin = float(cfg.get("CEA_TIEBREAK_MARGIN", "0.05"))
    ctx_tiebreak = flag(cfg, "CEA_CONTEXT_TIEBREAK", "False")
    ctx_margin = float(cfg.get("CEA_CONTEXT_MARGIN", "0.10")) if ctx_tiebreak else 0.0
    cea_llm_topk = int(cfg.get("CEA_LLM_TOPK", "5"))
    cta_margin = float(cfg.get("CTA_MARGIN", "0.3"))
    cta_topk = int(cfg.get("CTA_TOPK", "5"))
    allow_nil = flag(cfg, "ALLOW_NIL", "False")
    nil_label = cfg.get("NIL_LABEL", "NIL").strip() or "NIL"
    nil_threshold = float(cfg.get("NIL_SCORE_THRESHOLD", "0.0"))

    if "cea" in ctx.tasks or "cpa" in ctx.tasks:
        for (r, c), cands in ctx.cells.items():
            row_terms = ctx.row_terms(r, exclude_col=c) if ctx_margin > 0 else None
            qid = cea_mod.choose_cea(
                cands, ctx.type_pct.get(c, {}), llm=ctx.llm, use_slm=use_slm_cea,
                margin=margin, row_context=ctx.row_context(r),
                col_header=ctx.col_header(c),
                row_terms=row_terms, context_margin=ctx_margin,
                llm_topk=cea_llm_topk, allow_nil=allow_nil,
                nil_label=nil_label, nil_threshold=nil_threshold,
                nil_review=float(cfg.get("NIL_REVIEW_SCORE", "0.0")))
            if qid:
                if qid != nil_label:
                    ctx.cea_choice[(r, c)] = qid
                if "cea" in ctx.tasks:
                    ctx.writer.add_cea(ctx.tab_id, r, c, qid)
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

    if "cpa" in ctx.tasks:
        run_cpa(ctx, use_slm_cpa)