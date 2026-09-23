from cta import choose_cta, infer_literal_type
from cea import rank_cell
from method_base import run_cpa

def flag(cfg, key, default="True"):
    return cfg.get(key, default).lower() == "true"
def annotate(ctx):
    cfg = ctx.config
    use_verify = flag(cfg, "LLM_VERIFY")
    use_slm_cpa = flag(cfg, "CPA_USE_SLM")
    topk = int(cfg.get("LLM_TOPK", "10"))
    cta_from_sel = flag(cfg, "CTA_FROM_SELECTION")
    cta_topk = int(cfg.get("CTA_TOPK", "5"))

    if "cea" in ctx.tasks or "cpa" in ctx.tasks:
        items = []
        for (r, c), cands in ctx.cells.items():
            scored = rank_cell(cands, ctx.type_pct.get(c, {}))
            top = [cc for _, cc in scored[:topk]]
            if not top:
                continue
            items.append({"key": (r, c), "candidates": top,"mention": top[0]["mention"],"row_context": ctx.row_context(r),"col_header": ctx.col_header(c),"fallback": scored[0][1]["qid"]})

        qids = [qid for qid, _ in ctx.llm.debate_select_batch(items)]

        if use_verify:
            idx = [i for i, q in enumerate(qids) if q]
            verified = ctx.llm.verify_batch([{**items[i], "chosen": qids[i]} for i in idx])
            for i, vq in zip(idx, verified):
                qids[i] = vq
        for it, qid in zip(items, qids):
            qid = qid or it["fallback"]
            if qid:
                r, c = it["key"]
                ctx.cea_choice[(r, c)] = qid
                if "cea" in ctx.tasks:
                    ctx.writer.add_cea(ctx.tab_id, r, c, qid)

    if "cta" in ctx.tasks:
        if cta_from_sel:
            ctx.rebuild_cta_from_selection()
        for col in ctx.cta_cols:
            qid = None
            if col in ctx.cta_by_col:
                qid = choose_cta(ctx.cta_by_col[col], llm=ctx.llm, use_slm=True,language=ctx.language, col_values=ctx.col_values(col),col_header=ctx.col_header(col), margin=1.0, topk=cta_topk)
            else:
                qid = infer_literal_type(ctx.col_values(col))
            if qid:
                ctx.cta_choice[col] = qid
            ctx.writer.add_cta(ctx.tab_id, col, qid)
    if "cpa" in ctx.tasks:
        run_cpa(ctx, use_slm_cpa)