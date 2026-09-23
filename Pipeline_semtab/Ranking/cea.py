from scoring import Scoring_method
from scoring_method import DEFAULT_WEIGHTS, HeuristicScorer
from data_loader import candidates_frame
from cta import build_type_pct
from types import SimpleNamespace

_score = Scoring_method()

def cell_scores(cands, type_pct, weights, scorer):
    if scorer is None and all("score" in c for c in cands):
        return [c["score"] for c in cands]
    if scorer is None:
        scorer = HeuristicScorer({"CEA_WEIGHTS": ",".join(str(w) for w in weights)})
    context = SimpleNamespace(type_pct={0: type_pct})
    return scorer.score(candidates_frame(cands), context)["score"].tolist()

def score_candidate(cand, type_pct, weights=DEFAULT_WEIGHTS, scorer=None):
    return cell_scores([cand], type_pct, weights, scorer)[0]
def rank_cell(cands, type_pct, weights=DEFAULT_WEIGHTS, scorer=None):
    cands = [c for c in cands if c["qid"]]
    scored = list(zip(cell_scores(cands, type_pct, weights, scorer), cands))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored
def cand_text(cand):
    return " ".join([cand.get("label", ""), cand.get("description", ""),str(cand.get("aliases", "")).replace("|", " ")])
def context_tiebreak(scored, row_terms, context_margin):
    top = scored[0][0]
    near = [(s, c) for s, c in scored if top - s < context_margin]
    if len(near) <= 1:
        return None
    ov_top = _score.context_overlap(cand_text(scored[0][1]), row_terms)
    _, best_c = max(near, key=lambda sc:(_score.context_overlap(cand_text(sc[1]), row_terms), sc[0]))
    if best_c["qid"] != scored[0][1]["qid"]:
        if _score.context_overlap(cand_text(best_c), row_terms) > ov_top:
            return best_c["qid"]
    return None
    #Value for context_margin is set higher than the LLM margin because of the cost to call the first 
def choose_cea(cands, type_pct, llm=None, use_slm=True, margin=0.05,row_context="", col_header="", weights=DEFAULT_WEIGHTS,row_terms=None, context_margin=0.1, llm_topk=5,allow_nil=False, nil_label="NIL", nil_threshold=0.0, nil_review=0.0, scorer=None):
    scored = rank_cell(cands, type_pct, weights, scorer)
    if not scored:
        return nil_label if allow_nil else None
    top_score = scored[0][0]
    if len(scored) == 1:
        if allow_nil and top_score < nil_threshold:
            return nil_label
        return scored[0][1]["qid"]
    if context_margin > 0 and row_terms:
        choice = context_tiebreak(scored, row_terms, context_margin)
        if choice:
            return choice
    if not (use_slm and llm):
        if allow_nil and top_score < nil_threshold:
            return nil_label
        return scored[0][1]["qid"]
    needs_nil_review = allow_nil and nil_review > 0 and top_score < nil_review
    if scored[0][0] - scored[1][0] < margin or needs_nil_review:
        top_cands = [c for _, c in scored[:llm_topk]]
        choice = llm.select_best_entity(top_cands[0]["mention"], top_cands, row_context, col_header)
        if choice:
            return choice
    if allow_nil and top_score < nil_threshold:
        return nil_label
    return scored[0][1]["qid"]
