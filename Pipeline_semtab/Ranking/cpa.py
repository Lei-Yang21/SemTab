import re
from collections import Counter
from Pipeline_semtab.common.knowledge import WikidataClient

def to_float(s):
    if s is None:
        return None
    s = str(s).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None
def num_match(a, b, rel=0.02, absol=0.5):
    return abs(a - b) <= max(absol, rel * max(abs(a), abs(b)))
def str_norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())
def match_properties(subject_claims, object_value=None, object_qid=None,object_is_entity=False):

    pids = []
    f = None if object_is_entity else to_float(object_value)
    obj_norm = (None if object_is_entity or object_value is None else str_norm(object_value))
    for pid, vals in subject_claims.items():
        for v in vals:
            vtype, vval = v[0], v[1]
            if object_is_entity:
                if vtype == "entity" and object_qid and vval == object_qid:
                    pids.append(pid)
                    break
            else:
                if (vtype == "quantity" and f is not None
                        and isinstance(vval, (int, float)) and num_match(f, vval)):
                    pids.append(pid)
                    break
                if vtype == "time" and f is not None and 1000 <= f <= 2100:
                    m = re.search(r"([+-]?\d{4})", str(vval))
                    if m and int(m.group(1)) == int(f):
                        pids.append(pid)
                        break
                if vtype == "string" and obj_norm and str_norm(vval) == obj_norm:
                    pids.append(pid)
                    break
    return pids
def resolve_pair(sub_col, obj_col, n_rows, cea_choice, data_df, obj_is_entity, llm=None, use_slm=True, col_header="", sub_type="", obj_type="", knowledge=None):

    if knowledge is None:
        knowledge = WikidataClient()
    subj_qids = [cea_choice.get((r, sub_col)) for r in range(n_rows)]
    claims_map = knowledge.get_entity_claims([q for q in subj_qids if q])

    votes = Counter()
    sample_vals = []
    for r in range(n_rows):
        subj = cea_choice.get((r, sub_col))
        if not subj:
            continue
        claims = claims_map.get(subj, {})
        if obj_is_entity:
            obj_qid = cea_choice.get((r, obj_col))
            if not obj_qid:
                continue
            pids = match_properties(claims, object_qid=obj_qid, object_is_entity=True)
        else:
            try:
                obj_val = data_df.iat[r, obj_col]
            except (IndexError, KeyError):
                continue
            sample_vals.append(obj_val)
            pids = match_properties(claims, object_value=obj_val)
        for pid in set(pids):
            votes[pid] += 1

    if not votes:
        return None
    ranked = votes.most_common()
    if not (use_slm and llm) or len(ranked) == 1 or ranked[0][1] != ranked[1][1]:
        return ranked[0][0]
    
    tied = [pid for pid, c in ranked if c == ranked[0][1]]
    labels = knowledge.get_labels(tied)
    cands = [{"pid": pid, "label": labels.get(pid, pid)} for pid in tied]
    choice = llm.select_best_property(cands, col_header, sample_vals, sub_type=sub_type, obj_type=obj_type)
    return choice if choice else ranked[0][0]
