class TableLabels:
    def __init__(self, knowledge, language="en"):
        self.knowledge = knowledge
        self.language = language
        self.labels = {}

    def label(self, qid):
        if qid not in self.labels:
            info = self.knowledge.get_entities([qid], self.language)
            self.labels[qid] = info.get(qid, {}).get("label", qid)
        return self.labels[qid]

    def column_type(self, table, col):
        result = table.cta_by_col.get(col)
        if not result:
            return ""
        _, p31, p279 = result
        ranked = list((p31 or p279).items())
        return self.label(ranked[0][0]) if ranked else ""

    def final_type(self, table, result, col):
        qid = result.cta.get(col)
        return self.label(qid) if qid else self.column_type(table, col)

    def candidate_types(self, shortlist):
        qids = []
        for cand in shortlist:
            qids += (cand.get("P31") or []) + (cand.get("P279") or [])
        if not qids:
            return {}
        try:
            info = self.knowledge.get_entities(qids, self.language)
        except Exception:
            return {}
        return {q: v.get("label", "") for q, v in info.items()}
