def inverse_quality(quality):
    return 1.0 / quality

def identity_quality(quality):
    return float(quality)

QUALITY_METHODS = {"inverse": inverse_quality, "identity": identity_quality}

class Scoring_method:
    def __init__(self, quality_function=None):
        self.quality_function = quality_function if quality_function is not None else inverse_quality
    #Levenshtein distance minimum number of edits to transform one string into another
    def levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]
    # Levenshtein similarity is calculated as 1 - (levenshtein_distance / max_length)
    def levenshtein_similarity(self, s1, s2):
        s1, s2 = str(s1).lower(), str(s2).lower()
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 0.0
        return (max_len - self.levenshtein_distance(s1, s2)) / max_len
    # Jaccard similarity is calculated as the size of the intersection divided by the size of the union of two sets
    def token_jaccard(self, s1, s2):
        ta = set(str(s1).lower().split())
        tb = set(str(s2).lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)
    def type_coherence(self, candidate_types, type_pct):
        if not candidate_types or not type_pct:
            return 0.0
        return max((type_pct.get(t, 0.0) for t in candidate_types), default=0.0)

    def quality_score(self, quality):
        try:
            q = int(quality)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        return self.quality_function(q) if q > 0 else 0.0

    def best_string_sim(self, mention, label, aliases=None):
        best = max(self.levenshtein_similarity(mention, label),self.token_jaccard(mention, label))
        for a in aliases or []:
            if a:
                best = max(best, self.levenshtein_similarity(mention, a))
        return best
    def context_overlap(self, cand_text, context_terms):
        if not context_terms:
            return 0.0
        cand_tokens = set(str(cand_text).lower().split())
        if not cand_tokens:
            return 0.0
        hits = sum(1 for t in context_terms if t in cand_tokens)
        return hits / len(context_terms)
