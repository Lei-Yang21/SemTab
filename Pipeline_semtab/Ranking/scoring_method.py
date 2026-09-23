import math
from abc import ABC, abstractmethod
import pandas as pd

from scoring import Scoring_method, QUALITY_METHODS
from data_loader import cand_dict
from cta import cta_from_cea, build_type_pct

DEFAULT_FEATURES = ("string", "quality", "type")
DEFAULT_WEIGHTS = (0.5, 0.2, 0.3)

class ScoringMethod(ABC):
    requires_llm = False

    def __init__(self, config=None, llm=None):
        self.config = config if config is not None else {}
        self.llm = llm

    def score(self, candidates_df, context=None):
        result = candidates_df.copy()
        if result.empty:
            result["score"] = pd.Series(index=result.index, dtype=float)
            return result
        scores = self._score(candidates_df.copy(), context)
        if len(scores) != len(result):
            raise ValueError("Scoring must return one score per candidate")
        if isinstance(scores, pd.Series) and not scores.index.equals(result.index):
            raise ValueError("Scores must keep the candidate index and order")
        scores = pd.Series(scores, index=result.index, dtype=float)
        if not all(math.isfinite(s) for s in scores):
            raise ValueError("Candidate scores must be finite numbers")
        result["score"] = scores
        return result

    @abstractmethod
    def _score(self, candidates_df, context):
        raise NotImplementedError

class HeuristicScorer(ScoringMethod):
    def __init__(self, config=None, llm=None):
        super().__init__(config, llm)
        names = [n.strip() for n in self.config.get("CEA_FEATURES", ",".join(DEFAULT_FEATURES)).split(",")]
        features = {"string": self.string_score, "quality": self.quality_score, "type": self.type_score}
        unknown = [n for n in names if n not in features]
        if unknown:
            raise ValueError(f"Unknown CEA features: {', '.join(unknown)}")
        self.features = [features[n] for n in names]
        self.weights = tuple(float(x) for x in self.config.get("CEA_WEIGHTS", "0.5,0.2,0.3").split(","))
        if len(self.features) != len(self.weights):
            raise ValueError("CEA_FEATURES and CEA_WEIGHTS must have the same length")
        if not all(math.isfinite(w) for w in self.weights):
            raise ValueError("CEA_WEIGHTS must be finite")
        method = self.config.get("CEA_QUALITY_METHOD", "inverse").strip().lower()
        if method not in QUALITY_METHODS:
            raise ValueError(f"Unknown CEA quality method: {method}")
        self.metrics = Scoring_method(QUALITY_METHODS[method])

    def string_score(self, cand, type_pct):
        aliases = cand["aliases"].split("|") if cand["aliases"] else None
        return self.metrics.best_string_sim(cand["mention"], cand["label"], aliases)

    def quality_score(self, cand, type_pct):
        return self.metrics.quality_score(cand["quality"])

    def type_score(self, cand, type_pct):
        return self.metrics.type_coherence(cand["P31"] + cand["P279"], type_pct)

    def _score(self, candidates_df, context):
        type_pct = context.type_pct if context is not None else build_type_pct(cta_from_cea(candidates_df))
        scores = []
        for _, row in candidates_df.iterrows():
            cand = cand_dict(row)
            score = 0.0
            if cand["qid"]:
                for weight, feature in zip(self.weights, self.features):
                    score += weight * feature(cand, type_pct.get(int(row["columns"]), {}))
            scores.append(score)
        return scores

SCORING_METHODS = {"heuristic": HeuristicScorer}

def get_scoring_method(config):
    name = config.get("SCORING_METHOD", "heuristic").strip().lower()
    if name not in SCORING_METHODS:
        raise ValueError(f"Unknown scoring method: {name}")
    return SCORING_METHODS[name]

def build_scorer(config, llm=None):
    method = get_scoring_method(config)
    if method.requires_llm and llm is None:
        raise ValueError("The selected scoring method requires an LLM engine")
    return method(config, llm)
