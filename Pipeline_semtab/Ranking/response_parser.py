import re

NIL_RE = re.compile(r"\bNIL\b|\bnone of (?:the|these|them)\b|\bno match\b"r"|\bnot (?:in|among) the candidates\b|\bidk\b|\bI don'?t know\b", re.I)

class ResponseParser:
    def __init__(self, config):
        self.allow_nil = str(config.get("ALLOW_NIL", "false")).strip().lower() == "true"
        self.nil_label = config.get("NIL_LABEL", "NIL").strip() or "NIL"

    def said_nil(self, response):
        return bool(self.allow_nil and NIL_RE.search(response or ""))

    def match_id(self, response, candidate_ids):
        for candidate_id in candidate_ids:
            if candidate_id and candidate_id in response:
                return candidate_id
        if self.said_nil(response):
            return self.nil_label
        return None

    def final_id(self, response, candidate_ids):
        ids = {c for c in candidate_ids if c}
        m = re.search(r"(?:answer|final|winning\s*qid|qid)\s*[:\-]?\s*(Q\d+|NIL)", response, re.I)
        if m:
            answer = m.group(1)
            if answer.upper() == "NIL":
                return self.nil_label if self.allow_nil else None
            if answer in ids:
                return answer
        found = [x for x in re.findall(r"Q\d+", response) if x in ids]
        if found:
            return found[-1]
        if self.said_nil(response):
            return self.nil_label
        return None

    def verified_id(self, response, candidate_ids, chosen):
        if "NIL" in response.upper():
            return None
        return self.match_id(response, candidate_ids) or chosen

