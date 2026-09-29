import time
import requests

API_URL = "https://www.wikidata.org/w/api.php"
USER_AGENT = "SemTabThesisBot/1.0 (https://github.com/LeiY769; contact: leiyang677@gmail.com)"

class KnowledgeSource:
    def search_entities(self, query, language="en", limit=10):
        raise NotImplementedError

    def get_entity_data(self, qids, language="en"):
        raise NotImplementedError

    def get_labels(self, qids, language="en"):
        raise NotImplementedError

    def get_entity_claims(self, qids):
        raise NotImplementedError

    def get_entities(self, qids, language="en"):
        #Entity data plus label, fetched in batches for the whole qid list.
        qids = [q for q in dict.fromkeys(qids) if q]
        if not qids:
            return {}
        data = self.get_entity_data(qids, language=language)
        labels = self.get_labels(qids, language=language)
        out = {}
        for q in qids:
            entry = dict(data.get(q, {}))
            entry["label"] = labels.get(q, q)
            out[q] = entry
        return out

class WikidataClient(KnowledgeSource):
    def __init__(self, sleep=0.1, api_url=API_URL, session=None, wait=None, clock=None):
        self.api_url = api_url
        self.session = session if session is not None else requests.Session()
        self.wait = wait if wait is not None else time.sleep
        self.clock = clock if clock is not None else time.monotonic
        self.sleep = max(0.0, float(sleep))
        self.error_pause = 10.0
        self.max_backoff = 120.0
        self.lag_max_wait = 1800.0
        self.lag_grace = 60.0
        self.maxlag_off_window = 7200.0
        self.maxlag_off_until = 0.0

    def set_rate_limit(self, seconds):
        self.sleep = max(0.0, float(seconds))

    def retry_wait(self, r, attempt):
        try:
            retry_after = float(r.headers.get("Retry-After", 0))
        except (TypeError, ValueError):
            retry_after = 0.0
        return max(retry_after, min(max(self.error_pause, 2 ** attempt), self.max_backoff))

    def api_get(self, params, max_retries=8):
        headers = {"User-Agent": USER_AGENT}
        last_error = None
        attempt = 0
        lag_waited = 0.0
        while attempt < max_retries:
            use_maxlag = self.clock() >= self.maxlag_off_until
            try:
                req_params = {**params, "maxlag": 7} if use_maxlag else params
                r = self.session.get(self.api_url, params=req_params, headers=headers, timeout=30)
            except requests.RequestException as e:
                last_error = repr(e)
                attempt += 1
                self.wait(min(max(self.error_pause, 2 ** attempt), self.max_backoff))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last_error = f"HTTP {r.status_code}"
                attempt += 1
                self.wait(self.retry_wait(r, attempt))
                continue
            if r.status_code == 200:
                data = r.json()
                error = data.get("error")
                if error:
                    code = error.get("code", "")
                    if code in ("maxlag", "readonly") or code.startswith("cirrussearch"):
                        last_error = f"API error '{code}': {error.get('info', '')}"
                        wait = self.retry_wait(r, 0)
                        if lag_waited + wait > self.lag_max_wait:
                            raise RuntimeError(f"Wikidata still lagged after waiting {lag_waited:.0f}s, last error: {last_error}")
                        if code == "maxlag" and lag_waited + wait > self.lag_grace:
                            print(f"Wikidata still lagged after {lag_waited:.0f}s, retrying without maxlag (read-only requests)")
                            self.maxlag_off_until = self.clock() + self.maxlag_off_window
                            lag_waited += wait
                            self.wait(wait)
                            continue
                        if lag_waited == 0.0:
                            print(f"Wikidata lagged, waiting for it to catch up ({error.get('info', '')})")
                        lag_waited += wait
                        self.wait(wait)
                        continue
                    if code == "ratelimited" or code.startswith("internal_api_error"):
                        last_error = f"API error '{code}': {error.get('info', '')}"
                        attempt += 1
                        self.wait(self.retry_wait(r, attempt))
                        continue
                    print(f"Wikidata API error '{code}': {error.get('info', '')}")
                    return None
                if self.sleep:
                    self.wait(self.sleep)
                return data
            return None
        raise RuntimeError(f"Wikidata API failed after {max_retries} retries, last error: {last_error}")

    def search_entities(self, query, language="en", limit=10):
        if not query:
            return []
        data = self.api_get({"action": "wbsearchentities","search": query, "language": language,"limit": limit,"format": "json"})
        if not data:
            return []
        return [(e.get("label", ""), e.get("id", "")) for e in data.get("search", [])]

    def fulltext_search(self, query, language="en", limit=20):
        if not query:
            return []
        data = self.api_get({"action": "query", "list": "search","srsearch": query,"srlimit": limit,"format": "json"})
        if not data:
            return []
        qids = [r["title"] for r in data.get("query", {}).get("search", [])
                if str(r.get("title", "")).startswith("Q")]
        labels = self.get_labels(qids, language)
        return [(labels.get(q, q), q) for q in qids]

    def get_entity_data(self, qids, language="en"):
        out = {}
        qids = [q for q in dict.fromkeys(qids) if q]
        for i in range(0, len(qids), 50):
            chunk = qids[i:i + 50]
            data = self.api_get({"action": "wbgetentities", "ids": "|".join(chunk),  "props": "descriptions|aliases|claims|sitelinks", "languages": language,   "sitefilter": f"{language}wiki",  "format": "json"})
            if not data:
                continue
            for qid, entity in data.get("entities", {}).items():
                claims = entity.get("claims", {})
                def entity_ids(pid):
                    ids = []
                    for claim in claims.get(pid, []):
                        datavalue = claim.get("mainsnak", {}).get("datavalue")
                        if datavalue and datavalue.get("type") == "wikibase-entityid":
                            ids.append(datavalue["value"].get("id", ""))
                    return [x for x in ids if x]

                out[qid] = {"description": entity.get("descriptions", {}).get(language, {}).get("value", ""),"aliases": [a["value"] for a in entity.get("aliases", {}).get(language, [])],"P31": entity_ids("P31"),"P279": entity_ids("P279"), "sitelink": entity.get("sitelinks", {}).get(f"{language}wiki", {}).get("title", "")}
        return out

    def parse_datavalue(self, dv):
        if not dv:
            return None
        t = dv.get("type")
        v = dv.get("value")
        if t == "wikibase-entityid":
            return ["entity", v.get("id")]
        if t == "quantity":
            try:
                return ["quantity", float(v.get("amount"))]
            except (TypeError, ValueError):
                return None
        if t == "time":
            return ["time", v.get("time")]
        if t == "string":
            return ["string", v]
        if t == "monolingualtext":
            return ["string", v.get("text")]
        if t == "globecoordinate":
            return ["coord", [v.get("latitude"), v.get("longitude")]]
        return None

    def get_entity_claims(self, qids):
        out = {}
        qids = [q for q in dict.fromkeys(qids) if q]
        for i in range(0, len(qids), 50):
            chunk = qids[i:i + 50]
            data = self.api_get({"action": "wbgetentities", "ids": "|".join(chunk),
                         "props": "claims", "format": "json"})
            if not data:
                continue
            for qid, entity in data.get("entities", {}).items():
                props = {}
                for pid, claims in entity.get("claims", {}).items():
                    vals = []
                    for claim in claims:
                        parsed = self.parse_datavalue(claim.get("mainsnak", {}).get("datavalue"))
                        if parsed:
                            vals.append(parsed)
                    if vals:
                        props[pid] = vals
                out[qid] = props
        return out

    def get_labels(self, qids, language="en"):
        out = {}
        qids = [q for q in qids if q]
        for i in range(0, len(qids), 50):
            chunk = qids[i:i + 50]
            data = self.api_get({ "action": "wbgetentities", "ids": "|".join(chunk),  "props": "labels",  "languages": f"{language}|en",  "format": "json"})
            if not data:
                continue
            for qid, entity in data.get("entities", {}).items():
                labels = entity.get("labels", {})
                lab = labels.get(language) or labels.get("en")
                out[qid] = lab["value"] if lab else qid
        return out

