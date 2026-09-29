from Pipeline_semtab.common.knowledge import WikidataClient

LIST_SEP = "|"
ENRICHMENT_COLUMNS = ["description", "aliases", "P31", "P279", "sitelink"]
def enrich_rows(rows, language="en", knowledge=None):
    if knowledge is None:
        knowledge = WikidataClient()
    qids = sorted({r["QID"] for r in rows if r.get("QID")})
    data = knowledge.get_entity_data(qids, language)

    for row in rows:
        info = data.get(row.get("QID", ""), {})
        row["description"] = info.get("description", "")
        row["aliases"] = LIST_SEP.join(info.get("aliases", []))
        row["P31"] = LIST_SEP.join(info.get("P31", []))
        row["P279"] = LIST_SEP.join(info.get("P279", []))
        row["sitelink"] = info.get("sitelink", "")
    return rows
def split_list(value):
    if value is None:
        return []
    return [part for part in str(value).split(LIST_SEP) if part and part != "nan"]