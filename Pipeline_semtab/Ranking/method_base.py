from cta import cta_from_selection
from table_labels import TableLabels
from Pipeline_semtab.common.knowledge import WikidataClient
import cpa as cpa_mod

class AnnotationResult:
    def __init__(self):
        self.cea = {}
        self.cta = {}
        self.cpa = {}

class RankingSession:
    def __init__(self, table, config, llm=None, knowledge=None, result=None, labels=None):
        self.table = table
        self.config = config
        self.llm = llm
        self.knowledge = knowledge if knowledge is not None else WikidataClient(sleep=config.get("API_SLEEP", "0.1"))
        self.result = result if result is not None else AnnotationResult()
        self.language = config.get("LANGUAGE", "en")
        self.tasks = {t.strip().lower() for t in config.get("TASKS", "cea,cta,cpa").split(",") if t.strip()}
        self.labels = labels if labels is not None else TableLabels(self.knowledge, self.language)

    def selected_entities(self):
        nil_label = self.config.get("NIL_LABEL", "NIL").strip() or "NIL"
        return {cell: qid for cell, qid in self.result.cea.items() if qid and qid != nil_label}

    def rebuild_cta_from_selection(self):
        res = cta_from_selection(self.table.cand_df, self.selected_entities())
        if res is not None:
            self.table.cta_result = res
            self.table.cta_by_col = {c: (c, p31, p279) for c, p31, p279 in res}


def run_cpa(run, use_slm):
    ctx = run.table
    entities = run.selected_entities()
    for sub_col, obj_col in ctx.cpa_pairs:
        obj_is_entity = obj_col in ctx.cand_cols
        pid = cpa_mod.resolve_pair(
            sub_col, obj_col, ctx.n_rows, entities, ctx.data_df,
            obj_is_entity, llm=run.llm, use_slm=use_slm,
            col_header=ctx.col_header(obj_col),
            sub_type=run.labels.final_type(ctx, run.result, sub_col),
            obj_type=run.labels.final_type(ctx, run.result, obj_col) if obj_is_entity else "",
            knowledge=run.knowledge)
        if pid:
            run.result.cpa[(sub_col, obj_col)] = pid
