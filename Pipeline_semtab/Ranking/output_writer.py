import os
import pandas as pd

ENTITY_URI = "http://www.wikidata.org/entity/"
PROP_URI = "http://www.wikidata.org/prop/direct/"

class OutputWriter:
    def __init__(self, output_folder, as_uri=True, write_header=False, row_offset=1,
                 nil_label=None):
        self.output_folder = output_folder
        self.as_uri = as_uri
        self.write_header = write_header
        self.row_offset = row_offset
        self.nil_label = nil_label
        os.makedirs(output_folder, exist_ok=True)
        self.cea = []
        self.cta = []
        self.cpa = []

    def fmt(self, ident, prefix=ENTITY_URI):
        if not ident:
            return ""
        if self.nil_label and ident == self.nil_label:
            return ident
        return prefix + ident if self.as_uri else ident
    def add_cea(self, tab_id, row_id, col_id, qid):
        if not qid:
            return
        self.cea.append({"tab_id": tab_id, "row_id": int(row_id) + self.row_offset,"col_id": int(col_id), "entity": self.fmt(qid)})
    def add_cta(self, tab_id, col_id, qid):
        if not qid:
            return
        self.cta.append({"tab_id": tab_id, "col_id": int(col_id),"annotation": self.fmt(qid)})
    def add_cpa(self, tab_id, sub_col_id, obj_col_id, pid):
        if not pid:
            return
        self.cpa.append({"tab_id": tab_id, "sub_col_id": int(sub_col_id),"obj_col_id": int(obj_col_id),"property": self.fmt(pid, PROP_URI)})
    def add_result(self, tab_id, result, tasks=("cea", "cta", "cpa")):
        if "cea" in tasks:
            for (row, col), qid in result.cea.items():
                self.add_cea(tab_id, row, col, qid)
        if "cta" in tasks:
            for col, qid in result.cta.items():
                self.add_cta(tab_id, col, qid)
        if "cpa" in tasks:
            for (sub_col, obj_col), pid in result.cpa.items():
                self.add_cpa(tab_id, sub_col, obj_col, pid)

    def write(self, rows, columns, name):
        path = os.path.join(self.output_folder, name)
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False, header=self.write_header)
        return path
    def flush(self):
        cea_path = self.write(self.cea, ["tab_id", "row_id", "col_id", "entity"], "cea.csv")
        cta_path = self.write(self.cta, ["tab_id", "col_id", "annotation"], "cta.csv")
        cpa_path = self.write(self.cpa, ["tab_id", "sub_col_id", "obj_col_id", "property"], "cpa.csv")
        print(f"Wrote {len(self.cea)} CEA, {len(self.cta)} CTA, {len(self.cpa)} CPA rows")
        return cea_path, cta_path, cpa_path

