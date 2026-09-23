import os

from output_writer import OutputWriter
from logger_ranking import log_vram, reset_peaks, log_tokens, reset_tokens, set_run_name
from wikidata_api_ranking import set_rate_limit
from method_base import TableContext
from scoring_method import build_scorer, get_scoring_method
import method_limited_slm as m_limited_slm
import method_full_slm as m_full_slm
import method_slm_context as m_slm_context

def annotator(method):
    return {"full_slm": m_full_slm.annotate,"slm_context": m_slm_context.annotate}.get(method, m_limited_slm.annotate)
def needs_llm(config, method):
    if method in ("full_slm", "slm_context") or get_scoring_method(config).requires_llm:
        return True
    flags = ("CEA_USE_SLM", "CTA_USE_SLM", "CPA_USE_SLM")
    return any(config.get(f, "True").lower() == "true" for f in flags)
def rank_folder(config):
    input_folder = config["INPUT_FOLDER"]
    preprocess_folder = config.get("PREPROCESS_FOLDER")
    if not preprocess_folder or not os.path.exists(preprocess_folder):
        raise ValueError("PREPROCESS_FOLDER must be specified and exist in the config.")
    output_folder = config.get("OUTPUT_FOLDER") or (input_folder + "_ranked")
    set_run_name(output_folder)

    method = config.get("METHOD", "limited_slm").lower()
    as_uri = config.get("ENTITY_AS_URI", "True").lower() == "true"
    write_header = config.get("WRITE_HEADER", "False").lower() == "true"
    row_offset = int(config.get("ROW_OFFSET", "1"))
    allow_nil = config.get("ALLOW_NIL", "False").lower() == "true"
    nil_label = (config.get("NIL_LABEL", "NIL").strip() or "NIL") if allow_nil else None

    writer = OutputWriter(output_folder, as_uri=as_uri, write_header=write_header,row_offset=row_offset, nil_label=nil_label)
    set_rate_limit(config.get("API_SLEEP", "0.1"))

    llm = None

    if needs_llm(config, method):
        from llm_code_ranking import LLMEngine
        reset_peaks()
        reset_tokens()
        llm = LLMEngine(config)
        log_vram("model_loaded")

    scorer = build_scorer(config, llm)
    annotate = annotator(method)
    print(f"Method: {method}")

    files = sorted(f for f in os.listdir(input_folder) if f.endswith(".csv"))
    llm_calls = 0
    for filename in files:
        input_path = os.path.join(input_folder, filename)
        preprocess_path = os.path.join(preprocess_folder,filename.replace("_candidates", ""))
        if not os.path.exists(preprocess_path):
            print(f"Preprocess file missing for {filename}, skipping.")
            continue
        print(f"Ranking {filename}")
        ctx = TableContext(input_path, preprocess_path, config, llm, writer, scorer)
        llm_calls += annotate(ctx) or 0
        if llm is not None:
            log_tokens(filename)

    if llm_calls:
        print(f"LLM disambiguation calls: {llm_calls}")
    if llm is not None:
        log_vram("ranking_done")
        log_tokens("ranking_done")
    writer.flush()
