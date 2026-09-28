import os
import pandas as pd
import multiprocessing
import queue

from Pipeline_semtab.common.generation import GenerationPrompt, build_generation_engine

from logger_preprocessing import log_vram, reset_peaks, add_tokens, log_tokens, reset_tokens, set_run_name

def build_typo_engine(config, device_id=0):
    options = {"MODEL_DTYPE": "float16", "MAX_CTX": "model", "LLM_BATCH_SIZE": "32",
               **config, "DEVICE_ID": device_id}
    return build_generation_engine(options, token_callback=add_tokens)

def parse_fewshot_examples(raw):
    if raw is None or not isinstance(raw, str):
        return raw
    examples = []
    for pair in raw.split(";"):
        if "->" not in pair:
            continue
        src, tgt = pair.split("->", 1)
        examples.append((src.strip(), tgt.strip()))
    return examples if examples else None

def build_prompts(texts, system_prompt, fewshot_examples=None, use_fewshot=True):
    prompts = []
    for text in texts:
        messages = [{"role": "system", "content": system_prompt}]
        shots = ""
        if use_fewshot and fewshot_examples is not None:
            for source, target in fewshot_examples:
                messages.append({"role": "user", "content": source})
                messages.append({"role": "assistant", "content": target})
                shots += f"Original: {source}\nCorrected: {target}\n"
        messages.append({"role": "user", "content": text})
        fallback = f"{system_prompt}\n{shots}Original: {text}\nCorrected:"
        prompts.append(GenerationPrompt(messages, fallback))
    return prompts

def clean_correction(correction, original):
    correction = correction.strip().strip('"').strip("'").strip()
    correction = correction.replace("\n", " ").strip()
    if correction.endswith(".") and not original.strip().endswith("."):
        correction = correction[:-1].strip()
    if not correction or len(correction) > len(original) * 3:
        return original
    return correction

def correct_typo_with_llm(text, engine, SYSTEM_PROMPT, max_new_tokens=64, batch_size=32,
                          max_ctx=None, use_fewshot=True, fewshot_examples=None):
    if not text:
        return []
    prompts = build_prompts(text, SYSTEM_PROMPT, fewshot_examples, use_fewshot)
    options = {"max_new_tokens": max_new_tokens, "batch_size": batch_size, "do_sample": False}
    if max_ctx is not None:
        options["max_ctx"] = max_ctx
    decoded = engine.generate(prompts, **options)
    if len(decoded) != len(text):
        raise ValueError("Typo correction must return one response per input")
    return [clean_correction(correction, original) for correction, original in zip(decoded, text)]

def process_file_df(df, engine, system_prompt, batch_size=32, max_new_tokens=64, use_fewshot=True, fewshot_examples=None):

    text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    text_cols = [c for c in text_cols if not str(c).startswith("Metadata:")]
    cache = {}

    for col in text_cols:
        s = df[col]
        s_norm = s.where(s.isna(), s.astype(str))
        uniques = pd.unique(s_norm.dropna())
        uniques = [u for u in uniques if u != ""]

        to_do = [u for u in uniques if u not in cache]

        corrected = correct_typo_with_llm(to_do,engine=engine,SYSTEM_PROMPT=system_prompt,batch_size=batch_size,max_new_tokens=max_new_tokens,use_fewshot=use_fewshot,fewshot_examples=fewshot_examples)

        for u, c in zip(to_do, corrected):
            cache[u] = c

        df[col] = s_norm.map(lambda x: cache.get(x, x) if pd.notna(x) and x != "" else x)

    return df

def process_single_file(input_folder, output_folder, file_name, engine, config, device_id=None):
    in_path = os.path.join(input_folder, file_name)
    out_path = os.path.join(output_folder, file_name)
    df = pd.read_csv(in_path)
    df = process_file_df(df, engine, config.get("SYSTEM_PROMPT"),
                         batch_size=int(config.get("LLM_BATCH_SIZE", "32")),
                         max_new_tokens=int(config.get("MAX_NEW_TOKENS", "64")),
                         use_fewshot=config.get("USE_FEWSHOT", "True").lower() == "true",
                         fewshot_examples=parse_fewshot_examples(config.get("FEWSHOT_EXAMPLES")))
    df.to_csv(out_path, index=False)
    log_tokens(file_name, device_id=device_id)

def worker_process(file_queue, input_folder, config, output_folder, device_id):
    set_run_name(output_folder)
    reset_peaks(device_id)
    reset_tokens()
    engine = build_typo_engine(config, device_id)
    print(f"[GPU {device_id}] model loaded on", getattr(engine, "device", device_id))
    log_vram("model_loaded", device_id=device_id)

    while True:
        try:
            file_name = file_queue.get_nowait()
        except queue.Empty:
            break
        try:
            process_single_file(input_folder, output_folder, file_name, engine, config, device_id)
        except Exception as e:
            print(f"[GPU {device_id}] ERROR on {file_name}: {e}")
    log_vram("typo_correction_done", device_id=device_id)
    log_tokens("typo_correction_done", device_id=device_id)

def process_folder(config, engine=None):
    need_correctypo = config.get("NEED_CORRECTTYPO", "True").lower() == "true"
    if not need_correctypo:
        print("Typo correction is disabled in the configuration.")
        return
    input_folder = config["INPUT_FOLDER"]
    output_folder = config.get("OUTPUT_FOLDER") or (input_folder + "_typo_corrected")
    os.makedirs(output_folder, exist_ok=True)
    set_run_name(output_folder)

    num_gpus = 1
    backend = config.get("GENERATION_ENGINE", "huggingface").strip().lower()
    if engine is None and backend == "huggingface":
        import torch
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            num_gpus = max(1, min(int(config.get("NUM_GPUS", "1")), available_gpus))
            print(f"Using {num_gpus} GPUs out of {available_gpus} available")

    csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
    if num_gpus == 1:
        reset_peaks()
        reset_tokens()
        if engine is None:
            engine = build_typo_engine(config, int(config.get("DEVICE_ID", "0")))
        print("model device:", getattr(engine, "device", backend))
        log_vram("model_loaded")
        for file_name in csv_files:
            process_single_file(input_folder, output_folder, file_name, engine, config)
        log_vram("typo_correction_done")
        log_tokens("typo_correction_done")
    else:
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Manager() as manager:
            file_queue = manager.Queue()
            for file_name in csv_files:
                file_queue.put(file_name)
            processes = []
            for gpu_id in range(num_gpus):
                p = multiprocessing.Process(target=worker_process,
                    args=(file_queue, input_folder, config, output_folder, gpu_id))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            for gpu_id, p in enumerate(processes):
                if p.exitcode != 0:
                    print(f"WARNING: worker GPU {gpu_id} failed with code {p.exitcode}")
        print("All files processed successfully")
