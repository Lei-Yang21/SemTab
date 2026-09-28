import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from candidate_retrieval import candidate_retrieval_folder

def load_config(path):
    config = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(f"Invalid config line: {line}")
            key, value = line.split(":", 1)
            config[key.strip()] = value.strip().replace("\\n", "\n")
    return config


def launch_candidate_generation(config_path="config_candidate.txt"):
    config = load_config(config_path)
    input_folder = config["INPUT_FOLDER"]
    print("Start")
    start_time = time.time()
    candidate_retrieval_folder(input_folder, config)
    total_time = time.time() - start_time
    print(f"Total candidate generation time: {total_time/60:.2f} minutes")
    print("End")

if __name__ == "__main__":
    launch_candidate_generation(sys.argv[1] if len(sys.argv) > 1 else "config_candidate.txt")