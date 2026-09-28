import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hasAnnotation import hasAnnotation
from typo_method import process_folder as typo_process_folder
from noise import cleanup_folder as noise_cleanup_folder
import time

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

def launch_code_preprocessing(config_path="config_preprocessing.txt"):
    config = load_config(config_path)

    print("Start")
    start_time = time.time()
    noise_cleanup_folder(config)
    # The next stages read the previous stage's output, not the raw input
    config["INPUT_FOLDER"] = config["OUTPUT_FOLDER"]
    start_time2 = time.time()
    print(f"Noise removal time: {(start_time2 - start_time)/60:.2f} minutes")
    typo_process_folder(config)
    print(f"Typo correction time: {(time.time() - start_time2)/60:.2f} minutes")
    hasAnnotation(config)
    total_time = time.time() - start_time
    print(f"Total preprocessing time: {total_time/60:.2f} minutes")

    print("End")
    

if __name__ == "__main__":
    launch_code_preprocessing(sys.argv[1] if len(sys.argv) > 1 else "config_preprocessing.txt")