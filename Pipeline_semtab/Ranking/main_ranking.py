import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ranking import rank_folder

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

def launch_ranking(config_path="config_ranking.txt"):
    config = load_config(config_path)
    print("Start ranking")
    start_time = time.time()
    rank_folder(config)
    total_time = time.time() - start_time
    print(f"Total ranking time: {total_time / 60:.2f} minutes")
    print("End")

if __name__ == "__main__":
    launch_ranking(sys.argv[1] if len(sys.argv) > 1 else "config_ranking.txt")