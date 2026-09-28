import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))
for stage in ("Preprocessing", "Candidate_Retrieval", "Ranking"):
    sys.path.insert(0, os.path.join(BASE_DIR, stage))

from main_preprocessing import launch_code_preprocessing
from main_candidate import launch_candidate_generation
from main_ranking import launch_ranking

STAGES = [
    ("preprocessing", launch_code_preprocessing),
    ("candidate", launch_candidate_generation),
    ("ranking", launch_ranking),
]

if __name__ == "__main__":
    known = [name for name, _ in STAGES]
    configs = {}
    for arg in sys.argv[1:]:
        if "=" not in arg:
            print(f"Expected <stage>=<config>, got: {arg}")
            print(f"Stages: {', '.join(known)}")
            sys.exit(1)
        stage, config_path = arg.split("=", 1)
        stage = stage.strip().lower()
        if stage not in known:
            print(f"Unknown stage '{stage}'. Stages: {', '.join(known)}")
            sys.exit(1)
        configs[stage] = config_path.strip()

    if not configs:
        print("No stages specified.")
        sys.exit(1)

    for config_path in configs.values():
        if not os.path.isfile(config_path):
            print(f"Config unfound: {config_path}")
            sys.exit(1)

    start = time.time()
    for stage, launch in STAGES:
        if stage not in configs:
            continue
        stage_start = time.time()
        launch(configs[stage])
        print(f"Total {stage} time: {(time.time() - stage_start) / 60:.2f} minutes")
    print(f"Total time: {(time.time() - start) / 60:.2f} minutes")
