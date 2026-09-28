import os
import sys
import time

# Stage modules import their siblings as top-level modules (e.g. "from ranking import ..."),
# so each stage folder must be on sys.path. Module names are unique across stages.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))
for stage in ("Preprocessing", "Candidate_Retrieval", "Ranking"):
    sys.path.insert(0, os.path.join(BASE_DIR, stage))

from main_preprocessing import launch_code_preprocessing
from main_candidate import launch_candidate_generation
from main_ranking import launch_ranking

if __name__ == "__main__":
    args = sys.argv[1:]
    preprocessing_cfg = args[0] if len(args) > 0 else "config/config_preprocessing.txt"
    candidate_cfg = args[1] if len(args) > 1 else "config/config_candidate.txt"
    ranking_cfg = args[2] if len(args) > 2 else "config/config_ranking.txt"

    start = time.time()
    launch_code_preprocessing(preprocessing_cfg)
    t1 = time.time()
    print(f"Total preprocessing time: {(t1 - start) / 60:.2f} minutes")
    launch_candidate_generation(candidate_cfg)
    t2 = time.time()
    print(f"Total candidate retrieval time: {(t2 - t1) / 60:.2f} minutes")
    launch_ranking(ranking_cfg)
    t3 = time.time()
    print(f"Total ranking time: {(t3 - t2) / 60:.2f} minutes")
    print(f"Total pipeline time: {(t3 - start) / 60:.2f} minutes")
