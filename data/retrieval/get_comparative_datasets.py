import os
import json
import yaml
from pathlib import Path

# Define which property IDs we care about
TARGET_CATEGORIES = {
    "mountain-heights",
    "first-flights",
    "satellite-launches"
}

# Point to where the chainscope data is
INPUT_DIR = "temp_chainscope_repo/chainscope/data/questions/gt_NO_1"
OUTPUT_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

converted_count = 0

for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".yaml"):
        continue

    for cat in TARGET_CATEGORIES:
        if fname.startswith(cat + "_gt_NO_1_"):
            input_path = os.path.join(INPUT_DIR, fname)
            output_path = os.path.join(OUTPUT_DIR, f"{cat}.jsonl")

            with open(input_path, "r") as f:
                data = yaml.safe_load(f)

            qid_map = data.get("question_by_qid", {})
            sorted_items = sorted(qid_map.items())

            with open(output_path, "w") as out:
                for i in range(0, len(sorted_items) - 1, 2):
                    qid1, q1data = sorted_items[i]
                    qid2, q2data = sorted_items[i + 1]

                    q1 = q1data.get("q_str")
                    q2 = q2data.get("q_str")

                    if q1 and q2:
                        json.dump({
                            "id": f"{fname}::{i//2}",
                            "q1": q1.strip(),
                            "q2": q2.strip(),
                            "a1": None,
                            "a2": None,
                            "cot1": None,
                            "cot2": None,
                            "flags": []
                        }, out)
                        out.write("\n")
                        converted_count += 1

print(f"Converted {converted_count} question pairs to JSONL in '{OUTPUT_DIR}/'")
