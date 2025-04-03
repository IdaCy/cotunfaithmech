import os
import yaml
import json

input_dir = "data/gt_YES_1"
output_path = "data/comparative_pairs.jsonl"

pairs = []

for fname in os.listdir(input_dir):
    if not fname.endswith(".yaml"):
        continue

    with open(os.path.join(input_dir, fname), "r") as f:
        data = yaml.safe_load(f)
        qid_map = data.get("question_by_qid", {})

        # Sort question IDs for consistency
        sorted_items = sorted(qid_map.items())
        for i in range(0, len(sorted_items) - 1, 2):  # step by 2
            qid1, q1data = sorted_items[i]
            qid2, q2data = sorted_items[i + 1]

            q1 = q1data.get("q_str")
            q2 = q2data.get("q_str")

            if q1 and q2:
                pairs.append({
                    "id": f"{fname}::{i//2}",
                    "q1": q1.strip(),
                    "q2": q2.strip(),
                    "a1": None,
                    "a2": None,
                    "cot1": None,
                    "cot2": None,
                    "flags": []
                })

with open(output_path, "w") as out:
    for pair in pairs:
        json.dump(pair, out)
        out.write("\n")

print(f"✅ Extracted {len(pairs)} comparative question pairs to {output_path}")
