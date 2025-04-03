import yaml
import json

input_path = "data/gsm8k.yaml"
output_path = "data/gsm8k.jsonl"

with open(input_path, "r") as f:
    data = yaml.safe_load(f)

problems = data["problems-by-qid"]

with open(output_path, "w") as out:
    for qid, item in problems.items():
        question = item.get("q-str")
        answer = item.get("answer-without-reasoning") or item.get("answer")
        if question and answer:
            json.dump({
                "id": qid,
                "question": question.strip(),
                "answer": answer.strip(),
                "cot": None,
                "predicted_answer": None,
                "faithfulness_flags": []
            }, out)
            out.write("\n")

print(f"✅ Converted {len(problems)} GSM8K entries to {output_path}")
