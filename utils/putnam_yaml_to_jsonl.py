import yaml
import json

input_path = "data/minimal_fork_of_putnambench_with_clear_answers.yaml"
output_path = "data/putnambench.jsonl"

with open(input_path, "r") as f:
    data = yaml.safe_load(f)

with open(output_path, "w") as out:
    count = 0
    for i, item in enumerate(data):
        question = item.get("informal_statement")
        answer = item.get("informal_solution")
        if question and answer:
            json.dump({
                "id": f"putnam_{i:04d}",
                "question": question.strip(),
                "answer": answer.strip(),
                "cot": None,
                "predicted_answer": None,
                "faithfulness_flags": []
            }, out)
            out.write("\n")
            count += 1

print(f"✅ Converted {count} Putnam problems to {output_path}")
