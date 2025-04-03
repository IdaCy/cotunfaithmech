import os
import json
from datasets import load_dataset

def save_gsm8k(split, save_path):
    dataset = load_dataset("gsm8k", "main")[split]
    with open(save_path, "w", encoding="utf-8") as f:
        for example in dataset:
            json.dump({
                "question": example["question"].strip(),
                "answer": example["answer"].strip()
            }, f)
            f.write("\n")

if __name__ == "__main__":
    os.makedirs("data/gsm8k", exist_ok=True)
    save_gsm8k("train", "data/gsm8k/train.jsonl")
    save_gsm8k("test", "data/gsm8k/test.jsonl")
    print("✅ GSM8K downloaded and saved.")
