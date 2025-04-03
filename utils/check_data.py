import json

def count_jsonl(path):
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        print(f"✅ {path}: {len(lines)} entries")
    except FileNotFoundError:
        print(f"❌ {path}: Not found")

if __name__ == "__main__":
    print("📊 Dataset Counts:")
    count_jsonl("data/gsm8k.jsonl")             # Restoration Errors
    count_jsonl("data/putnambench.jsonl")       # Unfaithful Shortcuts
    count_jsonl("data/comparative_pairs.jsonl") # Comparative contradictions
