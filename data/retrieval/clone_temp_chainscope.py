import os
import json
import shutil
from pathlib import Path
from datasets import load_dataset
import subprocess


"""def download_gsm8k(save_dir="data/gsm8k"):
    print("Downloading GSM8K from Hugging Face...")
    dataset = load_dataset("gsm8k", "main")

    os.makedirs(save_dir, exist_ok=True)

    for split in ["train", "test"]:
        data = dataset[split]
        with open(os.path.join(save_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for item in data:
                json.dump(item, f)
                f.write("\n")
    print("GSM8K saved to", save_dir)"""


def clone_chainscope(save_dir="data/ivan_dataset"):
    repo_url = "https://github.com/jettjaniak/chainscope"
    target = "temp_chainscope_repo"

    print("Cloning Iván's Chainscope repo...")
    if os.path.exists(target):
        shutil.rmtree(target)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, target])

    os.makedirs(save_dir, exist_ok=True)

    """# Try copying over the comparative data (adjust if structure changes!)
    source_file = Path(target) / "data" / "comparative_pairs.jsonl"
    if source_file.exists():
        shutil.copy(source_file, Path(save_dir) / "comparative_pairs.jsonl")
        print("Comparative questions saved to", save_dir)
    else:
        print("comparative_pairs.jsonl not found in chainscope repo. Please check manually.")

    shutil.rmtree(target)"""


"""def clone_putnambench(save_dir="data/putnambench"):
    repo_url = "https://github.com/putnambench/putnambench"
    target = "temp_putnam_repo"

    print("Cloning PutnamBench repo...")
    if os.path.exists(target):
        shutil.rmtree(target)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, target])

    os.makedirs(save_dir, exist_ok=True)

    source_dir = Path(target) / "problems"
    if source_dir.exists():
        for fname in source_dir.glob("*.jsonl"):
            shutil.copy(fname, Path(save_dir) / fname.name)
        print("PutnamBench problems saved to", save_dir)
    else:
        print("⚠️ Problems directory not found in PutnamBench repo.")

    shutil.rmtree(target)"""


if __name__ == "__main__":
    print("Starting dataset download and setup...\n")
    #download_gsm8k()
    print()
    clone_chainscope()
    print()
    #clone_putnambench()
    #print("\nAll datasets downloaded.")
    print("\nACloned.")
