import os
import shutil
import subprocess
from pathlib import Path

def clone_and_generate_chainscope(save_path="data/ivan_dataset/comparative_pairs.jsonl"):
    tmp_dir = "temp_chainscope"
    repo_url = "https://github.com/jettjaniak/chainscope"

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    print("📥 Cloning Chainscope repo...")
    subprocess.run(["git", "clone", "--depth", "1", repo_url, tmp_dir], check=True)

    print("📦 Installing requirements (might need venv)...")
    subprocess.run(["pip", "install", "-r", "requirements.txt"], cwd=tmp_dir)

    print("⚙️  Running generation script...")
    subprocess.run(["python", "scripts/generate_comparative_data.py"], cwd=tmp_dir, check=True)

    print("📄 Copying comparative_pairs.jsonl...")
    output_path = Path(tmp_dir) / "data" / "comparative_pairs.jsonl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    shutil.copy(output_path, save_path)

    shutil.rmtree(tmp_dir)
    print("✅ Saved to", save_path)

if __name__ == "__main__":
    clone_and_generate_chainscope()
