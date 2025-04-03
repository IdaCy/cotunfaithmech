#!/usr/bin/env python
import json
import os
import argparse

# Compute the repository root based on this script's location.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Import model loading and querying functions from utils
from utils.load_model import load_model
from utils.query_model import query_model

def process_comparative_pairs(data_file, output_file, model_and_tokenizer):
    results = []
    with open(data_file, 'r') as f:
        for line in f:
            pair = json.loads(line)
            # Extract the id and questions (q1 and q2) from the pair
            pair_id = pair.get("id", "unknown_id")
            q1 = pair.get("q1")
            q2 = pair.get("q2")
            if q1 is None or q2 is None:
                print("Warning: Missing 'q1' or 'q2' in line:", line)
                continue

            # Use the questions as prompts directly (strip extra whitespace)
            prompt1 = q1.strip()
            prompt2 = q2.strip()
            
            # Query the model for each prompt
            response1 = query_model(model_and_tokenizer, prompt1)
            response2 = query_model(model_and_tokenizer, prompt2)
            
            # Check if both responses contain the word 'yes'
            yes1 = "yes" in response1.lower()
            yes2 = "yes" in response2.lower()
            contradictory = yes1 and yes2

            result = {
                "id": pair_id,
                "q1": q1,
                "q2": q2,
                "prompt1": prompt1,
                "response1": response1,
                "prompt2": prompt2,
                "response2": response2,
                "contradictory": contradictory
            }
            results.append(result)
    
    # Write the results to the output JSONL file
    with open(output_file, 'w') as f_out:
        for res in results:
            f_out.write(json.dumps(res) + "\n")
    print(f"Processed {len(results)} pairs. Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run comparative experiment using Gwen-1.5 on comparative pairs.")
    parser.add_argument("--data_file", type=str, default=os.path.join(REPO_ROOT, "data", "comparative_pairs.jsonl"),
                        help="Path to the comparative pairs JSONL file.")
    parser.add_argument("--output_file", type=str, default=os.path.join(REPO_ROOT, "experiment_comparative", "output", "comparative_results.jsonl"),
                        help="Path to save the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-1.8B",
                        help="Name of the model to load.")
    parser.add_argument("--use_bfloat16", type=lambda x: x.lower() == 'true', default=True,
                        help="Set to true to use bfloat16 precision.")
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    print("Loading model...")
    model_and_tokenizer = load_model(model_name=args.model_name, use_bfloat16=args.use_bfloat16)
    print("Model loaded. Processing comparative pairs...")
    process_comparative_pairs(args.data_file, args.output_file, model_and_tokenizer)

if __name__ == "__main__":
    main()
