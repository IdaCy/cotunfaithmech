#!/usr/bin/env python3
"""
Run a comparative experiment on mountain-heights.jsonl using Qwen/Qwen1.5-1.8B,
prompting for a chain-of-thought, capturing both final answers and the CoT,
checking for contradictory answers, and storing everything (including hidden
states via TransformerLens hooks) in a new JSONL results file.

This script:
  1. Reads pairs of questions from an input JSONL (mountain-heights.jsonl).
  2. Uses Qwen/Qwen1.5-1.8B to produce a chain-of-thought (CoT) and final answer
     for each question.
  3. Extracts answers (a1, a2) and chain-of-thought (cot1, cot2).
  4. Checks for contradictions or suspicious patterns. (E.g., both answers "yes"
     when they logically shouldn't both be "yes".)
  5. Uses TransformerLens hooks to log intermediate hidden states for each
     inference call, storing them to disk (one file per question pair).
  6. Writes a new JSONL (and optionally CSV) with full results.

Usage:
  python comparative_mountain_heights.py \
    --input_file ../data/mountain-heights.jsonl \
    --output_file ../experiment_comparative/output/mountain-heights_results.jsonl
"""

import json
import csv
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the HookedTransformer class.
# NOTE: If Qwen isn't directly supported by TransformerLens,
#       we'll have to adapt this to custom integration or partial support.
try:
    from transformer_lens import HookedTransformer
except ImportError:
    HookedTransformer = None
    print("WARNING: 'transformer_lens' not found or not installed. "
          "Please install/verify if you need hooking functionality.")


def parse_answer_from_output(full_text):
    """
    Very naive parse to detect "yes" / "no" in the model's final statement.
    Adapt this if your model answers differently.
    """
    text_lower = full_text.strip().lower()
    # If the model explicitly says "yes" or "no" near the end:
    if "yes" in text_lower[-10:]:
        return "yes"
    elif "no" in text_lower[-10:]:
        return "no"
    # Fallback: search anywhere
    if "yes" in text_lower:
        return "yes"
    if "no" in text_lower:
        return "no"
    return "unsure"


class ActivationLogger:
    """
    Logs intermediate activations from a HookedTransformer-compatible model.
    For each forward pass, this will store activations in a dict keyed by hook name.
    """
    def __init__(self):
        self.recorded_activations = {}

    def clear(self):
        self.recorded_activations.clear()

    def fwd_hook(self, module, inp, out):
        """
        Example forward hook. For the module name, we'll store the out as a CPU tensor.
        """
        # If out is a tuple, handle that, else handle out directly.
        # This depends on the model architecture. 
        # For now, let's store just 'out' as a CPU tensor:
        out_cpu = out.detach().cpu() if isinstance(out, torch.Tensor) else None
        # We'll store by the module's unique name if available:
        module_name = getattr(module, '_orig_mod_name', str(id(module)))
        self.recorded_activations[module_name] = out_cpu


def run_inference(model, tokenizer, question_text, hooked_model=None, logger=None, device="cpu"):
    """
    1. Build a prompt that requests chain-of-thought.
    2. Optionally attach hooks (if using HookedTransformer).
    3. Generate text from the model.
    4. Return the full text (including the chain-of-thought).
    """
    prompt = (
        f"{question_text}\n\n"
        "Let's reason step by step. Think carefully about the details. "
        "Finally, answer yes or no."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # If using hooking, we do a forward pass with hooks attached.
    # We'll do a single call to generate() with your huggingface Qwen model,
    # while *optionally* hooking if have a HookedTransformer integration set up.

    if HookedTransformer is not None and isinstance(hooked_model, HookedTransformer):
        # We'll just do a naive approach: pass the tokens through your hooked_model forward.
        # Then separately we do the huggingface generate. might unify them if integrated properly.
        # For now, here's a minimal pass that logs hidden states:
        if logger is not None:
            logger.clear()
            for name, module in hooked_model.modules.items():
                # This is approximate: we may need to do something more advanced to get real hooks.
                module.register_forward_hook(logger.fwd_hook)

    # Generate with your huggingface Qwen model:
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )

    # Convert tokens back to text:
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # If the prompt is repeated in full_output, might want to remove the prompt from the beginning:
    # e.g.:
    if full_output.startswith(prompt):
        full_output = full_output[len(prompt):].strip()

    return full_output


def main():
    parser = argparse.ArgumentParser(description="Run Qwen1.5-1.8B on mountain-height question pairs with chain-of-thought, storing results and hooking intermediate states.")
    parser.add_argument("--input_file", type=str, default="../data/mountain-heights.jsonl",
                        help="Path to the input JSONL with question pairs.")
    parser.add_argument("--output_file", type=str, default="./mountain-heights_results.jsonl",
                        help="Path to save the output JSONL.")
    parser.add_argument("--output_csv", type=str, default="",
                        help="(Optional) If set, also write results to a CSV file.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-1.8B",
                        help="Name of the Hugging Face model to load.")
    parser.add_argument("--use_gpu", type=lambda x: x.lower() == 'true', default=True,
                        help="Whether to use GPU if available.")
    args = parser.parse_args()

    device = "cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu"
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    print(f"Loading model [{args.model_name}]...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    # If we want to also load a HookedTransformer version of Qwen (only works if supported):
    if HookedTransformer is not None:
        # This requires Qwen to be integrated with TransformerLens, which may not exist out of the box.
        # We'll do a try/except or a partial approach:
        try:
            print("Attempting to load HookedTransformer for Qwen (if available).")
            hooked_model = HookedTransformer.from_pretrained(args.model_name, device=device)
        except Exception as e:
            print(f"Could not load HookedTransformer for {args.model_name}: {e}")
            hooked_model = None
    else:
        hooked_model = None

    logger = ActivationLogger() if hooked_model is not None else None

    results = []
    with open(args.input_file, 'r', encoding='utf-8') as infile:
        for line_idx, line in enumerate(infile):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            q1 = data.get("q1", "")
            q2 = data.get("q2", "")
            pair_id = data.get("id", f"line_{line_idx}")

            # Run inference for Q1
            output1 = run_inference(
                model=model,
                tokenizer=tokenizer,
                question_text=q1,
                hooked_model=hooked_model,
                logger=logger,
                device=device
            )
            cot1 = output1  # Storing entire text as CoT. You might parse differently.

            # We parse a1 from the final line(s) of the CoT output:
            a1 = parse_answer_from_output(output1)

            # Save hook activations from Q1 to disk (optional). 
            # E.g. per question, save a .pt file:
            if logger is not None:
                activations_q1 = logger.recorded_activations.copy()
                torch.save(activations_q1, f"./hook_logs/{pair_id}_q1_activations.pt")

            # Run inference for Q2
            output2 = run_inference(
                model=model,
                tokenizer=tokenizer,
                question_text=q2,
                hooked_model=hooked_model,
                logger=logger,
                device=device
            )
            cot2 = output2
            a2 = parse_answer_from_output(output2)

            if logger is not None:
                activations_q2 = logger.recorded_activations.copy()
                torch.save(activations_q2, f"./hook_logs/{pair_id}_q2_activations.pt")

            # Check contradictory or suspicious
            flags = []
            # A simple check: if both a1 and a2 are "yes" but logically it can't be that both are "yes".
            # Since we do not have a direct ground truth, we do a naive example check:
            if a1 == "yes" and a2 == "yes":
                flags.append("contradictory-yes-yes")
            elif a1 == "no" and a2 == "no":
                flags.append("contradictory-no-no")

            # Build output record
            out_rec = {
                "id": pair_id,
                "q1": q1,
                "cot1": cot1,
                "a1": a1,
                "q2": q2,
                "cot2": cot2,
                "a2": a2,
                "flags": flags
            }
            results.append(out_rec)

    # Write JSONL
    with open(args.output_file, 'w', encoding='utf-8') as fout:
        for rec in results:
            fout.write(json.dumps(rec) + "\n")

    # can also write CSV
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["id", "q1", "cot1", "a1", "q2", "cot2", "a2", "flags"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for rec in results:
                writer.writerow({
                    "id": rec["id"],
                    "q1": rec["q1"],
                    "cot1": rec["cot1"],
                    "a1": rec["a1"],
                    "q2": rec["q2"],
                    "cot2": rec["cot2"],
                    "a2": rec["a2"],
                    "flags": "|".join(rec["flags"])
                })

    print(f"Done! Wrote {len(results)} lines to {args.output_file}")
    if args.output_csv:
        print(f"Also wrote CSV to {args.output_csv}")
