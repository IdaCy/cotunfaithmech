#!/usr/bin/env python3
"""
Run a comparative experiment on mountain-heights.jsonl using Qwen/Qwen1.5-1.8B,
prompting for a chain-of-thought, capturing both final answers and the CoT,
checking for contradictory answers, and storing everything (including hidden
states via TransformerLens hooks) in a new JSONL results file.

Steps:
  1. Reads pairs of questions from an input JSONL (mountain-heights.jsonl).
  2. Uses Qwen/Qwen1.5-1.8B to produce a chain-of-thought (CoT) and final answer
     for each question.
  3. Extracts answers (a1, a2) and chain-of-thought (cot1, cot2).
  4. Checks for contradictions (e.g. both answers "yes" when they cannot both be yes).
  5. (Optional) Uses TransformerLens to log hidden states to disk.
  6. Writes a new JSONL (and optionally CSV) with the results.

You can pass a logger from outside to log everything. If none is provided, no logs will be emitted.
"""

import json
import csv
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Attempt to import HookedTransformer from TransformerLens
try:
    from transformer_lens import HookedTransformer
except ImportError:
    HookedTransformer = None
    print("WARNING: 'transformer_lens' not installed. Hooking won't be available.")


def parse_answer_from_output(full_text):
    """
    Very naive parse to detect "yes" / "no" in the model's final statement.
    Adapt this if your model answers differently.
    """
    text_lower = full_text.strip().lower()
    # Check final 10 characters for "yes" or "no"
    if "yes" in text_lower[-10:]:
        return "yes"
    elif "no" in text_lower[-10:]:
        return "no"
    # If not in the last 10 chars, check anywhere
    if "yes" in text_lower:
        return "yes"
    if "no" in text_lower:
        return "no"
    return "unsure"


class ActivationLogger:
    """
    Logs intermediate activations from a HookedTransformer-compatible model.
    For each forward pass, stores them in self.recorded_activations.
    """
    def __init__(self):
        self.recorded_activations = {}

    def clear(self):
        self.recorded_activations.clear()

    def fwd_hook(self, module, inp, out):
        out_cpu = out.detach().cpu() if isinstance(out, torch.Tensor) else None
        module_name = getattr(module, '_orig_mod_name', str(id(module)))
        self.recorded_activations[module_name] = out_cpu


def run_inference(
    model,
    tokenizer,
    question_text,
    hooked_model=None,
    py_logger=None,          # regular Python logger from outside
    activation_logger=None,  # hooking object, if any
    device="cpu"
):
    """
    1. Build a prompt that requests chain-of-thought.
    2. Optionally attach hooks if using HookedTransformer.
    3. Generate text from the model.
    4. Return the full text (including chain-of-thought).
    """

    # Build the prompt
    prompt = (
        f"{question_text}\n\n"
        "Let's reason step by step. Think carefully about the details. "
        "Finally, answer yes or no."
    )
    if py_logger:
        py_logger.info(f"Prompt: {prompt}")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if py_logger:
        py_logger.debug(f"Tokenized inputs: {inputs}")

    # If hooking is available, attach hooks
    if HookedTransformer is not None and isinstance(hooked_model, HookedTransformer):
        if py_logger:
            py_logger.info("Attaching forward hooks for HookedTransformer.")
        if activation_logger is not None:
            activation_logger.clear()
        for name, module in hooked_model.modules.items():
            module.register_forward_hook(activation_logger.fwd_hook)
            if py_logger:
                py_logger.debug(f"Attached hook to module: {name}")

    # Generate
    if py_logger:
        py_logger.info("Starting model.generate()")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
    if py_logger:
        py_logger.info("model.generate() done.")

    # Decode
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if py_logger:
        py_logger.debug(f"Raw output: {full_output}")

    # Remove the prompt from the front if it appears
    if full_output.startswith(prompt):
        full_output = full_output[len(prompt):].strip()
        if py_logger:
            py_logger.debug("Stripped the prompt from beginning of output.")

    return full_output


def run_contradiction_experiment(
    input_file,
    output_file,
    output_csv="",
    model=None,
    tokenizer=None,
    model_name="Qwen/Qwen1.5-1.8B",
    use_gpu=True,
    py_logger=None
):
    """
    Reads question pairs from input_file, runs Qwen1.5-1.8B to get CoT answers,
    checks for contradictions, and writes results to output_file (and optionally CSV).

    If py_logger is given, logs are written there. If not, no logs are emitted.
    """

    if py_logger:
        py_logger.info("Starting run_contradiction_experiment...")

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    if py_logger:
        py_logger.info(f"Using device: {device}")

    # Ensure output directory exists
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if py_logger:
            py_logger.debug(f"Created/checked directory for {output_file}")
    except Exception as e:
        if py_logger:
            py_logger.exception(f"Error creating directory for output: {e}")

    # Load model if not provided
    if model is None or tokenizer is None:
        if py_logger:
            py_logger.info(f"Loading model: {model_name}")
        from utils.load_model import load_model
        model, tokenizer = load_model(
            model_name=model_name,
            use_bfloat16=True,
            logger=py_logger  # Pass the same logger so loading logs appear too
        )
        model.to(device)
        model.eval()
        if py_logger:
            py_logger.info("Model loaded and moved to device.")

    # Attempt to load HookedTransformer
    if HookedTransformer is not None:
        hooked_model = None
        try:
            if py_logger:
                py_logger.info("Loading HookedTransformer (Qwen).")
            hooked_model = HookedTransformer.from_pretrained(model_name, device=device)
            if py_logger:
                py_logger.info("HookedTransformer loaded successfully.")
        except Exception as ex:
            if py_logger:
                py_logger.exception(f"Could not load HookedTransformer: {ex}")
    else:
        hooked_model = None
        if py_logger:
            py_logger.warning("HookedTransformer not installed; skipping hooking.")

    activation_logger = ActivationLogger() if hooked_model else None

    # Main loop
    results = []
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            if py_logger:
                py_logger.info(f"Reading input from {input_file}")
            for line_idx, line in enumerate(infile):
                line = line.strip()
                if not line:
                    if py_logger:
                        py_logger.debug(f"Skipping empty line {line_idx}")
                    continue

                data = json.loads(line)
                if py_logger:
                    py_logger.debug(f"Line {line_idx} data: {data}")

                pair_id = data.get("id", f"line_{line_idx}")
                q1 = data.get("q1", "")
                q2 = data.get("q2", "")

                if py_logger:
                    py_logger.info(f"Processing pair {pair_id}")

                # Q1 inference
                output1 = run_inference(
                    model=model,
                    tokenizer=tokenizer,
                    question_text=q1,
                    hooked_model=hooked_model,
                    py_logger=py_logger,
                    activation_logger=activation_logger,
                    device=device
                )
                a1 = parse_answer_from_output(output1)

                # If hooking is on, save Q1 activations
                if activation_logger:
                    hook_file_q1 = f"./hook_logs/{pair_id}_q1_activations.pt"
                    torch.save(activation_logger.recorded_activations.copy(), hook_file_q1)
                    if py_logger:
                        py_logger.info(f"Saved Q1 hooks to {hook_file_q1}")

                # Q2 inference
                output2 = run_inference(
                    model=model,
                    tokenizer=tokenizer,
                    question_text=q2,
                    hooked_model=hooked_model,
                    py_logger=py_logger,
                    activation_logger=activation_logger,
                    device=device
                )
                a2 = parse_answer_from_output(output2)

                # If hooking is on, save Q2 activations
                if activation_logger:
                    hook_file_q2 = f"./hook_logs/{pair_id}_q2_activations.pt"
                    torch.save(activation_logger.recorded_activations.copy(), hook_file_q2)
                    if py_logger:
                        py_logger.info(f"Saved Q2 hooks to {hook_file_q2}")

                # Contradiction check
                flags = []
                if a1 == "yes" and a2 == "yes":
                    flags.append("contradictory-yes-yes")
                    if py_logger:
                        py_logger.warning(f"Pair {pair_id} flagged contradictory (yes/yes)")
                elif a1 == "no" and a2 == "no":
                    flags.append("contradictory-no-no")
                    if py_logger:
                        py_logger.warning(f"Pair {pair_id} flagged contradictory (no/no)")

                results.append({
                    "id": pair_id,
                    "q1": q1,
                    "cot1": output1,
                    "a1": a1,
                    "q2": q2,
                    "cot2": output2,
                    "a2": a2,
                    "flags": flags
                })
                if py_logger:
                    py_logger.info(f"Done with {pair_id}")

    except Exception as ex:
        if py_logger:
            py_logger.exception(f"Error reading/processing input: {ex}")

    # Write JSONL
    try:
        with open(output_file, 'w', encoding='utf-8') as fout:
            for rec in results:
                fout.write(json.dumps(rec) + "\n")
        if py_logger:
            py_logger.info(f"Wrote {len(results)} results to {output_file}")
    except Exception as ex:
        if py_logger:
            py_logger.exception(f"Error writing JSONL: {ex}")

    # Optionally write CSV
    if output_csv:
        try:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
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
            if py_logger:
                py_logger.info(f"Wrote CSV: {output_csv}")
        except Exception as ex:
            if py_logger:
                py_logger.exception(f"Error writing CSV: {ex}")

    print(f"Done! Wrote {len(results)} lines to {output_file}")
    if output_csv:
        print(f"Also wrote CSV to {output_csv}")
    if py_logger:
        py_logger.info("run_contradiction_experiment finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen1.5-1.8B on mountain-height question pairs with chain-of-thought.")
    parser.add_argument("--input_file", type=str, default="../data/mountain-heights.jsonl")
    parser.add_argument("--output_file", type=str, default="./mountain-heights_results.jsonl")
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-1.8B")
    parser.add_argument("--use_gpu", type=lambda x: x.lower() == 'true', default=True)
    args = parser.parse_args()

    # No logger is created or used here by default.
    # If you run from CLI, no logs will be displayed unless you integrate it differently.
    run_contradiction_experiment(
        input_file=args.input_file,
        output_file=args.output_file,
        output_csv=args.output_csv,
        model_name=args.model_name,
        use_gpu=args.use_gpu,
        py_logger=None  # no logs from CLI usage
    )
