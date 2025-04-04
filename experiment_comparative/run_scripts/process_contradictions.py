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

# Import our comprehensive logger initializer.
from utils.logger import init_logger

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
        out_cpu = out.detach().cpu() if isinstance(out, torch.Tensor) else None
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
    if logger:
        logger.info(f"Built prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if logger:
        logger.debug(f"Tokenized inputs: {inputs}")

    if HookedTransformer is not None and isinstance(hooked_model, HookedTransformer):
        if logger:
            logger.info("Attaching forward hooks for HookedTransformer.")
        if logger is not None:
            # Clear previous activations
            logger.debug("Clearing previous activations in ActivationLogger.")
        if logger is not None:
            # Passed logger here is from ActivationLogger if hooking is used.
            logger.clear()
        for name, module in hooked_model.modules.items():
            module.register_forward_hook(logger.fwd_hook)
            if logger:
                logger.debug(f"Attached hook to module: {name}")

    if logger:
        logger.info("Starting model generation.")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
    if logger:
        logger.info("Model generation completed.")

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if logger:
        logger.debug(f"Raw model output: {full_output}")
    if full_output.startswith(prompt):
        full_output = full_output[len(prompt):].strip()
        if logger:
            logger.debug("Removed prompt from the output.")

    return full_output


def run_contradiction_experiment(input_file, output_file, output_csv="",
                                 model=None, tokenizer=None,
                                 model_name="Qwen/Qwen1.5-1.8B",
                                 use_gpu=True):
    # Initialize comprehensive logging using our provided logger.
    log = init_logger("logs/comparative_mountain_heights.log")
    log.info("Started comparative experiment script.")

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    log.info(f"Using device: {device}")
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        log.debug(f"Ensured output directory exists for {output_file}")
    except Exception as e:
        log.exception(f"Failed to create output directory: {e}")

    if model is None or tokenizer is None:
        log.info(f"Loading model [{model_name}]...")
        from utils.load_model import load_model
        model, tokenizer = load_model(model_name=model_name, use_bfloat16=True, logger=log)
        model.to(device)
        model.eval()
        log.info("Model loaded and moved to device.")

    if HookedTransformer is not None:
        try:
            log.info("Attempting to load HookedTransformer for Qwen (if available).")
            hooked_model = HookedTransformer.from_pretrained(model_name, device=device)
            log.info("HookedTransformer loaded successfully.")
        except Exception as e:
            log.exception(f"Could not load HookedTransformer for {model_name}: {e}")
            hooked_model = None
    else:
        hooked_model = None
        log.warning("HookedTransformer not available; skipping hooking functionality.")

    # Use ActivationLogger if hooked_model is loaded.
    activation_logger = ActivationLogger() if hooked_model is not None else None

    results = []
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            log.info(f"Opened input file: {input_file}")
            for line_idx, line in enumerate(infile):
                line = line.strip()
                if not line:
                    log.debug(f"Skipping empty line at index {line_idx}.")
                    continue
                data = json.loads(line)
                log.debug(f"Parsed JSON from line {line_idx}: {data}")

                q1 = data.get("q1", "")
                q2 = data.get("q2", "")
                pair_id = data.get("id", f"line_{line_idx}")
                log.info(f"Processing pair ID: {pair_id}")
                log.debug(f"q1: {q1}")
                log.debug(f"q2: {q2}")

                # Run inference for Q1
                log.info(f"Running inference for Q1 of pair {pair_id}.")
                output1 = run_inference(
                    model=model,
                    tokenizer=tokenizer,
                    question_text=q1,
                    hooked_model=hooked_model,
                    logger=activation_logger,
                    device=device
                )
                cot1 = output1
                log.debug(f"Inference output for Q1 (pair {pair_id}): {cot1}")

                a1 = parse_answer_from_output(output1)
                log.info(f"Parsed answer for Q1 (pair {pair_id}): {a1}")

                if activation_logger is not None:
                    activations_q1 = activation_logger.recorded_activations.copy()
                    hook_file_q1 = f"./hook_logs/{pair_id}_q1_activations.pt"
                    torch.save(activations_q1, hook_file_q1)
                    log.info(f"Saved Q1 hook activations to {hook_file_q1}")

                # Run inference for Q2
                log.info(f"Running inference for Q2 of pair {pair_id}.")
                output2 = run_inference(
                    model=model,
                    tokenizer=tokenizer,
                    question_text=q2,
                    hooked_model=hooked_model,
                    logger=activation_logger,
                    device=device
                )
                cot2 = output2
                log.debug(f"Inference output for Q2 (pair {pair_id}): {cot2}")

                a2 = parse_answer_from_output(output2)
                log.info(f"Parsed answer for Q2 (pair {pair_id}): {a2}")

                if activation_logger is not None:
                    activations_q2 = activation_logger.recorded_activations.copy()
                    hook_file_q2 = f"./hook_logs/{pair_id}_q2_activations.pt"
                    torch.save(activations_q2, hook_file_q2)
                    log.info(f"Saved Q2 hook activations to {hook_file_q2}")

                flags = []
                if a1 == "yes" and a2 == "yes":
                    flags.append("contradictory-yes-yes")
                    log.warning(f"Pair {pair_id} flagged as contradictory (yes/yes).")
                elif a1 == "no" and a2 == "no":
                    flags.append("contradictory-no-no")
                    log.warning(f"Pair {pair_id} flagged as contradictory (no/no).")

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
                log.info(f"Finished processing pair {pair_id}.")
    except Exception as e:
        log.exception(f"Error processing input file: {e}")

    try:
        with open(output_file, 'w', encoding='utf-8') as fout:
            for rec in results:
                fout.write(json.dumps(rec) + "\n")
        log.info(f"Wrote {len(results)} results to {output_file}")
    except Exception as e:
        log.exception(f"Error writing output JSONL file: {e}")

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
            log.info(f"Wrote CSV file to {output_csv}")
        except Exception as e:
            log.exception(f"Error writing output CSV file: {e}")

    print(f"Done! Wrote {len(results)} lines to {output_file}")
    if output_csv:
        print(f"Also wrote CSV to {output_csv}")
    log.info("Comparative experiment script finished successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Qwen1.5-1.8B on mountain-height question pairs with chain-of-thought, storing results and hooking intermediate states.")
    parser.add_argument("--input_file", type=str, default="../data/mountain-heights.jsonl")
    parser.add_argument("--output_file", type=str, default="./mountain-heights_results.jsonl")
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-1.8B")
    parser.add_argument("--use_gpu", type=lambda x: x.lower() == 'true', default=True)
    args = parser.parse_args()

    run_contradiction_experiment(
        input_file=args.input_file,
        output_file=args.output_file,
        output_csv=args.output_csv,
        model_name=args.model_name,
        use_gpu=args.use_gpu
    )
