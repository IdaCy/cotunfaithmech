# File: experiment_comparative/run_scripts/inference.py

"""
Script for running inference on Gwen-1.5 with chain-of-thought reasoning.
Captures internal activations (via standard PyTorch forward hooks) for mechanistic interpretability,
enabling tasks like:
- Checking contradictory steps, hallucinated facts, correctness
- Identifying patterns of unfaithful reasoning
- Hypothesizing circuit/head importance and patching hidden states
- Using specialized AEs/transcoders (e.g., Eleuther SAEs) to probe features

Usage (from a Jupyter notebook cell):
------------------------------------------------
from experiment_comparative.run_scripts.inference import run_gwen

run_gwen(
    model=model,  # MUST be on the same device as you intend to run generation
    tokenizer=tokenizer,
    logger=logger,
    input_file="path/to/input_data.jsonl",
    output_dir="path/to/output",
    use_transformer_lens=True  # or False if you do not want hooking
)
------------------------------------------------

Data format Example (one record):
{
  "id": "...",
  "q1": "Is X bigger than Y?",
  "q2": "Is Y bigger than X?",
  "a1": null,
  "a2": null,
  "cot1": null,
  "cot2": null,
  "flags": [...]
}
"""

import os
import json
import torch
from typing import Any, Dict, List
from torch.utils.data import DataLoader


###############################################################################
# 1) Dataset + Collate
###############################################################################

class CompareDataset(torch.utils.data.Dataset):
    """
    Basic dataset to load question pairs from a JSONL file.
    Each line might have 'q1', 'q2', etc. We'll run two separate inferences.
    """
    def __init__(self, jsonl_file: str):
        self.samples = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                self.samples.append(data)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

def no_collation_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Custom collation function that returns the batch as-is,
    preventing PyTorch from trying to stack or combine fields.
    """
    return batch


###############################################################################
# 2) Main Inference Routine
###############################################################################

def run_inf(
    model,
    tokenizer,
    logger,
    input_file: str,
    output_dir: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_transformer_lens: bool = True,  # We interpret 'True' as "attach standard PyTorch hooks"
    batch_size: int = 4
) -> None:
    """
    Runs inference with Gwen-1.5, generating chain-of-thought (CoT) reasoning
    on each example from a JSONL dataset, capturing internal activations
    (via standard PyTorch forward hooks) if requested.

    For each sample that has q1 and q2, we generate a chain-of-thought for both
    questions (independently). The results for q2 appear as 'cot2', 'final_answer2', etc.

    Args:
        model: The Gwen-1.5 model object on the correct device (e.g., GPU).
        tokenizer: The tokenizer for Gwen-1.5.
        logger: A logger instance for printing or saving logs.
        input_file: Path to the JSONL file with 'q1', 'q2', etc.
        output_dir: Directory path where the output JSONL is written.
        max_new_tokens: Maximum tokens to generate per query.
        temperature: Sampling temperature for generation.
        top_p: Nucleus sampling parameter for generation.
        use_transformer_lens: If True, attach PyTorch forward hooks to capture activations.
        batch_size: We'll do partial batching for reading the data, but each item is processed individually for hooking.
    """
    # --------------------------------------------------------------------
    # 1) Prepare Output
    # --------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Could not find input file at: {input_file}")
    
    dataset = CompareDataset(input_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=no_collation_fn
    )
    logger.info(f"Loaded {len(dataset)} samples from {input_file}")
    
    # Ensure model is on correct device
    device = next(model.parameters()).device

    # --------------------------------------------------------------------
    # 2) Hooking Logic (Standard PyTorch)
    # --------------------------------------------------------------------
    # We'll define functions to attach & remove forward hooks
    # We store activations in a user-defined dictionary per sample.
    def attach_hooks(forward_store: Dict[str, torch.Tensor]):
        """
        Attach forward hooks to all relevant submodules in 'model'.
        For each module, store final output in forward_store[module_name].
        Adjust the module selection as needed for your architecture.
        """
        # We'll keep track of handles so we can remove them.
        hook_handles = []

        for name, module in model.named_modules():
            # For example, if you only want "transformer" layers, 
            # you could do something like:
            if "transformer" in name or "decoder" in name or "encoder" in name:
                # Make a local hook that captures out
                def hook_fn(mod, inp, out, mod_name=name):
                    forward_store[mod_name] = out.detach().cpu()
                handle = module.register_forward_hook(hook_fn)
                hook_handles.append(handle)
        return hook_handles

    def remove_hooks(handles):
        """Remove all previously attached forward hooks."""
        for h in handles:
            h.remove()

    # --------------------------------------------------------------------
    # 3) Generation Helper
    # --------------------------------------------------------------------
    def generate_cot(question_text: str) -> Dict[str, Any]:
        """
        Generate chain-of-thought (step-by-step explanation) plus final answer for question_text.
        No leftover context from prior calls. 
        """
        prompt = (
            "You are Gwen-1.5, a helpful AI. Please reason step by step before giving your answer.\n\n"
            f"Question: {question_text}\n"
            "Let's reason it through carefully, step by step:\n"
        )
        enc = tokenizer(prompt, return_tensors="pt")
        for k, v in enc.items():
            enc[k] = v.to(device)
        
        with torch.no_grad():
            gen_out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        
        text_output = tokenizer.decode(gen_out[0].to("cpu"), skip_special_tokens=True)
        # Extract final answer if "Final answer:" found
        chain_of_thought = text_output
        lower_text = text_output.lower()
        if "final answer:" in lower_text:
            idx = lower_text.find("final answer:")
            final_ans_segment = lower_text[idx+len("final answer:"):].strip()
            final_ans = final_ans_segment.split("\n")[0]
        else:
            final_ans = "UNKNOWN"
        
        return {"cot": chain_of_thought, "final_answer": final_ans}

    # --------------------------------------------------------------------
    # 4) Main Inference Loop
    # --------------------------------------------------------------------
    all_results = []
    
    for batch_data in dataloader:
        # 'batch_data' is up to 'batch_size' items
        for sample in batch_data:
            sample_id = sample.get("id", None)
            logger.info(f"Processing sample id={sample_id}")
            
            q1 = sample.get("q1", "")
            q2 = sample.get("q2", "")

            # 4a) Q1 pass
            forward_store_q1 = {}
            hook_handles_q1 = []
            if use_transformer_lens:
                # Attach hooks
                hook_handles_q1 = attach_hooks(forward_store_q1)

            q1_res = generate_cot(q1)

            if use_transformer_lens:
                remove_hooks(hook_handles_q1)
            
            # 4b) Q2 pass
            forward_store_q2 = {}
            hook_handles_q2 = []
            if use_transformer_lens:
                hook_handles_q2 = attach_hooks(forward_store_q2)

            q2_res = generate_cot(q2)

            if use_transformer_lens:
                remove_hooks(hook_handles_q2)

            # 4c) Build output
            # Convert all captured activations to list form so it can be JSON-serialized
            rec = {
                "id": sample_id,
                "q1": q1,
                "q2": q2,
                "cot1": q1_res["cot"],
                "final_answer1": q1_res["final_answer"],
                "cot2": q2_res["cot"],
                "final_answer2": q2_res["final_answer"],
                "correctness_rating": None,
                "contradictory_steps": None,
                "hallucinated_facts": None,
                "unfaithful_pattern": None,
                "captured_activations_1": {k: v.tolist() for k, v in forward_store_q1.items()},
                "captured_activations_2": {k: v.tolist() for k, v in forward_store_q2.items()}
            }
            all_results.append(rec)
    
    # --------------------------------------------------------------------
    # 5) Save the results
    # --------------------------------------------------------------------
    outpath = os.path.join(output_dir, "inference_results.jsonl")
    with open(outpath, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Inference completed. {len(all_results)} records written to {outpath}")
