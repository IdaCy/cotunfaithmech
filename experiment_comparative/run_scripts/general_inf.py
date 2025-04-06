#!/usr/bin/env python
import os
import math
import json
import logging
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.load_model import load_model

# Default parameters
DEFAULT_OUTPUT_DIR = "output/"
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_TOP_K_LOGITS = 10

DEFAULT_GENERATION_KWARGS = {
    "do_sample": True,
    "max_new_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.2
}

def parse_cot_and_answer(generated_text: str):
    """
    Minimal helper to parse out 'Chain-of-Thought:' and 'Final-Answer:' from the generated_text.
    Expects something like:
      Chain-of-Thought:
      ...some reasoning...
      Final-Answer:
      ...final answer...

    Returns: (chain_of_thought, final_answer).
    If not found, returns partial or empty strings.
    """
    # We'll do naive searches. Tweak as needed for your prompt style.
    cot_start_tag = "chain-of-thought:"
    ans_start_tag = "final-answer:"
    
    text_lower = generated_text.lower()
    chain_of_thought = ""
    final_answer = ""
    
    # Find chain-of-thought if present
    cot_idx = text_lower.find(cot_start_tag)
    if cot_idx >= 0:
        # after the tag
        after_cot = generated_text[cot_idx + len(cot_start_tag):]
        
        # If there's also a final-answer tag, we want everything up to that
        ans_idx = after_cot.lower().find(ans_start_tag)
        if ans_idx >= 0:
            chain_of_thought = after_cot[:ans_idx].strip()
            # then parse final answer
            after_ans = after_cot[ans_idx + len(ans_start_tag):].strip()
            final_answer = after_ans.split("\n")[0].strip()
        else:
            # no final-answer found
            chain_of_thought = after_cot.strip()
    
    else:
        # no chain-of-thought found
        # let's see if we can find final-answer alone
        ans_idx_global = text_lower.find(ans_start_tag)
        if ans_idx_global >= 0:
            after_ans = generated_text[ans_idx_global + len(ans_start_tag):]
            final_answer = after_ans.split("\n")[0].strip()
    
    return chain_of_thought, final_answer

def run_inf(model,
            tokenizer,
            data,
            output_dir=DEFAULT_OUTPUT_DIR,
            batch_size=DEFAULT_BATCH_SIZE,
            max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
            extract_hidden_layers=None,
            extract_attention_layers=None,
            top_k_logits=DEFAULT_TOP_K_LOGITS,
            logger=None,
            generation_kwargs=None):
    """
    Runs inference on provided data, saving results batchwise. 
    Also extracts a chain of thought and final answer from each generated output.
    """
    if logger is None:
        logger = logging.getLogger("polAIlogger")

    if extract_hidden_layers is None:
        extract_hidden_layers = [0, 5, 10, 15]
    if extract_attention_layers is None:
        extract_attention_layers = [0, 5, 10, 15]

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Clearing CUDA cache before starting.")
    torch.cuda.empty_cache()

    if generation_kwargs is None:
        generation_kwargs = DEFAULT_GENERATION_KWARGS

    total_samples = len(data)
    total_batches = math.ceil(total_samples / batch_size)
    logger.warning(f"=== Starting inference. #samples={total_samples}, batch_size={batch_size} ===")

    for batch_idx in range(total_batches):
        start_i = batch_idx * batch_size
        end_i = min((batch_idx + 1) * batch_size, total_samples)
        batch_items = data[start_i:end_i]
        batch_indices = [x[0] for x in batch_items]
        batch_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": x[1]}],
                tokenize=False,
                add_generation_prompt=True
            )
            for x in batch_items
        ]

        if batch_idx % 20 == 0:
            logger.info(f"Processing batch {batch_idx+1}/{total_batches} (samples {start_i}-{end_i-1})")

        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].cuda()
        attention_mask = encodings["attention_mask"].cuda()

        try:
            with torch.no_grad():
                # Forward pass for hidden/attention extraction:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=True
                )

                # Generation pass for actual produced text:
                gen_out = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **generation_kwargs
                )

            # -- Extract hidden states + attentions:
            hidden_map = {}
            for layer_idx in extract_hidden_layers:
                if layer_idx < len(outputs.hidden_states):
                    hidden_map[f"layer_{layer_idx}"] = outputs.hidden_states[layer_idx].cpu()

            attn_map = {}
            for layer_idx in extract_attention_layers:
                if layer_idx < len(outputs.attentions):
                    attn_map[f"layer_{layer_idx}"] = outputs.attentions[layer_idx].cpu()

            # -- Extract top-k logits:
            logits = outputs.logits
            topk_vals, topk_indices = torch.topk(logits, k=top_k_logits, dim=-1)
            topk_vals = topk_vals.cpu()
            topk_indices = topk_indices.cpu()

            # -- Decode final predictions:
            decoded_preds = [
                tokenizer.decode(o, skip_special_tokens=True)
                for o in gen_out.cpu()
            ]

            # For each item, parse chain-of-thought and final answer
            cot_list = []
            answer_list = []
            for dpred in decoded_preds:
                c_of_t, f_ans = parse_cot_and_answer(dpred)
                cot_list.append(c_of_t)
                answer_list.append(f_ans)

            # -- Build output dictionary to save
            out_dict = {
                "hidden_states": hidden_map,     # shape [batch, seq, hidden] per layer
                "attentions": attn_map,          # shape [batch, heads, seq, seq] per layer
                "topk_vals": topk_vals,          # [batch, seq, top_k_logits]
                "topk_indices": topk_indices,    # [batch, seq, top_k_logits]
                "input_ids": input_ids.cpu(),
                "final_predictions": decoded_preds,  # entire text generation
                "chain_of_thought": cot_list,         # extracted chain of thought
                "final_answers": answer_list,         # extracted final answers
                "original_indices": batch_indices     # which sample IDs this corresponds to
            }

            save_name = f"activations_{start_i:05d}_{end_i:05d}.pt"
            save_path = os.path.join(output_dir, save_name)
            torch.save(out_dict, save_path)
            logger.debug(f"Saved batch => {save_path}")

        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM error on batch {batch_idx}. Clearing cache and continuing.")
            torch.cuda.empty_cache()
        except Exception as ex:
            logger.exception(f"Error on batch {batch_idx}: {ex}")

    logger.warning("=== Inference Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a dataset.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--top_k_logits", type=int, default=DEFAULT_TOP_K_LOGITS)
    args = parser.parse_args()

    # Setup a basic logger if needed
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("polAIlogger")
    
    # For demonstration, create some dummy data: a list of (index, text) pairs.
    dummy_data = [(i, f"Sample text {i}") for i in range(10)]

    # Load model and tokenizer (you can pass command-line args as needed)
    model, tokenizer = load_model(logger=logger)

    run_inf(model=model,
            tokenizer=tokenizer,
            data=dummy_data,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            top_k_logits=args.top_k_logits,
            logger=logger)
