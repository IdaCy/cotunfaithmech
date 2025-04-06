# File: evaluate_answers.py

"""
Script to:
1) Read .pt files from your inference runs (which contain final_predictions, chain_of_thought, final_answers, etc.).
2) For certain pairs of IDs, check if they have the same final answer ("YES"/"NO" or some variant).
3) Create:
   (a) an output file that states if each pair has the same result of yes/no.
   (b) a separate file that logs each sample's CoT and final answer.
   (c) a copy of each .pt file adding a label for "same_as_other" or "not_same".

Usage from a Jupyter notebook:
  from evaluate_answers import evaluate_outputs

  evaluate_outputs(
      input_dir="experiment_comparative/output",
      pair_list=[(1,2), (3,4), (4,5)],     # or any list of sample IDs
      same_result_output_file="same_result_info.txt",
      cot_output_file="chain_of_thoughts.json",
      rewrites_dir="experiment_comparative/output_with_labels"
  )

Then check those outputs or continue your analysis in your notebook.
"""

import os
import glob
import json
import torch

def evaluate_outputs(
    input_dir: str,
    pair_list: list,
    same_result_output_file: str,
    cot_output_file: str,
    rewrites_dir: str
):
    """
    Reads all .pt files in `input_dir`, collects final answers per sample ID,
    checks if certain pairs have the same result or not,
    writes out the results, plus copies the .pt files with an added label.

    Args:
      input_dir: Directory with .pt files from your inference script (e.g. "activations_00000_00004.pt", etc.).
      pair_list: A list of pairs of sample IDs, e.g. [(1,2), (3,4), (4,5)].
      same_result_output_file: Path to a text file listing each pair's "same/different" info.
      cot_output_file: Path to a file listing chain-of-thought & final answer for each sample.
      rewrites_dir: Directory to store the updated .pt files with added label "same_as_other" or "not_same".
    """
    os.makedirs(rewrites_dir, exist_ok=True)

    # 1) Collect data from .pt files
    # We'll store in a dict: { sample_id -> { "cot":..., "final_answer":..., "topk_vals":..., etc. } }
    # Because each .pt file can have a batch of items. We'll store them individually keyed by ID.
    all_data = {}

    pt_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))
    if not pt_files:
        print(f"No .pt files found in {input_dir}")
        return

    print(f"Found {len(pt_files)} .pt files in {input_dir}. Reading them...")

    for ptf in pt_files:
        data_dict = torch.load(ptf, map_location="cpu")  # a dictionary from your script
        # The shape is typically:
        # {
        #   "hidden_states": ...
        #   "attentions": ...
        #   "topk_vals": ...
        #   "topk_indices": ...
        #   "input_ids": ...
        #   "final_predictions": [str, str, ...],
        #   "chain_of_thought": [str, str, ...],
        #   "final_answers": [str, str, ...],
        #   "original_indices": [1, 2, ...]
        # }
        # We'll store them in a loop:
        final_preds = data_dict.get("final_predictions", [])
        cots = data_dict.get("chain_of_thought", [])
        final_ans = data_dict.get("final_answers", [])
        indices = data_dict.get("original_indices", [])

        # We also keep the rest if you want, e.g. hidden_states, topk_vals, etc.
        # We'll store them as "extras" so we can re-save them later with a label.

        # We'll re-save everything in an item-based structure:
        batch_size = len(indices)
        for i in range(batch_size):
            sid = indices[i]
            item_dict = {
                "pt_filename": os.path.basename(ptf),
                "final_prediction": final_preds[i] if i < len(final_preds) else "",
                "chain_of_thought": cots[i] if i < len(cots) else "",
                "final_answer": final_ans[i] if i < len(final_ans) else "",
                # store hidden states, attentions, topk, etc. to re-save
                "hidden_states": data_dict.get("hidden_states", {}),
                "attentions": data_dict.get("attentions", {}),
                "topk_vals": data_dict.get("topk_vals", None),
                "topk_indices": data_dict.get("topk_indices", None),
                "input_ids": data_dict.get("input_ids", None)
            }
            all_data[sid] = item_dict

    print(f"Collected data for {len(all_data)} unique sample IDs.")

    # 2) For the pairs in pair_list, check if final_answer is the same
    # We'll define "YES" or "NO" or "unknown" based on substring or direct equality.
    # This is just an example, adjust as needed for your actual final answers.

    results_for_pairs = []
    for (idA, idB) in pair_list:
        itemA = all_data.get(idA)
        itemB = all_data.get(idB)
        if itemA is None or itemB is None:
            # We skip if we don't have data for both
            line = f"Pair ({idA}, {idB}): MISSING one or both => cannot compare.\n"
            results_for_pairs.append(line)
            continue

        # check final_answer for itemA and itemB
        ansA = itemA["final_answer"].strip().lower()
        ansB = itemB["final_answer"].strip().lower()
        # naive approach: if both contain "yes", or both contain "no", => same
        # or do a direct equality check ansA == ansB
        same = False
        if ansA and ansB:
            # direct equality is simplest
            same = (ansA == ansB)

        if same:
            line = f"Pair ({idA}, {idB}): same final answer => {ansA}"
            # optionally store a label
            itemA["same_as_other"] = True
            itemB["same_as_other"] = True
        else:
            line = f"Pair ({idA}, {idB}): different final answer => {ansA} vs. {ansB}"
            itemA["same_as_other"] = False
            itemB["same_as_other"] = False

        results_for_pairs.append(line + "\n")

    # 3) Write out the pair result info to `same_result_output_file`
    with open(same_result_output_file, "w", encoding="utf-8") as f_s:
        f_s.writelines(results_for_pairs)
    print(f"Wrote same-result info to {same_result_output_file}")

    # 4) Write out chain-of-thought & final answer for each sample ID => `cot_output_file`
    # We'll store a JSON array of {id, cot, final_answer}
    cot_list_for_write = []
    for sid, val in sorted(all_data.items(), key=lambda x: x[0]):  # sort by ID
        cot_list_for_write.append({
            "id": sid,
            "cot": val["chain_of_thought"],
            "final_answer": val["final_answer"]
        })
    with open(cot_output_file, "w", encoding="utf-8") as f_cot:
        json.dump(cot_list_for_write, f_cot, indent=2)
    print(f"Wrote chain-of-thoughts & final answers to {cot_output_file}")

    # 5) Make a copy of each .pt file with the new label "same_as_other" or not
    # We need to read them again or we can store them in memory. We'll do it from memory for clarity.
    # We'll group all items from each original .pt by 'pt_filename'.

    # Build a dict mapping pt_filename => list of (sample_id, updated item)
    ptfile_to_items = {}
    for sid, itemdata in all_data.items():
        ptf = itemdata["pt_filename"]
        if ptf not in ptfile_to_items:
            ptfile_to_items[ptf] = []
        ptfile_to_items[ptf].append((sid, itemdata))

    # Now re-load each .pt, reassemble the batch data.
    for ptf in pt_files:
        base_ptf = os.path.basename(ptf)
        if base_ptf not in ptfile_to_items:
            continue  # no items from this file? skip
        # load the original dictionary
        orig_dict = torch.load(ptf, map_location="cpu")

        # We re-build final_predictions, chain_of_thought, final_answers, etc., plus the new label
        # We'll do a list for each item in the batch. We'll rely on "original_indices" to reorder them.
        batch_indices = orig_dict["original_indices"]
        new_cot = []
        new_answers = []
        new_preds = []
        same_labels = []
        for i, sid in enumerate(batch_indices):
            itemdata = all_data[sid]
            # incorporate label
            same_as_other = itemdata.get("same_as_other", False)
            new_cot.append(itemdata["chain_of_thought"])
            new_answers.append(itemdata["final_answer"])
            new_preds.append(itemdata["final_prediction"])
            # store the label in some new structure
            # e.g., "same_as_other_labels": [bool1, bool2, ...]
            same_labels.append(same_as_other)

        # Now we add to orig_dict
        orig_dict["chain_of_thought"] = new_cot
        orig_dict["final_answers"] = new_answers
        orig_dict["final_predictions"] = new_preds
        orig_dict["same_as_other_labels"] = same_labels

        # rewrite with the new info
        newname = f"{os.path.splitext(base_ptf)[0]}_withlabel.pt"
        newpath = os.path.join(rewrites_dir, newname)
        torch.save(orig_dict, newpath)

    print(f"Created updated PT files in {rewrites_dir}")

def main():
    """
    Example usage from CLI:
    python evaluate_answers.py \
      --input_dir=experiment_comparative/output \
      --same_result_output_file=experiment_comparative/same_result_info.txt \
      --cot_output_file=experiment_comparative/chain_of_thoughts.json \
      --rewrites_dir=experiment_comparative/output_with_labels \
      --pairs 1 2  3 4  4 5
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--same_result_output_file", type=str, required=True)
    parser.add_argument("--cot_output_file", type=str, required=True)
    parser.add_argument("--rewrites_dir", type=str, required=True)
    parser.add_argument(
        "--pairs",
        nargs="*",
        type=int,
        help="List of IDs in pairs. E.g. '--pairs 1 2 3 4 4 5' => pairs of (1,2), (3,4), (4,5)."
    )
    args = parser.parse_args()

    # interpret the pairs
    if not args.pairs or len(args.pairs) % 2 != 0:
        print("Please provide an even number of IDs via --pairs, e.g. --pairs 1 2 3 4")
        return
    pair_list = []
    for i in range(0, len(args.pairs), 2):
        pair_list.append((args.pairs[i], args.pairs[i+1]))

    evaluate_outputs(
        input_dir=args.input_dir,
        pair_list=pair_list,
        same_result_output_file=args.same_result_output_file,
        cot_output_file=args.cot_output_file,
        rewrites_dir=args.rewrites_dir
    )

if __name__ == "__main__":
    main()
