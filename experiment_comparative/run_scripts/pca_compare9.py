#!/usr/bin/env python
import os
import argparse
import logging
from glob import glob

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # So it does not require an X server
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

def convert_bfloat16_to_float32(obj):
    """
    Recursively convert any bfloat16 tensors in a structure (dict, list, tensor)
    into float32 tensors. Returns the modified structure.
    """
    if isinstance(obj, torch.Tensor):
        if obj.dtype == torch.bfloat16:
            return obj.to(torch.float32)
        else:
            return obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = convert_bfloat16_to_float32(v)
        return obj
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = convert_bfloat16_to_float32(obj[i])
        return obj
    else:
        return obj

def perform_pca_and_plot(input_dir, output_dir, layer, logger=None):
    """
    Perform a single PCA across all subdirectories combined.
    Each subdirectory gets a different color; within each color,
    samples with label "same" (i.e. same final answer as its paired question)
    are plotted with higher alpha (darker) and those with "different" (different final answer)
    with lower alpha (lighter).

    :param input_dir:   str, path to the directory containing subdirectories of .pt inference files.
    :param output_dir:  str, path to where the PCA plot (PNG) will be saved.
    :param layer:       int, which layer index to use (e.g. 0, 5, 10, etc.).
    :param logger:      optional logger object.
    """
    if logger is None:
        logger = logging.getLogger("pcaLogger")
        logger.setLevel(logging.INFO)

    os.makedirs(output_dir, exist_ok=True)

    # Gather subdirectories in input_dir
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    if not subdirs:
        logger.warning(f"No subdirectories found in {input_dir}. Exiting.")
        return

    global_vectors = []
    global_subdirs = []
    global_labels = []  # Labels from "same_as_other_labels": "same" if True, "different" if False
    all_shapes = []   # To record each embedding's shape

    # Define a simple color palette for subdirectories.
    base_colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    subdir_color_map = {}
    for i, s in enumerate(sorted(subdirs)):
        subdir_color_map[s] = base_colors[i % len(base_colors)]

    pt_files = []
    for subdir in subdirs:
        subdir_path = os.path.join(input_dir, subdir)
        files = sorted(glob(os.path.join(subdir_path, "*.pt")))
        pt_files.extend(files)

    logger.info(f"Found {len(pt_files)} .pt files across subdirectories.")

    # First pass: collect all embeddings and record their shapes.
    temp_vectors = []
    for pt_file in pt_files:
        try:
            data_dict = torch.load(pt_file, map_location="cpu", weights_only=True)
            data_dict = convert_bfloat16_to_float32(data_dict)
            
            if "same_as_other_labels" not in data_dict:
                logger.warning(f"No 'same_as_other_labels' found in {pt_file}; skipping.")
                continue

            classifications = data_dict["same_as_other_labels"]  # List of booleans

            layer_key = f"layer_{layer}"
            if "hidden_states" not in data_dict or layer_key not in data_dict["hidden_states"]:
                logger.warning(f"Missing {layer_key} in {pt_file}; skipping.")
                continue

            hidden_states = data_dict["hidden_states"][layer_key]  # shape: (B, S, H)
            batch_size, seq_len, hidden_dim = hidden_states.shape
            hidden_mean = hidden_states.mean(dim=1)  # shape: (B, H)

            subdir = os.path.basename(os.path.dirname(pt_file))
            for i in range(batch_size):
                emb = hidden_mean[i].cpu().numpy().flatten()
                temp_vectors.append((emb, subdir, classifications[i]))
                all_shapes.append(emb.shape)
        except Exception as e:
            logger.error(f"Failed loading {pt_file} due to {e}")

    if not temp_vectors:
        logger.warning("No valid data found in any subdirectory. Nothing to plot.")
        return

    # Determine target dimension: maximum dimension across all samples.
    target_dim = max(s[0] for s in all_shapes)
    logger.info(f"Target embedding dimension for PCA will be {target_dim}.")

    # Second pass: adjust each embedding to the target dimension.
    for emb, subdir, cls in temp_vectors:
        current_dim = emb.shape[0]
        if current_dim < target_dim:
            logger.warning(f"Padding embedding from shape {emb.shape} to ({target_dim},).")
            emb = np.pad(emb, (0, target_dim - current_dim), mode='constant')
        # If current_dim > target_dim, we leave it as is (since target_dim is max)
        global_vectors.append(emb)
        label = "same" if cls else "different"
        global_labels.append(label)
        global_subdirs.append(subdir)

    try:
        global_vectors = np.vstack(global_vectors)
    except Exception as e:
        logger.error(f"Error stacking global_vectors: {e}")
        return
    global_subdirs = np.array(global_subdirs)
    global_labels = np.array(global_labels)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(global_vectors)

    plt.figure(figsize=(6, 5))
    plt.title(f"Combined PCA - Layer {layer}")

    for subdir in sorted(set(global_subdirs)):
        color = subdir_color_map[subdir]
        for lab in ["same", "different"]:
            mask = (global_subdirs == subdir) & (global_labels == lab)
            if not np.any(mask):
                continue
            alpha_val = 0.8 if lab == "same" else 0.3
            legend_label = f"{subdir} ({lab})"
            plt.scatter(
                pca_coords[mask, 0],
                pca_coords[mask, 1],
                c=color,
                alpha=alpha_val,
                s=10,
                label=legend_label
            )
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    fig_path = os.path.join(output_dir, f"pca_all_layer{layer}.png")
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved single PCA plot for all data: {fig_path}")

def main():
    parser = argparse.ArgumentParser(description="Perform PCA on hidden-states from multiple subdirectories, combining them into one plot.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with subdirs that contain .pt files")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save PCA plots")
    parser.add_argument("--layer", type=int, default=0, help="Which layer index to use for PCA (e.g. 0, 5, 10, etc.)")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("pcaLogger")

    perform_pca_and_plot(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        layer=args.layer,
        logger=logger
    )

if __name__ == "__main__":
    main()
