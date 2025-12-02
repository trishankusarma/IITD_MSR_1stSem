"""
visualizePlots.py
Generate maze visualizations showing:
- Input maze adjacency + origin + target
- Ground-truth TARGETPATH
- Predicted PATH from Transformer (greedy decoding)

Usage:
    python visualizePlots.py
"""

import os
import re
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import json

from dataset_handler import MazeDataset
from transformer_model import TransformerMazeSolver


# ---------- Config ----------
MODEL_PATH = "./runs/transformer_optimized_20251127_010641/best_model.pth"
VOCAB_PATH = "./runs/transformer_optimized_20251127_010641/vocabulary.json"
TRAIN_CSV = "dataset/test_6x6_mazes.csv"
N_SAMPLES = 5
MAX_GEN_LEN = 150
PLOTS_DIR = "plots_transformer"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROWS, COLS = 6, 6

# TOKEN PARSING HELPERS
def extract_between(tag, text):
    """Extract content between <TAG_START> and <TAG_END>."""
    pattern = rf"<{tag}_START>(.*?)<{tag}_END>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def parse_coords(s):
    """Parse coordinate tuple from string like '(1, 2)'."""
    if not s:
        return None
    nums = re.findall(r"-?\d+", s)
    return tuple(map(int, nums)) if len(nums) == 2 else None


def parse_edges(adj_section):
    """Parse edge pairs from adjacency list section."""
    edge_pairs = re.findall(
        r"\(\s*-?\d+\s*,\s*-?\d+\s*\)\s*<-->\s*\(\s*-?\d+\s*,\s*-?\d+\s*\)",
        adj_section
    )
    edges = []
    for edge_match in edge_pairs:
        coords = re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", edge_match)
        if len(coords) == 2:
            a = parse_coords(coords[0])
            b = parse_coords(coords[1])
            edges.append((a, b))
    return edges


def parse_path(text, tag):
    """Parse path coordinates from <TAG_START>...<TAG_END> section."""
    pattern = rf"<{tag}_START>\s*((?:\(\s*-?\d+\s*,\s*-?\d+\s*\)\s*)+)\s*<{tag}_END>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    coords = re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", match.group(1))
    return [parse_coords(c) for c in coords]

# MAZE PLOTTING
def plot_maze(tokens, rows=6, cols=6):

    # Convert tokens to text
    text = " ".join(tokens) if isinstance(tokens, (list, tuple)) else str(tokens)
    
    # Extract all sections
    adj_section = extract_between("ADJLIST", text)
    if not adj_section:
        raise ValueError("No adjacency list found in tokens/text")
    
    origin = parse_coords(extract_between("ORIGIN", text))
    target = parse_coords(extract_between("TARGET", text))
    true_path = parse_path(text, "TARGETPATH")
    pred_path = parse_path(text, "PATH")
    edges = parse_edges(adj_section)
    
    # Initialize walls (all walls present initially)
    vertical_walls = np.ones((rows, cols + 1), dtype=bool)
    horizontal_walls = np.ones((rows + 1, cols), dtype=bool)
    
    # Remove walls based on edges
    for (r1, c1), (r2, c2) in edges:
        if r1 == r2:  # Horizontal edge
            c_between = min(c1, c2) + 1
            if 0 <= r1 < rows and 0 <= c_between < cols + 1:
                vertical_walls[r1, c_between] = False
        elif c1 == c2:  # Vertical edge
            r_between = min(r1, r2) + 1
            if 0 <= r_between < rows + 1 and 0 <= c1 < cols:
                horizontal_walls[r_between, c1] = False
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')
    
    # Draw light grid
    for r in range(rows):
        for c in range(cols):
            x0, x1 = c, c + 1
            y_top = rows - r
            y_bot = rows - r - 1
            ax.plot([x0, x1], [y_top, y_top], color='lightgray', lw=1)
            ax.plot([x0, x1], [y_bot, y_bot], color='lightgray', lw=1)
            ax.plot([x0, x0], [y_bot, y_top], color='lightgray', lw=1)
            ax.plot([x1, x1], [y_bot, y_top], color='lightgray', lw=1)
    
    # Draw walls
    for r in range(rows):
        for c in range(cols + 1):
            if vertical_walls[r, c]:
                x = c
                y_top = rows - r
                y_bot = rows - r - 1
                ax.plot([x, x], [y_bot, y_top], color='black', lw=4, solid_capstyle='butt')
    
    for r in range(rows + 1):
        for c in range(cols):
            if horizontal_walls[r, c]:
                y = rows - r
                ax.plot([c, c + 1], [y, y], color='black', lw=4, solid_capstyle='butt')
    
    # Shade true path cells (light green)
    if true_path:
        for r, c in true_path:
            x0, y0 = c, rows - r - 1
            rect = plt.Rectangle((x0, y0), 1, 1, facecolor=(0.9, 1, 0.9), 
                                edgecolor=None, zorder=0)
            ax.add_patch(rect)
    
    # Draw true path (green line)
    if true_path:
        true_x = [c + 0.5 for r, c in true_path]
        true_y = [rows - r - 0.5 for r, c in true_path]
        ax.plot(true_x, true_y, linestyle='-', linewidth=3, color='tab:green', 
                label='True Path', zorder=5)
        ax.scatter(true_x[0], true_y[0], marker='o', s=80, color='tab:green', zorder=6)
        ax.scatter(true_x[-1], true_y[-1], marker='s', s=80, color='tab:green', zorder=6)
    
    # Draw predicted path (red dashed line)
    if pred_path:
        pred_x = [c + 0.5 + 0.08 for r, c in pred_path]
        pred_y = [rows - r - 0.5 - 0.08 for r, c in pred_path]
        ax.plot(pred_x, pred_y, linestyle='--', linewidth=2.5, color='tab:red', 
                label='Predicted Path', zorder=8)
        ax.scatter(pred_x[0], pred_y[0], marker='o', s=70, color='tab:red', zorder=9)
        ax.scatter(pred_x[-1], pred_y[-1], marker='x', s=70, color='tab:red', zorder=9)
    
    # Draw origin and target markers
    if origin:
        ox, oy = origin[1] + 0.5, rows - origin[0] - 0.5
        ax.scatter(ox, oy, c='blue', s=60, marker='o', zorder=7)
    if target:
        tx, ty = target[1] + 0.5, rows - target[0] - 0.5
        ax.scatter(tx, ty, c='blue', s=60, marker='x', zorder=7)
    
    # Configure axes
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(cols + 1)[:-1])
    ax.set_yticks(np.arange(rows + 1)[:-1])
    plt.yticks([])
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    
    return fig

#   TRANSFORMER LOAD + GREEDY DECODE
def load_transformer(path, token_to_idx, device):
    """
    Load best Transformer model from checkpoint.
    """
    ckpt = torch.load(path, map_location=device)
    h = ckpt["hyperparameters"]

    model = TransformerMazeSolver(
        vocab_size=len(token_to_idx),
        d_model=h["d_model"],
        nhead=h["nhead"],
        num_layers=h["num_layers"],
        dim_feedforward=h["dim_feedforward"],
        dropout=h["dropout"]
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def greedy_decode(model, src, token_to_idx, device, max_len=120):
    """
    Greedy autoregressive decoding for the PATH section.
    """
    pad = token_to_idx["<PAD>"]
    start = token_to_idx["<PATH_START>"]
    end = token_to_idx["<PATH_END>"]

    memory, src_mask = model.encode(src, src_pad_idx=pad)

    tgt = torch.tensor([[start]], device=device)
    finished = False

    for _ in range(max_len):
        out = model.decode_step(
            tgt,
            memory,
            tgt_pad_idx=pad,
            memory_padding_mask=src_mask
        )

        next_tok = out[:, -1, :].argmax(dim=-1)      # greedy

        tgt = torch.cat([tgt, next_tok.unsqueeze(1)], dim=1)

        if next_tok.item() == end:
            break

    return tgt.squeeze(0).tolist()

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    device = torch.device(DEVICE)
    print("[INFO] Using device:", device)

    # Load vocabulary
    with open(VOCAB_PATH, "r") as f:
        vocab_json = json.load(f)

    token_to_idx = vocab_json["token_to_idx"]
    idx_to_token = {int(k): v for k, v in vocab_json["idx_to_token"].items()}

    # Load dataset
    dataset = MazeDataset(TRAIN_CSV, token_to_idx)
    print("[INFO] Dataset size:", len(dataset))

    # Load model
    model = load_transformer(MODEL_PATH, token_to_idx, device)
    print("[INFO] Loaded Transformer checkpoint.")

    # Pick samples
    indices = random.sample(range(len(dataset)), min(N_SAMPLES, len(dataset)))
    figs = []

    # Generate & plot
    for i, idx in enumerate(indices):
        # MazeDataset returns a tuple (input_indices, output_indices)
        input_indices, output_indices = dataset[idx]

        # Convert input to tensor for model
        src_ids = torch.tensor(input_indices, device=device).unsqueeze(0)
        tgt_ids = output_indices

        # Decode prediction
        pred_ids = greedy_decode(model, src_ids, token_to_idx, device, MAX_GEN_LEN)

        # Convert integer tokens back to string tokens
        src_toks = [idx_to_token[int(i)] for i in input_indices]
        tgt_toks = [idx_to_token[int(i)] for i in tgt_ids]
        pred_toks = [idx_to_token[int(i)] for i in pred_ids]

        print(f"\n=== SAMPLE {i+1} ===")
        print("GT   :", " ".join(tgt_toks))
        print("PRED :", " ".join(pred_toks))

        # Build whole token stream for plotting
        merged_tokens = (
            src_toks
            + ["<TARGETPATH_START>"] + tgt_toks + ["<TARGETPATH_END>"]
            + ["<PATH_START>"] + pred_toks + ["<PATH_END>"]
        )

        fig = plot_maze(merged_tokens, rows=ROWS, cols=COLS)
        figs.append(fig)

    # Combine plots horizontally
    combined, axs = plt.subplots(1, len(figs), figsize=(4 * len(figs), 4))

    if len(figs) == 1:
        axs = [axs]

    for ax, f in zip(axs, figs):
        f.canvas.draw()
        img = np.array(f.canvas.renderer.buffer_rgba())
        ax.imshow(img)
        ax.axis("off")

    out_path = os.path.join(PLOTS_DIR, f"maze_preds_{len(figs)}_test.png")
    combined.savefig(out_path, dpi=200)
    print("[INFO] Saved maze visualization →", out_path)


if __name__ == "__main__":
    main()
