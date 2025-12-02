

import os
import re
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from vocabulary import Vocabulary
from dataset import MazeDataset
from model import Encoder, Decoder, Seq2Seq

# ---------- Config ----------
MODEL_PATH = "best_rnn_model.pt"
VOCAB_PATH = "vocabulary.json"
TRAIN_CSV = "data/train_6x6_mazes.csv"
TRAIN_SPLIT = 0.9
N_SAMPLES = 5
MAX_GEN_LEN = 100
PLOTS_DIR = "plots"
BATCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAZE_ROWS, MAZE_COLS = 6, 6
# ----------------------------


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


def load_model(model_path, vocab_size, device):
    """Load trained Seq2Seq model from checkpoint."""
    encoder = Encoder(vocab_size, embedding_dim=128, hidden_dim=512, num_layers=2).to(device)
    decoder = Decoder(vocab_size, embedding_dim=128, hidden_dim=512, num_layers=2).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    state = torch.load(model_path, map_location=device)
    
    try:
        model.load_state_dict(state)
    except:
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        else:
            # Remove 'module.' prefix if present
            state_dict = {k.replace('module.', ''): v for k, v in state.items()}
            model.load_state_dict(state_dict)
    
    model.eval()
    return model


def build_inverse_vocab(vocab):
    try:
        # Try to convert string keys to int
        return {int(k): v for k, v in vocab.idx2token.items()}
    except:
        # If already int keys or need fallback
        if vocab.idx2token:
            return vocab.idx2token
        else:
            return {i: t for t, i in vocab.token2idx.items()}


def main():
    device = torch.device(BATCH_DEVICE)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("Using device:", device)
    
    # Load vocabulary
    vocab = Vocabulary()
    vocab.load_vocab(VOCAB_PATH)
    pad_idx = vocab.token2idx.get('<PAD>', 0)
    path_start_idx = vocab.token2idx.get('<PATH_START>')
    
    # Load validation dataset
    val_dataset = MazeDataset(TRAIN_CSV, vocab, split='val', train_split=TRAIN_SPLIT)
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty. Check TRAIN_CSV and TRAIN_SPLIT.")
    
    # Load model
    model = load_model(MODEL_PATH, len(vocab), device)
    print("Loaded model weights from:", MODEL_PATH)
    
    # Build inverse vocabulary
    inv_vocab = build_inverse_vocab(vocab)
    
    # Select random samples
    sample_indices = random.sample(range(len(val_dataset)), min(N_SAMPLES, len(val_dataset)))
    figs = []
    
    for si, idx in enumerate(sample_indices, start=1):
        item = val_dataset[idx]
        input_indices = item['input_indices']
        target_indices = item['output_indices']
        
        # Prepare input tensor
        input_tensor = torch.tensor(input_indices, dtype=torch.long, device=device).unsqueeze(0)
        input_lengths = torch.tensor([len(input_indices)], dtype=torch.long, device=device)
        
        # Determine start token
        if path_start_idx is None:
            start_token = target_indices[0] if target_indices else pad_idx
            print("Warning: <PATH_START> not found in vocab")
        else:
            start_token = path_start_idx
        
        # Generate prediction
        with torch.no_grad():
            generated = model.generate(input_tensor, input_lengths, 
                                      max_length=MAX_GEN_LEN, start_token_idx=start_token)
            gen_list = generated[0].cpu().tolist()
        
        # Decode tokens to strings
        input_tokens = [inv_vocab.get(int(i), '<UNK>') for i in input_indices if int(i) != pad_idx]
        target_tokens = [inv_vocab.get(int(i), '<UNK>') for i in target_indices if int(i) != pad_idx]
        gen_tokens = [inv_vocab.get(int(i), '<UNK>') for i in gen_list if int(i) != pad_idx]
        
        # Trim at PATH_END if present
        if '<PATH_END>' in gen_tokens:
            pos = gen_tokens.index('<PATH_END>')
            gen_tokens = gen_tokens[:pos + 1]
        
        # Build full token list for visualization
        maze_text = " ".join(input_tokens)
        tgt_text = "<TARGETPATH_START> " + " ".join(target_tokens) + " <TARGETPATH_END>"
        pred_text = "<PATH_START> " + " ".join(gen_tokens) + " <PATH_END>"
        full_tokens = (maze_text + " " + tgt_text + " " + pred_text).split()
        
        print(f"\n=== SAMPLE {si} (dataset idx={idx}) ===")
        print("GT path:", " ".join(target_tokens))
        print("Pred  :", " ".join(gen_tokens))
        
        # Create maze plot
        fig = plot_maze(full_tokens, rows=MAZE_ROWS, cols=MAZE_COLS)
        figs.append(fig)
    
    # Combine all figures into one image
    combined_fig, axs = plt.subplots(1, len(figs), figsize=(4 * len(figs), 4))
    if len(figs) == 1:
        axs = [axs]
    
    for ax, fig in zip(axs, figs):
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save combined figure
    output_path = os.path.join(PLOTS_DIR, f"maze_predictions_{len(figs)}.png")
    combined_fig.savefig(output_path, dpi=200)
    print(f"\nSaved combined maze predictions to: {output_path}")


if __name__ == "__main__":
    main()