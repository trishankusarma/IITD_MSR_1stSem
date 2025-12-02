import torch
import pandas as pd
import sys
from transformer_model import TransformerMazeSolver
from rnn_model import RNNMazeSolver
from dataset_handler import MazeDataset, collate_fn
from train_transformer import evaluate_no_teacher_forcing, logging
import torch.nn as nn

VAL_SPLIT = 0.1
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 5e-4
SEED = 42
LABEL_SMOOTHING = 0.1
MAX_GRAD_NORM = 1.0

def load_transformer(path, device):
    checkpoint = torch.load(path, map_location=device)
    token_to_idx = checkpoint["token_to_idx"]
    idx_to_token = checkpoint["idx_to_token"]
    vocab_size = len(token_to_idx)

    model = TransformerMazeSolver(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, token_to_idx, idx_to_token


def load_rnn(path, device):
    checkpoint = torch.load(path, map_location=device)
    token_to_idx = checkpoint["token_to_idx"]
    idx_to_token = checkpoint["idx_to_token"]
    vocab_size = len(token_to_idx)

    model = RNNMazeSolver(
        vocab_size=vocab_size,
        embedding_dim=,
        hidden_dim=h["hidden_dim"],
        num_layers=h["num_layers"],
        dropout=h["dropout"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, token_to_idx, idx_to_token


def main():
    logging(s="Q2-eval")

    if len(sys.argv) != 5:
        print("Usage: python eval.py <model_path> <model_type> <test_csv> <output_csv>")
        sys.exit(1)

    model_path = sys.argv[1]
    model_type = sys.argv[2].lower()   # rnn / transformer
    test_csv = sys.argv[3]
    output_csv = sys.argv[4]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load correct model
    if model_type == "transformer":
        model, token_to_idx, idx_to_token = load_transformer(model_path, device)
    elif model_type == "rnn":
        model, token_to_idx, idx_to_token = load_rnn(model_path, device)
    else:
        print("Error: model_type must be 'rnn' or 'transformer'")
        sys.exit(1)

    pad_idx = token_to_idx["<PAD>"]

    # DataLoader
    test_dataset = MazeDataset(test_csv, token_to_idx)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx)
    )

    print("\nEvaluating full test dataset...")
    token_acc, seq_acc, f1, precision, recall, predictions, gt = evaluate_no_teacher_forcing(
        model, test_loader, device, token_to_idx, return_sequences=True
    )

    print("\nFINAL TEST METRICS")
    print(f"Token Accuracy: {token_acc:.4f}")
    print(f"Sequence Accuracy: {seq_acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Save output CSV
    # Must match sample_submission.csv
    df_out = pd.read_csv(test_csv)

    df_out["predicted"] = [" ".join(seq) for seq in predictions]

    df_out.to_csv(output_csv, index=False)
    print(f"\nSaved output CSV → {output_csv}")


if __name__ == "__main__":
    main()
