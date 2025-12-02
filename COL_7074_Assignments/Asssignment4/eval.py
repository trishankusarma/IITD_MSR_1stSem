import torch
import pandas as pd
import sys
import torch.nn as nn
import json
import os
from tqdm import tqdm

# RNN imports
from Q1.model import Encoder, Decoder, Seq2Seq
from Q1.dataset import MazeDataset as MazeDatasetRNN, collate_fn as collate_fnRNN
from Q1.vocabulary import Vocabulary  

# Transformer imports
from Q2.transformer_model import TransformerMazeSolver
from Q2.dataset_handler import MazeDataset as MazeDatasetTF, collate_fn


# Hyperparameters (Transformer)
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

# Hyperparameters (RNN)
EMBEDDING_DIM = 128
HIDDEN_DIM = 512
NUM_LAYERS_RNN = 2


# -----------------------------
# Vocabulary loaders
# -----------------------------
def load_vocab_json1():
    """Load RNN vocabulary from Q1/vocabulary.json"""
    vocab_path = os.path.join("Q1", "vocabulary.json")
    vocab = Vocabulary()
    vocab.load_vocab(vocab_path)
    return vocab


def load_vocab_json2():
    """Load Transformer vocabulary from Q2/vocabulary.json"""
    vocab_path = os.path.join("Q2", "vocabulary.json")

    with open(vocab_path, "r") as f:
        data = json.load(f)

    if "token_to_idx" not in data:
        token_to_idx = data
        idx_to_token = {v: k for k, v in token_to_idx.items()}
    else:
        token_to_idx = data["token_to_idx"]
        idx_to_token = data["idx_to_token"]

    return token_to_idx, idx_to_token


# -----------------------------
# Model loaders
# -----------------------------
def load_transformer(path, device):
    """
    Load Transformer model from checkpoint (weights only)
    """
    print(f"\nLoading Transformer from: {path}")
    
    # Load vocabulary from file (not from checkpoint)
    print("Loading vocabulary from file")
    token_to_idx, idx_to_token = load_vocab_json2()
    
    vocab_size = len(token_to_idx)
    print(f"Vocabulary size: {vocab_size}")
    
    # Initialize model
    model = TransformerMazeSolver(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)
    
    # Load checkpoint (just weights)
    checkpoint = torch.load(path, map_location=device)
    
    # Load weights - checkpoint contains only model weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("Loading from 'model_state_dict' key")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Loading weights directly")
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Transformer model loaded successfully")
    
    return model, token_to_idx, idx_to_token


def load_rnn(path, device):
    """
    Load RNN model from checkpoint (weights only)
    """
    print(f"\nLoading RNN from: {path}")
    
    # Load vocabulary from file (not from checkpoint)
    vocab = load_vocab_json1()
    vocab_size = len(vocab.token2idx)
    print(f"Vocabulary size: {vocab_size}")

    # Initialize model
    encoder = Encoder(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS_RNN).to(device)
    decoder = Decoder(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS_RNN).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Load checkpoint (just weights)
    checkpoint = torch.load(path, map_location=device)
    
    # Load weights - checkpoint contains only model weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("Loading from 'model_state_dict' key")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Loading weights directly")
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("RNN model loaded successfully")

    return model, vocab


# -----------------------------
# Metrics helper
# -----------------------------
def calculate_position_aware_f1(pred_sequences, target_sequences, pad_idx):
    """
    Calculate F1 for ORDERED sequence matching (position-aware)
    """
    batch_size = pred_sequences.size(0)
    f1_scores = []
    precisions = []
    recalls = []
    
    for i in range(batch_size):
        pred_seq = pred_sequences[i]
        target_seq = target_sequences[i]
        
        # Remove padding from target
        mask = (target_seq != pad_idx)
        target_seq_clean = target_seq[mask]
        pred_seq_clean = pred_seq[mask]
        
        if len(target_seq_clean) == 0:
            continue
        
        # Handle length mismatch
        min_len = min(len(pred_seq_clean), len(target_seq_clean))
        
        if min_len == 0:
            f1_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            continue
        
        # Position-by-position matching
        correct_positions = (pred_seq_clean[:min_len] == target_seq_clean[:min_len]).sum().item()
        
        # Precision: correct positions / predicted length
        pred_len = len(pred_seq_clean)
        prec = correct_positions / pred_len if pred_len > 0 else 0.0
        
        # Recall: correct positions / target length
        target_len = len(target_seq_clean)
        rec = correct_positions / target_len if target_len > 0 else 0.0
        
        # F1 score
        if prec + rec > 0:
            f1 = 2 * (prec * rec) / (prec + rec)
        else:
            f1 = 0.0
        
        f1_scores.append(f1)
        precisions.append(prec)
        recalls.append(rec)
    
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_prec = sum(precisions) / len(precisions) if precisions else 0.0
    avg_rec = sum(recalls) / len(recalls) if recalls else 0.0
    
    return avg_f1, avg_prec, avg_rec


# -----------------------------
# Transformer evaluation
# -----------------------------
@torch.no_grad()
def evaluate_no_teacher_forcing(model, dataloader, device, token_to_idx, max_len=100, 
                                return_predictions=False):
    """
    Evaluate the Transformer WITHOUT teacher forcing.
    Uses autoregressive greedy decoding.
    """
    model.eval()

    pad_idx = token_to_idx["<PAD>"]
    path_start_idx = token_to_idx["<PATH_START>"]
    path_end_idx = token_to_idx["<PATH_END>"]

    total_correct_tokens = 0
    total_tokens = 0
    total_correct_sequences = 0
    total_sequences = 0

    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    # Storage for predictions and ground truths
    all_predictions = []
    all_ground_truths = []

    for src, tgt_output in tqdm(dataloader, desc="Evaluating (No Teacher Forcing)", leave=False):
        src = src.to(device)
        tgt_output = tgt_output.to(device)
        batch_size = src.size(0)

        # Start decoder with <PATH_START>
        decoder_input = torch.full(
            (batch_size, 1), path_start_idx, dtype=torch.long, device=device
        )

        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Greedy Decoding
        for _ in range(max_len):
            output = model(src, decoder_input, pad_idx, pad_idx)

            next_token_logits = output[:, -1, :]
            next_tokens = next_token_logits.argmax(dim=-1)

            decoder_input = torch.cat(
                [decoder_input, next_tokens.unsqueeze(1)], dim=1
            )

            finished |= (next_tokens == path_end_idx)

            if finished.all():
                break

        # Remove <PATH_START> before comparison
        generated_output = decoder_input[:, 1:]
        
        # Store predictions if requested
        if return_predictions:
            for i in range(batch_size):
                pred_seq = generated_output[i].cpu().tolist()
                gt_seq = tgt_output[i].cpu().tolist()
                all_predictions.append(pred_seq)
                all_ground_truths.append(gt_seq)

        # Align lengths by padding
        max_compare_len = max(generated_output.size(1), tgt_output.size(1))

        if generated_output.size(1) < max_compare_len:
            pad = torch.full(
                (batch_size, max_compare_len - generated_output.size(1)),
                pad_idx, device=device
            )
            generated_output = torch.cat([generated_output, pad], dim=1)
        else:
            generated_output = generated_output[:, :max_compare_len]

        if tgt_output.size(1) < max_compare_len:
            pad = torch.full(
                (batch_size, max_compare_len - tgt_output.size(1)),
                pad_idx, device=device
            )
            tgt_output = torch.cat([tgt_output, pad], dim=1)
        else:
            tgt_output = tgt_output[:, :max_compare_len]

        # Token-Level Accuracy
        mask = (tgt_output != pad_idx)
        correct_tokens = ((generated_output == tgt_output) & mask).sum().item()
        total_correct_tokens += correct_tokens
        total_tokens += mask.sum().item()

        # Sequence-Level Accuracy
        exact_match = ((generated_output == tgt_output) | ~mask).all(dim=1)
        total_correct_sequences += exact_match.sum().item()
        total_sequences += batch_size

        # F1 Score
        f1, p, r = calculate_position_aware_f1(
            generated_output, tgt_output, pad_idx
        )
        f1_scores.append(f1)
        precision_scores.append(p)
        recall_scores.append(r)

    # Final Metrics
    token_acc = total_correct_tokens / total_tokens if total_tokens > 0 else 0
    seq_acc = total_correct_sequences / total_sequences if total_sequences > 0 else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    avg_prec = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    avg_rec = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    
    if return_predictions:
        # Convert token indices to strings
        idx_to_token = {v: k for k, v in token_to_idx.items()}
        
        pred_token_sequences = []
        gt_token_sequences = []
        
        for pred_indices, gt_indices in zip(all_predictions, all_ground_truths):
            # Convert predictions (remove padding)
            pred_tokens = []
            for idx in pred_indices:
                if idx == pad_idx:
                    break  # Stop at first padding
                pred_tokens.append(idx_to_token[idx])
            pred_token_sequences.append(pred_tokens)
            
            # Convert ground truths (remove padding)
            gt_tokens = []
            for idx in gt_indices:
                if idx == pad_idx:
                    break  # Stop at first padding
                gt_tokens.append(idx_to_token[idx])
            gt_token_sequences.append(gt_tokens)a
        
        return token_acc, seq_acc, avg_f1, avg_prec, avg_rec, pred_token_sequences, gt_token_sequences
    
    return token_acc, seq_acc, avg_f1, avg_prec, avg_rec

# CSV cleaning helper
def fix_empty_list(x):
    # Only fix NaN -> "[]"
    if pd.isna(x):
        return "[]"
    return x

# MAIN
def main():
    if len(sys.argv) != 5:
        print("Usage: python eval.py <model_path> <model_type> <test_csv> <output_csv>")
        print("\nExample:")
        print("  python eval.py runs/transformer_best/best_model.pth transformer test.csv output.csv")
        print("  python eval.py runs/rnn_best/best_model.pth rnn test.csv output.csv")
        sys.exit(1)

    model_path = sys.argv[1]
    model_type = sys.argv[2].lower()
    test_csv = sys.argv[3]
    output_csv = sys.argv[4]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df_tmp = pd.read_csv(test_csv)
    if "output_path" in df_tmp.columns:
        df_tmp["output_path"] = df_tmp["output_path"].apply(fix_empty_list)
    df_tmp.to_csv(test_csv, index=False)
    
    # Validate inputs
    if not os.path.exists(model_path):
        print(f"Error: Model path does not exist: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(test_csv):
        print(f"Error: Test CSV does not exist: {test_csv}")
        sys.exit(1)
    
    if model_type not in ['transformer', 'rnn']:
        print(f"Error: model_type must be 'rnn' or 'transformer', got: {model_type}")
        sys.exit(1)

    # TRANSFORMER EVALUATION
    if model_type == "transformer":
        model, token_to_idx, idx_to_token = load_transformer(model_path, device)

        pad_idx = token_to_idx["<PAD>"]

        test_dataset = MazeDatasetTF(test_csv, token_to_idx)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, pad_idx)
        )

        print("\n" + "="*60)
        print("EVALUATING TRANSFORMER")
        print("="*60)
        
        token_acc, seq_acc, f1, precision, recall, predictions, gt = evaluate_no_teacher_forcing(
            model, test_loader, device, token_to_idx, 
            max_len=100,
            return_predictions=True
        )

        pred_tokens = predictions  # List[List[str]]
        vocab_obj = None  

    # RNN EVALUATION (GREEDY)
    elif model_type == "rnn":
        model, vocab = load_rnn(model_path, device)

        pad_idx = vocab.token2idx["<PAD>"]
        start_idx = vocab.token2idx["<PATH_START>"]

        test_dataset = MazeDatasetRNN(test_csv, vocab, split='test')
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=lambda batch: collate_fnRNN(batch, pad_idx)
        )

        print("\n" + "="*60)
        print("EVALUATING RNN")
        print("="*60)

        pred_tokens = []
        model.eval()
        max_decode_len = 100

        for batch in tqdm(test_loader, desc="Greedy decoding for RNN"):
            input_seqs  = batch["input_padded"].to(device)
            input_lengths = batch["input_lengths"]
            batch_size = input_seqs.size(0)

            # Dummy target just to control decoding length; teacher_forcing_ratio=0
            dummy_target = torch.full(
                (batch_size, max_decode_len + 1),
                start_idx,
                dtype=torch.long,
                device=device
            )

            # Forward pass with pure autoregressive decoding
            outputs = model(
                input_seqs,
                input_lengths,
                dummy_target,
                teacher_forcing_ratio=0.0
            )  # shape: (batch, T, vocab)

            # Drop the first position (corresponding to initial <SOS>/<PATH_START>)
            pred_batch = outputs[:, 1:, :].argmax(dim=2)  # (batch, T)

            for seq in pred_batch:
                pred_tokens.append(seq.cpu().tolist())

        vocab_obj = vocab
        # No ground truth in test CSV ⇒ dummy metrics
        token_acc = 0.0
        seq_acc = 0.0
        f1 = 0.0

    # PRINT RESULTS
    print("\n" + "="*60)
    print("FINAL TEST METRICS")
    print("="*60)
    print(f"Token Accuracy:    {token_acc:.4f}")
    print(f"Sequence Accuracy: {seq_acc:.4f}")
    print(f"F1 Score:          {f1:.4f}")
    print("="*60)
    
    # -------------------------
    # SAVE OUTPUT CSV
    # -------------------------
    df_out = pd.read_csv(test_csv)

    # Ensure same number of predictions and test rows
    if len(pred_tokens) != len(df_out):
        print(f"\nERROR: Number of predictions ({len(pred_tokens)}) does not match dataset size ({len(df_out)})")
        sys.exit(1)

    # Convert predictions to Python list string format (matching the input format)
    if model_type == "transformer":
        # pred_tokens is List[List[str]] - already token strings
        df_out["output_path"] = [str(seq) for seq in pred_tokens]
        
    else:  # RNN
        predicted_paths = []
        for seq in pred_tokens:
            # Filter out padding and convert to tokens
            tokens = [vocab_obj.idx2token[int(idx)] for idx in seq if idx != pad_idx]
            predicted_paths.append(str(tokens))
        df_out["output_path"] = predicted_paths

    df_out.to_csv(output_csv, index=False)
    print(f"\nSaved predictions to: {output_csv}")
    print(f"  Total predictions: {len(pred_tokens)}")
    print(f"  Format: Python list string representation")
    
    # Show a few examples
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS (first 3 rows)")
    print("="*60)
    for i in range(min(3, len(pred_tokens))):
        if model_type == "transformer":
            tokens_preview = pred_tokens[i][:5]
            print(f"Row {i+1}: {tokens_preview}... (total: {len(pred_tokens[i])} tokens)")
        else:
            tokens = [vocab_obj.idx2token[int(idx)] for idx in pred_tokens[i][:5] if idx != pad_idx]
            total_tokens = len([idx for idx in pred_tokens[i] if idx != pad_idx])
            print(f"Row {i+1}: {tokens}... (total: {total_tokens} tokens)")
    print("="*60)


if __name__ == "__main__":
    main()
