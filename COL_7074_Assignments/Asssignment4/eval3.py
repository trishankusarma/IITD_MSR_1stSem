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
from Q1.train_rnn import evaluate
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


# Load vocabulary for RNN (Q1)
def load_vocab_json1():
    vocab_path = os.path.join("Q1", "vocabulary.json")
    vocab = Vocabulary()
    vocab.load_vocab(vocab_path)
    return vocab


# Load vocabulary for Transformer (Q2)
def load_vocab_json2():
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


# Load Transformer model
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
        # If checkpoint has nested structure
        print("Loading from 'model_state_dict' key")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct state dict (just weights)
        print("Loading weights directly")
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✓ Transformer model loaded successfully")
    
    return model, token_to_idx, idx_to_token


# Load RNN model
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
        # If checkpoint has nested structure
        print("Loading from 'model_state_dict' key")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct state dict (just weights)
        print("Loading weights directly")
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✓ RNN model loaded successfully")

    return model, vocab


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


# For transformer
@torch.no_grad()
def evaluate_no_teacher_forcing(model, dataloader, device, token_to_idx, max_len=100, 
                                return_predictions=False):
    """
    Evaluate the model WITHOUT teacher forcing.
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
            # Convert predictions (remove padding, stop at first <PATH_END>)
            pred_tokens = []
            for idx in pred_indices:
                if idx == pad_idx:
                    break  # Stop at first padding
                token = idx_to_token[idx]
                pred_tokens.append(token)
                if idx == path_end_idx:
                    break  # Stop at first <PATH_END>
            pred_token_sequences.append(pred_tokens)
            
            # Convert ground truths (remove padding, stop at first <PATH_END>)
            gt_tokens = []
            for idx in gt_indices:
                if idx == pad_idx:
                    break  # Stop at first padding
                token = idx_to_token[idx]
                gt_tokens.append(token)
                if idx == path_end_idx:
                    break  # Stop at first <PATH_END>
            gt_token_sequences.append(gt_tokens)
        
        return token_acc, seq_acc, avg_f1, avg_prec, avg_rec, pred_token_sequences, gt_token_sequences
    
    return token_acc, seq_acc, avg_f1, avg_prec, avg_rec


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
    
    # Validate inputs
    import os
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
        vocab_obj = None  # Not needed for transformer
    
    # RNN EVALUATION
    elif model_type == "rnn":
        model, vocab = load_rnn(model_path, device)

        pad_idx = vocab.token2idx["<PAD>"]
        path_start_idx = vocab.token2idx["<PATH_START>"]
        path_end_idx = vocab.token2idx["<PATH_END>"]
        
        # Load test data
        df_test = pd.read_csv(test_csv)
        
        # Check if we have ground truth
        has_ground_truth = False
        try:
            first_output = df_test.iloc[0]['output_path']
            if pd.notna(first_output) and first_output != '':
                eval(first_output)  # Try to parse it
                has_ground_truth = True
        except:
            has_ground_truth = False
        
        # If no ground truth, create dummy output_path for dataset
        if not has_ground_truth:
            print("No ground truth - creating dummy output paths for dataset loading")
            df_test['output_path'] = df_test['output_path'].fillna("['<PATH_START>', '<PATH_END>']")
            # Save temporary file
            temp_csv = test_csv.replace('.csv', '_temp.csv')
            df_test.to_csv(temp_csv, index=False)
            test_csv_to_load = temp_csv
        else:
            test_csv_to_load = test_csv

        test_dataset = MazeDatasetRNN(test_csv_to_load, vocab, split='test')
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=lambda batch: collate_fnRNN(batch, pad_idx)
        )

        print("\n" + "="*60)
        print("EVALUATING RNN (Greedy Decoding)")
        print("="*60)
        
        if has_ground_truth:
            print("Ground truth available - computing metrics")
        else:
            print("No ground truth available - generating predictions only")

        # Greedy decoding for RNN
        all_pred_tokens = []
        
        if has_ground_truth:
            total_correct_tokens = 0
            total_tokens = 0
            total_correct_sequences = 0
            total_sequences = 0
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Generating predictions"):
                input_seqs = batch["input_padded"].to(device)
                input_lengths = batch["input_lengths"]
                batch_size = input_seqs.size(0)
                max_len = 100

                # Encode input
                encoder_outputs, hidden = model.encoder(input_seqs, input_lengths)
                
                # Create mask for attention
                mask = (input_seqs != pad_idx).long()
                
                # Initialize decoder input with <PATH_START> token
                decoder_input = torch.full((batch_size,), path_start_idx, 
                                          dtype=torch.long, device=device)
                
                # Initialize decoder hidden state
                decoder_hidden = hidden
                
                # Storage for generated sequences
                generated_tokens = []
                
                # Greedy decoding step by step
                for t in range(max_len):
                    # Decoder forward (expects 1D input: batch_size)
                    # The decoder will unsqueeze internally
                    output, decoder_hidden, _ = model.decoder(
                        decoder_input,  # (batch_size,) - NO unsqueeze needed!
                        decoder_hidden, 
                        encoder_outputs,
                        mask
                    )
                    
                    # Get predictions (greedy - take argmax)
                    # output shape: (batch_size, vocab_size)
                    next_tokens = output.argmax(dim=-1)  # (batch_size,)
                    
                    # Store predictions
                    generated_tokens.append(next_tokens.unsqueeze(1))
                    
                    # Use predicted token as next input
                    decoder_input = next_tokens
                    
                    # Check if all sequences have generated <PATH_END>
                    if (next_tokens == path_end_idx).all():
                        break
                
                # Concatenate all generated tokens
                generated_output = torch.cat(generated_tokens, dim=1)  # (batch_size, seq_len)
                
                # Convert to token strings
                for i in range(batch_size):
                    pred_indices = generated_output[i].cpu().tolist()
                    pred_tokens = []
                    for idx in pred_indices:
                        token = vocab.idx2token[idx]
                        pred_tokens.append(token)
                        if idx == path_end_idx:
                            break
                    all_pred_tokens.append(pred_tokens)
                
                # Calculate metrics if ground truth available
                if has_ground_truth:
                    target_seqs = batch["output_padded"].to(device)
                    target_seqs_no_start = target_seqs[:, 1:]
                    
                    max_compare_len = max(generated_output.size(1), target_seqs_no_start.size(1))
                    
                    if generated_output.size(1) < max_compare_len:
                        pad_tensor = torch.full(
                            (batch_size, max_compare_len - generated_output.size(1)),
                            pad_idx, device=device
                        )
                        generated_output = torch.cat([generated_output, pad_tensor], dim=1)
                    else:
                        generated_output = generated_output[:, :max_compare_len]
                    
                    if target_seqs_no_start.size(1) < max_compare_len:
                        pad_tensor = torch.full(
                            (batch_size, max_compare_len - target_seqs_no_start.size(1)),
                            pad_idx, device=device
                        )
                        target_seqs_no_start = torch.cat([target_seqs_no_start, pad_tensor], dim=1)
                    else:
                        target_seqs_no_start = target_seqs_no_start[:, :max_compare_len]
                    
                    mask = (target_seqs_no_start != pad_idx)
                    correct_tokens = ((generated_output == target_seqs_no_start) & mask).sum().item()
                    total_correct_tokens += correct_tokens
                    total_tokens += mask.sum().item()
                    
                    exact_match = ((generated_output == target_seqs_no_start) | ~mask).all(dim=1)
                    total_correct_sequences += exact_match.sum().item()
                    total_sequences += batch_size
        
        # Clean up temporary file if created
        if not has_ground_truth:
            import os
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
        
        # Calculate final metrics
        if has_ground_truth:
            token_acc = total_correct_tokens / total_tokens if total_tokens > 0 else 0
            seq_acc = total_correct_sequences / total_sequences if total_sequences > 0 else 0
        else:
            token_acc = seq_acc = 0.0
        
        f1 = 0.0

        pred_tokens = all_pred_tokens
        vocab_obj = vocab

    
    # PRINT RESULTS
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    if token_acc > 0 or seq_acc > 0:  # If metrics were computed
        print(f"Token Accuracy:    {token_acc:.4f}")
        print(f"Sequence Accuracy: {seq_acc:.4f}")
        print(f"F1 Score:          {f1:.4f}")
    else:
        print("Predictions generated (no ground truth for evaluation)")
    print("="*60)

    
    # SAVE OUTPUT CSV
    df_out = pd.read_csv(test_csv)

    # Ensure same number of predictions and test rows
    if len(pred_tokens) != len(df_out):
        print(f"\nERROR: Number of predictions ({len(pred_tokens)}) does not match dataset size ({len(df_out)})")
        sys.exit(1)

    # Convert predictions to Python list string format (matching the training format)
    # Both transformer and RNN now return List[List[str]] (token strings)
    df_out["output_path"] = [str(seq) for seq in pred_tokens]

    # Save to CSV
    df_out.to_csv(output_csv, index=False)
    print(f"\n✓ Saved predictions to: {output_csv}")
    print(f"  Total predictions: {len(pred_tokens)}")
    print(f"  Format: Python list string representation")
    
    # Show a few examples
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS (first 3 rows)")
    print("="*60)
    for i in range(min(3, len(pred_tokens))):
        # Show first few tokens from the list
        tokens_preview = pred_tokens[i][:5] if len(pred_tokens[i]) > 5 else pred_tokens[i]
        print(f"Row {i+1}: {tokens_preview}... (total: {len(pred_tokens[i])} tokens)")
    print("="*60)


if __name__ == "__main__":
    main()