"""
train_transformer.py

Usage:
    python train_transformer.py --train_csv train.csv --test_csv test.csv
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import json
from datetime import datetime
import time
import sys
from tqdm import tqdm

from .utils.loggerUtils import Logger
from .utils.plotUtils import plot_training_curves
from .transformer_model import TransformerMazeSolver
from .dataset_handler import (build_vocabulary, create_dataloaders, 
                             analyze_dataset_statistics)

# DATASET_PATHS
TRAIN_SET_PATH = './dataset/train_6x6_mazes.csv'
TEST_SET_PATH = './dataset/test_6x6_mazes.csv'

# DEFAULT HYPER-PARAMETERS
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

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

os.makedirs("logs", exist_ok=True)
os.makedirs("runs", exist_ok=True)

def set_seed(seed=SEED):
    """
    Set all random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make cuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"All random seeds set to {seed} for reproducibility")

def logging(s='1', output_directory = "logs"):
    global current_logger
    log_path = os.path.join(f"{output_directory}", f'output_{s}.txt')

    # If stdout is already a Logger, unwrap to real stdout
    if isinstance(sys.stdout, Logger):
        sys.stdout = sys.stdout.terminal

    # Initialize new logger (always starts fresh)
    current_logger = Logger(log_path)
    sys.stdout = current_logger

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"Logging started at {timestamp}")
    print(f"Log file   : {log_path}")
    print(f"Using device: {device}")
    print("=" * 70)

def train_one_epoch(model, dataloader, optimizer, criterion, device, pad_idx, epoch):
    model.train()
    total_loss = 0
    total_correct_tokens = 0
    total_tokens = 0
    batch_times = []
    
    for batch_idx, (src, tgt) in enumerate(dataloader):
        batch_start = time.time()
        
        # Move to GPU (non_blocking for async transfer)
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        
        # Shift target for teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # Regular forward pass
        output = model(src, tgt_input, pad_idx, pad_idx)
        output_flat = output.reshape(-1, output.shape[-1])
        tgt_output_flat = tgt_output.reshape(-1)
        loss = criterion(output_flat, tgt_output_flat)
            
        # Regular backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy (moved to CPU only when needed)
        with torch.no_grad():
            mask = (tgt_output_flat != pad_idx)
            predictions = output_flat.argmax(dim=-1)
            correct = ((predictions == tgt_output_flat) & mask).sum().item()
            total_correct_tokens += correct
            total_tokens += mask.sum().item()
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Print progress
        if (batch_idx + 1) % 50 == 0:
            batch_acc = correct / mask.sum().item() if mask.sum().item() > 0 else 0
            avg_batch_time = sum(batch_times[-50:]) / len(batch_times[-50:])
            print(f"  Batch {batch_idx+1}/{len(dataloader)}, "
                  f"Loss: {loss.item():.4f}, "
                  f"Acc: {batch_acc:.4f}, "
                  f"Time: {avg_batch_time:.3f}s/batch")
    
    avg_loss = total_loss / len(dataloader)
    token_accuracy = total_correct_tokens / total_tokens if total_tokens > 0 else 0
    avg_batch_time = sum(batch_times) / len(batch_times)
    
    return avg_loss, token_accuracy, avg_batch_time

def calculate_position_aware_f1(pred_sequences, target_sequences, pad_idx):
    """
    FIXED: Calculate F1 for ORDERED sequence matching (position-aware)
    This gives partial credit based on position-by-position matches
    
    For maze paths, order matters! This is the correct F1 for sequential data.
    """
    batch_size = pred_sequences.size(0)
    f1_scores = []
    precisions = []
    recalls = []
    
    for i in range(batch_size):
        pred_seq = pred_sequences[i]
        target_seq = target_sequences[i]
        
        # Remove padding from target (ground truth determines length)
        mask = (target_seq != pad_idx)
        target_seq_clean = target_seq[mask]
        pred_seq_clean = pred_seq[mask]
        
        if len(target_seq_clean) == 0:
            continue
        
        # Handle length mismatch: compare up to minimum length
        min_len = min(len(pred_seq_clean), len(target_seq_clean))
        
        if min_len == 0:
            f1_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            continue
        
        # Position-by-position matching (order matters!)
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

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, pad_idx):
    """
    FIXED evaluation with correct F1 score calculation
    
    Returns:
        avg_loss, token_accuracy, seq_accuracy, f1, precision, recall
    """
    model.eval()
    total_loss = 0
    total_correct_tokens = 0
    total_tokens = 0
    total_correct_sequences = 0
    total_sequences = 0
    
    # For F1 calculation - accumulate scores per batch instead of concatenating
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for src, tgt in dataloader:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # Forward pass
        output = model(src, tgt_input, pad_idx, pad_idx)
        output_flat = output.reshape(-1, output.shape[-1])
        tgt_output_flat = tgt_output.reshape(-1)
        loss = criterion(output_flat, tgt_output_flat)
        
        total_loss += loss.item()
        
        # Get predictions
        predictions = output_flat.argmax(dim=-1)
        pred_sequences = output.argmax(dim=-1)
        
        # Token accuracy
        mask = (tgt_output_flat != pad_idx)
        total_correct_tokens += ((predictions == tgt_output_flat) & mask).sum().item()
        total_tokens += mask.sum().item()
        
        # Sequence accuracy (exact match)
        mask_2d = (tgt_output != pad_idx)
        correct_seq = ((pred_sequences == tgt_output) | ~mask_2d).all(dim=1)
        total_correct_sequences += correct_seq.sum().item()
        total_sequences += src.size(0)
        
        # Calculate F1 for this batch
        batch_f1, batch_prec, batch_rec = calculate_position_aware_f1(
            pred_sequences, tgt_output, pad_idx
        )
        f1_scores.append(batch_f1)
        precision_scores.append(batch_prec)
        recall_scores.append(batch_rec)
    
    # Calculate overall metrics
    avg_loss = total_loss / len(dataloader)
    token_accuracy = total_correct_tokens / total_tokens if total_tokens > 0 else 0
    seq_accuracy = total_correct_sequences / total_sequences if total_sequences > 0 else 0
    
    # Average F1 scores across all batches
    seq_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    seq_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    seq_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    
    return (avg_loss, token_accuracy, seq_accuracy, 
            seq_f1, seq_precision, seq_recall)

import torch
from tqdm import tqdm

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

        # ---- Start decoder with <PATH_START> ----
        decoder_input = torch.full(
            (batch_size, 1), path_start_idx, dtype=torch.long, device=device
        )

        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # ---- Greedy Decoding ----
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

        # ---- Remove <PATH_START> before comparison ----
        generated_output = decoder_input[:, 1:]
        
        # ---- Store predictions if requested ----
        if return_predictions:
            # Store raw token indices for this batch
            for i in range(batch_size):
                pred_seq = generated_output[i].cpu().tolist()
                gt_seq = tgt_output[i].cpu().tolist()
                all_predictions.append(pred_seq)
                all_ground_truths.append(gt_seq)

        # ---- Align lengths by padding ----
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

        # ---- Token-Level Accuracy ----
        mask = (tgt_output != pad_idx)
        correct_tokens = ((generated_output == tgt_output) & mask).sum().item()
        total_correct_tokens += correct_tokens
        total_tokens += mask.sum().item()

        # ---- Sequence-Level Accuracy ----
        exact_match = ((generated_output == tgt_output) | ~mask).all(dim=1)
        total_correct_sequences += exact_match.sum().item()
        total_sequences += batch_size

        # ---- F1 Score ----
        f1, p, r = calculate_position_aware_f1(
            generated_output, tgt_output, pad_idx
        )
        f1_scores.append(f1)
        precision_scores.append(p)
        recall_scores.append(r)

    # ---- Final Metrics ----
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
            gt_token_sequences.append(gt_tokens)
        
        return token_acc, seq_acc, avg_f1, avg_prec, avg_rec, pred_token_sequences, gt_token_sequences
    
    return token_acc, seq_acc, avg_f1, avg_prec, avg_rec

def print_gpu_stats():
    """Print GPU memory statistics"""
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def main(args):
    # Set random seeds
    set_seed(seed=SEED)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"runs/transformer_optimized_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    logging(s="Q2", output_directory = output_dir)
    
    print(f"\n{'='*60}")
    print(f"GPU-OPTIMIZED TRANSFORMER TRAINING (FIXED F1)")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
        print_gpu_stats()
    else:
        print("WARNING: CUDA not available, using CPU (will be slow)")
    
    # Analyze dataset
    analyze_dataset_statistics(args.train_csv)
    
    # Build vocabulary
    token_to_idx, idx_to_token = build_vocabulary(args.train_csv)
    vocab_size = len(token_to_idx)
    pad_idx = token_to_idx['<PAD>']
    
    # Save vocabulary
    vocab_path = os.path.join(output_dir, 'vocabulary.json')
    with open(vocab_path, 'w') as f:
        json.dump({
            'token_to_idx': token_to_idx,
            'idx_to_token': idx_to_token
        }, f, indent=2)
    
    # Create dataloaders with optimized settings
    print("\n" + "="*60)
    print("Creating dataloaders...")
    print("="*60)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        args.train_csv,
        args.test_csv,
        token_to_idx,
        batch_size=args.batch_size,
        val_split=args.val_split
    )
    
    # Initialize model
    print("\n" + "="*60)
    print("Initializing model...")
    print("="*60)
    
    model = TransformerMazeSolver(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if torch.cuda.is_available():
        print_gpu_stats()
    
    # Initialize optimizer and loss
    optimizer = optim.AdamW(  # AdamW is more stable than Adam
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_token_acc': [],
        'val_loss': [],
        'val_token_acc': [],
        'val_seq_acc': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'batch_time': []
    }
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    best_val_seq_acc = 0.0
    best_val_f1 = 0.0
    total_train_time = 0
    
    for epoch in tqdm(range(args.epochs), desc = "Training Episodes"):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_token_acc, avg_batch_time = train_one_epoch(
            model, train_loader, optimizer, criterion, device, pad_idx, 
            epoch
        )
        
        # Validate
        (val_loss, val_token_acc, val_seq_acc, 
         val_f1, val_precision, val_recall) = evaluate(
            model, val_loader, criterion, device, pad_idx
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_token_acc'].append(train_token_acc)
        history['val_loss'].append(val_loss)
        history['val_token_acc'].append(val_token_acc)
        history['val_seq_acc'].append(val_seq_acc)
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['batch_time'].append(avg_batch_time)
        
        # Print results
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train - Loss: {train_loss:.4f}, Token Acc: {train_token_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Token Acc: {val_token_acc:.4f}, Seq Acc: {val_seq_acc:.4f}")
        print(f"  Val   - F1: {val_f1:.4f} (P: {val_precision:.4f}, R: {val_recall:.4f})")
        print(f"  Time  - Epoch: {epoch_time:.2f}s, Avg Batch: {avg_batch_time:.3f}s")
        print(f"  LR    - {optimizer.param_groups[0]['lr']:.6f}")
        
        if torch.cuda.is_available():
            print_gpu_stats()
        
        # Save best model (based on sequence accuracy)
        if val_seq_acc > best_val_seq_acc:
            best_val_seq_acc = val_seq_acc
            best_val_f1 = val_f1
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_seq_acc': val_seq_acc,
                'val_f1': val_f1,
                'hyperparameters': vars(args),
                'token_to_idx': token_to_idx,
                'idx_to_token': idx_to_token
            }
            torch.save(checkpoint, os.path.join(output_dir, 'best_model.pth'))
            print(f"✓ Saved best model (Seq Acc: {val_seq_acc:.4f}, F1: {val_f1:.4f})")
    
    # Plot training curves
    print("\n" + "="*60)
    print("GENERATING TRAINING PLOTS")
    print("="*60)
    plot_training_curves(history, output_dir)
    
    # Final evaluation on test set WITH teacher forcing
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET (WITH TEACHER FORCING)")
    print("="*60)
    
    test_start = time.time()
    (test_loss, test_token_acc, test_seq_acc,
     test_f1, test_precision, test_recall) = evaluate(
        model, test_loader, criterion, device, pad_idx
    )
    test_time = time.time() - test_start
    
    print(f"\nTest Results (Teacher Forcing):")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Token Accuracy: {test_token_acc:.4f}")
    print(f"  Sequence Accuracy: {test_seq_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  Evaluation Time: {test_time:.2f}s")
    
    # Evaluation WITHOUT teacher forcing (True generative test)
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET (NO TEACHER FORCING)")
    print("="*60)
    print("Running greedy decoding on test set...")
    
    ntf_start = time.time()
    test_token_acc_ntf, test_seq_acc_ntf, test_f1_ntf, test_prec_ntf, test_rec_ntf = \
        evaluate_no_teacher_forcing(model, test_loader, device, token_to_idx, max_len=100)
    ntf_time = time.time() - ntf_start
    
    print(f"\nTest Results (No Teacher Forcing):")
    print(f"  Token Accuracy: {test_token_acc_ntf:.4f}")
    print(f"  Sequence Accuracy: {test_seq_acc_ntf:.4f}")
    print(f"  F1 Score: {test_f1_ntf:.4f}")
    print(f"  Precision: {test_prec_ntf:.4f}")
    print(f"  Recall: {test_rec_ntf:.4f}")
    print(f"  Evaluation Time: {ntf_time:.2f}s")
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON: TEACHER FORCING vs NO TEACHER FORCING")
    print("="*60)
    print(f"Token Accuracy:    {test_token_acc:.4f} vs {test_token_acc_ntf:.4f} " +
          f"(Δ: {(test_token_acc - test_token_acc_ntf)*100:+.2f}%)")
    print(f"Sequence Accuracy: {test_seq_acc:.4f} vs {test_seq_acc_ntf:.4f} " +
          f"(Δ: {(test_seq_acc - test_seq_acc_ntf)*100:+.2f}%)")
    print(f"F1 Score:          {test_f1:.4f} vs {test_f1_ntf:.4f} " +
          f"(Δ: {(test_f1 - test_f1_ntf)*100:+.2f}%)")

    # Save final results
    results = {
        'best_val_seq_acc': best_val_seq_acc,
        'best_val_f1': best_val_f1,
        'test_with_teacher_forcing': {
            'loss': test_loss,
            'token_acc': test_token_acc,
            'seq_acc': test_seq_acc,
            'f1': test_f1,
            'precision': test_precision,
            'recall': test_recall,
            'eval_time': test_time
        },
        'test_no_teacher_forcing': {
            'token_acc': test_token_acc_ntf,
            'seq_acc': test_seq_acc_ntf,
            'f1': test_f1_ntf,
            'precision': test_prec_ntf,
            'recall': test_rec_ntf,
            'eval_time': ntf_time
        },
        'total_train_time': total_train_time,
        'avg_epoch_time': total_train_time / args.epochs,
        'history': history,
        'hyperparameters': vars(args)
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total Training Time: {total_train_time/60:.2f} minutes")
    print(f"\nBest Validation Performance:")
    print(f"  Sequence Accuracy: {best_val_seq_acc:.4f}")
    print(f"  F1 Score: {best_val_f1:.4f}")
    print(f"\nFinal Test Performance (Teacher Forcing):")
    print(f"  Sequence Accuracy: {test_seq_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"\nFinal Test Performance (No Teacher Forcing):")
    print(f"  Sequence Accuracy: {test_seq_acc_ntf:.4f}")
    print(f"  F1 Score: {test_f1_ntf:.4f}")
    print(f"\nAll results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPU-Optimized Transformer Training (Fixed F1)')

    # Data arguments
    parser.add_argument('--train_csv', type=str, default=TRAIN_SET_PATH,
                       help='Path to training CSV file')
    parser.add_argument('--test_csv', type=str, default=TEST_SET_PATH,
                       help='Path to test CSV file')
    parser.add_argument('--val_split', type=float, default=VAL_SPLIT,
                       help='Validation split ratio')
    
    # Model hyperparameters (as specified in assignment)
    parser.add_argument('--d_model', type=int, default=D_MODEL,
                       help='Embedding dimension')
    parser.add_argument('--nhead', type=int, default=NHEAD,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS,
                       help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=DIM_FEEDFORWARD,
                       help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=DROPOUT,
                       help='Dropout rate')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    main(args)