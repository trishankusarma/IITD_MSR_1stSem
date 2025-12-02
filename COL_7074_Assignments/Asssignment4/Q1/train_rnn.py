# train_rnn.py
import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from .dataset import MazeDataset, collate_fn
from .model import Encoder, Decoder, Seq2Seq
from .vocabulary import Vocabulary

class Logger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.log = open(file_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        try:
            self.log.close()
        except Exception:
            pass

def start_logging(name="run"):
    global current_logger, original_stdout
    original_stdout = sys.stdout
    log_path = os.path.join("logs", f"output_{name}.txt")
    current_logger = Logger(log_path)
    sys.stdout = current_logger
    print("Logging started — all terminal output will also go to:", log_path)


def save_state_dict_only(model, filename='best_rnn_model.pt'):
    """Save model.state_dict() only."""
    torch.save(model.state_dict(), filename)
    print(f"Saved model.state_dict() -> {filename}")

def save_checkpoint(model, optimizer, epoch, best_val_seq_acc,token_to_idx,idx_to_token, filename='best_checkpoint.pth', extra=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_seq_acc': best_val_seq_acc,
        'time': time.asctime(),
        'token_to_idx': token_to_idx,
        'idx_to_token': idx_to_token

    }
    if extra is not None:
        checkpoint.update(extra)
    torch.save(checkpoint, filename)
    print(f"Saved checkpoint -> {filename}")

#hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 128
HIDDEN_DIM = 512
NUM_LAYERS = 2
TEACHER_FORCING_RATIO = 0.5


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    try:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except Exception:
        pass

#helper funcitons for metrics
def calculate_accuracy(predictions, targets, pad_idx):
    pred_tokens = predictions.argmax(dim=2)
    mask = (targets != pad_idx)
    if mask.sum().item() == 0:
        return 0.0
    correct = ((pred_tokens == targets) & mask).sum().item()
    return correct / mask.sum().item()

def calculate_sequence_accuracy(predictions, targets, pad_idx):
    pred_tokens = predictions.argmax(dim=2)
    mask = (targets != pad_idx)

    correct_sequences = ((pred_tokens == targets) | ~mask).all(dim=1)
    return correct_sequences.sum().item() / pred_tokens.size(0)

def calculate_sklearn_f1(predictions, targets, pad_idx, average='micro'):
    pred_tokens = predictions.argmax(dim=2)
    mask = (targets != pad_idx)
    if mask.sum().item() == 0:
        return 0.0
    pred_flat = pred_tokens[mask].cpu().numpy()
    target_flat = targets[mask].cpu().numpy()
    return f1_score(target_flat, pred_flat, average=average, zero_division=0)

#training and evaluation loops
def train_epoch(model, dataloader, optimizer, criterion, pad_idx, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0.0
    epoch_token_acc = 0.0
    epoch_seq_acc = 0.0
    epoch_f1 = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        input_seqs = batch['input_padded'].to(device)
        target_seqs = batch['output_padded'].to(device)
        input_lengths = batch['input_lengths']

        optimizer.zero_grad()
        outputs = model(input_seqs, input_lengths, target_seqs, teacher_forcing_ratio)

        #ignored first token
        outputs_flat = outputs[:, 1:, :].reshape(-1, outputs.size(2))
        targets_flat = target_seqs[:, 1:].reshape(-1)

        loss = criterion(outputs_flat, targets_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            token_acc = calculate_accuracy(outputs[:, 1:, :], target_seqs[:, 1:], pad_idx)
            seq_acc = calculate_sequence_accuracy(outputs[:, 1:, :], target_seqs[:, 1:], pad_idx)
            f1 = calculate_sklearn_f1(outputs[:, 1:, :], target_seqs[:, 1:], pad_idx, average='micro')

        epoch_loss += loss.item()
        epoch_token_acc += token_acc
        epoch_seq_acc += seq_acc
        epoch_f1 += f1

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'token_acc': f'{token_acc:.4f}',
            'seq_acc': f'{seq_acc:.4f}',
            'f1': f'{f1:.4f}'
        })

    n = len(dataloader)
    return (epoch_loss / n,
            epoch_token_acc / n,
            epoch_seq_acc / n,
            epoch_f1 / n)

def evaluate(model, dataloader, criterion, pad_idx, sklearn_average='micro', send_preds = False):
    model.eval()
    epoch_loss = 0.0
    epoch_token_acc = 0.0
    epoch_seq_acc = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_seqs = batch['input_padded'].to(device)
            target_seqs = batch['output_padded'].to(device)
            input_lengths = batch['input_lengths']

            outputs = model(input_seqs, input_lengths, target_seqs, teacher_forcing_ratio=0.0)

            outputs_flat = outputs[:, 1:, :].reshape(-1, outputs.size(2))
            targets_flat = target_seqs[:, 1:].reshape(-1)
            loss = criterion(outputs_flat, targets_flat)

            token_acc = calculate_accuracy(outputs[:, 1:, :], target_seqs[:, 1:], pad_idx)
            seq_acc = calculate_sequence_accuracy(outputs[:, 1:, :], target_seqs[:, 1:], pad_idx)

            # collect for sklearn F1 (non-pad positions)
            pred_tokens = outputs[:, 1:, :].argmax(dim=2)
            mask = (target_seqs[:, 1:] != pad_idx)
            if mask.sum().item() > 0:
                all_preds.append(pred_tokens[mask].cpu().numpy())
                all_targets.append(target_seqs[:, 1:][mask].cpu().numpy())

            epoch_loss += loss.item()
            epoch_token_acc += token_acc
            epoch_seq_acc += seq_acc

    if len(all_preds) > 0:
        import numpy as _np
        preds_concat = _np.concatenate(all_preds)
        targets_concat = _np.concatenate(all_targets)
        from sklearn.metrics import f1_score as _f1
        dataset_f1 = _f1(targets_concat, preds_concat, average=sklearn_average, zero_division=0)
    else:
        dataset_f1 = 0.0

    n_batches = len(dataloader)

    if send_preds:
        # Convert last batch predictions → Python list of list of ints
        pred_tokens_list = pred_tokens.cpu().numpy().tolist()

        return (epoch_loss / n_batches,
                epoch_token_acc / n_batches,
                epoch_seq_acc / n_batches,
                dataset_f1,
                pred_tokens_list)

    return (epoch_loss / n_batches,
            epoch_token_acc / n_batches,
            epoch_seq_acc / n_batches,
            dataset_f1, pred_tokens)

#main
def main():
    start_logging("rnn_run")

    #loading vocabulary
    vocab = Vocabulary()
    vocab.load_vocab('vocabulary.json')
    pad_idx = vocab.token2idx['<PAD>']

    print(f"\nVocabulary size: {len(vocab)}")
    print(f"PAD token index: {pad_idx}")

    #creating dataset
    train_dataset = MazeDataset('data/train_6x6_mazes.csv', vocab, split='train', train_split=0.9)
    val_dataset = MazeDataset('data/train_6x6_mazes.csv', vocab, split='val', train_split=0.9)
    test_dataset = MazeDataset('data/test_6x6_mazes.csv', vocab, split='test')

    #dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda batch: collate_fn(batch, pad_idx=pad_idx))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=lambda batch: collate_fn(batch, pad_idx=pad_idx))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=lambda batch: collate_fn(batch, pad_idx=pad_idx))

    #model
    encoder = Encoder(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    decoder = Decoder(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    #loss and optimizers
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 12 13 0 0 0
    # 15 0 0 0 0 

    # 12 13
    # 15

    #history
    history = {
        'train_loss': [], 'train_token_acc': [], 'train_seq_acc': [], 'train_f1': [],
        'val_loss': [], 'val_token_acc': [], 'val_seq_acc': [], 'val_f1': [],
        'test_loss': [], 'test_token_acc': [], 'test_seq_acc': [], 'test_f1': []
    }

    best_val_seq_acc = 0.0

    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_token_acc, train_seq_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, pad_idx, TEACHER_FORCING_RATIO
        )

        val_loss, val_token_acc, val_seq_acc, val_f1 = evaluate(
            model, val_loader, criterion, pad_idx, sklearn_average='micro'
        )

        test_loss, test_token_acc, test_seq_acc, test_f1 = evaluate(
            model, test_loader, criterion, pad_idx, sklearn_average='micro'
        )

        history['train_loss'].append(train_loss)
        history['train_token_acc'].append(train_token_acc)
        history['train_seq_acc'].append(train_seq_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_token_acc'].append(val_token_acc)
        history['val_seq_acc'].append(val_seq_acc)
        history['val_f1'].append(val_f1)
        history['test_loss'].append(test_loss)
        history['test_token_acc'].append(test_token_acc)
        history['test_seq_acc'].append(test_seq_acc)
        history['test_f1'].append(test_f1)

        #persist history each epoch (safe if job interrupted)
        with open('rnn_training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Print
        print(f"\nTrain Loss: {train_loss:.4f} | Token Acc: {train_token_acc:.4f} | Seq Acc: {train_seq_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Token Acc: {val_token_acc:.4f} | Seq Acc: {val_seq_acc:.4f} | F1: {val_f1:.4f}")
        print(f"Test  Loss: {test_loss:.4f} | Token Acc: {test_token_acc:.4f} | Seq Acc: {test_seq_acc:.4f} | F1: {test_f1:.4f}")

        #Saveing best model (based on validation sequence accuracy)
        if val_seq_acc > best_val_seq_acc:
            best_val_seq_acc = val_seq_acc

            save_state_dict_only(model, filename='best_rnn_model.pt')
            # saving checkpoint with optimizer for resuming
            save_checkpoint(model, optimizer, epoch, best_val_seq_acc,vocab.token2idx,vocab.idx2token, filename='best_checkpoint.pth',
                            extra={'vocab_size': len(vocab)})

    # final save of history and final model state
    with open('rnn_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # final evaluation (print)
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation sequence accuracy: {best_val_seq_acc:.4f}")

    print("\n" + "="*60)
    print("EVALUATING ON TEST SET (FINAL)")
    print("="*60)
    test_loss, test_token_acc, test_seq_acc, test_f1 = evaluate(
        model, test_loader, criterion, pad_idx, sklearn_average='micro'
    )
    print(f"\nTest Loss: {test_loss:.4f} | Token Acc: {test_token_acc:.4f} | Seq Acc: {test_seq_acc:.4f} | F1: {test_f1:.4f}")

    save_state_dict_only(model, filename='final_rnn_model.pt')

    #stop logging and restore stdout
    try:
        sys.stdout = original_stdout
        current_logger.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
