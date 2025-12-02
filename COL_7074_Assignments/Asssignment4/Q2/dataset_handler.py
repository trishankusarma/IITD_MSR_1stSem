"""
dataset_handler.py
Dataset loading, vocabulary building, and data processing
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter

class MazeDataset(Dataset):
    """
    Dataset class for maze pathfinding
    """
    def __init__(self, csv_path, token_to_idx):
        """
        Args:
            csv_path: path to CSV file
            token_to_idx: dictionary mapping tokens to indices
        """
        self.data = pd.read_csv(csv_path)
        self.token_to_idx = token_to_idx
        
        print(f"Loaded {len(self.data)} samples from {csv_path}")
        
        # Count maze types
        forked_count = (self.data['maze_type'] == 'forked').sum()
        forkless_count = (self.data['maze_type'] == 'forkless').sum()
        print(f"  Forked mazes: {forked_count}")
        print(f"  Forkless mazes: {forkless_count}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq = eval(self.data.iloc[idx]['input_sequence'])
        output_seq = eval(self.data.iloc[idx]['output_path'])
        
        input_indices = [self.token_to_idx.get(token, self.token_to_idx['<UNK>']) 
                        for token in input_seq]
        output_indices = [self.token_to_idx.get(token, self.token_to_idx['<UNK>'])
                        for token in output_seq]
        
        return torch.tensor(input_indices), torch.tensor(output_indices)

def build_vocabulary(csv_path, min_freq=1):
    """
    Returns:
        token_to_idx: dict mapping tokens to indices
        idx_to_token: dict mapping indices to tokens
    """
    print("\n" + "="*60)
    print("Building vocabulary from training data...")
    print("="*60)
    
    df = pd.read_csv(csv_path)
    
    # Count all tokens
    token_counter = Counter()
    
    for i in range(len(df)):
        input_seq = eval(df.iloc[i]['input_sequence'])
        output_seq = eval(df.iloc[i]['output_path'])
        
        token_counter.update(input_seq)
        token_counter.update(output_seq)
    
    print(f"Found {len(token_counter)} unique tokens")
    
    # # Filter by frequency
    filtered_tokens = {token for token, count in token_counter.items() 
                      if count >= min_freq}
    
    # # Define special tokens
    special_tokens = [
        '<PAD>',         # Padding token (index 0)
        '<UNK>',         # Unknown token
    ]
    
    # Build vocabulary: special tokens first, then sorted regular tokens
    vocab = special_tokens + sorted(list(filtered_tokens - set(special_tokens)))
    
    # Create mappings
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    
    print(f"Vocabulary size: {len(token_to_idx)}")
    print(f"Special tokens: {special_tokens}")
    print(f"PAD index: {token_to_idx['<PAD>']}")
    
    return token_to_idx, idx_to_token

def collate_fn(batch, pad_idx):
    """
    Collate function for DataLoader - handles padding
    
    Args:
        batch: list of (input, output) tensor tuples
        pad_idx: index of padding token
    
    Returns:
        padded_inputs: (batch_size, max_input_len)
        padded_outputs: (batch_size, max_output_len)
    """
    inputs, outputs = zip(*batch)
    
    # Pad sequences to the max length in this batch
    padded_inputs = torch.nn.utils.rnn.pad_sequence(
        inputs, 
        batch_first=True, 
        padding_value=pad_idx
    )
    
    padded_outputs = torch.nn.utils.rnn.pad_sequence(
        outputs, 
        batch_first=True, 
        padding_value=pad_idx
    )
    
    return padded_inputs, padded_outputs


def analyze_dataset_statistics(csv_path):
    """
    Print useful statistics about the dataset
    """
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    df = pd.read_csv(csv_path)
    
    input_lengths = []
    output_lengths = []
    
    for i in range(len(df)):
        input_seq = eval(df.iloc[i]['input_sequence']) # to get the list of input tokens
        output_seq = eval(df.iloc[i]['output_path']) # to get the list of output tokens
        
        input_lengths.append(len(input_seq))
        output_lengths.append(len(output_seq))
    
    print(f"Total samples: {len(df)}")
    print(f"\nInput sequence lengths:")
    print(f"  Min: {min(input_lengths)}, Max: {max(input_lengths)}")
    
    print(f"\nOutput sequence lengths:")
    print(f"  Min: {min(output_lengths)}, Max: {max(output_lengths)}")
    
    print("="*60)


def create_dataloaders(train_csv, test_csv, token_to_idx, batch_size=32, 
                       val_split=0.1):
    """
    Create train, validation, and test dataloaders
    
    Args:
        train_csv: path to training CSV
        test_csv: path to test CSV
        token_to_idx: vocabulary mapping
        batch_size: batch size
        val_split: fraction of training data to use for validation
    
    Returns:
        train_loader, val_loader, test_loader
    """
    pad_idx = token_to_idx['<PAD>']
    
    # Create full training dataset
    full_train_dataset = MazeDataset(train_csv, token_to_idx)
    
    # Split into train and validation
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"\nDataset splits:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    
    # Create test dataset
    test_dataset = MazeDataset(test_csv, token_to_idx)
    print(f"  Test: {len(test_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader