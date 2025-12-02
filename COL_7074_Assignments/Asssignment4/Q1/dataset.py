import torch
from torch.utils.data import Dataset
import pandas as pd
import ast

class MazeDataset(Dataset):
    def __init__(self, csv_path, vocab, split='train', train_split=0.9):
       
        self.vocab = vocab
        self.data = pd.read_csv(csv_path)
        
        # Split data if it's the training CSV
        if split in ['train', 'val']:
            train_size = int(len(self.data) * train_split)
            if split == 'train':
                self.data = self.data[:train_size].reset_index(drop=True)
            else:  # val
                self.data = self.data[train_size:].reset_index(drop=True)
        
        print(f"{split.upper()} set: {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_tokens = ast.literal_eval(row['input_sequence'])
        output_tokens = ast.literal_eval(row['output_path'])
        maze_type = row['maze_type']

        input_indices = self.vocab.tokens_to_indices(input_tokens)
        output_indices = self.vocab.tokens_to_indices(output_tokens)

        return {
            'input_indices': input_indices,
            'output_indices': output_indices,
            'input_length': len(input_indices),
            'output_length': len(output_indices),
            'maze_type': maze_type
        }

# here dataloader will call to merge a list of examples into a single batch
def collate_fn(batch, pad_idx):
    max_input_len = max(item['input_length'] for item in batch)
    max_output_len = max(item['output_length'] for item in batch)

    batch_size = len(batch)
    input_padded = torch.full((batch_size, max_input_len), pad_idx, dtype=torch.long)
    output_padded = torch.full((batch_size, max_output_len), pad_idx, dtype=torch.long)
    input_lengths = torch.zeros(batch_size, dtype=torch.long)
    output_lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        input_seq = item['input_indices']
        output_seq = item['output_indices']
        input_padded[i, :len(input_seq)] = torch.tensor(input_seq, dtype=torch.long)
        output_padded[i, :len(output_seq)] = torch.tensor(output_seq, dtype=torch.long)
        input_lengths[i] = item['input_length']
        output_lengths[i] = item['output_length']

    input_mask = (input_padded != pad_idx)
    return {
        'input_padded': input_padded,
        'output_padded': output_padded,
        'input_lengths': input_lengths,
        'output_lengths': output_lengths,
        'input_mask': input_mask
    }
