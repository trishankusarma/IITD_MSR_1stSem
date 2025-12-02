# vocabulary.py
import pandas as pd
import ast
from collections import Counter
import json

class Vocabulary:
    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        self.token_counts = Counter()
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        
        # Initialize with PAD and UNK
        self.add_token(self.PAD_TOKEN)  # usually idx 0
        self.add_token(self.UNK_TOKEN)  # usually idx 1
        
    def add_token(self, token):
 
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        self.token_counts[token] += 1
    
    def build_vocab(self, data_path, train_split=0.9):

        print("Building vocabulary from training split only...")
        data = pd.read_csv(data_path)
        
        #using first train_split fraction as training for vocab-building
        train_size = int(len(data) * train_split)
        train_data = data[:train_size]
        
        print(f"Using {len(train_data)} examples out of {len(data)} for vocabulary building")
        
        for idx, row in train_data.iterrows():
            try:
                input_seq = ast.literal_eval(row['input_sequence'])
            except Exception:
                input_seq = []
            for token in input_seq:
                self.add_token(token)
            
            try:
                output_path = ast.literal_eval(row['output_path'])
            except Exception:
                output_path = []
            for token in output_path:
                self.add_token(token)
        
        print(f"Vocabulary built! Total unique tokens: {len(self.token2idx)}")
        print(f"Special tokens: {self.PAD_TOKEN} (idx: {self.token2idx[self.PAD_TOKEN]}), "
              f"{self.UNK_TOKEN} (idx: {self.token2idx[self.UNK_TOKEN]})")
        keys = list(self.token2idx.keys())
        if len(keys) > 2:
            print(f"Sample tokens: {keys[2:12]}")
        return self
    
    def tokens_to_indices(self, tokens):
        unk_idx = self.token2idx[self.UNK_TOKEN]
        return [self.token2idx.get(token, unk_idx) for token in tokens]
    
    def indices_to_tokens(self, indices):

        return [self.idx2token.get(int(idx), self.UNK_TOKEN) for idx in indices]
    
    def __len__(self):
        return len(self.token2idx)
    
    def save_vocab(self, filepath):
        vocab_data = {
            'token2idx': self.token2idx,
            'idx2token': {str(k): v for k, v in self.idx2token.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        print(f"Vocabulary saved to {filepath}")
    
    def load_vocab(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        self.token2idx = vocab_data['token2idx']

        self.idx2token = {int(k): v for k, v in vocab_data['idx2token'].items()}
        
        self.token_counts = Counter()
        print(f"Vocabulary loaded from {filepath}")
        print(f"Vocabulary size: {len(self.token2idx)}")
        return self

# Quick test when run directly
# if __name__ == "__main__":
#     vocab = Vocabulary()
#     # build with 90% train split by default
#     vocab.build_vocab('data/train_6x6_mazes.csv', train_split=0.9)
#     vocab.save_vocab('vocabulary.json')
    
#     print("\n" + "="*60)
#     print(f"FINAL VOCABULARY SIZE: {len(vocab)}")
#     print("="*60)
    
#     # Test encode/decode
#     test_tokens = ['<ADJLIST_START>', '(0,0)', '<-->', '(1,0)', ';']
#     test_indices = vocab.tokens_to_indices(test_tokens)
#     decoded_tokens = vocab.indices_to_tokens(test_indices)
    
#     print("\nTest encoding/decoding:")
#     print("Original tokens:", test_tokens)
#     print("Encoded indices:", test_indices)
#     print("Decoded tokens: ", decoded_tokens)