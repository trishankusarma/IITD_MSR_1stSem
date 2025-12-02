"""
transformer_model.py
Transformer-based Encoder-Decoder model for maze path prediction
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding as described in 
    'Attention Is All You Need' paper (Section 3.5)
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        # Compute the div_term for the sinusoidal functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerMazeSolver(nn.Module):
    """
    Transformer-based Encoder-Decoder model for maze path prediction
    """
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=6,
                 dim_feedforward=512, dropout=0.1, max_len=5000):
        super(TransformerMazeSolver, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Shared embedding layer (Section 3.4 of Transformer paper)
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        # Output projection layer
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Will play a bit with initialization
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz):
        # Generate causal mask for decoder self-attention
        # Prevents attending to future tokens
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def make_padding_mask(self, seq, pad_idx):
        """
        Create padding mask
        Args:
            seq: (batch_size, seq_len)
            pad_idx: index of padding token
        Returns:
            mask: (batch_size, seq_len) with True for padding positions
        """
        return (seq == pad_idx)
    
    def forward(self, src, tgt, src_pad_idx, tgt_pad_idx):
        """
        Forward pass
        Args:
            src: source sequence (batch_size, src_len)
            tgt: target sequence (batch_size, tgt_len)
            src_pad_idx: padding index for source
            tgt_pad_idx: padding index for target
        Returns:
            output: (batch_size, tgt_len, vocab_size)
        """
        # Create masks
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        src_padding_mask = self.make_padding_mask(src, src_pad_idx)
        tgt_padding_mask = self.make_padding_mask(tgt, tgt_pad_idx)
        
        # Embed and add positional encoding
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Encode
        memory = self.transformer_encoder(
            src_emb,
            src_key_padding_mask=src_padding_mask
        )
        
        # Decode
        output = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        # Project to vocabulary
        output = self.fc_out(output)
        
        return output
    
    def encode(self, src, src_pad_idx):
        """Encode source sequence"""
        src_padding_mask = self.make_padding_mask(src, src_pad_idx)
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        memory = self.transformer_encoder(
            src_emb,
            src_key_padding_mask=src_padding_mask
        )
        return memory, src_padding_mask
    
    def decode_step(self, tgt, memory, tgt_pad_idx, memory_padding_mask=None):
        """Decode one step"""
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        tgt_padding_mask = self.make_padding_mask(tgt, tgt_pad_idx)
        
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        output = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask
        )
        
        output = self.fc_out(output)
        return output