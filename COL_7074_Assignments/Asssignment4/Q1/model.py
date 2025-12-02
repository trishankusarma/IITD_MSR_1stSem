import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.2):
       
        super(Encoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        #embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        #unidirectional RNN
        self.rnn = nn.RNN(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, input_seqs, input_lengths):
      
        batch_size = input_seqs.size(0)
        
        #embeding input sequences
        embedded = self.embedding(input_seqs)  
        
        #Pack sequences for efficient RNN processing
        packed = pack_padded_sequence(
            embedded, 
            input_lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        packed_output, hidden = self.rnn(packed)
        
        #Unpacking sequences
        outputs, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        return outputs, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
      
        super(BahdanauAttention, self).__init__()
        
        #Attention parameters (from Bahdanau paper)
        self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)  
        self.U_a = nn.Linear(hidden_dim , hidden_dim, bias=False)  
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)  
        
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
 
        batch_size = encoder_outputs.size(0)
        max_seq_len = encoder_outputs.size(1)
        
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, max_seq_len, 1)
        
        energy = self.v_a(torch.tanh(
            self.W_a(decoder_hidden_expanded) + self.U_a(encoder_outputs)
        ))  
        
        energy = energy.squeeze(2) 
           
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        #Computing attention weights
        attention_weights = torch.softmax(energy, dim=1) 
        
  
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  
        context = context.squeeze(1)  
        
        return context, attention_weights

# A FLUFFY BLUE CREATURE
# E1, E2, E3, E4

# E1, E2 + DEL E1, 
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.2):
      
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.attention = BahdanauAttention(hidden_dim)
        
        self.rnn = nn.RNN(
            embedding_dim + hidden_dim,  
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.out = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, target_token, decoder_hidden, encoder_outputs, mask=None):
       
        embedded = self.embedding(target_token).unsqueeze(1)
        
        last_hidden = decoder_hidden[-1]  
        context, attention_weights = self.attention(last_hidden, encoder_outputs, mask)
        
        context = context.unsqueeze(1)  
        rnn_input = torch.cat([embedded, context], dim=2)  
        
        rnn_output, decoder_hidden = self.rnn(rnn_input, decoder_hidden)
        output = self.out(rnn_output.squeeze(1))  
        
        return output, decoder_hidden, attention_weights
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, input_seqs, input_lengths, target_seqs, teacher_forcing_ratio=0.5):
       
        batch_size = input_seqs.size(0)
        target_len = target_seqs.size(1)
        vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
        
        encoder_outputs, encoder_hidden = self.encoder(input_seqs, input_lengths)
        
        decoder_hidden = encoder_hidden  
        
        mask = (input_seqs != 0).long()
        decoder_input = target_seqs[:, 0]
        
        for t in range(1, target_len):
            output, decoder_hidden, _ = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, mask
            )
            
            outputs[:, t, :] = output
            
            # Decide whether to use teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            
            # Get next input
            if use_teacher_forcing:
                decoder_input = target_seqs[:, t]
            else:
                decoder_input = output.argmax(1)
        
        return outputs
    
    def generate(self, input_seqs, input_lengths, max_length=50, start_token_idx=None):
        self.eval()
        with torch.no_grad():
            batch_size = input_seqs.size(0)
            
            encoder_outputs, encoder_hidden = self.encoder(input_seqs, input_lengths)
            
            decoder_hidden = encoder_hidden
            mask = (input_seqs != 0).long()
            
            decoder_input = torch.full((batch_size,), start_token_idx, dtype=torch.long).to(self.device)
            
            generated_tokens = [decoder_input]
            
            for t in range(max_length):
                output, decoder_hidden, _ = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs, mask
                )
                
                decoder_input = output.argmax(1)
                generated_tokens.append(decoder_input)
                
            
            generated_seqs = torch.stack(generated_tokens, dim=1)  
        return generated_seqs

