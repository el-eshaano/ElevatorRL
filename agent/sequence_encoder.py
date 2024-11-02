import torch
import torch.nn as nn

class SequenceEncoder(nn.Module):
    def __init__(self, input_size: int, seq_len: int, encoding_size: int, heads: int, normalize: bool = True):
        super(SequenceEncoder, self).__init__()
        
        self.normalize = normalize
        self.encoding_size = encoding_size
        self.heads = heads
        self.seq_len = seq_len
        self.embedding = nn.Linear(input_size, encoding_size)
        self.attention = nn.MultiheadAttention(encoding_size, heads)
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(encoding_size, encoding_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(encoding_size, encoding_size, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.norm = nn.BatchNorm1d(encoding_size)
        
        self.fc = nn.Linear(seq_len, 1)
        
    def forward(self, x):
        
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
            
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        # pad with zeros
        x = nn.functional.pad(x, (0, 0, 0, self.seq_len - x.shape[1]), "constant", 0)
        
        # shape of x is now (batch_size, seq_len, input_size)
        
        x = self.embedding(x) # (batch_size, seq_len, encoding_size)
        x = x.permute(1, 0, 2) # (seq_len, batch_size, encoding_size)
        
        x, _ = self.attention(x, x, x)
        
        x = x.permute(1, 2, 0) # (batch_size, encoding_size, seq_len)
        
        x_ = self.conv_layers(x) # (batch_size, encoding_size, seq_len)
        x = x + x_ # (batch_size, encoding_size, seq_len)
        
        if self.normalize:
            x = self.norm(x)
        
        x = self.fc(x)
        return x