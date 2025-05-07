import torch.nn as nn
from .attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feedforward network."""
    def __init__(self, d, num_heads, dropout=0.01):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.mha = MultiHeadAttention(d, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4*d),
            nn.ReLU(),
            nn.Linear(4*d, d)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention
        x = x + self.mha(self.norm1(x))
        # Feedforward network
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x
