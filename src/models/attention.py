import torch
import torch.nn as nn

class HeadAttention(nn.Module):
    """Single attention head for transformer architecture."""
    def __init__(self, d, d_head, dropout=0.01):
        super().__init__()
        self.Q = nn.Linear(d, d_head)
        self.K = nn.Linear(d, d_head)
        self.V = nn.Linear(d, d_head)
        self.sqrt_d = torch.sqrt(torch.tensor(d_head))
        self.drop_att = nn.Dropout(dropout)

    def forward(self, x):
        Q = self.Q(x).unsqueeze(2)     # [bs, n, 1, d_head]
        K = self.K(x).unsqueeze(1)     # [bs, 1, n, d_head]
        V = self.V(x)                  # [bs, n, d_head]

        Att = (Q * K).sum(dim=3) / self.sqrt_d  # [bs, n, n]
        Att = torch.softmax(Att, dim=1)         # [bs, n, n]
        Att = self.drop_att(Att)               # [bs, n, n]

        return Att @ V                         # [bs, n, d_head]

class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, d, num_heads, dropout=0.01):
        super().__init__()
        d_head = d // num_heads
        self.heads = nn.ModuleList([HeadAttention(d, d_head, dropout) for _ in range(num_heads)])
        self.WO = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x_heads = [head(x) for head in self.heads]
        x = self.WO(torch.cat(x_heads, dim=2))
        return self.drop(x)
