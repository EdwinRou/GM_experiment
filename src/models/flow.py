import torch
import torch.nn as nn
from .transformer import TransformerBlock

class Flow(nn.Module):
    """Flow Matching model with transformer-based architecture."""
    def __init__(self, dim: int = 64, h_dim: int = 256, num_heads: int = 2, num_layers: int = 2, dropout: float = 0.01):
        super().__init__()
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, dim)
        )

        # Position embedding
        self.pos_embed = nn.Embedding(100, dim)  # Fixed for 100 points

        # Input projection
        self.input_proj = nn.Linear(1, dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.norm_final = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        # Shape [batch_size, n_points] -> [batch_size, n_points, 1]
        x = x_t.unsqueeze(-1)

        # Input embedding
        h = self.input_proj(x)  # [batch_size, n_points, dim]

        # Add positional embedding
        pos = torch.arange(x.shape[1], device=x.device)
        h = h + self.pos_embed(pos).unsqueeze(0)

        # Add time embedding
        t_emb = self.time_embed(t.view(-1, 1))  # [batch_size, dim]
        h = h + t_emb.unsqueeze(1)

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h)

        # Output projection
        h = self.norm_final(h)
        output = self.output_proj(h)

        return output.squeeze(-1)

    def step(self, x_t: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
        """
        Perform one step of the flow matching process.
        
        Args:
            x_t: Current state [batch_size, n_points]
            t_start: Starting time [batch_size or 1]
            t_end: Ending time [batch_size or 1]
            
        Returns:
            Next state [batch_size, n_points]
        """
        bs = x_t.shape[0]
        t_start = t_start.expand(bs)
        t_end = t_end.expand(bs)
        dt = (t_end - t_start).unsqueeze(-1)

        # Midpoint method with improved stability
        t_mid = t_start + dt.squeeze(-1)/2
        dx_start = self(t=t_start, x_t=x_t)
        x_mid = x_t + dx_start * dt
        dx_mid = self(t=t_mid, x_t=x_mid)

        return x_t + dx_mid * dt
