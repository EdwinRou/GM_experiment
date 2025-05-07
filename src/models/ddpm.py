import torch
import torch.nn as nn
from .transformer import TransformerBlock

class UNet(nn.Module):
    """UNet architecture for denoising diffusion model."""
    def __init__(self, num_steps, dim=64, h_dim=256, n_heads=2, n_layers=2, dropout=0.01):
        super().__init__()
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Embedding(num_steps, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, dim)
        )

        # Position embedding
        self.pos_embed = nn.Embedding(100, dim)  # Fixed for 100 points

        # Input projection
        self.input_proj = nn.Linear(1, dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.norm_final = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
        # Shape [batch_size, n_points] -> [batch_size, n_points, 1]
        x = x.unsqueeze(-1)

        # Input embedding
        h = self.input_proj(x)  # [batch_size, n_points, dim]

        # Add positional embedding
        pos = torch.arange(x.shape[1], device=x.device)
        h = h + self.pos_embed(pos).unsqueeze(0)

        # Add time embedding
        t_emb = self.time_embed(t)  # [batch_size, dim]
        h = h + t_emb.unsqueeze(1)

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h)

        # Output projection
        h = self.norm_final(h)
        output = self.output_proj(h)

        return output.squeeze(-1)

class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model."""
    def __init__(self, num_steps, beta_1, beta_T, device='cuda'):
        super().__init__()
        self.num_steps = num_steps
        self.device = device

        # Define noise schedule
        self.beta = torch.linspace(beta_1, beta_T, num_steps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # Create UNet model
        self.model = UNet(num_steps).to(device)

    def forward_process(self, x_0, t, eps=None):
        """Forward diffusion process."""
        if eps is None:
            eps = torch.randn_like(x_0)

        alpha_bar_t = self.alpha_bar[t].view(-1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * eps

        return x_t, eps

    def backward_process(self, x_t, t):
        """Predict noise in backward diffusion process."""
        return self.model(x_t, t)

    def generate_samples(self, num_samples, device, x_coords):
        """Generate new samples using the trained model."""
        with torch.no_grad():
            # Start from pure noise
            x_t = torch.randn(num_samples, len(x_coords)).to(device)

            # Gradually denoise
            for t in range(self.num_steps - 1, -1, -1):
                t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
                predicted_noise = self.backward_process(x_t, t_batch)

                # Compute parameters for denoising step
                alpha_t = self.alpha[t]
                alpha_bar_t = self.alpha_bar[t]
                beta_t = self.beta[t]

                # Add noise (except for the last step)
                noise = torch.randn_like(x_t) if t > 0 else 0

                # Apply denoising step
                x_t = (1 / torch.sqrt(alpha_t)) * (
                    x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
                ) + torch.sqrt(beta_t) * noise

        return x_t
