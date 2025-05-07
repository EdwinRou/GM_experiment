import matplotlib.pyplot as plt
import torch
from typing import List, Dict

def plot_diffusion_progress(losses: List[float], epoch: int):
    """Plot training progress for diffusion model."""
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title(f'Diffusion Training Loss\nEpoch {epoch}')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    plt.show()

def plot_flow_progress(losses: List[float], epoch: int):
    """Plot training progress for flow matching model."""
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title(f'Flow Matching Training Loss\nEpoch {epoch}')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    plt.show()

def plot_model_distributions(
    diffusion_samples: torch.Tensor,
    flow_samples: torch.Tensor,
    x_coords: torch.Tensor,
    test_data: torch.Tensor,
    num_samples: int = 100
):
    """Compare distributions of both models using mean/std visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # Plot means and std for both models
    # Diffusion distribution
    axes[0].plot(x_coords.cpu(), diffusion_samples.cpu().mean(dim=0), 'b-', label='Diffusion (mean)')
    axes[0].fill_between(
        x_coords.cpu(),
        diffusion_samples.cpu().mean(dim=0) - diffusion_samples.cpu().std(dim=0),
        diffusion_samples.cpu().mean(dim=0) + diffusion_samples.cpu().std(dim=0),
        alpha=0.2,
        color='b',
        label='Diffusion (±1 std)'
    )

    # Flow matching distribution
    axes[0].plot(x_coords.cpu(), flow_samples.cpu().mean(dim=0), 'r-', label='Flow (mean)')
    axes[0].fill_between(
        x_coords.cpu(),
        flow_samples.cpu().mean(dim=0) - flow_samples.cpu().std(dim=0),
        flow_samples.cpu().mean(dim=0) + flow_samples.cpu().std(dim=0),
        alpha=0.2,
        color='r',
        label='Flow (±1 std)'
    )

    # Target distribution
    axes[0].plot(x_coords.cpu(), test_data.cpu().mean(dim=0), 'k--', label='Target (mean)')
    axes[0].fill_between(
        x_coords.cpu(),
        test_data.cpu().mean(dim=0) - test_data.cpu().std(dim=0),
        test_data.cpu().mean(dim=0) + test_data.cpu().std(dim=0),
        alpha=0.2,
        color='gray',
        label='Target (±1 std)'
    )

    axes[0].set_title('Distribution Comparison (Mean ± Std)')
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_ylim(-2.0, 2.0)

    # Single sample comparison
    axes[1].plot(x_coords.cpu(), diffusion_samples[0].cpu(), 'b-', label='Diffusion sample')
    axes[1].plot(x_coords.cpu(), flow_samples[0].cpu(), 'r-', label='Flow sample')
    axes[1].plot(x_coords.cpu(), test_data[0].cpu(), 'k--', label='Target sample')
    axes[1].set_title('Individual Sample Comparison')
    axes[1].grid(True)
    axes[1].legend()
    axes[1].set_ylim(-2.0, 2.0)

    plt.tight_layout()
    plt.show()

def visualize_flow_evolution(
    x_samples: List[torch.Tensor],
    x_coords: torch.Tensor,
    time_steps: List[float],
    num_steps: int = 5
):
    """Visualize flow matching model's sample generation process."""
    fig, axes = plt.subplots(1, num_steps, figsize=(20, 4))
    step_indices = [0, len(time_steps)//4, len(time_steps)//2, 3*len(time_steps)//4, -1]

    # Initial noise
    axes[0].plot(x_coords.cpu(), x_samples[0][0].cpu(), 'r--', label='Initial noise')
    axes[0].set_title('t = 0.0')
    axes[0].set_ylim(-2.0, 2.0)
    axes[0].grid(True)
    axes[0].legend()

    for i, idx in enumerate(step_indices[1:]):
        axes[i+1].plot(x_coords.cpu(), x_samples[idx][0].cpu(), 'b-', label='Sample')
        axes[i+1].set_title(f't = {time_steps[idx]:.1f}')
        axes[i+1].set_ylim(-2.0, 2.0)
        axes[i+1].grid(True)
        axes[i+1].legend()

    plt.suptitle('Flow Matching: Single Sample Evolution', y=1.05)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.85, wspace=0.3)
    plt.show()
