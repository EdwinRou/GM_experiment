import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils.visualization import plot_diffusion_progress, plot_flow_progress
from src.utils.metrics import evaluate_generated_samples
from src.models.ddpm import DDPM
from src.models.flow import Flow

def evaluate_model_samples(model, x_coords, test_data, device):
    """Evaluate model samples against test data."""
    with torch.no_grad():
        if isinstance(model, DDPM):
            samples = model.generate_samples(len(test_data), device, x_coords)
        else:  # Flow model
            samples = torch.randn(len(test_data), len(x_coords)).to(device)
            time_steps = torch.linspace(0, 1.0, 100, device=device)
            for i in range(len(time_steps)-1):
                samples = model.step(
                    x_t=samples,
                    t_start=time_steps[i],
                    t_end=time_steps[i+1]
                )
        return samples, evaluate_generated_samples(samples.cpu(), test_data.cpu())

def train_diffusion(
    train_loader: DataLoader,
    test_loader: DataLoader,
    x_coords: torch.Tensor,
    num_steps: int = 100,
    num_epochs: int = 100,
    save_interval: int = 50,
    device: str = 'cuda'
) -> tuple:
    """Train the diffusion model."""
    # Initialize model
    beta_1 = 1e-4
    beta_T = 0.02
    model = DDPM(num_steps, beta_1, beta_T, device=device).to(device)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=5)

    # Training loop
    train_losses = []
    print("\nTraining Diffusion model...")
    pbar = tqdm(range(num_epochs), desc="Training")

    for epoch in pbar:
        model.train()
        batch_losses = []

        for x_0 in train_loader:
            x_0 = x_0.to(device)
            t = torch.randint(0, num_steps, (len(x_0),), device=device)

            # Forward process
            noise = torch.randn_like(x_0)
            x_t, eps = model.forward_process(x_0, t, noise)

            # Predict noise
            predicted_noise = model.backward_process(x_t, t)

            # Compute loss
            loss = torch.nn.functional.mse_loss(predicted_noise, eps)

            # Update model
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_losses.append(loss.item())

        # Update progress
        avg_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(avg_loss)
        pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
        scheduler.step(avg_loss)

        # Visualize progress
        if (epoch + 1) % save_interval == 0:
            plot_diffusion_progress(train_losses, epoch + 1)
            test_batch = next(iter(test_loader)).to(device)
            samples, metrics = evaluate_model_samples(model, x_coords, test_batch, device)
            print(f"\nEpoch {epoch + 1} metrics:", metrics)

    print("\nDiffusion model training completed!")
    return model, x_coords, train_losses

def train_flow(
    train_loader: DataLoader,
    test_loader: DataLoader,
    x_coords: torch.Tensor,
    num_steps: int = 100,
    num_epochs: int = 100,
    save_interval: int = 50,
    device: str = 'cuda'
) -> tuple:
    """Train the flow matching model."""
    # Initialize model
    flow = Flow().to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.95, patience=5
    )
    loss_fn = nn.MSELoss()

    train_losses = []
    print("\nTraining Flow Matching model...")
    pbar = tqdm(range(num_epochs), desc="Training")

    for epoch in pbar:
        batch_losses = []

        for x_1 in train_loader:
            x_1 = x_1.to(device)
            x_0 = torch.randn_like(x_1, device=device)
            t = torch.rand(len(x_1), device=device)

            x_t = (1 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * x_1
            dx_t = x_1 - x_0

            optimizer.zero_grad()
            pred = flow(t=t, x_t=x_t)
            loss = loss_fn(pred, dx_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            optimizer.step()

            batch_losses.append(loss.item())

        avg_epoch_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(avg_epoch_loss)

        # Update learning rate
        scheduler.step(avg_epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{avg_epoch_loss:.6f}',
            'lr': f'{current_lr:.2e}'
        })

        if (epoch + 1) % save_interval == 0:
            plot_flow_progress(train_losses, epoch+1)
            test_batch = next(iter(test_loader)).to(device)
            samples, metrics = evaluate_model_samples(flow, x_coords, test_batch, device)
            print(f"\nEpoch {epoch + 1} metrics:", metrics)

    print("\nFlow Matching model training completed!")
    return flow, x_coords, test_batch, train_losses
