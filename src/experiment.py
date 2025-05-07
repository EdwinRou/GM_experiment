import torch
from src.data.dataset import get_dataloaders
from src.utils.training import train_diffusion, train_flow
from src.utils.visualization import plot_model_distributions

# Constants
DIFFUSION_STEPS = 100  # Number of steps for diffusion process
FLOW_STEPS = 100       # Number of steps for flow generation
BATCH_SIZE = 512       # Batch size for training
NUM_EPOCHS = 200       # Number of training epochs
SAVE_INTERVAL = 100    # Interval for saving progress and visualizations

def run_experiment(
    num_train: int = 4000,
    num_test: int = 1000,
    seed: int = 42,
    device: str = None
):
    """
    Run the complete experiment comparing DDPM and Flow Matching.

    Args:
        num_train (int): Number of training samples
        num_test (int): Number of test samples
        seed (int): Random seed for reproducibility
        device (str): Device to use for computation (cuda/cpu)
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    # Get data
    print("\nPreparing datasets...")
    train_loader, test_loader, x_coords = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_train=num_train,
        num_test=num_test,
        seed=seed
    )
    x_coords = x_coords.to(device)

    # Train Diffusion model
    print("\nTraining Diffusion Model...")
    diffusion_model, _, diffusion_losses = train_diffusion(
        train_loader=train_loader,
        test_loader=test_loader,
        x_coords=x_coords,
        num_steps=DIFFUSION_STEPS,
        num_epochs=NUM_EPOCHS,
        save_interval=SAVE_INTERVAL,
        device=device
    )

    # Train Flow Matching model
    print("\nTraining Flow Matching Model...")
    flow_model, _, test_data, flow_losses = train_flow(
        train_loader=train_loader,
        test_loader=test_loader,
        x_coords=x_coords,
        num_steps=FLOW_STEPS,
        num_epochs=NUM_EPOCHS,
        save_interval=SAVE_INTERVAL,
        device=device
    )

    # Generate samples for comparison
    print("\nGenerating samples for comparison...")
    with torch.no_grad():
        num_samples = 100
        diffusion_samples = diffusion_model.generate_samples(
            num_samples=num_samples,
            device=device,
            x_coords=x_coords
        )

        flow_samples = torch.randn(num_samples, len(x_coords)).to(device)
        time_steps = torch.linspace(0, 1.0, FLOW_STEPS, device=device)
        for i in range(len(time_steps)-1):
            flow_samples = flow_model.step(
                x_t=flow_samples,
                t_start=time_steps[i],
                t_end=time_steps[i+1]
            )

    # Compare results
    print("\nPlotting comparison results...")
    plot_model_distributions(
        diffusion_samples=diffusion_samples,
        flow_samples=flow_samples,
        x_coords=x_coords,
        test_data=test_data,
        num_samples=num_samples
    )

if __name__ == "__main__":
    run_experiment()
