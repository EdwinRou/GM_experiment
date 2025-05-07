import torch
import math
from torch.utils.data import Dataset, DataLoader

def generate_mixture_data(num_samples=256, num_points=100, range_min=-5, range_max=5):
    """
    Generate samples from a mixture of Gaussian distributions.

    Args:
        num_samples (int): Number of samples to generate
        num_points (int): Number of points per sample
        range_min (float): Minimum x value
        range_max (float): Maximum x value

    Returns:
        tuple: (samples, x_values) where samples has shape [num_samples, num_points]
    """
    # Generate x values
    x = torch.linspace(range_min, range_max, num_points)

    # Generate mixture parameters
    components = [
        {'mean_range': (-3, -1), 'sigma': 0.5},
        {'mean_range': (1, 3), 'sigma': 0.5}
    ]

    # Generate data
    p = torch.zeros(num_samples, num_points)
    for comp in components:
        mean_range = comp['mean_range']
        sigma = comp['sigma']
        mu = torch.rand(num_samples) * (mean_range[1] - mean_range[0]) + mean_range[0]
        component_p = torch.exp(-(x.unsqueeze(0) - mu.unsqueeze(1))**2 / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))
        p += component_p

    # Normalize to [-1, 1]
    min_vals, _ = p.min(dim=1, keepdim=True)
    max_vals, _ = p.max(dim=1, keepdim=True)
    p = 2 * (p - min_vals) / (max_vals - min_vals) - 1

    return p, x

def generate_and_save_datasets(num_train=4000, num_test=1000, seed=42):
    """
    Generate and save training and test datasets.

    Args:
        num_train (int): Number of training samples
        num_test (int): Number of test samples
        seed (int): Random seed for reproducibility
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)

    print("Generating training data...")
    train_data, x_coords = generate_mixture_data(num_train)

    print("Generating test data...")
    test_data, _ = generate_mixture_data(num_test)

    print("Saving datasets...")
    torch.save({
        'train_data': train_data,
        'test_data': test_data,
        'x_coords': x_coords,
        'metadata': {
            'num_train': num_train,
            'num_test': num_test,
            'seed': seed
        }
    }, 'mixture_datasets.pt')

    print("Datasets saved successfully!")
    return train_data, test_data, x_coords

class MixtureDataset(Dataset):
    """Dataset wrapper for mixture distribution samples."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_datasets():
    """Load pre-generated mixture datasets."""
    data = torch.load('mixture_datasets.pt')
    return data['train_data'], data['test_data'], data['x_coords']

def get_dataloaders(batch_size=128, num_train=4000, num_test=1000, seed=42):
    """
    Create DataLoaders for training and testing.

    Args:
        batch_size (int): Batch size for DataLoaders
        num_train (int): Number of training samples
        num_test (int): Number of test samples
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_loader, test_loader, x_coords)
    """
    try:
        train_data, test_data, x_coords = load_datasets()
    except FileNotFoundError:
        train_data, test_data, x_coords = generate_and_save_datasets(
            num_train=num_train,
            num_test=num_test,
            seed=seed
        )

    train_dataset = MixtureDataset(train_data)
    test_dataset = MixtureDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, x_coords
