# Diffusion and Flow Matching Experiment

This project implements and compares two state-of-the-art generative modeling techniques: Denoising Diffusion Probabilistic Models (DDPM) and Flow Matching. Both models are implemented using transformer-based architectures and are trained on a mixture of Gaussian distributions dataset.

## Overview

The experiment provides:
- Implementation of DDPM and Flow Matching models
- Transformer-based architecture for both models
- Training and evaluation pipelines
- Comprehensive visualization tools
- Performance metrics comparison

## Installation

1. Clone the repository:
```bash
git clone https://github.com/EdwinRou/GM_experiment.git
cd GM_experiment
```

2. Install PyTorch:
   Visit the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) to install the correct version for your system.

3. Install dependencies and development setup:
```bash
# Install dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

The development installation allows you to modify the code while using the package, and includes test dependencies.

## Usage

### Original Implementation

The original implementation is available as a standalone Jupyter notebook:
```bash
jupyter notebook notebooks/original_implementation.ipynb
```

This notebook contains the complete experiment in a single file, which has been refactored into the modular structure described below.

### Tutorial

Get started with the tutorial notebook:
```bash
jupyter notebook tutorials/basic_experiment.ipynb
```

The tutorial demonstrates:
- Data generation and loading
- Model training and configuration
- Sample generation and visualization
- Model evaluation and comparison
- Flow evolution visualization

### Running from Python

You can also use the modular components directly in Python:

```python
from src.data.dataset import get_dataloaders
from src.utils.training import train_diffusion, train_flow
from src.utils.visualization import plot_model_distributions

# Get data
train_loader, test_loader, x_coords = get_dataloaders(
    batch_size=512,
    num_train=4000,
    num_test=1000
)

# Train models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_coords = x_coords.to(device)

# Train DDPM
diffusion_model, _, _ = train_diffusion(
    train_loader=train_loader,
    test_loader=test_loader,
    x_coords=x_coords,
    device=device
)

# Train Flow Matching
flow_model, _, test_data, _ = train_flow(
    train_loader=train_loader,
    test_loader=test_loader,
    x_coords=x_coords,
    device=device
)

# Generate samples for comparison
with torch.no_grad():
    # Generate Diffusion samples
    diffusion_samples = diffusion_model.generate_samples(100, device, x_coords)
    
    # Generate Flow samples
    flow_samples = torch.randn(100, len(x_coords)).to(device)
    time_steps = torch.linspace(0, 1.0, 100, device=device)
    for i in range(len(time_steps)-1):
        flow_samples = flow_model.step(
            x_t=flow_samples,
            t_start=time_steps[i],
            t_end=time_steps[i+1]
        )

# Compare results
plot_model_distributions(
    diffusion_samples=diffusion_samples,
    flow_samples=flow_samples,
    x_coords=x_coords,
    test_data=test_data
)
```

## Project Structure

```
.
├── notebooks/          # Reference implementations
│   └── original_implementation.ipynb
├── tutorials/          # Tutorial notebooks
│   └── basic_experiment.ipynb
├── src/               # Source code
│   ├── models/        # Model implementations
│   │   ├── attention.py    # Attention mechanisms
│   │   ├── transformer.py  # Transformer block
│   │   ├── ddpm.py        # DDPM implementation
│   │   └── flow.py        # Flow Matching implementation
│   ├── data/         # Data handling
│   │   └── dataset.py     # Dataset generation and loading
│   ├── utils/        # Utilities
│   │   ├── metrics.py     # Distribution metrics
│   │   ├── training.py    # Training loops
│   │   └── visualization.py# Plotting functions
│   └── experiment.py  # Main experiment runner
└── tests/            # Test files
```

## Key Components

### Models

1. **DDPM (Denoising Diffusion Probabilistic Model)**
   - Gradually adds and removes noise using a transformer-based architecture
   - Implements forward and backward diffusion processes
   - Uses noise prediction for sample generation

2. **Flow Matching**
   - Learns continuous transformation between distributions
   - Implements ODE-based flow using transformer architecture
   - Uses vector field prediction for sample generation

### Architecture

Both models share common components:
- Multi-head attention mechanisms
- Transformer blocks
- Time embeddings
- Positional encodings

### Utilities

1. **Data Handling**
   - Mixture of Gaussians dataset generation
   - DataLoader creation
   - Dataset normalization

2. **Training**
   - Configurable training loops
   - Learning rate scheduling
   - Progress visualization

3. **Evaluation**
   - Distribution metrics (KL, JS, Wasserstein)
   - Sample quality assessment
   - Model comparison tools

## Results

The project provides various visualization and evaluation tools:

1. **Training Progress**
   - Loss curves
   - Sample evolution
   - Distribution comparisons

2. **Evaluation Metrics**
   - KL Divergence
   - JS Divergence
   - Wasserstein Distance

3. **Flow Matching Analysis**
   - Step count impact
   - Generation time analysis
   - Quality vs. computation trade-offs

## License

MIT License

Copyright (c) 2023

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. In Advances in Neural Information Processing Systems.
2. Lipman, Y., Igashov, D., Chen, Y., & Huang, C.-W. (2023). Flow Matching for Generative Modeling. In International Conference on Learning Representations.
3. Vaswani, A., et al. (2017). Attention Is All You Need. In Advances in Neural Information Processing Systems.
