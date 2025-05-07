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
git clone [repository-url]
cd [repository-name]
```

2. Install PyTorch:
   Visit the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) to install the correct version for your system.

3. Install other dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Experiment

```python
# Generate datasets
_, _, _ = generate_and_save_datasets()

# Train models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train DDPM
diffusion_model, x_coords, diffusion_losses = train_diffusion(
    num_epochs=200,
    batch_size=512,
    save_interval=100,
    device=device
)

# Train Flow Matching
flow_model, x_coords, test_data, flow_losses = train_flow(
    num_epochs=200,
    batch_size=512,
    save_interval=100,
    device=device
)

# Compare results
plot_model_distributions(diffusion_model, flow_model, x_coords, test_data, device)
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
