# H3 Optimizer

**H**ierarchical **H**essian-informed **H**yperparameter Optimizer for Efficient Deep Learning

## Overview

H3 Optimizer is a novel PyTorch optimizer that combines three key innovations:

1. **Information-weighted sampling**: Prioritizes training samples based on their information content
2. **Lipschitz-adaptive learning rates**: Adjusts learning rates per layer based on gradient smoothness
3. **Energy-efficient training**: Tracks and optimizes computational efficiency

## Features

- Drop-in replacement for standard PyTorch optimizers (Adam, SGD, etc.)
- Layer-wise adaptive learning rates based on local Lipschitz estimates
- Intelligent sample prioritization for faster convergence
- Built-in energy and efficiency tracking
- Compatible with existing PyTorch training pipelines

## Installation

### From source

```bash
git clone https://github.com/yourusername/h3-optimizer.git
cd h3-optimizer
pip install -e .
```

### Using pip (coming soon)

```bash
pip install h3-optimizer
```

## Quick Start

```python
import torch
from h3.optimizer import H3Optimizer
from h3.scheduler import LipschitzAdaptiveLR
from h3.energy_tracker import EnergyTracker

# Initialize your model
model = YourModel()

# Create H3 optimizer
optimizer = H3Optimizer(model.parameters(), lr=1e-3)

# Add Lipschitz-adaptive scheduler
scheduler = LipschitzAdaptiveLR(optimizer)

# Track energy consumption
energy_tracker = EnergyTracker(device='cuda')

# Training loop
for epoch in range(num_epochs):
    energy_tracker.start()

    for batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch.x), batch.y)
        loss.backward()
        optimizer.step()

    energy_metrics = energy_tracker.stop()
    scheduler.step()
```

## Examples

- `examples/quickstart.py`: Basic usage with a simple neural network
- `examples/cifar10_demo.py`: Complete CIFAR-10 training example

## Project Structure

```
H3-Optimizer/
├── h3/                   # Main package
│   ├── optimizer.py      # H3Optimizer class
│   ├── sampler.py        # Information-weighted sampler
│   ├── scheduler.py      # Lipschitz-adaptive scheduler
│   ├── energy_tracker.py # Energy measurement
│   └── utils.py          # Utility functions
├── examples/             # Usage examples
├── notebooks/            # Jupyter notebooks
├── benchmarks/           # Performance benchmarks
├── tests/                # Unit tests
└── docs/                 # Documentation
```

## Requirements

- Python >= 3.7
- PyTorch >= 1.9.0
- NumPy >= 1.19.0

See `requirements.txt` for complete list of dependencies.

## Development Status

H3 Optimizer is currently in **alpha** stage. The API may change as we continue development.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use H3 Optimizer in your research, please cite:

```bibtex
@software{h3_optimizer,
  title={H3 Optimizer: Hierarchical Hessian-informed Hyperparameter Optimization},
  author={H3 Development Team},
  year={2024},
  url={https://github.com/yourusername/h3-optimizer}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and feedback, please open an issue on GitHub.
