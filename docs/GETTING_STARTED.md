# Getting Started with H3 Optimizer 🚀

**Your first steps with thermodynamically efficient deep learning**

This guide will walk you through everything you need to know to start using H3 Optimizer, from installation to your first training run.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Your First H3 Program](#your-first-h3-program)
4. [Understanding the Results](#understanding-the-results)
5. [Choosing the Right Approach](#choosing-the-right-approach)
6. [Next Steps](#next-steps)

---

## Prerequisites

### Required Knowledge

- Basic Python programming
- Familiarity with PyTorch (basic training loops)
- Understanding of neural networks (at a high level)

**Don't have PyTorch experience?** Check out the [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/basics/intro.html) first!

### System Requirements

- **Python:** 3.8 or higher
- **PyTorch:** 1.12.0 or higher
- **Hardware:** CPU or GPU (GPU recommended for best performance)
- **OS:** Linux, macOS, or Windows

### Optional (for energy tracking)

- **NVIDIA GPU:** Install `pynvml` for real power monitoring
  ```bash
  pip install pynvml
  ```
- **Apple Silicon:** Use `powermetrics` (requires sudo access)

---

## Installation

### Quick Install (Recommended)

```bash
pip install h3-optimizer
```

That's it! This installs the validated package from PyPI (version 0.2.1).

### Verify Installation

```python
import h3
print(h3.__version__)  # Should print: 0.2.1
```

### From Source (for development)

```bash
git clone https://github.com/nfocardoso/EMSTI.git
cd EMSTI
pip install -e .
```

---

## Your First H3 Program

Let's start with the **simplest possible H3 example** using AutoH3.

### Option 1: AutoH3 (Recommended for Beginners)

**The absolute simplest API** - complete automation with one function call!

```python
"""
Your First H3 Program - AutoH3
Complete automation with zero configuration required!
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from h3 import AutoH3

# 1. Define your model (any PyTorch model works!)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Load your dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# 3. Create model and AutoH3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)

# Single line creates EVERYTHING!
auto = AutoH3(
    model.parameters(),
    train_dataset,
    mode="balanced",  # Options: "safe", "balanced", "green", "extreme"
    name="my_first_h3_run"
)

# 4. Start tracking
auto.start()

# 5. Training loop (super simple!)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    # Get the data loader for this epoch
    train_loader = auto.get_loader(batch_size=64)

    # Training
    model.train()
    for data, target, indices in train_loader:
        data, target = data.to(device), target.to(device)

        # One-line training step!
        loss = auto.training_step(model, data, target, criterion, indices)

    # Evaluation
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_dataset)

    # One-line epoch evaluation!
    auto.evaluate_epoch(model, test_loader)

    print(f"Epoch {epoch+1}/10 - Accuracy: {accuracy:.2f}%")

# 6. Get comprehensive results
results = auto.finish()

print("\n" + "="*50)
print("🎉 Training Complete!")
print("="*50)
print(f"Final Accuracy: {results['final_accuracy']:.2f}%")
print(f"Total Energy: {results['total_energy_j']:.2f} J")
print(f"Efficiency: {results['efficiency_bits_per_j']:.6f} bits/J")
print(f"Total Time: {results['total_time_s']:.2f} s")
print("="*50)
```

**Run it:**
```bash
python my_first_h3.py
```

**Expected output:**
```
Epoch 1/10 - Accuracy: 96.32%
Epoch 2/10 - Accuracy: 97.84%
...
Epoch 10/10 - Accuracy: 99.15%

==================================================
🎉 Training Complete!
==================================================
Final Accuracy: 99.15%
Total Energy: 1172.02 J
Efficiency: 0.002816 bits/J
Total Time: 20.27 s
==================================================
```

---

### Option 2: Manual H3 (More Control)

Want more control? Here's the manual approach:

```python
"""
Your First H3 Program - Manual Setup
More control over configuration
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from h3 import (
    H3Optimizer,
    LossTracker,
    InformationWeightedSampler,
    IndexedDataset,
    EnergyTracker
)

# ... (same model and dataset setup as above) ...

# Setup H3 components
indexed_dataset = IndexedDataset(train_dataset)
loss_tracker = LossTracker(len(train_dataset), smoothing=0.1)
optimizer = H3Optimizer(model.parameters(), lr=1e-3)
energy_tracker = EnergyTracker(device=device)
criterion = nn.CrossEntropyLoss(reduction='none')  # Important: reduction='none'

# Start energy tracking
energy_tracker.start()

# Training with three-phase strategy
num_epochs = 10
warmup_epochs = 1
consolidation_epochs = 1

for epoch in range(num_epochs):
    # Determine training phase
    if epoch < warmup_epochs:
        phase = 'warmup'
        sampler = InformationWeightedSampler(
            indexed_dataset,
            loss_tracker,
            uniform_mix=1.0  # 100% uniform sampling
        )
    elif epoch >= (num_epochs - consolidation_epochs):
        phase = 'consolidation'
        sampler = InformationWeightedSampler(
            indexed_dataset,
            loss_tracker,
            uniform_mix=1.0  # 100% uniform sampling
        )
    else:
        phase = 'thermodynamic'
        sampler = InformationWeightedSampler(
            indexed_dataset,
            loss_tracker,
            keep_frac=0.65,    # Keep top 65% by loss
            uniform_mix=0.2    # 80% weighted, 20% uniform
        )

    # Create data loader with sampler
    train_loader = DataLoader(indexed_dataset, sampler=sampler, batch_size=64)

    # Training
    model.train()
    for data, target, indices in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        # Compute per-sample losses
        loss_vector = criterion(output, target)
        loss = loss_vector.mean()

        # Update loss tracker for intelligent sampling
        loss_tracker.update(indices, loss_vector.detach())

        loss.backward()
        optimizer.step()

    # Evaluation (same as AutoH3 example)
    # ...

    print(f"Epoch {epoch+1}/{num_epochs} [{phase:15s}] - Accuracy: {accuracy:.2f}%")

# Get final results
results = energy_tracker.stop()
print(f"\nThermodynamic Efficiency: {results['efficiency_bits_per_j']:.6f} bits/J")
```

---

### Option 3: Using Presets (Quick & Easy)

**Best of both worlds** - simple setup with good defaults:

```python
from h3.presets import h3_mnist_fast
from torch.utils.data import DataLoader

# One-liner setup!
optimizer, loss_tracker, indexed_dataset = h3_mnist_fast(
    model.parameters(),
    train_dataset,
    lr=1e-3
)

# Get sampler for each epoch
from h3.presets import MNISTFast
preset = MNISTFast(model.parameters(), train_dataset)

for epoch in range(10):
    phase, sampler = preset.get_sampler(epoch, total_epochs=10)
    loader = DataLoader(indexed_dataset, sampler=sampler, batch_size=64)
    # ... training loop ...
```

See [PRESETS.md](PRESETS.md) for all available presets!

---

## Understanding the Results

### Key Metrics Explained

When H3 finishes training, you'll see these metrics:

#### 1. **Accuracy**
```
Final Accuracy: 99.15%
```
- Your model's performance on the test set
- H3 typically maintains accuracy within ±0.2pp of baseline

#### 2. **Energy Consumption**
```
Total Energy: 1172.02 J
```
- Real energy consumed during training (in joules)
- Measured via:
  - **NVIDIA GPUs:** NVML (real power monitoring)
  - **Apple Silicon:** PowerMetrics
  - **Others:** TDP-based estimation

#### 3. **Efficiency (η)**
```
Efficiency: 0.002816 bits/J
```
- **The core H3 metric!**
- Formula: η = ΔI / E (bits of information per joule)
- Higher is better
- Typical improvement: 15-20% over baseline

#### 4. **Training Time**
```
Total Time: 20.27 s
```
- Wall-clock time for training
- H3 typically achieves 10-15% speedup

### What's a Good Result?

**Compared to baseline (Adam):**

| Metric | Good Result | Excellent Result |
|--------|-------------|------------------|
| Speedup | 10-15% | 20-30% |
| Energy savings | 10-15% | 20-30% |
| Efficiency gain (η) | 15-20% | 25-35% |
| Accuracy | ±0.5pp | ±0.2pp |

**Your results will vary based on:**
- Dataset complexity (simple datasets = better results)
- Model architecture
- Training duration (longer = better)
- Hardware capabilities

---

## Choosing the Right Approach

### Decision Tree

```
START HERE
│
├─ Are you NEW to H3?
│  └─ YES → Use AutoH3 (Option 1)
│
├─ Want ZERO configuration?
│  └─ YES → Use AutoH3 (Option 1)
│
├─ Want quick setup with GOOD DEFAULTS?
│  └─ YES → Use Presets (Option 3)
│
├─ Need FULL CONTROL over parameters?
│  └─ YES → Use Manual Setup (Option 2)
│
└─ Want to PROFILE FIRST (zero risk)?
   └─ YES → Use H3Profiler (see Example 04)
```

### Comparison Table

| Approach | Setup Lines | Control | Auto-tuning | Best For |
|----------|-------------|---------|-------------|----------|
| **AutoH3** | 1 line | Low | ✅ Yes | Beginners, quick experiments |
| **Presets** | 2-3 lines | Medium | No | Production, known dataset types |
| **Manual** | 5-10 lines | High | No | Research, custom requirements |
| **H3Profiler** | 2 lines | N/A | N/A | Risk-free exploration |

---

## Common Patterns

### Pattern 1: Quick Experiment

```python
from h3 import AutoH3

auto = AutoH3(model.parameters(), dataset, mode="balanced")
auto.start()

for epoch in range(epochs):
    loader = auto.get_loader(batch_size=64)
    for data, target, indices in loader:
        loss = auto.training_step(model, data, target, criterion, indices)
    auto.evaluate_epoch(model, test_loader)

results = auto.finish()
```

### Pattern 2: Production Setup

```python
from h3.presets import h3_mnist_fast
from h3.hooks import ThermoAuditLogger

# Setup with logging
optimizer, loss_tracker, indexed_dataset = h3_mnist_fast(
    model.parameters(), dataset
)

logger = ThermoAuditLogger(
    "production_run",
    metadata={"model": "ResNet-18", "dataset": "CIFAR-10"}
)

# ... training with logging ...

# Analyze results
from h3 import explain_thermo_log
print(explain_thermo_log(logger.get_path()))
```

### Pattern 3: Profile-First Workflow

```python
from h3 import H3Profiler

# 1. Profile your current optimizer (Adam, SGD, etc.)
profiler = H3Profiler(device='cuda', name="baseline_adam")
profiler.start()

# ... your normal training loop (no changes!) ...

results = profiler.stop()
print(profiler.get_report())

# 2. If results suggest H3 could help, try AutoH3
from h3 import AutoH3
auto = AutoH3(model.parameters(), dataset, mode="balanced")
# ... training ...
```

---

## Troubleshooting Common Issues

### Issue 1: "Using fallback energy estimation"

**What it means:** Real power monitoring unavailable

**Solutions:**
- **NVIDIA GPU:** Install `pip install pynvml`
- **Apple Silicon:** Use sudo (if you have permissions)
- **Otherwise:** Fallback works fine for comparisons!

### Issue 2: Accuracy drops significantly

**What it means:** Sampling too aggressive for your dataset

**Solutions:**
```python
# Use more conservative mode
auto = AutoH3(model.parameters(), dataset, mode="safe")

# Or with manual setup, increase keep_frac
sampler = InformationWeightedSampler(
    dataset, loss_tracker,
    keep_frac=0.75,  # Was 0.65, now more conservative
    uniform_mix=0.4   # Was 0.2, now more exploration
)
```

### Issue 3: Loss fluctuates heavily

**What it means:** Needs more warmup or smoother tracking

**Solutions:**
```python
# Increase loss tracker smoothing
loss_tracker = LossTracker(len(dataset), smoothing=0.2)  # Was 0.1

# Or increase warmup epochs
warmup_epochs = 3  # Was 1
```

### Issue 4: Import errors

**Problem:**
```python
ImportError: No module named 'h3'
```

**Solution:**
```bash
pip install h3-optimizer  # Make sure it's installed!
```

For more issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## Next Steps

### Level Up Your H3 Skills

1. **Try all examples:**
   - [Example 01: Fashion MNIST](../examples/01_fashion_mnist_beginner.py) - Complete tutorial
   - [Example 02: AutoH3](../examples/02_autoh3_zero_config.py) - Zero config
   - [Example 03: η-Controller](../examples/03_eta_controller_autotuning.py) - Auto-tuning
   - [Example 04: H3Profiler](../examples/04_profiler_any_optimizer.py) - Profiling
   - [Example 05: Presets](../examples/05_presets_showcase.py) - Pre-tuned configs
   - [Example 06: Custom Data](../examples/06_custom_datasets.py) - Your own data

2. **Read the documentation:**
   - [API Reference](API_REFERENCE.md) - Complete API docs
   - [Presets Guide](PRESETS.md) - Pre-tuned configurations
   - [Theory](THEORY.md) - Understand the thermodynamics
   - [Advanced Usage](ADVANCED.md) - Power user features

3. **Experiment with your data:**
   - Start with H3Profiler to establish baseline
   - Try AutoH3 with balanced mode
   - Compare results
   - Fine-tune based on your needs

4. **Join the community:**
   - Star the repo: https://github.com/nfocardoso/EMSTI
   - Report issues: https://github.com/nfocardoso/EMSTI/issues
   - Read the paper: https://zenodo.org/records/14357760

---

## Quick Reference Card

```python
# OPTION 1: AutoH3 (Simplest)
from h3 import AutoH3
auto = AutoH3(model.parameters(), dataset, mode="balanced")
auto.start()
# ... training with auto.training_step() ...
results = auto.finish()

# OPTION 2: Presets (Quick & Good Defaults)
from h3.presets import h3_mnist_fast
optimizer, loss_tracker, indexed_dataset = h3_mnist_fast(
    model.parameters(), dataset
)
# ... training loop ...

# OPTION 3: Manual (Full Control)
from h3 import H3Optimizer, LossTracker, InformationWeightedSampler
optimizer = H3Optimizer(model.parameters(), lr=1e-3)
loss_tracker = LossTracker(len(dataset))
sampler = InformationWeightedSampler(dataset, loss_tracker)
# ... training loop ...

# OPTION 4: Profile First (Zero Risk)
from h3 import H3Profiler
profiler = H3Profiler(device='cuda')
profiler.start()
# ... your normal training (no changes!) ...
results = profiler.stop()
```

---

## Summary

**You've learned:**
- ✅ How to install H3 Optimizer
- ✅ Three ways to use H3 (AutoH3, Presets, Manual)
- ✅ How to interpret results
- ✅ Common patterns and troubleshooting
- ✅ Where to go next

**Remember:**
1. **Start simple** with AutoH3
2. **Profile first** if unsure
3. **Use presets** for production
4. **Read examples** to learn patterns

---

**Ready to optimize your training?** 🚀

Go to: [Examples](../examples/README.md) | [API Reference](API_REFERENCE.md) | [Main Docs](../DOCUMENTATION.md)

---

**Questions?** Open an issue: https://github.com/nfocardoso/EMSTI/issues

⚡ Happy training with H3! 📊
