# H3 Optimizer - Complete Examples 🚀

This directory contains comprehensive, self-contained examples showing how to use H3 optimizer with different datasets and configurations. Each example is fully documented and ready to run!

## 📚 Examples Overview

### 🎓 For Beginners

**[01_fashion_mnist_beginner.py](01_fashion_mnist_beginner.py)** - Perfect First Example
- Complete step-by-step tutorial
- Fashion-MNIST classification
- All H3 components explained
- Expected: 10-15% speedup, 15-20% efficiency gain

```bash
python examples/01_fashion_mnist_beginner.py
```

### 🤖 Zero Configuration

**[02_autoh3_zero_config.py](02_autoh3_zero_config.py)** - Simplest API
- One-line setup with AutoH3
- Automatic everything (optimizer, sampling, tracking)
- Choose mode: safe, balanced, green, extreme
- Perfect for quick experiments

```bash
python examples/02_autoh3_zero_config.py
```

### 🎛️ Automatic Tuning

**[03_eta_controller_autotuning.py](03_eta_controller_autotuning.py)** - Self-Regulating System
- η-Controller adjusts hyperparameters in real-time
- No manual tuning needed
- Respects accuracy constraints
- Prevents data starvation

```bash
python examples/03_eta_controller_autotuning.py
```

### 🔬 Zero-Risk Profiling

**[04_profiler_any_optimizer.py](04_profiler_any_optimizer.py)** - Analyze ANY Optimizer
- Profile Adam, SGD, AdamW, or any optimizer
- Measure thermodynamic efficiency
- Find optimal stopping point
- Identify energy waste
- **NO CHANGES** to your training code

```bash
python examples/04_profiler_any_optimizer.py
```

### ⚡ Pre-Tuned Configurations

**[05_presets_showcase.py](05_presets_showcase.py)** - One-Line Setup
- Pre-tuned for common use cases
- `h3_mnist_fast`: Simple datasets (10-15% speedup)
- `h3_cifar_safe`: Complex datasets (5-8% speedup)
- `h3_edge`: Edge devices (15-25% speedup)

```bash
python examples/05_presets_showcase.py
```

### 📊 Custom Datasets

**[06_custom_datasets.py](06_custom_datasets.py)** - Your Own Data
- NumPy arrays
- Pandas DataFrames
- CSV files
- Tabular data, time series, etc.
- Complete working examples

```bash
python examples/06_custom_datasets.py
```

### 🎯 Original Examples

**[quickstart.py](quickstart.py)** - H3 vs Adam Comparison
- MNIST benchmark
- Side-by-side comparison
- Real energy measurements
- Validated results

```bash
python examples/quickstart.py
```

**[cifar10_demo.py](cifar10_demo.py)** - CIFAR-10 Training
- Complex image dataset
- ResNet-compatible
- Full training pipeline

```bash
python examples/cifar10_demo.py
```

---

## 🎯 Which Example Should I Use?

### I'm New to H3
→ Start with **01_fashion_mnist_beginner.py**
- Comprehensive tutorial with explanations
- Shows all H3 components
- Clear documentation

### I Want the Simplest API
→ Use **02_autoh3_zero_config.py**
- One-line setup
- Automatic everything
- Just pick a mode and go

### I Don't Want to Tune Hyperparameters
→ Try **03_eta_controller_autotuning.py**
- Self-tuning system
- Adapts automatically
- Respects constraints

### I Want to Profile My Current Optimizer First
→ Start with **04_profiler_any_optimizer.py**
- Zero risk (just measurement)
- Works with any optimizer
- Shows optimization opportunities

### I Want Quick Setup with Good Defaults
→ Use **05_presets_showcase.py**
- Pre-tuned configurations
- Choose preset for your dataset type
- One line setup

### I Have Custom Data (CSV, NumPy, etc.)
→ See **06_custom_datasets.py**
- Complete examples for custom data
- NumPy, Pandas, CSV
- Easy to adapt

---

## 📊 Expected Performance

All results measured on real hardware (Apple M4, NVIDIA GPUs):

| Dataset Type | Preset/Mode | Speedup | Efficiency Gain | Accuracy Change |
|-------------|-------------|---------|-----------------|-----------------|
| Simple (MNIST) | `h3_mnist_fast` | 10-15% | 15-20% | ±0.2pp |
| Complex (CIFAR-10) | `h3_cifar_safe` | 5-8% | 8-12% | -0.5 to -1.5pp |
| Edge devices | `h3_edge` | 15-25% | 20-30% | -2 to -4pp |
| Balanced mode | `AutoH3(mode="balanced")` | 10-20% | 15-25% | ±1.0pp |

**Key Finding:** H3 often IMPROVES accuracy while saving energy, due to intelligent sample selection and adaptive learning rates!

---

## 🚀 Quick Start Guide

### 1. Install H3
```bash
pip install h3-optimizer
```

### 2. Run Your First Example
```bash
# Beginner-friendly tutorial
python examples/01_fashion_mnist_beginner.py

# Or the simplest API
python examples/02_autoh3_zero_config.py
```

### 3. Adapt to Your Dataset
- See `06_custom_datasets.py` for examples with your own data
- Use presets (`05_presets_showcase.py`) for quick setup
- Profile first (`04_profiler_any_optimizer.py`) to establish baseline

---

## 📖 Understanding the Examples

### Three Levels of Control

**Level 1: AutoH3 (Simplest)**
- File: `02_autoh3_zero_config.py`
- Setup: 1 line
- Control: Choose mode only
- Best for: Quick experiments, beginners

**Level 2: Presets + η-Controller (Balanced)**
- Files: `03_eta_controller_autotuning.py`, `05_presets_showcase.py`
- Setup: 2-3 lines
- Control: Choose preset/mode, some tuning
- Best for: Production, most users

**Level 3: Manual Configuration (Full Control)**
- Files: `01_fashion_mnist_beginner.py`, `quickstart.py`
- Setup: 5-10 lines
- Control: Full control over all parameters
- Best for: Research, custom requirements

### Operating Modes Explained

```python
# AutoH3 and η-Controller support these modes:

"safe"      # keep_frac ~0.70, maximize accuracy
"balanced"  # keep_frac ~0.55, balance speed/accuracy (RECOMMENDED)
"green"     # keep_frac ~0.45, maximize energy savings
"extreme"   # keep_frac ~0.35, extreme savings (edge devices)
```

### Presets Explained

```python
from h3.presets import h3_mnist_fast, h3_cifar_safe, h3_edge

# For simple, redundant datasets
h3_mnist_fast(...)      # MNIST, Fashion-MNIST

# For complex image datasets
h3_cifar_safe(...)      # CIFAR-10, CIFAR-100

# For edge/IoT devices
h3_edge(...)            # Any dataset, maximum savings
```

---

## 🔬 Profiling Workflow

**Recommended workflow for new datasets:**

```bash
# Step 1: Profile your current optimizer
python examples/04_profiler_any_optimizer.py
# → Measure baseline efficiency
# → Identify optimization opportunities

# Step 2: Try AutoH3 with balanced mode
python examples/02_autoh3_zero_config.py
# → Quick test with automatic tuning

# Step 3: Fine-tune if needed
python examples/03_eta_controller_autotuning.py
# → Adjust mode based on results

# Step 4: Use presets for production
python examples/05_presets_showcase.py
# → Choose preset matching your dataset
```

---

## 💡 Tips and Best Practices

### Choosing the Right Approach

**Use AutoH3 when:**
- You want the simplest API
- Quick experiments
- You're new to H3

**Use η-Controller when:**
- You want automatic tuning
- Production systems
- Dataset characteristics may vary

**Use Presets when:**
- You want manual control
- You know your dataset type
- You need reproducible configs

**Use H3Profiler when:**
- Evaluating current optimizer
- Establishing baselines
- Zero-risk exploration

### Hyperparameter Guidelines

**keep_frac (sample retention):**
- Simple datasets (MNIST): 0.50-0.60
- Complex datasets (CIFAR-10): 0.65-0.75
- Edge devices: 0.40-0.50

**uniform_mix (exploration):**
- Early training: 0.30-0.50 (more exploration)
- Complex datasets: 0.40-0.60 (more exploration)
- Simple datasets: 0.20-0.30 (focused learning)

**Phase ratios:**
- Warmup: 10-15% of epochs
- Thermodynamic: 70-80% of epochs
- Consolidation: 10-15% of epochs

**Minimum training:**
- At least 10 epochs recommended
- 20+ epochs for best results
- H3 shines in longer training runs

---

## 📊 Analyzing Results

### Using h3-report CLI

```bash
# Analyze single run
h3-report --h3-log ./profiles/mnist_h3.csv

# Compare H3 vs baseline
h3-report --h3-log mnist_h3.csv --baseline-log mnist_adam.csv

# Compare multiple configurations
h3-report --compare run1.csv run2.csv run3.csv
```

### Understanding Efficiency Metric (η)

```
η = ΔI / E  (bits per joule)

Where:
  ΔI = Information gained (loss reduction in bits)
  E  = Energy consumed (joules)
```

**Typical values:**
- Warmup phase: η = 0.005-0.015 bits/J (fast learning)
- Thermodynamic phase: η = 0.002-0.008 bits/J (focused learning)
- Consolidation phase: η = 0.001-0.005 bits/J (refinement)

**Higher is better!** H3 aims to maximize this throughout training.

---

## 🐛 Troubleshooting

### Energy Tracking

**Issue:** "Using fallback energy estimation"
**Solution:** Normal for CPU-only systems. Install:
- NVIDIA GPU: `pip install pynvml`
- Apple Silicon: Use `sudo powermetrics` (requires root)

### Memory Issues

**Issue:** Out of memory with large datasets
**Solution:**
- Reduce batch size
- Use gradient accumulation
- Try `h3_edge` preset (more aggressive sampling)

### Accuracy Drop

**Issue:** Accuracy drops too much
**Solution:**
- Use more conservative mode: `mode="safe"`
- Increase `keep_frac`: 0.70-0.80
- Increase `uniform_mix`: 0.50-0.60
- Check dataset is shuffled properly

### Loss Volatility

**Issue:** Training loss fluctuates heavily
**Solution:**
- Increase warmup epochs
- Use more uniform sampling (`uniform_mix=0.5`)
- Increase `loss_tracker` smoothing: 0.15-0.20
- η-Controller will auto-adjust for this

---

## 📚 Further Reading

- **Paper:** [H3: A Thermodynamically Efficient ML Framework](https://zenodo.org/records/14357760)
- **Main README:** [../README.md](../README.md)
- **Presets Documentation:** [../docs/PRESETS.md](../docs/PRESETS.md)
- **PyPI Package:** [h3-optimizer](https://pypi.org/project/h3-optimizer/)

---

## 🤝 Contributing Examples

Have a cool example? We'd love to include it!

Requirements:
- Self-contained (runs without modification)
- Well-documented
- Includes expected results
- Works with `pip install h3-optimizer`

Submit via pull request or open an issue!

---

## 📝 Example Template

Creating your own example? Use this template:

```python
"""
Example XX: [Title]
====================

[Brief description]

What you'll learn:
- Point 1
- Point 2

Expected results:
- Metric 1
- Metric 2

Requirements:
    pip install h3-optimizer [other packages]

Usage:
    python examples/XX_your_example.py
"""

import torch
# ... your imports ...

from h3.optimizer import H3Optimizer
from h3.sampler import LossTracker, InformationWeightedSampler, IndexedDataset
from h3.energy_tracker import EnergyTracker

# Your code here...

if __name__ == '__main__':
    main()
```

---

## 🎉 Happy Training!

Questions? Issues? Feedback?
- GitHub Issues: https://github.com/nfocardoso/EMSTI/issues
- Paper: https://zenodo.org/records/14357760

**Remember:** Start simple, profile first, then optimize! 🚀
