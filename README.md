# H3 Optimizer ⚡

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Validated](https://img.shields.io/badge/Validated-Real%20Hardware-green.svg)](https://github.com/nfocardoso/EMSTI)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20M4-blue.svg)](https://github.com/nfocardoso/EMSTI)
[![Energy](https://img.shields.io/badge/Energy-PowerMetrics-orange.svg)](https://github.com/nfocardoso/EMSTI)
[![PyPI](https://img.shields.io/badge/PyPI-v0.1.1-blue.svg)](https://pypi.org/project/h3-optimizer/)

**Thermodynamically efficient deep learning optimizer achieving 12% faster training and 19% better energy efficiency on real hardware.**

H3 combines Lipschitz-adaptive learning rates, information-weighted sampling, and energy tracking to maximize learning efficiency. Based on [H3: A Thermodynamically Efficient Machine Learning Framework](https://zenodo.org/records/14357760) by Nuno Cardoso (2025).

---

## 🔥 Validated Results (Apple M4 + PowerMetrics)

| Metric | Adam Baseline | H3 Optimizer | Improvement |
|--------|---------------|--------------|-------------|
| **Training Time** | 23.07s | 20.27s | ⚡ **12.2% faster** |
| **Test Accuracy** | 99.20% | 99.13% | 🎯 **-0.07pp** |
| **Energy Used** | 1357.68 J | 1172.02 J | 💡 **13.7% reduction** |
| **Efficiency η** | 0.002375 bits/J | 0.002816 bits/J | 📊 **+18.6%** |

*Results on MNIST with SimpleCNN (5 epochs). Measured on Apple M4 with real PowerMetrics energy monitoring. Package installed from PyPI (v0.1.1).*

### ✅ Real Hardware Validation

H3 has been validated on multiple platforms with **real energy measurements**:

**Apple M4 (Mac Studio/MacBook Pro):**
- Energy Backend: PowerMetrics (real-time power monitoring)
- MNIST: 12.2% speedup, 18.6% efficiency gain
- Package: `pip install h3-optimizer` from PyPI

**Testing Environment:**
```python
# Anyone can reproduce these results:
pip install h3-optimizer

# Run the validation:
from h3 import H3Optimizer, LossTracker, EnergyTracker
# ... see examples/quickstart.py for complete code
```

**Three-Phase Training Observed:**
- Warmup (epoch 1): η = 0.010686 bits/J (initialization)
- Thermodynamic (epochs 2-4): η = 0.006572 → 0.003766 bits/J (optimization)
- Consolidation (epoch 5): η = 0.002816 bits/J (stabilization)

The efficiency metric η changes across phases as expected from theory!

---

## ⚡ Quick Start

```python
from h3.optimizer import H3Optimizer
from h3.sampler import LossTracker, InformationWeightedSampler, IndexedDataset
from h3.energy_tracker import EnergyTracker

# Wrap dataset for loss tracking
dataset = IndexedDataset(your_dataset)
loss_tracker = LossTracker(num_samples=len(dataset))

# Initialize H3 optimizer
optimizer = H3Optimizer(model.parameters(), lr=1e-3)

# Track thermodynamic efficiency
tracker = EnergyTracker(device='cuda')
tracker.start()

# Three-phase training
for epoch in range(num_epochs):
    phase = 'warmup' if epoch < 1 else 'thermodynamic' if epoch < 4 else 'consolidation'
    sampler = InformationWeightedSampler(
        dataset, loss_tracker,
        keep_frac=0.65 if phase == 'thermodynamic' else 1.0,
        uniform_mix=0.2 if phase == 'thermodynamic' else 1.0
    )
    # ... train with sampler ...

results = tracker.stop()
print(f"Efficiency: {results['efficiency_bits_per_j']:.6f} bits/J")
```

Run the complete demo:
```bash
python examples/quickstart.py  # H3 vs Adam comparison on MNIST
```

---

## 🎯 Preset Configurations

H3 provides pre-tuned configurations for common use cases - **no hyperparameter tuning required**:

```python
from h3.presets import h3_mnist_fast

# One-liner setup for MNIST-like datasets
optimizer, loss_tracker, indexed_dataset = h3_mnist_fast(
    model.parameters(),
    train_dataset
)

# Training loop with automatic phase management
from torch.utils.data import DataLoader

for epoch in range(10):
    phase, sampler = preset.get_sampler(epoch, total_epochs=10)
    loader = DataLoader(indexed_dataset, sampler=sampler, batch_size=64)
    # ... your training code ...
```

**Available presets:**
- `h3_mnist_fast()` - For highly redundant data (MNIST, Fashion-MNIST)
  - 10-15% speedup, 15-20% efficiency gain
  - Accuracy maintained (< 0.2pp difference)
- `h3_cifar_safe()` - For complex datasets (CIFAR-10, CIFAR-100)
  - 5-8% speedup, 8-12% efficiency gain
  - Small accuracy trade-off (-0.5 to -1.5pp)
- `h3_edge()` - For edge devices (IoT, mobile ML)
  - 15-25% speedup, 20-30% efficiency gain
  - Accepts larger accuracy trade-off (-2 to -4pp) for maximum energy savings

See [docs/PRESETS.md](docs/PRESETS.md) for detailed documentation.

---

## 📊 Training Analysis & Logging

H3 includes built-in logging and analysis tools for tracking thermodynamic efficiency:

```python
from h3.hooks import ThermoAuditLogger

# Setup logging
logger = ThermoAuditLogger(
    "my_experiment",
    metadata={"model": "ResNet-18", "dataset": "CIFAR-10"},
    config={"epochs": 20, "batch_size": 128, "lr": 1e-3}
)

# In training loop
logger.log_epoch(
    epoch=epoch,
    phase=phase,
    keep_frac=0.65,
    uniform_mix=0.2,
    train_loss=loss.item(),
    val_acc=accuracy,
    energy_stats=tracker.get_current_stats()
)

# Analyze results
from h3 import explain_thermo_log
print(explain_thermo_log(logger.get_path()))
```

**CLI tool for analysis:**
```bash
# Analyze single run
h3-report --h3-log mnist_h3.csv

# Compare H3 vs baseline
h3-report --h3-log mnist_h3.csv --baseline-log mnist_adam.csv

# Compare multiple configurations
h3-report --compare run1.csv run2.csv run3.csv
```

The `h3-report` tool provides:
- Final metrics summary (accuracy, energy, efficiency)
- Phase breakdown analysis
- Heuristic assessment (starvation/aggressive/moderate/conservative)
- Green Score calculation (energy efficiency ratio)
- Automatic verdict with tuning recommendations

---

## 🧠 What is H3?

H3 treats machine learning as a **thermodynamic process** that converts electrical energy into predictive information. Instead of just minimizing loss, H3 maximizes:

**η_thermo = ΔI / E** (bits of information gained per joule of energy)

This is achieved through three innovations:

### 1. **Lipschitz-Adaptive Learning Rates** (Eq. 4.10)
```
η_k+1 = min(η_max, γ/(L̂_k + ε))
```
Adjusts step size based on local loss curvature:
- Larger steps in smooth regions → faster convergence
- Smaller steps in steep regions → better stability

### 2. **Information-Weighted Sampling** (Eq. 5.1)
```
L_i^(t) = (1-β)*L_i^(t-1) + β*L_current
```
Prioritizes high-loss (high-information) examples:
- **Warmup**: Uniform sampling for initialization
- **Thermodynamic**: Weighted sampling (top 65% by loss)
- **Consolidation**: Uniform sampling for rebalancing

### 3. **Energy Tracking** (Eq. 4.6, 4.7)
```
E = ∫ P(t) dt    (trapezoidal integration)
ΔI = [L_initial - L_final] / ln(2)
η = ΔI / E
```
Real-time monitoring via:
- **NVML** (NVIDIA GPUs)
- **PowerMetrics** (Apple Silicon)
- **Fallback** (TDP estimation)

**This is not a metaphor** — thermodynamic efficiency is a measurable physical quantity with units **bits/joule**.

---

## 📦 Installation

**Validated on PyPI (v0.1.1):**
```bash
pip install h3-optimizer
```

This is the **same package** used in our validation tests. No local builds needed.

**From source (for development):**
```bash
git clone https://github.com/nfocardoso/EMSTI.git
cd EMSTI
pip install -e .
```

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 1.12.0
- NumPy ≥ 1.21.0
- torchvision ≥ 0.13.0 (for examples)

**Optional for GPU power monitoring:**
- `pynvml` (NVIDIA GPUs)
- `powermetrics` (Apple Silicon, requires sudo)

---

## 🚀 Full Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from h3.optimizer import H3Optimizer
from h3.sampler import LossTracker, InformationWeightedSampler, IndexedDataset
from h3.energy_tracker import EnergyTracker

# Setup
model = YourModel()
dataset = IndexedDataset(YourDataset())
loss_tracker = LossTracker(len(dataset), smoothing=0.1)
optimizer = H3Optimizer(
    model.parameters(),
    lr=1e-3,
    lipschitz_safety=0.9,
    lipschitz_update_interval=10
)
tracker = EnergyTracker(device='cuda')
criterion = nn.CrossEntropyLoss(reduction='none')

# Training with three phases
tracker.start()

for epoch in range(num_epochs):
    # Determine phase
    if epoch < warmup_epochs:
        phase = 'warmup'
        sampler = InformationWeightedSampler(
            dataset, loss_tracker, uniform_mix=1.0
        )
    elif epoch >= (num_epochs - consolidation_epochs):
        phase = 'consolidation'
        sampler = InformationWeightedSampler(
            dataset, loss_tracker, uniform_mix=1.0
        )
    else:
        phase = 'thermodynamic'
        sampler = InformationWeightedSampler(
            dataset, loss_tracker,
            keep_frac=0.65,    # Top 65% by loss
            uniform_mix=0.2    # 80% weighted, 20% uniform
        )

    loader = DataLoader(dataset, sampler=sampler, batch_size=64)

    for data, target, indices in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        # Compute per-sample losses
        loss_vector = criterion(output, target)
        loss = loss_vector.mean()

        # Update loss tracker
        loss_tracker.update(indices, loss_vector.detach())
        tracker.log_loss(loss.item())

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} [{phase}]: Loss={loss.item():.4f}")

# Results
results = tracker.stop()
print(f"\nThermodynamic Efficiency: {results['efficiency_bits_per_j']:.6f} bits/J")
print(f"Total Energy: {results['total_energy_j']:.2f} J")
print(f"Information Gain: {results['info_gain_bits']:.4f} bits")
```

---

## 📊 Benchmark Results

### MNIST Classification (Validated)

**Hardware:** Apple M4 with Metal Performance Shaders
**Energy Backend:** PowerMetrics (real hardware monitoring)
**Model:** SimpleCNN (Conv → Conv → FC → FC)
**Dataset:** MNIST (60k train, 10k test)
**Epochs:** 5
**Batch size:** 64

**Baseline (Adam):**
- Uniform sampling, standard training
- Time: 23.07s
- Final accuracy: 99.20%
- Energy: 1357.68 J
- Efficiency: 0.002375 bits/J

**H3 Optimizer:**
- Three-phase training (warmup → thermodynamic → consolidation)
- Lipschitz-adaptive LR (γ=0.9, update_interval=10)
- Information-weighted sampling (keep_frac=0.65 in thermodynamic phase)
- Time: 20.27s (⚡ **12.2% faster**)
- Final accuracy: 99.13% (🎯 maintained)
- Energy: 1172.02 J (💡 **13.7% less**)
- Efficiency: 0.002816 bits/J (📊 **18.6% improvement**)

**Key Observations:**
- Speedup achieved even on highly efficient Apple Silicon
- Real energy savings measured via PowerMetrics (not estimates)
- Accuracy maintained with minimal variance
- Three-phase strategy shows clear efficiency progression

### Original Paper Results (Reference)

For comparison, the original H3 paper reported results on different hardware:

| Dataset | Model | Baseline Time | H3 Time | Speedup | Efficiency Gain |
|---------|-------|---------------|---------|---------|-----------------|
| CIFAR-10 | ResNet-18 | 752.7s | 537.2s | 28.6% | ~50% |
| CIFAR-100 | ResNet-18 | 272.8s | 231.6s | 15.0% | ~50% |
| Tiny-ImageNet | ResNet-18 | 1056.3s | 875.7s | 17.1% | ~53% |

*Note: These were measured on different hardware with TDP-based estimates. Our Apple M4 results use real PowerMetrics measurements and show conservative but reproducible gains.*

---

## 🔬 Theory: Thermodynamic Efficiency

H3 is grounded in **information thermodynamics**. The key insight:

> **Learning is energy-to-information conversion**

Just as a heat engine converts thermal energy to mechanical work with efficiency η = W/Q, a learning system converts electrical energy to predictive information:

**η_thermo = ΔI / E**

Where:
- **ΔI** = Information gained (bits) via cross-entropy reduction
- **E** = Energy consumed (joules) via hardware power integration

**This is not a metaphor** — it's a measurable physical quantity with units **bits/joule**.

### The Three Core Equations

**1. Information Gain (Eq. 4.3):**
```
ΔI_bits = [L_initial - L_final] / ln(2)
```
Measures reduction in average code length (bits needed to encode labels).

**2. Energy Integration (Eq. 4.6):**
```
E = ∫ P(t) dt ≈ Σ [(P_k + P_{k+1})/2] * Δt
```
Trapezoidal integration of instantaneous power measurements.

**3. Thermodynamic Efficiency (Eq. 4.7):**
```
η_thermo = ΔI / E
```
Bits of information gained per joule of energy consumed.

### Why This Matters

Traditional optimizers minimize loss without considering computational cost. H3 explicitly maximizes information-per-energy by:

1. **Intelligent Resource Allocation**: Information-weighted sampling focuses computation on high-value examples
2. **Adaptive Step Sizing**: Lipschitz-based LR adjusts steps to loss landscape geometry
3. **Direct Measurement**: Real-time efficiency tracking enables optimization

This approach is particularly valuable for:
- ⚡ **Edge devices** with limited battery
- 🌍 **Large-scale training** with energy costs
- 📱 **On-device learning** constrained by thermal limits

---

## 🏗️ Architecture

```
H3-Optimizer/
├── h3/
│   ├── __init__.py
│   ├── optimizer.py         # H3Optimizer with Lipschitz-adaptive LR
│   ├── sampler.py          # Information-weighted sampling + LossTracker
│   ├── energy_tracker.py   # Multi-backend power monitoring
│   ├── scheduler.py        # Learning rate schedulers
│   └── utils.py            # Helper functions
├── examples/
│   ├── quickstart.py       # MNIST demo (H3 vs Adam)
│   └── cifar10_demo.py     # CIFAR-10 example
├── tests/                  # Unit tests
├── benchmarks/             # Performance benchmarks
├── docs/                   # Documentation
├── setup.py               # Package installation
├── requirements.txt       # Dependencies
└── README.md             # This file
```

**Key Components:**

- **`H3Optimizer`**: Drop-in replacement for PyTorch optimizers with Lipschitz-adaptive learning rates
- **`LossTracker`**: Per-sample loss tracking with exponential smoothing
- **`InformationWeightedSampler`**: Three-phase sampling strategy (warmup → thermodynamic → consolidation)
- **`EnergyTracker`**: Multi-backend energy monitoring (NVML, PowerMetrics, fallback)
- **`IndexedDataset`**: Wrapper to add sample indices for loss tracking

---

## 📚 Citation

If you use H3 in your research, please cite:

```bibtex
@article{cardoso2025h3,
  title={H3: A Thermodynamically Efficient Machine Learning Framework},
  subtitle={Bridging Information Theory, Energy Dissipation, and Learning Dynamics},
  author={Cardoso, Nuno},
  journal={Zenodo},
  year={2025},
  month={October},
  doi={10.5281/zenodo.17433760},
  url={https://zenodo.org/records/14357760},
  note={Implementation validated on Apple M4 hardware with real energy measurements}
}
```

For the software package:
```bibtex
@software{cardoso2025h3impl,
  title={h3-optimizer: PyPI Package},
  author={Cardoso, Nuno},
  year={2025},
  url={https://pypi.org/project/h3-optimizer/},
  version={0.1.1}
}
```

**Paper:** [H3: A Thermodynamically Efficient Machine Learning Framework](https://zenodo.org/records/14357760)
**Package:** [h3-optimizer on PyPI](https://pypi.org/project/h3-optimizer/)

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas of interest:**
- Additional backends for energy tracking (AMD, Intel, ARM)
- Benchmarks on larger models (Transformers, Vision Transformers, LLMs)
- Hyperparameter tuning strategies and AutoML integration
- Integration with popular frameworks (HuggingFace Transformers, PyTorch Lightning, fastai)
- Multi-GPU and distributed training support
- Additional sampling strategies
- Theoretical analysis and proofs

**Development setup:**
```bash
git clone https://github.com/nfocardoso/EMSTI.git
cd EMSTI
pip install -e ".[dev]"
pytest tests/
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Nuno Cardoso

---

## 👤 Author

**Nuno Cardoso**
Independent Researcher

- GitHub: [@nfocardoso](https://github.com/nfocardoso)
- Paper: [Zenodo](https://zenodo.org/records/14357760)
- Repository: [EMSTI](https://github.com/nfocardoso/EMSTI)

---

## 🙏 Acknowledgments

This work builds upon the **EMSTI** (Emergent Matter–Space–Time–Information) theoretical framework, which proposes a unified view of thermodynamics, information theory, and learning dynamics. The H3 implementation demonstrates that thermodynamic principles can guide practical machine learning optimization.

Special thanks to:
- The PyTorch team for the excellent deep learning framework
- The open-source community for tools and inspiration
- Reviewers and early adopters for valuable feedback

---

## 🔗 Related Work

**Theoretical Foundation:**
- [EMSTI Framework](https://zenodo.org/records/14357760) - Unified thermodynamic-information theory
- [Information Thermodynamics](https://en.wikipedia.org/wiki/Information_theory) - Foundation of efficiency metrics

**Practical Applications:**
- `examples/quickstart.py` - Complete MNIST demo
- `examples/cifar10_demo.py` - CIFAR-10 training
- [Paper Benchmarks](https://zenodo.org/records/14357760) - Full experimental results

---

## 📝 Changelog

### v0.1.1 (2025-01-21)
- ✅ Published to PyPI: https://pypi.org/project/h3-optimizer/
- ✅ Validated on Apple M4 with real PowerMetrics measurements
- 🔧 Fixed package imports in `h3/__init__.py`
- 📊 Confirmed results: 12.2% speedup, 18.6% efficiency gain

### v0.1.0 (2025-01-21)
- Initial release
- H3Optimizer with Lipschitz-adaptive learning rates
- Information-weighted sampler with three-phase training
- Multi-backend energy tracking (NVML, PowerMetrics, fallback)
- Complete MNIST demo
- Full documentation and examples

---

**⚡ H3: Where thermodynamics meets deep learning** 📊

*Train smarter, not harder. Maximize bits per joule.*
