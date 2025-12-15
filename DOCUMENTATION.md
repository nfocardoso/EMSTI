# H3 Optimizer - Complete Documentation Index 📚

**Thermodynamically Efficient Deep Learning Framework**

This is the central documentation hub for the H3 Optimizer. All documentation is organized hierarchically for easy navigation.

---

## 🚀 Quick Navigation

### New to H3?
1. **[Getting Started Guide](docs/GETTING_STARTED.md)** - Step-by-step tutorial for beginners
2. **[Examples Overview](examples/README.md)** - Complete code examples
3. **[Main README](README.md)** - Project overview and quick start

### Already familiar?
- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation
- **[Presets Guide](docs/PRESETS.md)** - Pre-tuned configurations
- **[Theory & Concepts](docs/THEORY.md)** - Thermodynamic principles

### Production Use
- **[Advanced Usage](docs/ADVANCED.md)** - Advanced features and optimization
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Contributing](docs/CONTRIBUTING.md)** - How to contribute

---

## 📖 Documentation Structure

### Level 1: Getting Started

| Document | Description | Audience |
|----------|-------------|----------|
| [README.md](README.md) | Project overview, quick start, installation | Everyone |
| [Getting Started](docs/GETTING_STARTED.md) | Complete beginner tutorial | Beginners |
| [Examples](examples/README.md) | 6 complete examples + original demos | All levels |

**Start here if you're new to H3!**

---

### Level 2: Core Documentation

#### 🎯 Essential Guides

| Document | Description | When to Use |
|----------|-------------|-------------|
| [API Reference](docs/API_REFERENCE.md) | Complete API documentation | Need detailed parameter info |
| [Presets Guide](docs/PRESETS.md) | Pre-tuned configurations | Want quick setup |
| [Theory](docs/THEORY.md) | Thermodynamic principles | Want to understand how it works |

#### 💡 Examples by Use Case

| Example | Description | Complexity |
|---------|-------------|------------|
| [01_fashion_mnist_beginner.py](examples/01_fashion_mnist_beginner.py) | Complete tutorial with explanations | Beginner |
| [02_autoh3_zero_config.py](examples/02_autoh3_zero_config.py) | Simplest API (one-liner) | Beginner |
| [03_eta_controller_autotuning.py](examples/03_eta_controller_autotuning.py) | Self-tuning system | Intermediate |
| [04_profiler_any_optimizer.py](examples/04_profiler_any_optimizer.py) | Profile existing optimizers | All levels |
| [05_presets_showcase.py](examples/05_presets_showcase.py) | Pre-tuned configs showcase | Intermediate |
| [06_custom_datasets.py](examples/06_custom_datasets.py) | Custom data (CSV, NumPy, etc.) | Intermediate |
| [quickstart.py](examples/quickstart.py) | H3 vs Adam comparison | All levels |
| [cifar10_demo.py](examples/cifar10_demo.py) | Complex dataset training | Advanced |

---

### Level 3: Advanced Topics

| Document | Description | For |
|----------|-------------|-----|
| [Advanced Usage](docs/ADVANCED.md) | η-Controller, AutoH3, H3Profiler | Power users |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues and solutions | Everyone |
| [Contributing](docs/CONTRIBUTING.md) | Development guide | Contributors |

---

## 🔍 Quick Reference by Task

### "I want to..."

#### Install H3
```bash
pip install h3-optimizer
```
See: [README.md](README.md#installation)

#### Get started quickly
→ [Getting Started Guide](docs/GETTING_STARTED.md)
→ [Example 01: Fashion MNIST](examples/01_fashion_mnist_beginner.py)

#### Use the simplest API possible
→ [Example 02: AutoH3](examples/02_autoh3_zero_config.py)
→ [API Reference: AutoH3](docs/API_REFERENCE.md#autoh3)

#### Profile my current optimizer first
→ [Example 04: H3Profiler](examples/04_profiler_any_optimizer.py)
→ [API Reference: H3Profiler](docs/API_REFERENCE.md#h3profiler)

#### Use pre-tuned configurations
→ [Presets Guide](docs/PRESETS.md)
→ [Example 05: Presets Showcase](examples/05_presets_showcase.py)

#### Let H3 tune itself automatically
→ [Example 03: η-Controller](examples/03_eta_controller_autotuning.py)
→ [API Reference: η-Controller](docs/API_REFERENCE.md#eta-controller)

#### Work with custom data (CSV, NumPy, etc.)
→ [Example 06: Custom Datasets](examples/06_custom_datasets.py)

#### Understand the theory
→ [Theory Guide](docs/THEORY.md)
→ [Research Paper](https://zenodo.org/records/14357760)

#### Compare H3 vs baseline
→ [quickstart.py](examples/quickstart.py)
→ [CLI: h3-report](docs/API_REFERENCE.md#cli-tools)

#### Troubleshoot issues
→ [Troubleshooting Guide](docs/TROUBLESHOOTING.md)

#### Contribute to H3
→ [Contributing Guide](docs/CONTRIBUTING.md)

---

## 📊 Feature Comparison Matrix

| Feature | Manual H3 | Presets | η-Controller | AutoH3 | H3Profiler |
|---------|-----------|---------|--------------|--------|------------|
| **Setup complexity** | High | Low | Medium | Very Low | Minimal |
| **Hyperparameter tuning** | Manual | Pre-tuned | Automatic | Automatic | N/A |
| **Works with any optimizer** | No | No | No | No | ✅ YES |
| **Real-time adaptation** | No | No | ✅ YES | ✅ YES | No |
| **Zero risk** | No | No | No | No | ✅ YES |
| **Best for** | Research | Quick start | Production | Beginners | Profiling |
| **Documentation** | [API Ref](docs/API_REFERENCE.md#h3optimizer) | [Presets](docs/PRESETS.md) | [Example 03](examples/03_eta_controller_autotuning.py) | [Example 02](examples/02_autoh3_zero_config.py) | [Example 04](examples/04_profiler_any_optimizer.py) |

---

## 📦 Core Components

### Optimizers & Sampling

- **[H3Optimizer](docs/API_REFERENCE.md#h3optimizer)** - Thermodynamic optimizer with Lipschitz-adaptive LR
- **[LossTracker](docs/API_REFERENCE.md#losstracker)** - Per-sample loss tracking
- **[InformationWeightedSampler](docs/API_REFERENCE.md#informationweightedsampler)** - Three-phase sampling
- **[IndexedDataset](docs/API_REFERENCE.md#indexeddataset)** - Dataset wrapper for tracking

### Automation & Control

- **[AutoH3](docs/API_REFERENCE.md#autoh3)** - Zero-config complete automation
- **[η-Controller](docs/API_REFERENCE.md#eta-controller)** - Automatic hyperparameter tuning
- **[Presets](docs/PRESETS.md)** - Pre-tuned configurations (MNIST, CIFAR, Edge)

### Monitoring & Analysis

- **[EnergyTracker](docs/API_REFERENCE.md#energytracker)** - Multi-backend energy monitoring
- **[H3Profiler](docs/API_REFERENCE.md#h3profiler)** - Zero-risk profiling for any optimizer
- **[ThermoAuditLogger](docs/API_REFERENCE.md#thermoauditlogger)** - Experiment tracking
- **[CLI Tools](docs/API_REFERENCE.md#cli-tools)** - h3-report for analysis

---

## 🎓 Learning Path

### Path 1: Complete Beginner (Recommended)

1. Read [Getting Started Guide](docs/GETTING_STARTED.md)
2. Run [Example 01: Fashion MNIST](examples/01_fashion_mnist_beginner.py)
3. Try [Example 02: AutoH3](examples/02_autoh3_zero_config.py)
4. Read [Presets Guide](docs/PRESETS.md)
5. Explore [API Reference](docs/API_REFERENCE.md) as needed

**Time: ~2 hours** | **Outcome: Ready to use H3**

### Path 2: Quick Start (Experienced Users)

1. Install: `pip install h3-optimizer`
2. Run [Example 02: AutoH3](examples/02_autoh3_zero_config.py)
3. Skim [API Reference](docs/API_REFERENCE.md)
4. Check [Presets Guide](docs/PRESETS.md) for your dataset
5. Go!

**Time: ~30 minutes** | **Outcome: Working H3 setup**

### Path 3: Research/Deep Understanding

1. Read [Research Paper](https://zenodo.org/records/14357760)
2. Study [Theory Guide](docs/THEORY.md)
3. Review [API Reference](docs/API_REFERENCE.md)
4. Examine all [Examples](examples/README.md)
5. Explore [Advanced Usage](docs/ADVANCED.md)

**Time: ~8 hours** | **Outcome: Expert understanding**

### Path 4: Profile-First (Cautious Approach)

1. Run [Example 04: H3Profiler](examples/04_profiler_any_optimizer.py) on your current optimizer
2. Analyze baseline efficiency
3. Try [Example 02: AutoH3](examples/02_autoh3_zero_config.py) with balanced mode
4. Compare results
5. Fine-tune with [Presets](docs/PRESETS.md) or [η-Controller](examples/03_eta_controller_autotuning.py)

**Time: ~1 hour** | **Outcome: Data-driven decision**

---

## 🔬 Technical Documentation

### Architecture & Design

```
h3/
├── optimizer.py           # H3Optimizer (Lipschitz-adaptive LR)
├── sampler.py            # LossTracker + InformationWeightedSampler
├── energy_tracker.py     # Multi-backend energy monitoring
├── profiler.py           # H3Profiler (zero-risk analysis)
├── eta_controller.py     # Automatic hyperparameter tuning
├── auto.py               # AutoH3 (complete automation)
├── presets.py            # Pre-tuned configurations
├── hooks.py              # ThermoAuditLogger + logging
├── explain.py            # Analysis and reporting tools
├── cli.py                # Command-line interface
├── scheduler.py          # Learning rate schedulers
├── adaptive.py           # Adaptive annealing strategies
└── utils.py              # Helper functions
```

See: [API Reference](docs/API_REFERENCE.md) for detailed documentation

### Thermodynamic Equations

**Efficiency Metric:**
```
η = ΔI / E  (bits per joule)

Where:
  ΔI = [L_initial - L_final] / ln(2)  [Information gain in bits]
  E  = ∫ P(t) dt                       [Energy consumed in joules]
```

**Lipschitz-Adaptive Learning Rate:**
```
η_k+1 = min(η_max, γ/(L̂_k + ε))

Where:
  L̂_k = estimated local Lipschitz constant
  γ   = safety factor (default 0.9)
  ε   = numerical stability term
```

See: [Theory Guide](docs/THEORY.md) for complete mathematical foundations

---

## 📈 Performance Expectations

### Validated Results (Real Hardware)

| Dataset Type | Preset/Mode | Speedup | Efficiency Gain | Accuracy |
|-------------|-------------|---------|-----------------|----------|
| Simple (MNIST) | `h3_mnist_fast` | 10-15% | 15-20% | ±0.2pp |
| Complex (CIFAR-10) | `h3_cifar_safe` | 5-8% | 8-12% | -0.5 to -1.5pp |
| Edge devices | `h3_edge` | 15-25% | 20-30% | -2 to -4pp |
| Balanced mode | `AutoH3(mode="balanced")` | 10-20% | 15-25% | ±1.0pp |

**Hardware:** Apple M4, NVIDIA GPUs
**Energy Backend:** PowerMetrics, NVML
**Validation:** Real hardware measurements

See: [README.md](README.md#validated-performance) for detailed benchmarks

---

## 🌐 External Resources

### Official Links

- **PyPI Package:** https://pypi.org/project/h3-optimizer/
- **GitHub Repository:** https://github.com/nfocardoso/EMSTI
- **Research Paper:** https://zenodo.org/records/14357760
- **Project Website:** [docs/index.html](docs/index.html)

### Community & Support

- **Issues:** https://github.com/nfocardoso/EMSTI/issues
- **Discussions:** GitHub Discussions (coming soon)
- **Citation:** See [README.md](README.md#citation)

---

## 📋 Deployment & Release Documentation

Internal development documentation:

- [Changelog](docs/deployment/CHANGELOG.md) - Version history
- [Deployment Status](docs/deployment/DEPLOYMENT_STATUS.md) - Current deployment state
- [Release Notes](docs/deployment/) - Version-specific notes

---

## 🤝 Contributing

Want to contribute? See:

- [Contributing Guide](docs/CONTRIBUTING.md) - Development workflow
- [API Reference](docs/API_REFERENCE.md) - Code architecture
- [GitHub Issues](https://github.com/nfocardoso/EMSTI/issues) - Open issues

---

## 📝 Document Status

| Document | Status | Last Updated |
|----------|--------|--------------|
| README.md | ✅ Complete | 2025-01-21 |
| GETTING_STARTED.md | 🚧 In Progress | - |
| API_REFERENCE.md | 🚧 In Progress | - |
| PRESETS.md | ✅ Complete | 2025-01-21 |
| THEORY.md | 🚧 In Progress | - |
| ADVANCED.md | 📝 Planned | - |
| TROUBLESHOOTING.md | 📝 Planned | - |
| CONTRIBUTING.md | 📝 Planned | - |
| examples/README.md | ✅ Complete | 2025-01-21 |

---

## 🎯 Quick Command Reference

```bash
# Installation
pip install h3-optimizer

# Run examples
python examples/01_fashion_mnist_beginner.py
python examples/02_autoh3_zero_config.py
python examples/04_profiler_any_optimizer.py

# CLI analysis
h3-report --h3-log mnist_h3.csv
h3-report --h3-log h3.csv --baseline-log adam.csv
h3-report --compare run1.csv run2.csv run3.csv

# Development
pip install -e ".[dev]"
pytest tests/
```

---

**Last Updated:** 2025-01-21
**Version:** 0.2.1
**Maintained by:** Nuno Cardoso ([@nfocardoso](https://github.com/nfocardoso))

---

⚡ **H3: Where thermodynamics meets deep learning** 📊
