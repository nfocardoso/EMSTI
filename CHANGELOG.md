# Changelog

All notable changes to h3-optimizer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - UNRELEASED

### Added
- **H3Profiler** - Zero-risk thermodynamic profiler that works with ANY optimizer (Adam, SGD, AdamW, etc.)
  - Non-invasive energy and information flow measurement
  - Automatic diminishing returns detection
  - Waste analysis and early stopping recommendations
  - Works without changing existing training code
  - Export to CSV/JSON for further analysis
- **η-Controller** - Automatic hyperparameter adjustment based on real-time efficiency
  - Thermodynamic coherence control (EMSTI-inspired)
  - Operating modes: SAFE, BALANCED, GREEN, EXTREME
  - Auto-adjusts keep_frac and uniform_mix to maintain target efficiency
  - Respects accuracy constraints while maximizing energy savings
  - Automatic phase transitions (warmup → thermodynamic → consolidation)
  - Loss volatility detection and starvation prevention
  - `create_controlled_h3()` convenience function for one-liner setup
- Thermo-Audit logging system (`ThermoAuditLogger`) for standardized training metrics
- Preset configurations for common use cases:
  - `MNISTFast` - Aggressive preset for highly redundant datasets
  - `CIFAR10Safe` - Conservative preset for complex datasets
  - `EdgeDevice` - Ultra-aggressive preset for edge devices
- CLI tool: `h3-report` for analyzing and comparing training runs
- Explain utilities (`explain_thermo_log`, `green_score`, `compare_runs`)
- Adaptive annealing schedules (cosine, linear, exponential) for automatic data fraction scheduling
- Comprehensive CIFAR-10 optimization experiments framework
- Analysis and visualization tools for experiment results

### Changed
- Upgraded package status to Beta (Development Status :: 4 - Beta)
- Enhanced package metadata with green-ai and energy-efficient keywords
- Moved pandas and matplotlib to core dependencies (needed for analysis tools)
- Improved documentation with preset configuration guides

### Fixed
- N/A (development version)

## [0.1.1] - 2025-01-21

### Added
- Initial PyPI release
- Core H3Optimizer implementation with three-phase training strategy
- Lipschitz-adaptive learning rates (Eq. 4.10)
- Information-weighted sampling (Eq. 5.1)
- EnergyTracker with multi-backend support:
  - NVML for NVIDIA GPUs
  - PowerMetrics for Apple Silicon
  - TDP fallback estimation
- Three-phase training strategy (warmup → thermodynamic → consolidation)
- LossTracker for per-sample loss monitoring
- InformationWeightedSampler for intelligent data selection
- Examples for MNIST and CIFAR-10
- Documentation and README

### Validated
- **MNIST (Apple M4):**
  - 12.2% faster training (23.07s → 20.27s)
  - 18.6% better energy efficiency (0.002375 → 0.002816 bits/J)
  - 13.7% energy reduction (1357.68J → 1172.02J)
  - Accuracy maintained: 99.20% → 99.13% (-0.07pp)

- **CIFAR-10 (Baseline with fixed keep_frac=0.65):**
  - 5.7% speedup
  - 14.2% efficiency gain
  - Accuracy trade-off: 82.04% → 80.49% (-1.55pp)
  - Identified as needing adaptive annealing for better generalization

### Technical Details
- Thermodynamic efficiency: η = ΔI/E (bits per joule)
- Information gain calculation based on cross-entropy reduction
- GPU/CPU energy tracking with graceful degradation
- PyTorch-native implementation with minimal dependencies

## [0.1.0] - 2025-01-15

### Added
- Initial research prototype
- Basic optimizer implementation
- Proof-of-concept experiments

---

## Version Naming Convention

- **Major (X.0.0)**: Breaking API changes, major architectural changes
- **Minor (0.X.0)**: New features, backward-compatible improvements
- **Patch (0.0.X)**: Bug fixes, documentation updates

## Development Status

- **Alpha**: Early development, API may change
- **Beta**: Feature-complete, API stabilizing, testing in progress
- **Stable**: Production-ready, API locked
