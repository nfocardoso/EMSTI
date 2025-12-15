# H3 Optimizer - API Reference 📘

**Complete API documentation for all H3 components**

Version: 0.2.1

---

## Table of Contents

- [Core Components](#core-components)
  - [H3Optimizer](#h3optimizer)
  - [LossTracker](#losstracker)
  - [InformationWeightedSampler](#informationweightedsampler)
  - [IndexedDataset](#indexeddataset)
- [Automation](#automation)
  - [AutoH3](#autoh3)
  - [η-Controller](#eta-controller)
  - [Presets](#presets)
- [Monitoring & Analysis](#monitoring--analysis)
  - [EnergyTracker](#energytracker)
  - [H3Profiler](#h3profiler)
  - [ThermoAuditLogger](#thermoauditlogger)
- [CLI Tools](#cli-tools)
- [Utility Functions](#utility-functions)

---

## Core Components

### H3Optimizer

**Thermodynamically-efficient optimizer with Lipschitz-adaptive learning rates.**

The core optimizer that implements adaptive learning rates based on local loss curvature.

#### Class Definition

```python
class H3Optimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lipschitz_safety: float = 0.8,
        lipschitz_update_interval: int = 20,
        lipschitz_ema_beta: float = 0.9,
        decoupled_weight_decay: bool = True
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | Iterable | Required | Model parameters to optimize (`model.parameters()`) |
| `lr` | float | `1e-3` | Base learning rate |
| `weight_decay` | float | `0.0` | Weight decay factor (L2 penalty) |
| `lipschitz_safety` | float | `0.8` | Safety margin in (0,1]. Lower = more aggressive |
| `lipschitz_update_interval` | int | `20` | Steps between Lipschitz constant updates |
| `lipschitz_ema_beta` | float | `0.9` | Exponential moving average smoothing factor |
| `decoupled_weight_decay` | bool | `True` | Use AdamW-style decoupled weight decay |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `lipschitz_estimate` | float | Current estimate of local Lipschitz constant |
| `step_count` | int | Number of optimization steps performed |

#### Methods

##### `step(closure=None)`

Performs a single optimization step with Lipschitz-adaptive learning rate.

**Parameters:**
- `closure` (callable, optional): A closure that reevaluates the model and returns the loss

**Returns:**
- Loss value if closure is provided, else None

**Example:**
```python
from h3 import H3Optimizer

optimizer = H3Optimizer(model.parameters(), lr=1e-3)

for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

#### How It Works

The effective learning rate is computed as:

```
η_eff = base_lr * (safety / L̂)

Where:
  L̂ = estimated local Lipschitz constant
  safety = lipschitz_safety parameter
```

This adapts the learning rate to the local loss curvature:
- **Smooth regions (low L̂):** Larger steps → faster convergence
- **Steep regions (high L̂):** Smaller steps → better stability

#### Key Changes in v0.2.1

- ✅ Decoupled weight decay (AdamW-style) by default
- ✅ More aggressive defaults (`lipschitz_safety=0.8` vs `0.9`)
- ✅ Better Lipschitz constant initialization
- ✅ Robust parameter validation

---

### LossTracker

**Tracks per-sample losses with exponential smoothing.**

Maintains smoothed loss estimates for each training sample to identify high-information examples.

#### Class Definition

```python
class LossTracker:
    def __init__(
        self,
        num_samples: int,
        smoothing: float = 0.1,
        initial_loss: float = 1.0
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_samples` | int | Required | Total number of samples in dataset |
| `smoothing` | float | `0.1` | Smoothing factor β ∈ [0,1]. Higher = more weight to recent losses |
| `initial_loss` | float | `1.0` | Initial loss value for unseen samples |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `losses` | Tensor | Tensor of shape `(num_samples,)` with smoothed losses |
| `initialized` | Tensor | Boolean tensor indicating which samples have been seen |

#### Methods

##### `update(indices, losses)`

Updates loss estimates for given samples using exponential smoothing:

```
L_i^(t) = (1 - β) * L_i^(t-1) + β * L_current
```

**Parameters:**
- `indices` (Tensor): Sample indices, shape `(batch_size,)`
- `losses` (Tensor): Loss values, shape `(batch_size,)`

**Example:**
```python
from h3 import LossTracker
import torch.nn as nn

loss_tracker = LossTracker(num_samples=60000, smoothing=0.1)
criterion = nn.CrossEntropyLoss(reduction='none')  # Important: reduction='none'

# In training loop:
output = model(data)
loss_vector = criterion(output, target)  # Per-sample losses
loss_tracker.update(indices, loss_vector.detach())
loss = loss_vector.mean()  # For backward pass
```

##### `get_weights()`

Returns normalized sampling weights based on current losses.

**Returns:**
- `Tensor`: Weights of shape `(num_samples,)`, summing to 1.0

##### `get_statistics()`

Returns statistics about current loss distribution.

**Returns:**
- Dictionary with keys: `mean`, `std`, `min`, `max`, `median`

---

### InformationWeightedSampler

**PyTorch sampler implementing information-weighted sample selection.**

Prioritizes high-loss (high-information) samples during training.

#### Class Definition

```python
class InformationWeightedSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset: IndexedDataset,
        loss_tracker: LossTracker,
        keep_frac: float = 1.0,
        uniform_mix: float = 0.0,
        replacement: bool = False
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | IndexedDataset | Required | Indexed dataset to sample from |
| `loss_tracker` | LossTracker | Required | Loss tracker with per-sample losses |
| `keep_frac` | float | `1.0` | Fraction of samples to keep (top by loss). `1.0` = all samples |
| `uniform_mix` | float | `0.0` | Fraction of uniform random sampling. `0.0` = pure weighted, `1.0` = pure uniform |
| `replacement` | bool | `False` | Whether to sample with replacement |

#### Usage Examples

**Warmup Phase (uniform sampling):**
```python
sampler = InformationWeightedSampler(
    dataset, loss_tracker,
    uniform_mix=1.0  # 100% uniform
)
```

**Thermodynamic Phase (weighted sampling):**
```python
sampler = InformationWeightedSampler(
    dataset, loss_tracker,
    keep_frac=0.65,   # Keep top 65% by loss
    uniform_mix=0.2   # 80% weighted, 20% uniform
)
```

**Consolidation Phase (uniform sampling):**
```python
sampler = InformationWeightedSampler(
    dataset, loss_tracker,
    uniform_mix=1.0  # 100% uniform
)
```

#### Three-Phase Training

| Phase | Epochs | `keep_frac` | `uniform_mix` | Purpose |
|-------|--------|-------------|---------------|---------|
| **Warmup** | 1-2 | `1.0` | `1.0` | Stable initialization |
| **Thermodynamic** | 70-80% | `0.55-0.75` | `0.2-0.4` | Focused learning |
| **Consolidation** | 1-2 | `1.0` | `1.0` | Ensure coverage |

---

### IndexedDataset

**Wrapper that adds sample indices to dataset outputs.**

Required for loss tracking - adds index to each sample.

#### Class Definition

```python
class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset` | Dataset | PyTorch dataset to wrap |

#### Returns

Each `__getitem__` call returns `(data, target, index)` instead of just `(data, target)`.

#### Example

```python
from h3 import IndexedDataset
from torchvision import datasets

# Wrap any PyTorch dataset
original_dataset = datasets.MNIST(root='./data', train=True, download=True)
indexed_dataset = IndexedDataset(original_dataset)

# Now returns (data, target, index)
for data, target, index in DataLoader(indexed_dataset, batch_size=64):
    # index contains sample indices for loss tracking
    loss_tracker.update(index, losses)
```

---

## Automation

### AutoH3

**Complete zero-config automation combining all H3 features.**

The simplest H3 API - automatically handles optimizer, sampling, tracking, and logging.

#### Class Definition

```python
class AutoH3:
    def __init__(
        self,
        params: Iterable,
        dataset: torch.utils.data.Dataset,
        mode: str = "balanced",
        accuracy_tolerance: float = 1.0,
        name: str = "auto_h3_run",
        lr: float = 1e-3,
        device: str = "auto"
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | Iterable | Required | Model parameters (`model.parameters()`) |
| `dataset` | Dataset | Required | Training dataset |
| `mode` | str | `"balanced"` | Operating mode: `"safe"`, `"balanced"`, `"green"`, `"extreme"` |
| `accuracy_tolerance` | float | `1.0` | Max acceptable accuracy loss (percentage points) |
| `name` | str | `"auto_h3_run"` | Name for logging and outputs |
| `lr` | float | `1e-3` | Learning rate |
| `device` | str | `"auto"` | Device for energy tracking: `"auto"`, `"cuda"`, `"cpu"` |

#### Operating Modes

| Mode | `keep_frac` | Energy Priority | Accuracy Priority | Use Case |
|------|-------------|-----------------|-------------------|----------|
| `"safe"` | ~0.70 | Low | High | Maximum accuracy |
| `"balanced"` | ~0.55 | Medium | Medium | **Recommended** |
| `"green"` | ~0.45 | High | Low | Maximum energy savings |
| `"extreme"` | ~0.35 | Very High | Very Low | Edge/IoT devices |

#### Methods

##### `start()`

Starts energy tracking and logging.

##### `get_loader(batch_size=64, num_workers=0)`

Returns DataLoader for current epoch with appropriate sampler.

**Parameters:**
- `batch_size` (int): Batch size
- `num_workers` (int): Number of data loading workers

**Returns:**
- `DataLoader`: Configured data loader

##### `training_step(model, data, target, criterion, indices)`

Performs single training step with automatic loss tracking.

**Parameters:**
- `model`: PyTorch model
- `data`: Input data tensor
- `target`: Target labels tensor
- `criterion`: Loss function (must support `reduction='none'`)
- `indices`: Sample indices tensor

**Returns:**
- `Tensor`: Loss value

##### `evaluate_epoch(model, test_loader)`

Evaluates model and logs metrics.

**Parameters:**
- `model`: PyTorch model
- `test_loader`: Test data loader

**Returns:**
- `float`: Test accuracy

##### `finish()`

Stops tracking and returns comprehensive results.

**Returns:**
- Dictionary with keys:
  - `final_accuracy`: Final test accuracy
  - `total_energy_j`: Total energy consumed (joules)
  - `efficiency_bits_per_j`: Thermodynamic efficiency (bits/J)
  - `total_time_s`: Total training time (seconds)
  - `speedup`: Speedup vs baseline (if available)
  - `energy_savings`: Energy savings vs baseline (if available)

#### Complete Example

```python
from h3 import AutoH3
import torch.nn as nn

# Setup
auto = AutoH3(
    model.parameters(),
    train_dataset,
    mode="balanced",
    name="mnist_experiment"
)

auto.start()
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(10):
    loader = auto.get_loader(batch_size=64)

    model.train()
    for data, target, indices in loader:
        data, target = data.to(device), target.to(device)
        loss = auto.training_step(model, data, target, criterion, indices)

    # Evaluation
    model.eval()
    accuracy = auto.evaluate_epoch(model, test_loader)

    print(f"Epoch {epoch+1}: Accuracy {accuracy:.2f}%")

# Results
results = auto.finish()
print(f"Efficiency: {results['efficiency_bits_per_j']:.6f} bits/J")
```

---

### η-Controller

**Automatic hyperparameter tuning based on real-time efficiency measurements.**

Dynamically adjusts `keep_frac` and `uniform_mix` to maximize efficiency while respecting constraints.

#### Class Definition

```python
class EtaController:
    def __init__(
        self,
        total_epochs: int,
        mode: OperatingMode = OperatingMode.BALANCED,
        coherence_target: CoherenceTarget = CoherenceTarget.MODERATE,
        accuracy_tolerance: float = 1.0
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `total_epochs` | int | Required | Total number of training epochs |
| `mode` | OperatingMode | `BALANCED` | Operating mode enum |
| `coherence_target` | CoherenceTarget | `MODERATE` | Sampling aggressiveness |
| `accuracy_tolerance` | float | `1.0` | Max acceptable accuracy loss (pp) |

#### Enums

**OperatingMode:**
- `OperatingMode.SAFE` - Maximize accuracy
- `OperatingMode.BALANCED` - Balance efficiency and accuracy
- `OperatingMode.GREEN` - Maximize energy savings
- `OperatingMode.EXTREME` - Maximum savings (edge devices)

**CoherenceTarget:**
- `CoherenceTarget.AGGRESSIVE` - More data skipping
- `CoherenceTarget.MODERATE` - Balanced approach
- `CoherenceTarget.CONSERVATIVE` - Less data skipping

#### Methods

##### `control_step()`

Computes hyperparameters for current epoch.

**Returns:**
- Tuple `(keep_frac, uniform_mix, phase)` where:
  - `keep_frac` (float): Fraction of samples to keep
  - `uniform_mix` (float): Fraction of uniform sampling
  - `phase` (str): Current phase name

##### `observe(eta, accuracy, loss, energy, info)`

Updates controller state based on observed metrics.

**Parameters:**
- `eta` (float): Current efficiency (bits/J)
- `accuracy` (float): Current accuracy
- `loss` (float): Current loss
- `energy` (float): Energy consumed
- `info` (dict): Additional info

##### `get_report()`

Returns detailed status report.

**Returns:**
- String with current state and recommendations

#### Convenience Function

```python
def create_controlled_h3(
    params: Iterable,
    dataset: Dataset,
    mode: str = "balanced",
    accuracy_tolerance: float = 1.0,
    total_epochs: int = 20,
    lr: float = 1e-3
) -> Tuple[H3Optimizer, EtaController, LossTracker, IndexedDataset]
```

Returns configured optimizer, controller, loss tracker, and indexed dataset.

#### Example

```python
from h3 import create_controlled_h3, InformationWeightedSampler
from torch.utils.data import DataLoader

# One-line setup
optimizer, controller, loss_tracker, indexed_dataset = create_controlled_h3(
    model.parameters(),
    train_dataset,
    mode="balanced",
    accuracy_tolerance=1.0,
    total_epochs=20
)

# Training loop
for epoch in range(20):
    # Controller provides parameters
    keep_frac, uniform_mix, phase = controller.control_step()

    # Create sampler with tuned parameters
    sampler = InformationWeightedSampler(
        indexed_dataset, loss_tracker,
        keep_frac=keep_frac,
        uniform_mix=uniform_mix
    )
    loader = DataLoader(indexed_dataset, sampler=sampler, batch_size=128)

    # ... training ...

    # Controller learns from results
    controller.observe(eta, accuracy, loss, energy, {})

    if epoch % 5 == 0:
        print(controller.get_report())
```

---

### Presets

**Pre-tuned configurations for common use cases.**

Eliminate manual hyperparameter tuning with dataset-specific presets.

#### Available Presets

##### `h3_mnist_fast()`

For simple, highly redundant datasets (MNIST, Fashion-MNIST).

```python
def h3_mnist_fast(
    params: Iterable,
    dataset: Dataset,
    lr: float = 1e-3
) -> Tuple[H3Optimizer, LossTracker, IndexedDataset]
```

**Configuration:**
- Data retention: 65% (skip 35%)
- Exploration: 20% uniform mix
- Warmup: 1 epoch
- Consolidation: 1 epoch

**Expected Results:**
- ✅ 10-15% speedup
- ✅ 15-20% efficiency gain
- ✅ Accuracy maintained (< 0.2pp)

##### `h3_cifar_safe()` / `h3_cifar10_safe_adamw()`

For complex image datasets (CIFAR-10, CIFAR-100, STL-10).

```python
def h3_cifar10_safe_adamw(
    params: Iterable,
    dataset: Dataset,
    lr: float = 1e-3
) -> Tuple[H3Optimizer, LossTracker, IndexedDataset]
```

**Configuration:**
- Data retention: 75% (skip 25%)
- Exploration: 40% uniform mix
- Warmup: 3 epochs
- Consolidation: 3 epochs

**Expected Results:**
- ✅ 5-8% speedup
- ✅ 8-12% efficiency gain
- ⚠️ Accuracy: -0.5 to -1.5pp

##### `h3_cifar10_balanced()`

Balanced configuration for CIFAR-10.

```python
def h3_cifar10_balanced(
    params: Iterable,
    dataset: Dataset,
    lr: float = 1e-3
) -> Tuple[H3Optimizer, LossTracker, IndexedDataset]
```

**Configuration:**
- Data retention: 55% (skip 45%)
- More aggressive than safe mode
- Better energy savings

##### `h3_cifar10_green()`

Green/energy-focused configuration for CIFAR-10.

```python
def h3_cifar10_green(
    params: Iterable,
    dataset: Dataset,
    lr: float = 1e-3
) -> Tuple[H3Optimizer, LossTracker, IndexedDataset]
```

**Configuration:**
- Maximum energy savings
- Accepts larger accuracy trade-off

#### Preset Classes

All presets are instances of `H3Preset` base class:

```python
from h3.presets import MNISTFast, CIFAR10Safe, EdgeDevice

# Using preset classes
preset = MNISTFast(model.parameters(), train_dataset, lr=1e-3)
optimizer, loss_tracker, indexed_dataset = preset.get_components()

for epoch in range(10):
    phase, sampler = preset.get_sampler(epoch, total_epochs=10)
    loader = DataLoader(indexed_dataset, sampler=sampler, batch_size=64)
    # ... training ...
```

#### Example

```python
from h3.presets import h3_mnist_fast
from torch.utils.data import DataLoader

# One-liner setup
optimizer, loss_tracker, indexed_dataset = h3_mnist_fast(
    model.parameters(),
    train_dataset
)

# Training loop
for epoch in range(10):
    # Manually create sampler based on phase
    if epoch < 1:  # Warmup
        sampler = InformationWeightedSampler(
            indexed_dataset, loss_tracker, uniform_mix=1.0
        )
    elif epoch >= 9:  # Consolidation
        sampler = InformationWeightedSampler(
            indexed_dataset, loss_tracker, uniform_mix=1.0
        )
    else:  # Thermodynamic
        sampler = InformationWeightedSampler(
            indexed_dataset, loss_tracker,
            keep_frac=0.65, uniform_mix=0.2
        )

    loader = DataLoader(indexed_dataset, sampler=sampler, batch_size=64)
    # ... training ...
```

See [PRESETS.md](PRESETS.md) for complete documentation.

---

## Monitoring & Analysis

### EnergyTracker

**Multi-backend energy monitoring for real power measurements.**

Tracks energy consumption during training with automatic backend selection.

#### Class Definition

```python
class EnergyTracker:
    def __init__(
        self,
        device: str = "auto",
        fallback_tdp: Optional[float] = None
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | str | `"auto"` | Device to track: `"auto"`, `"cuda"`, `"cpu"`, `"mps"` |
| `fallback_tdp` | float, optional | `None` | TDP watts for fallback estimation |

#### Supported Backends

| Backend | Hardware | Requires | Accuracy |
|---------|----------|----------|----------|
| **NVML** | NVIDIA GPU | `pynvml` | ✅ Real power |
| **PowerMetrics** | Apple Silicon | `sudo` access | ✅ Real power |
| **TDP Fallback** | Any | Nothing | ⚠️ Estimation |

#### Methods

##### `start()`

Starts energy tracking.

##### `stop()`

Stops tracking and returns results.

**Returns:**
- Dictionary with keys:
  - `total_energy_j`: Total energy in joules
  - `total_time_s`: Total time in seconds
  - `average_power_w`: Average power in watts
  - `efficiency_bits_per_j`: Efficiency (if loss info available)
  - `info_gain_bits`: Information gain (if loss info available)

##### `log_loss(loss)`

Logs loss value for efficiency calculation.

**Parameters:**
- `loss` (float): Current loss value

##### `get_current_stats()`

Returns current statistics without stopping.

**Returns:**
- Dictionary with current metrics

#### Example

```python
from h3 import EnergyTracker

tracker = EnergyTracker(device='cuda')
tracker.start()

initial_loss = None

for epoch in range(num_epochs):
    for batch in train_loader:
        # ... training ...
        tracker.log_loss(loss.item())

        if initial_loss is None:
            initial_loss = loss.item()

# Get results
results = tracker.stop()
print(f"Energy: {results['total_energy_j']:.2f} J")
print(f"Efficiency: {results['efficiency_bits_per_j']:.6f} bits/J")
```

---

### H3Profiler

**Zero-risk profiling for ANY optimizer (Adam, SGD, AdamW, etc.).**

Measures thermodynamic efficiency without modifying your training code.

#### Class Definition

```python
class H3Profiler:
    def __init__(
        self,
        device: str = "auto",
        name: str = "profiler_run",
        save_interval: int = 1
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | str | `"auto"` | Device for energy tracking |
| `name` | str | `"profiler_run"` | Name for saved profiles |
| `save_interval` | int | `1` | Epochs between saves |

#### Methods

##### `start()`

Starts profiling.

##### `stop()`

Stops profiling and returns results.

**Returns:**
- Dictionary with comprehensive metrics

##### `log_batch(loss)`

Logs batch loss.

**Parameters:**
- `loss` (float): Batch loss value

##### `log_epoch(accuracy=None, **kwargs)`

Logs epoch completion.

**Parameters:**
- `accuracy` (float, optional): Epoch accuracy
- `**kwargs`: Additional metrics to log

##### `get_report()`

Returns detailed analysis report.

**Returns:**
- String with:
  - Efficiency metrics
  - Optimal stopping point detection
  - Energy waste analysis
  - H3 suitability assessment

##### `export_csv(path)`

Exports profiling data to CSV.

**Parameters:**
- `path` (str): Output file path

#### Example

```python
from h3 import H3Profiler
import torch.optim as optim

# Setup profiler (doesn't change your training!)
profiler = H3Profiler(device='cuda', name="baseline_adam")
profiler.start()

# Your normal training loop with Adam
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Just log the loss
        profiler.log_batch(loss.item())

    # Log epoch accuracy
    accuracy = evaluate(model, test_loader)
    profiler.log_epoch(accuracy=accuracy)

# Get comprehensive analysis
results = profiler.stop()
print(profiler.get_report())
profiler.export_csv("./profiles/adam_run.csv")
```

**Sample Report:**
```
=== H3 Profiler Report ===
Optimizer: Adam (baseline)
Total Energy: 1357.68 J
Final Accuracy: 99.20%
Efficiency: 0.002375 bits/J

Optimal Stopping: Epoch 3 (diminishing returns detected)
Energy Waste: 20% beyond optimal point

Recommendation: H3 could improve efficiency by 15-20%
Try: mode="balanced" or h3_mnist_fast preset
```

---

### ThermoAuditLogger

**Experiment tracking and logging for H3 training runs.**

Logs all training metrics to CSV for later analysis.

#### Class Definition

```python
class ThermoAuditLogger:
    def __init__(
        self,
        name: str,
        metadata: Optional[dict] = None,
        config: Optional[dict] = None,
        output_dir: str = "./thermo_logs"
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Experiment name |
| `metadata` | dict, optional | `None` | Metadata (model, dataset, etc.) |
| `config` | dict, optional | `None` | Configuration (epochs, lr, etc.) |
| `output_dir` | str | `"./thermo_logs"` | Output directory |

#### Methods

##### `log_epoch(**kwargs)`

Logs epoch metrics.

**Common kwargs:**
- `epoch` (int): Epoch number
- `phase` (str): Training phase
- `keep_frac` (float): Sample retention fraction
- `uniform_mix` (float): Uniform sampling fraction
- `train_loss` (float): Training loss
- `val_acc` (float): Validation accuracy
- `energy_stats` (dict): Energy statistics

##### `get_path()`

Returns path to log file.

**Returns:**
- `str`: CSV file path

#### Example

```python
from h3.hooks import ThermoAuditLogger

logger = ThermoAuditLogger(
    "my_experiment",
    metadata={"model": "ResNet-18", "dataset": "CIFAR-10"},
    config={"epochs": 20, "batch_size": 128, "lr": 1e-3}
)

for epoch in range(20):
    # ... training ...

    logger.log_epoch(
        epoch=epoch,
        phase=phase,
        keep_frac=0.65,
        uniform_mix=0.2,
        train_loss=train_loss,
        val_acc=val_accuracy,
        energy_stats=tracker.get_current_stats()
    )

print(f"Log saved to: {logger.get_path()}")
```

---

## CLI Tools

### h3-report

**Command-line tool for analyzing H3 logs.**

Analyzes thermodynamic logs and generates detailed reports.

#### Installation

Installed automatically with h3-optimizer package.

#### Usage

```bash
# Analyze single run
h3-report --h3-log mnist_h3.csv

# Compare H3 vs baseline
h3-report --h3-log mnist_h3.csv --baseline-log mnist_adam.csv

# Compare multiple configurations
h3-report --compare run1.csv run2.csv run3.csv
```

#### Options

| Option | Description |
|--------|-------------|
| `--h3-log PATH` | Path to H3 training log |
| `--baseline-log PATH` | Path to baseline log for comparison |
| `--compare PATH [PATH ...]` | Compare multiple runs |
| `--output PATH` | Save report to file |
| `--format {text,json,html}` | Output format |

#### Output

The tool provides:
- Final metrics summary
- Phase breakdown analysis
- Heuristic assessment (starvation/aggressive/moderate/conservative)
- Green Score (energy efficiency ratio)
- Automatic verdict with tuning recommendations

#### Example Output

```
=== H3 Training Report ===

Final Metrics:
  Accuracy: 99.13%
  Energy: 1172.02 J
  Efficiency: 0.002816 bits/J
  Time: 20.27s

Phase Breakdown:
  Warmup (1 epoch): η=0.010686 bits/J
  Thermodynamic (3 epochs): η=0.004512 bits/J
  Consolidation (1 epoch): η=0.002816 bits/J

Assessment: MODERATE
  - Good balance between efficiency and accuracy
  - No signs of data starvation
  - Efficiency curve smooth

Green Score: 1.19
  (19% more efficient than baseline)

Verdict: ✅ SUCCESS
  H3 delivered 12.2% speedup and 18.6% efficiency gain
  with minimal accuracy impact (-0.07pp)
```

---

## Utility Functions

### explain_thermo_log()

Analyzes thermodynamic log and returns human-readable explanation.

```python
def explain_thermo_log(
    log_path: str,
    verbose: bool = True
) -> str
```

**Parameters:**
- `log_path` (str): Path to CSV log file
- `verbose` (bool): Include detailed analysis

**Returns:**
- String with analysis

**Example:**
```python
from h3 import explain_thermo_log

analysis = explain_thermo_log("./thermo_logs/experiment.csv")
print(analysis)
```

---

### green_score()

Calculates green score (efficiency ratio) between two runs.

```python
def green_score(
    h3_log: str,
    baseline_log: str
) -> float
```

**Parameters:**
- `h3_log` (str): Path to H3 log
- `baseline_log` (str): Path to baseline log

**Returns:**
- `float`: Efficiency ratio (>1.0 means H3 is more efficient)

**Example:**
```python
from h3 import green_score

score = green_score("h3_run.csv", "adam_run.csv")
print(f"H3 is {(score-1)*100:.1f}% more efficient")
```

---

### compare_runs()

Compares multiple training runs.

```python
def compare_runs(
    log_paths: List[str],
    metrics: List[str] = ["accuracy", "energy", "efficiency"]
) -> dict
```

**Parameters:**
- `log_paths` (List[str]): Paths to log files
- `metrics` (List[str]): Metrics to compare

**Returns:**
- Dictionary with comparison results

**Example:**
```python
from h3 import compare_runs

results = compare_runs([
    "safe_mode.csv",
    "balanced_mode.csv",
    "green_mode.csv"
])

for run, metrics in results.items():
    print(f"{run}: {metrics['efficiency']:.6f} bits/J")
```

---

### create_baseline_logger()

Creates logger for baseline (non-H3) training runs.

```python
def create_baseline_logger(
    name: str,
    metadata: Optional[dict] = None,
    config: Optional[dict] = None
) -> ThermoAuditLogger
```

**Parameters:**
- `name` (str): Experiment name
- `metadata` (dict, optional): Metadata
- `config` (dict, optional): Configuration

**Returns:**
- `ThermoAuditLogger`: Configured logger

---

## Type Hints and Imports

```python
from typing import Iterable, Optional, Tuple, List, Dict, Callable, Iterator

import torch
from torch.optim import Optimizer
from torch.utils.data import Dataset, Sampler, DataLoader

# Core components
from h3 import (
    H3Optimizer,
    LossTracker,
    InformationWeightedSampler,
    IndexedDataset,
)

# Automation
from h3 import (
    AutoH3,
    auto_h3,
    EtaController,
    OperatingMode,
    CoherenceTarget,
    create_controlled_h3,
)

# Presets
from h3.presets import (
    h3_mnist_fast,
    h3_cifar10_safe_adamw,
    h3_cifar10_balanced,
    h3_cifar10_green,
    load_preset,
)

# Monitoring
from h3 import (
    EnergyTracker,
    H3Profiler,
    ThermoAuditLogger,
    create_baseline_logger,
)

# Analysis
from h3 import (
    explain_thermo_log,
    green_score,
    compare_runs,
)

# CLI
from h3.cli import h3_report_cli
```

---

## Version History

### v0.2.1 (Current)

**Core Improvements:**
- Decoupled weight decay (AdamW-style) by default
- More aggressive Lipschitz defaults
- Better initialization
- Robust parameter validation

**New Features:**
- Multiple preset configurations
- η-Controller for auto-tuning
- H3Profiler for zero-risk profiling
- AutoH3 for complete automation

See [CHANGELOG.md](../CHANGELOG.md) for complete history.

---

## See Also

- [Getting Started Guide](GETTING_STARTED.md) - Learn H3 basics
- [Presets Guide](PRESETS.md) - Pre-tuned configurations
- [Theory](THEORY.md) - Thermodynamic principles
- [Examples](../examples/README.md) - Complete code examples
- [Main Documentation](../DOCUMENTATION.md) - Documentation index

---

**Questions?** Open an issue: https://github.com/nfocardoso/EMSTI/issues

⚡ Complete API reference for H3 Optimizer 📊
