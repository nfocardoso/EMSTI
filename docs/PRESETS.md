# H3 Preset Configurations

H3 provides pre-tuned configurations for common use cases. These presets eliminate the need for manual hyperparameter tuning while delivering optimal performance for specific dataset types.

## Available Presets

### 1. MNISTFast - For Highly Redundant Data

**Best for:** MNIST, Fashion-MNIST, simple digit/character recognition

**Strategy:**
- Data retention: 65% (skip 35%)
- Exploration: 20% uniform mix
- Warmup: 1 epoch
- Consolidation: 1 epoch

**Expected Results:**
- ✅ 10-15% speedup
- ✅ 15-20% efficiency gain
- ✅ Accuracy maintained (< 0.2pp difference)

**Usage:**
```python
from h3.presets import MNISTFast
from torch.utils.data import DataLoader

preset = MNISTFast(model.parameters(), train_dataset, lr=1e-3)
optimizer, loss_tracker, indexed_dataset = preset.get_components()

for epoch in range(10):
    phase, sampler = preset.get_sampler(epoch, total_epochs=10)
    loader = DataLoader(indexed_dataset, sampler=sampler, batch_size=64)
    # ... your training loop ...
```

---

### 2. CIFAR10Safe - For Complex Datasets

**Best for:** CIFAR-10, CIFAR-100, STL-10

**Strategy:**
- Data retention: 55% (skip 45%)
- Exploration: 30% uniform mix
- Warmup: 3 epochs
- Consolidation: 3 epochs

**Expected Results:**
- ✅ 5-8% speedup
- ✅ 8-12% efficiency gain
- ⚠️ Accuracy trade-off: -0.5 to -1.5pp

**Usage:**
```python
from h3.presets import CIFAR10Safe

preset = CIFAR10Safe(model.parameters(), train_dataset, lr=1e-3)
optimizer, loss_tracker, indexed_dataset = preset.get_components()

for epoch in range(20):
    phase, sampler = preset.get_sampler(epoch, total_epochs=20)
    # ... training ...
```

---

### 3. EdgeDevice - Maximum Energy Savings

**Best for:** IoT devices, mobile ML, embedded systems

**Strategy:**
- Data retention: 40% (skip 60%)
- Exploration: 50% uniform mix
- Aggressive energy optimization

**Expected Results:**
- ✅ 15-25% speedup
- ✅ 20-30% efficiency gain
- ⚠️ Accuracy trade-off: -2 to -4pp

**Use when:**
- Energy is severely constrained
- Slightly lower accuracy is acceptable
- Fast training/deployment is critical

**Usage:**
```python
from h3.presets import EdgeDevice

preset = EdgeDevice(model.parameters(), train_dataset, lr=1e-3)
optimizer, loss_tracker, indexed_dataset = preset.get_components()
# ... training with maximum energy savings ...
```

---

## Quick Start Functions

For even simpler usage, use the convenience functions:

```python
from h3.presets import h3_mnist_fast, h3_cifar_safe, h3_edge

# MNIST
optimizer, loss_tracker, indexed_dataset = h3_mnist_fast(
    model.parameters(),
    train_dataset,
    lr=1e-3
)

# CIFAR-10
optimizer, loss_tracker, indexed_dataset = h3_cifar_safe(
    model.parameters(),
    train_dataset,
    lr=1e-3
)

# Edge devices
optimizer, loss_tracker, indexed_dataset = h3_edge(
    model.parameters(),
    train_dataset,
    lr=1e-3
)
```

---

## Complete Example with Logging

```python
import torch
from torch.utils.data import DataLoader
import torchvision
from h3.presets import MNISTFast
from h3.hooks import ThermoAuditLogger
from h3.energy_tracker import EnergyTracker

# Load data
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True,
    transform=torchvision.transforms.ToTensor()
)

test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True,
    transform=torchvision.transforms.ToTensor()
)

# Setup model
model = SimpleCNN()

# Setup H3 with preset
preset = MNISTFast(model.parameters(), train_dataset)
optimizer, loss_tracker, indexed_dataset = preset.get_components()

# Setup logging and energy tracking
logger = ThermoAuditLogger(
    "mnist_fast_run",
    metadata={"model": "SimpleCNN", "dataset": "MNIST"},
    config={"epochs": 10, "batch_size": 64, "lr": 1e-3}
)
energy_tracker = EnergyTracker()

# Training loop
criterion = torch.nn.CrossEntropyLoss(reduction='none')

for epoch in range(10):
    energy_tracker.start_epoch()

    # Get sampler for current phase
    phase, sampler = preset.get_sampler(epoch, total_epochs=10)
    loader = DataLoader(indexed_dataset, sampler=sampler, batch_size=64)

    # Training
    model.train()
    running_loss = 0.0
    for data, target, indices in loader:
        optimizer.zero_grad()
        output = model(data)
        loss_vec = criterion(output, target)
        loss = loss_vec.mean()

        # Update loss tracker
        loss_tracker.update(indices, loss_vec.detach())

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Evaluation
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in DataLoader(test_dataset, batch_size=64):
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    energy_tracker.end_epoch()

    # Log metrics
    acc = 100. * correct / len(test_dataset)
    logger.log_epoch(
        epoch=epoch,
        phase=phase,
        keep_frac=0.65 if phase == "thermodynamic" else 1.0,
        uniform_mix=0.2 if phase == "thermodynamic" else 1.0,
        train_loss=running_loss / len(loader),
        val_acc=acc,
        energy_stats=energy_tracker.get_current_stats()
    )

    print(f"Epoch {epoch}: {phase:15s} | Acc: {acc:.2f}%")

print(f"\nTraining complete!")
print(f"Log saved to: {logger.get_path()}")

# Analyze results
from h3 import explain_thermo_log
print(explain_thermo_log(logger.get_path()))
```

---

## Choosing the Right Preset

| Dataset Type | Redundancy | Preset | Expected Speedup | Accuracy Trade-off |
|-------------|-----------|--------|-----------------|-------------------|
| MNIST-like | High | `MNISTFast` | 10-15% | < 0.2pp |
| CIFAR-like | Medium | `CIFAR10Safe` | 5-8% | 0.5-1.5pp |
| ImageNet | Low | Not Recommended | - | - |
| Edge/IoT | Any | `EdgeDevice` | 15-25% | 2-4pp |

**Decision Guide:**

1. **Dataset Redundancy Check:**
   - High redundancy: Many similar/duplicated samples (MNIST, simple datasets)
   - Medium redundancy: Diverse but learnable (CIFAR-10, CIFAR-100)
   - Low redundancy: Highly diverse, complex (ImageNet, medical imaging)

2. **Accuracy Requirements:**
   - Mission-critical (< 0.2pp acceptable): Use `MNISTFast` or manual tuning
   - Standard (0.5-1.5pp acceptable): Use `CIFAR10Safe`
   - Flexible (2-4pp acceptable): Use `EdgeDevice`

3. **Energy Constraints:**
   - Minimal constraints: Start with `CIFAR10Safe` (balanced)
   - Moderate constraints: Use `MNISTFast` if dataset allows
   - Severe constraints: Use `EdgeDevice` and accept accuracy trade-off

---

## Custom Presets

You can create custom presets by subclassing `H3Preset`:

```python
from h3.presets import H3Preset
from h3.sampler import InformationWeightedSampler

class MyCustomPreset(H3Preset):
    def get_smoothing(self) -> float:
        return 0.15  # Custom smoothing

    def get_lipschitz_safety(self) -> float:
        return 0.92  # Custom safety margin

    def get_sampler(self, epoch, total_epochs):
        # Custom sampling logic
        if epoch < 5:
            return "warmup", InformationWeightedSampler(
                self.indexed_dataset,
                self.loss_tracker,
                uniform_mix=1.0
            )
        elif epoch >= total_epochs - 3:
            return "consolidation", InformationWeightedSampler(
                self.indexed_dataset,
                self.loss_tracker,
                uniform_mix=1.0
            )
        else:
            return "thermodynamic", InformationWeightedSampler(
                self.indexed_dataset,
                self.loss_tracker,
                keep_frac=0.60,
                uniform_mix=0.25
            )

# Use custom preset
preset = MyCustomPreset(model.parameters(), train_dataset)
optimizer, loss_tracker, indexed_dataset = preset.get_components()
```

---

## FAQ

**Q: Can I use presets with my own model architecture?**
A: Yes! Presets work with any PyTorch model. Just pass `model.parameters()` to the preset constructor.

**Q: What if my dataset doesn't fit any preset?**
A: Start with `CIFAR10Safe` (most conservative). Monitor results using `h3-report` and adjust manually if needed.

**Q: Can I modify preset parameters during training?**
A: Presets are designed to be static for reproducibility. For dynamic adjustments, consider creating a custom preset or using the adaptive annealing schedules (see `h3/adaptive.py`).

**Q: How do I know which preset to use?**
A: Consider dataset redundancy:
- High redundancy (same-ish samples) → `MNISTFast`
- Medium redundancy → `CIFAR10Safe`
- Low redundancy → Manual tuning or avoid H3
- Energy-critical → `EdgeDevice`

**Q: Can I use presets for transfer learning?**
A: Yes, but be cautious. Pre-trained features may be less redundant than training from scratch. Start with `CIFAR10Safe` and monitor accuracy carefully.

**Q: What about other optimizers (SGD, AdamW)?**
A: H3 implements its own optimizer with Lipschitz-adaptive learning rates. If you need SGD/AdamW features, consider using H3's sampling strategy separately (manual implementation).

**Q: How do presets handle class imbalance?**
A: H3's information-weighted sampling naturally handles class imbalance by focusing on high-loss (difficult) samples. No special handling needed.

---

## Performance Tips

1. **Batch Size:** Larger batches (128-256) work better with H3 due to reduced sampling overhead.

2. **Learning Rate:** Start with the same LR as your baseline optimizer. H3 adapts internally.

3. **Monitoring:** Always use `ThermoAuditLogger` and analyze results with `h3-report` to verify performance.

4. **Consolidation Phase:** Critical for accuracy recovery. Never skip the final consolidation epochs.

5. **First Run:** On a new dataset, run both baseline and `CIFAR10Safe` to establish trade-off characteristics.

---

## Related Documentation

- [Main README](../README.md) - Project overview and quick start
- [Examples](../examples/) - Complete working examples
- [API Reference](../h3/) - Detailed API documentation

---

## Citation

If you use H3 presets in your research, please cite:

```bibtex
@software{h3_optimizer_2025,
  title = {H3 Optimizer: Thermodynamically Efficient Deep Learning},
  author = {Cardoso, Nuno},
  year = {2025},
  url = {https://github.com/nfocardoso/EMSTI},
  version = {0.2.0}
}
```
