"""
Example 05: H3 Presets - Pre-Tuned Configurations
==================================================

H3 provides pre-tuned presets for common use cases, so you don't need
to tune hyperparameters yourself. Just pick a preset and go!

Available Presets:
    - h3_mnist_fast: For highly redundant datasets (MNIST, Fashion-MNIST)
    - h3_cifar_safe: For complex datasets (CIFAR-10, CIFAR-100)
    - h3_edge: For edge devices (IoT, mobile ML)

Each preset provides:
    ✓ Pre-configured optimizer
    ✓ Loss tracker
    ✓ Indexed dataset
    ✓ Automatic phase management
    ✓ get_sampler() method for easy training

This is the easiest way to use H3 if you want manual control
but don't want to tune hyperparameters.

Requirements:
    pip install h3-optimizer torch torchvision

Usage:
    python examples/05_presets_showcase.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time

from h3.presets import h3_mnist_fast, h3_cifar_safe, h3_edge
from h3.energy_tracker import EnergyTracker


# ============================================================================
# Model Definitions
# ============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN for MNIST."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return F.log_softmax(self.fc2(x), dim=1)


# ============================================================================
# Training with Presets
# ============================================================================

def train_with_preset(preset_func, preset_name, model, train_dataset,
                     test_loader, device, num_epochs=10, batch_size=64):
    """
    Train using an H3 preset.

    This demonstrates the simple preset API.
    """
    print("\n" + "="*70)
    print(f"Training with: {preset_name}")
    print("="*70)

    # ONE LINE SETUP with preset!
    preset = preset_func(model.parameters(), train_dataset)

    # The preset provides:
    # - preset.optimizer: H3Optimizer configured for this dataset type
    # - preset.loss_tracker: LossTracker configured for this dataset type
    # - preset.indexed_dataset: Your dataset wrapped with indices
    # - preset.get_sampler(epoch, total_epochs): Returns sampler for current epoch

    # Energy tracking
    energy_tracker = EnergyTracker(device=str(device))
    energy_tracker.start()
    start_time = time.time()

    # Training loop
    for epoch in range(num_epochs):
        model.train()

        # Get sampler for current epoch (preset handles phase transitions!)
        phase, sampler = preset.get_sampler(epoch, total_epochs=num_epochs)

        # Create loader with preset-managed sampler
        train_loader = DataLoader(
            preset.indexed_dataset,
            sampler=sampler,
            batch_size=batch_size
        )

        total_loss = 0
        for data, target, indices in train_loader:
            data, target = data.to(device), target.to(device)

            preset.optimizer.zero_grad()
            output = model(data)

            loss_vector = F.nll_loss(output, target, reduction='none')
            loss = loss_vector.mean()

            # Update loss tracker (preset provides this)
            preset.loss_tracker.update(indices, loss_vector.detach())

            loss.backward()
            preset.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        energy_tracker.log_loss(avg_loss)

        # Evaluate
        test_accuracy = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1:2d}/{num_epochs} [{phase:>14s}]: "
              f"Loss={avg_loss:.4f}, Acc={test_accuracy:.2f}%")

    # Results
    elapsed_time = time.time() - start_time
    results = energy_tracker.stop()

    print(f"\nResults:")
    print(f"  Time: {elapsed_time:.2f}s")
    print(f"  Accuracy: {test_accuracy:.2f}%")
    print(f"  Energy: {results['total_energy_j']:.2f} J")
    print(f"  Efficiency: {results['efficiency_bits_per_j']:.6f} bits/J")

    return {
        'time': elapsed_time,
        'accuracy': test_accuracy,
        'energy': results['total_energy_j'],
        'efficiency': results['efficiency_bits_per_j']
    }


def evaluate(model, loader, device):
    """Standard evaluation."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                data, target, _ = batch
            else:
                data, target = batch

            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return 100.0 * correct / total


# ============================================================================
# Main Demo
# ============================================================================

def main():
    """Demonstrate different H3 presets."""

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )

    print("="*70)
    print("H3 Presets Showcase")
    print("="*70)
    print(f"Device: {device}")
    print("\nPresets provide pre-tuned configurations for common use cases.")
    print("No hyperparameter tuning needed!")
    print("="*70)

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("\nLoading MNIST dataset...")
    train_dataset = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # ========================================================================
    # Demo 1: h3_mnist_fast preset
    # ========================================================================

    print("\n" + "="*70)
    print("DEMO 1: h3_mnist_fast - For Highly Redundant Datasets")
    print("="*70)
    print("Best for: MNIST, Fashion-MNIST, simple datasets")
    print("Expected: 10-15% speedup, 15-20% efficiency gain")
    print("Accuracy: Maintained (< 0.2pp difference)")

    model1 = SimpleCNN().to(device)
    result1 = train_with_preset(
        h3_mnist_fast,
        "h3_mnist_fast",
        model1,
        train_dataset,
        test_loader,
        device,
        num_epochs=8
    )

    # ========================================================================
    # Demo 2: h3_cifar_safe preset
    # ========================================================================

    print("\n" + "="*70)
    print("DEMO 2: h3_cifar_safe - For Complex Datasets")
    print("="*70)
    print("Best for: CIFAR-10, CIFAR-100, complex image datasets")
    print("Expected: 5-8% speedup, 8-12% efficiency gain")
    print("Accuracy: Small trade-off (-0.5 to -1.5pp)")

    model2 = SimpleCNN().to(device)
    result2 = train_with_preset(
        h3_cifar_safe,
        "h3_cifar_safe",
        model2,
        train_dataset,
        test_loader,
        device,
        num_epochs=8
    )

    # ========================================================================
    # Demo 3: h3_edge preset
    # ========================================================================

    print("\n" + "="*70)
    print("DEMO 3: h3_edge - For Edge Devices")
    print("="*70)
    print("Best for: IoT, mobile ML, battery-constrained devices")
    print("Expected: 15-25% speedup, 20-30% efficiency gain")
    print("Accuracy: Larger trade-off (-2 to -4pp) for max savings")

    model3 = SimpleCNN().to(device)
    result3 = train_with_preset(
        h3_edge,
        "h3_edge",
        model3,
        train_dataset,
        test_loader,
        device,
        num_epochs=8
    )

    # ========================================================================
    # Final Comparison
    # ========================================================================

    print("\n" + "="*70)
    print("PRESET COMPARISON")
    print("="*70)

    results = {
        "mnist_fast": result1,
        "cifar_safe": result2,
        "edge": result3
    }

    print(f"\n{'Preset':<15} {'Time (s)':<12} {'Accuracy':<12} {'Energy (J)':<12} {'η (bits/J)':<15}")
    print("-"*70)

    for name, res in results.items():
        print(f"{name:<15} "
              f"{res['time']:<12.2f} "
              f"{res['accuracy']:<12.2f} "
              f"{res['energy']:<12.2f} "
              f"{res['efficiency']:<15.6f}")

    print("\n" + "="*70)
    print("CHOOSING THE RIGHT PRESET:")
    print("="*70)
    print("""
1. h3_mnist_fast:
   • Use for: Simple, highly redundant datasets
   • Examples: MNIST, Fashion-MNIST, simple grayscale images
   • Strategy: Aggressive sampling (keep_frac=0.55)
   • Goal: Maximum speedup with maintained accuracy

2. h3_cifar_safe:
   • Use for: Complex datasets with diverse samples
   • Examples: CIFAR-10, CIFAR-100, complex images
   • Strategy: Conservative sampling (keep_frac=0.70)
   • Goal: Balanced speedup with minimal accuracy loss

3. h3_edge:
   • Use for: Energy-constrained environments
   • Examples: IoT devices, mobile phones, embedded systems
   • Strategy: Very aggressive sampling (keep_frac=0.45)
   • Goal: Maximum energy savings (accepts accuracy trade-off)

TIP: Start with the appropriate preset, run it, then tune if needed!
    """)

    print("="*70)


def show_preset_details():
    """Show what each preset configures."""

    print("\n" + "="*70)
    print("Preset Configuration Details")
    print("="*70)

    presets = {
        "h3_mnist_fast": {
            "target_dataset": "Highly redundant (MNIST, Fashion-MNIST)",
            "keep_frac": "0.55 (aggressive)",
            "uniform_mix": "0.30 (focused sampling)",
            "warmup_epochs": "1",
            "consolidation_epochs": "1",
            "lipschitz_safety": "0.9",
            "expected_speedup": "10-15%",
            "expected_accuracy": "±0.2pp"
        },
        "h3_cifar_safe": {
            "target_dataset": "Complex images (CIFAR-10, CIFAR-100)",
            "keep_frac": "0.70 (conservative)",
            "uniform_mix": "0.50 (balanced sampling)",
            "warmup_epochs": "2",
            "consolidation_epochs": "2",
            "lipschitz_safety": "0.95",
            "expected_speedup": "5-8%",
            "expected_accuracy": "-0.5 to -1.5pp"
        },
        "h3_edge": {
            "target_dataset": "Edge/IoT devices (any dataset)",
            "keep_frac": "0.45 (very aggressive)",
            "uniform_mix": "0.25 (highly focused)",
            "warmup_epochs": "1",
            "consolidation_epochs": "1",
            "lipschitz_safety": "0.85",
            "expected_speedup": "15-25%",
            "expected_accuracy": "-2 to -4pp"
        }
    }

    for preset_name, config in presets.items():
        print(f"\n{preset_name}:")
        print("-" * 70)
        for key, value in config.items():
            print(f"  {key.replace('_', ' ').title():<25}: {value}")

    print("\n" + "="*70)
    print("All presets include:")
    print("  • Automatic phase transitions (warmup → thermodynamic → consolidation)")
    print("  • Lipschitz-adaptive learning rate")
    print("  • Information-weighted sampling")
    print("  • Loss tracking with exponential smoothing")
    print("="*70)


if __name__ == '__main__':
    # Uncomment to see preset details
    # show_preset_details()

    # Run demos
    main()
