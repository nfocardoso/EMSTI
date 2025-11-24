"""
Example 04: H3Profiler - Profile ANY Optimizer (Zero Risk)
===========================================================

H3Profiler is a zero-risk tool that lets you measure the thermodynamic
efficiency of ANY optimizer (Adam, SGD, AdamW, etc.) without changing
your training code.

What it measures:
- Thermodynamic efficiency (η = bits/joule) over time
- Energy consumption per epoch
- Optimal stopping point (diminishing returns)
- Potential energy waste
- Whether H3 could help

This is perfect for:
1. Understanding your current training efficiency
2. Identifying optimization opportunities
3. Establishing baselines before trying H3

NO RISK: This only measures, it doesn't change your training!

Requirements:
    pip install h3-optimizer torch torchvision

Usage:
    python examples/04_profiler_any_optimizer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time

from h3.profiler import H3Profiler


# ============================================================================
# Model Definition
# ============================================================================

class SimpleNet(nn.Module):
    """Simple network for demonstration."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return F.log_softmax(self.fc3(x), dim=1)


# ============================================================================
# Training Functions (Your Normal Code - No Changes!)
# ============================================================================

def train_epoch(model, loader, optimizer, device, profiler=None):
    """
    Your normal training loop - NO CHANGES NEEDED!

    Just optionally pass profiler.log_batch() to track losses.
    """
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Optional: log individual batch losses for detailed analysis
        if profiler:
            profiler.log_batch(loss.item())

    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Standard evaluation - no changes."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return 100.0 * correct / total


# ============================================================================
# Profile Different Optimizers
# ============================================================================

def profile_optimizer(optimizer_name, optimizer_class, optimizer_kwargs,
                     device, num_epochs=10):
    """
    Profile any optimizer with H3Profiler.

    This is your normal training loop with minimal profiler integration.
    """
    print("\n" + "="*70)
    print(f"Profiling: {optimizer_name}")
    print("="*70)

    # Load data (your normal code)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Initialize model and optimizer (your normal code)
    model = SimpleNet().to(device)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    # ONLY ADDITION: Wrap with H3Profiler
    profiler = H3Profiler(
        device=str(device),
        name=f"mnist_{optimizer_name.lower()}"
    )

    print(f"Energy backend: {profiler.energy_tracker.backend}")

    # Start profiling
    profiler.start()

    # Your normal training loop!
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, profiler)
        test_accuracy = evaluate(model, test_loader, device)

        # Log epoch results (minimal addition)
        profiler.log_epoch(accuracy=test_accuracy)

        print(f"Epoch {epoch+1:2d}/{num_epochs}: "
              f"Loss={train_loss:.4f}, Acc={test_accuracy:.2f}%")

    # Stop and get comprehensive analysis
    results = profiler.stop()

    # Get detailed report
    print("\n" + "-"*70)
    print("PROFILER ANALYSIS:")
    print("-"*70)
    print(profiler.get_report())

    # Export for later analysis
    csv_path = profiler.export_csv("./profiles")
    print(f"\nProfile saved to: {csv_path}")

    return results


# ============================================================================
# Main Comparison
# ============================================================================

def main():
    """Profile multiple optimizers and compare."""

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )

    print("="*70)
    print("H3Profiler - Zero-Risk Optimizer Analysis")
    print("="*70)
    print(f"Device: {device}")
    print("\nWe'll profile different optimizers to measure their")
    print("thermodynamic efficiency WITHOUT changing your training code!")
    print("="*70)

    NUM_EPOCHS = 10

    # Profile different optimizers
    optimizers_to_profile = [
        {
            "name": "Adam",
            "class": torch.optim.Adam,
            "kwargs": {"lr": 0.001}
        },
        {
            "name": "SGD",
            "class": torch.optim.SGD,
            "kwargs": {"lr": 0.01, "momentum": 0.9}
        },
        {
            "name": "AdamW",
            "class": torch.optim.AdamW,
            "kwargs": {"lr": 0.001, "weight_decay": 0.01}
        }
    ]

    results = {}

    for opt_config in optimizers_to_profile:
        result = profile_optimizer(
            opt_config["name"],
            opt_config["class"],
            opt_config["kwargs"],
            device,
            NUM_EPOCHS
        )
        results[opt_config["name"]] = result

        # Small delay between runs
        time.sleep(2)

    # ========================================================================
    # Final Comparison
    # ========================================================================

    print("\n" + "="*70)
    print("OPTIMIZER COMPARISON")
    print("="*70)

    print(f"\n{'Optimizer':<12} {'Time (s)':<12} {'Accuracy':<12} {'Energy (J)':<12} {'η (bits/J)':<15}")
    print("-"*70)

    for name, res in results.items():
        print(f"{name:<12} "
              f"{res['total_time_s']:<12.2f} "
              f"{res['final_accuracy']:<12.2f} "
              f"{res['total_energy_j']:<12.2f} "
              f"{res['efficiency_bits_per_j']:<15.6f}")

    # Find best
    best_efficiency = max(results.values(), key=lambda x: x['efficiency_bits_per_j'])
    best_name = [k for k, v in results.items() if v == best_efficiency][0]

    print("\n" + "="*70)
    print(f"Most efficient optimizer: {best_name}")
    print(f"Efficiency: {best_efficiency['efficiency_bits_per_j']:.6f} bits/J")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)

    for name, res in results.items():
        waste_potential = res.get('energy_waste_percent', 0)
        optimal_epoch = res.get('optimal_stopping_epoch', NUM_EPOCHS)

        print(f"\n{name}:")
        print(f"  • Final efficiency: {res['efficiency_bits_per_j']:.6f} bits/J")

        if waste_potential > 15:
            print(f"  ⚠️  Potential energy waste: {waste_potential:.1f}%")
            print(f"  💡 Consider stopping at epoch {optimal_epoch}")
            print(f"  💡 H3 could improve efficiency by focusing on high-loss samples")
        elif waste_potential > 5:
            print(f"  ℹ️  Moderate energy waste: {waste_potential:.1f}%")
            print(f"  💡 H3 might provide 10-20% efficiency gains")
        else:
            print(f"  ✅ Efficient training (waste: {waste_potential:.1f}%)")
            print(f"  ℹ️  H3 gains may be modest on this dataset")

    print("\n" + "="*70)
    print("Next Steps:")
    print("  1. Review the profiles/ directory for detailed CSV logs")
    print("  2. Use 'h3-report --compare' to analyze multiple runs")
    print("  3. If waste > 15%, try H3 optimizer for improvements")
    print("  4. Profiles show WHERE energy is wasted in training")
    print("="*70)


def demonstrate_profiler_features():
    """Show what H3Profiler can tell you."""

    print("\n" + "="*70)
    print("What H3Profiler Tells You")
    print("="*70)

    features = {
        "Thermodynamic Efficiency (η)": [
            "Bits of information per joule of energy",
            "Measured continuously throughout training",
            "Shows which epochs are most efficient"
        ],
        "Optimal Stopping Point": [
            "Detects diminishing returns automatically",
            "Suggests when to stop training",
            "Prevents wasted computation"
        ],
        "Energy Waste Analysis": [
            "Calculates % of energy wasted after optimal point",
            "Identifies inefficient training phases",
            "Quantifies potential savings"
        ],
        "H3 Opportunity Score": [
            "Estimates if H3 could help your training",
            "Based on loss distribution and redundancy",
            "Recommends next steps"
        ],
        "Phase Analysis": [
            "Shows efficiency in early/mid/late training",
            "Identifies where most learning happens",
            "Guides hyperparameter tuning"
        ]
    }

    for feature, details in features.items():
        print(f"\n{feature}:")
        for detail in details:
            print(f"  • {detail}")

    print("\n" + "="*70)
    print("Key Advantage: ZERO RISK")
    print("  • Works with ANY optimizer (Adam, SGD, AdamW, etc.)")
    print("  • No changes to your training code")
    print("  • Only measures - doesn't modify training")
    print("  • Establishes baseline before trying H3")
    print("="*70)


if __name__ == '__main__':
    # Uncomment to see feature overview
    # demonstrate_profiler_features()

    # Run profiling comparison
    main()
