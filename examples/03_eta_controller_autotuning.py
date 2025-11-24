"""
Example 03: η-Controller - Automatic Hyperparameter Tuning
===========================================================

This example shows how to use the η-Controller to automatically tune
H3's hyperparameters (keep_frac, uniform_mix) in real-time based on
thermodynamic efficiency feedback.

The controller:
- Monitors thermodynamic efficiency (η = bits/joule)
- Adjusts sampling aggressiveness automatically
- Prevents data starvation (loss volatility detection)
- Respects accuracy constraints
- Self-regulates like a thermostat

This is perfect when you want H3's benefits but don't want to
manually tune hyperparameters for each dataset.

Requirements:
    pip install h3-optimizer torch torchvision

Usage:
    python examples/03_eta_controller_autotuning.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time

from h3.eta_controller import create_controlled_h3, ETAController
from h3.sampler import InformationWeightedSampler
from h3.energy_tracker import EnergyTracker


# ============================================================================
# Model Definition
# ============================================================================

class FashionCNN(nn.Module):
    """Simple CNN for Fashion-MNIST."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return F.log_softmax(self.fc2(x), dim=1)


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(model, indexed_dataset, loss_tracker, optimizer,
                controller, device, batch_size=64):
    """
    Train one epoch with η-Controller guidance.

    The controller automatically provides optimized keep_frac and uniform_mix.
    """
    model.train()

    # Get current hyperparameters from controller
    keep_frac, uniform_mix, phase = controller.control_step()

    print(f"  Controller → keep_frac={keep_frac:.2f}, "
          f"uniform_mix={uniform_mix:.2f}, phase={phase}")

    # Create sampler with controller-optimized parameters
    sampler = InformationWeightedSampler(
        indexed_dataset,
        loss_tracker,
        keep_frac=keep_frac,
        uniform_mix=uniform_mix
    )

    loader = DataLoader(indexed_dataset, sampler=sampler, batch_size=batch_size)

    total_loss = 0
    losses = []

    for data, target, indices in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss_vector = F.nll_loss(output, target, reduction='none')
        loss = loss_vector.mean()

        loss_tracker.update(indices, loss_vector.detach())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        losses.append(loss.item())

    avg_loss = total_loss / len(loader)

    # Calculate loss volatility (for starvation detection)
    loss_std = torch.tensor(losses).std().item()

    return avg_loss, loss_std, keep_frac, uniform_mix


def evaluate(model, loader, device):
    """Evaluate model accuracy."""
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
# Main Training Loop with η-Controller
# ============================================================================

def main():
    """Train Fashion-MNIST with automatic η-Controller tuning."""

    # Configuration
    NUM_EPOCHS = 15
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    CONTROLLER_MODE = "balanced"  # "safe", "balanced", "green", "extreme"
    ACCURACY_TOLERANCE = 1.5  # Max acceptable accuracy drop (percentage points)

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )

    print("="*70)
    print("η-Controller - Automatic Hyperparameter Tuning")
    print("="*70)
    print(f"Device: {device}")
    print(f"Mode: {CONTROLLER_MODE}")
    print(f"Accuracy tolerance: ±{ACCURACY_TOLERANCE}pp")
    print(f"Epochs: {NUM_EPOCHS}")
    print("="*70)

    # ========================================================================
    # Load Dataset
    # ========================================================================

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("\nLoading Fashion-MNIST dataset...")
    train_dataset_base = torchvision.datasets.FashionMNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        './data', train=False, download=True, transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # ========================================================================
    # Setup H3 with η-Controller
    # ========================================================================

    print("\nInitializing H3 with η-Controller...")
    model = FashionCNN().to(device)

    # ONE LINE SETUP with controller!
    optimizer, controller, loss_tracker, indexed_dataset = create_controlled_h3(
        model.parameters(),
        train_dataset_base,
        lr=LEARNING_RATE,
        mode=CONTROLLER_MODE,
        accuracy_tolerance=ACCURACY_TOLERANCE,
        total_epochs=NUM_EPOCHS
    )

    # Energy tracker
    energy_tracker = EnergyTracker(device=str(device))
    print(f"Energy backend: {energy_tracker.backend}")
    print(f"Controller mode: {CONTROLLER_MODE}")

    # ========================================================================
    # Training Loop - Controller Learns and Adapts
    # ========================================================================

    print("\n" + "="*70)
    print("Training with Adaptive η-Controller")
    print("="*70)

    energy_tracker.start()
    start_time = time.time()

    baseline_accuracy = None

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # Train with controller guidance
        train_loss, loss_std, keep_frac, uniform_mix = train_epoch(
            model, indexed_dataset, loss_tracker,
            optimizer, controller, device, BATCH_SIZE
        )

        # Evaluate
        test_accuracy = evaluate(model, test_loader, device)

        # Track baseline
        if baseline_accuracy is None:
            baseline_accuracy = test_accuracy

        # Log for energy tracking
        energy_tracker.log_loss(train_loss)
        stats = energy_tracker.get_current_stats()

        # Calculate information gain (simplified)
        info_gain = abs(stats['info_gain_bits'])

        # FEEDBACK TO CONTROLLER
        # Controller learns from these metrics and adapts
        controller.observe(
            eta=stats['current_efficiency_bits_per_j'],
            accuracy=test_accuracy,
            loss=train_loss,
            energy=stats['total_energy_j'],
            info=info_gain,
            loss_volatility=loss_std,
            baseline_accuracy=baseline_accuracy
        )

        # Print status
        accuracy_delta = test_accuracy - baseline_accuracy
        print(f"  Loss: {train_loss:.4f} (std: {loss_std:.4f})")
        print(f"  Accuracy: {test_accuracy:.2f}% (Δ: {accuracy_delta:+.2f}pp)")
        print(f"  Efficiency: {stats['current_efficiency_bits_per_j']:.6f} bits/J")
        print(f"  Energy so far: {stats['total_energy_j']:.2f} J")

        # Controller status every 3 epochs
        if epoch % 3 == 0:
            print("\n  📊 Controller Status:")
            report = controller.get_report()
            print(f"     {report}")

    # ========================================================================
    # Final Results
    # ========================================================================

    elapsed_time = time.time() - start_time
    results = energy_tracker.stop()

    print("\n" + "="*70)
    print("Training Complete - η-Controller Performance")
    print("="*70)

    print("\nFinal Metrics:")
    print("-"*70)
    print(f"  Total time:              {elapsed_time:.2f}s")
    print(f"  Final accuracy:          {test_accuracy:.2f}%")
    print(f"  Accuracy change:         {test_accuracy - baseline_accuracy:+.2f}pp")
    print(f"  Total energy:            {results['total_energy_j']:.2f} J")
    print(f"  Thermodynamic efficiency: {results['efficiency_bits_per_j']:.6f} bits/J")

    print("\nController Learning Summary:")
    print("-"*70)
    final_report = controller.get_report()
    print(f"  {final_report}")

    print("\n" + "="*70)
    print("Key Benefits of η-Controller:")
    print("  ✓ No manual hyperparameter tuning needed")
    print("  ✓ Adapts to dataset characteristics automatically")
    print("  ✓ Prevents data starvation via loss monitoring")
    print("  ✓ Respects accuracy constraints")
    print("  ✓ Self-regulating thermodynamic system")
    print("="*70)


def demonstrate_controller_modes():
    """
    Show how different controller modes behave.
    """
    print("\n" + "="*70)
    print("η-Controller Modes Explained")
    print("="*70)

    modes_info = {
        "safe": {
            "target_keep_frac": "0.65-0.75",
            "adjustment_rate": "Slow (conservative)",
            "accuracy_priority": "High",
            "best_for": "Production systems, critical applications"
        },
        "balanced": {
            "target_keep_frac": "0.50-0.65",
            "adjustment_rate": "Moderate",
            "accuracy_priority": "Medium",
            "best_for": "General use (RECOMMENDED)"
        },
        "green": {
            "target_keep_frac": "0.40-0.55",
            "adjustment_rate": "Fast (aggressive)",
            "accuracy_priority": "Lower",
            "best_for": "Energy-constrained environments"
        },
        "extreme": {
            "target_keep_frac": "0.30-0.45",
            "adjustment_rate": "Very fast",
            "accuracy_priority": "Minimum",
            "best_for": "Edge devices, IoT"
        }
    }

    for mode, info in modes_info.items():
        print(f"\n{mode.upper()} MODE:")
        print(f"  Target keep_frac:   {info['target_keep_frac']}")
        print(f"  Adjustment rate:    {info['adjustment_rate']}")
        print(f"  Accuracy priority:  {info['accuracy_priority']}")
        print(f"  Best for:           {info['best_for']}")

    print("\n" + "="*70)
    print("The controller continuously monitors η and adjusts parameters to")
    print("maximize thermodynamic efficiency while respecting constraints.")
    print("="*70)


if __name__ == '__main__':
    # Uncomment to see mode explanations
    # demonstrate_controller_modes()

    # Run training
    main()
