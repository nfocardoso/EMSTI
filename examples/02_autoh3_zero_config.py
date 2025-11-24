"""
Example 02: AutoH3 - Zero Configuration Training
=================================================

The simplest way to use H3! AutoH3 handles everything automatically:
- Optimizer setup
- Loss tracking
- Information-weighted sampling
- Energy tracking
- Phase transitions
- Final reporting

Just provide your model, dataset, and mode - that's it!

Operating Modes:
    - "safe": Maximize accuracy (conservative sampling)
    - "balanced": Balance speed/accuracy (recommended)
    - "green": Maximize energy savings (aggressive sampling)
    - "extreme": Maximum savings (accepts accuracy trade-off)

Requirements:
    pip install h3-optimizer torch torchvision

Usage:
    python examples/02_autoh3_zero_config.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from h3.auto import AutoH3


# ============================================================================
# Model Definition
# ============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN for MNIST-like datasets."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ============================================================================
# Main Training with AutoH3
# ============================================================================

def main():
    """Train with AutoH3 - completely automatic!"""

    # Configuration
    NUM_EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    MODE = "balanced"  # Try: "safe", "balanced", "green", "extreme"

    # Device
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )

    print("="*70)
    print("AutoH3 - Zero Configuration Training")
    print("="*70)
    print(f"Mode: {MODE.upper()}")
    print(f"Device: {device}")
    print(f"Epochs: {NUM_EPOCHS}")
    print("="*70)

    # ========================================================================
    # Load Dataset
    # ========================================================================

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

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False
    )

    # ========================================================================
    # Initialize Model and AutoH3
    # ========================================================================

    print("\nInitializing model and AutoH3...")
    model = SimpleCNN(num_classes=10).to(device)

    # ONE LINE SETUP! AutoH3 handles everything
    auto = AutoH3(
        model.parameters(),
        train_dataset,
        mode=MODE,
        lr=LEARNING_RATE,
        name=f"mnist_auto_{MODE}"
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Start tracking
    auto.start()

    # ========================================================================
    # Training Loop - Simple and Clean!
    # ========================================================================

    print("\n" + "="*70)
    print("Training Started")
    print("="*70)

    for epoch in range(NUM_EPOCHS):
        model.train()

        # Get data loader (AutoH3 handles sampling automatically)
        train_loader = auto.get_loader(batch_size=BATCH_SIZE)

        epoch_loss = 0
        for data, target, indices in train_loader:
            data, target = data.to(device), target.to(device)

            # ONE LINE TRAINING STEP!
            # AutoH3 handles: loss tracking, energy logging, phase management
            loss = auto.training_step(model, data, target, criterion, indices)
            epoch_loss += loss

        avg_loss = epoch_loss / len(train_loader)

        # ONE LINE EVALUATION!
        accuracy = auto.evaluate_epoch(model, test_loader, device)

        # Get current status
        status = auto.get_status()

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} [{status['phase']:>14s}] | "
              f"Loss: {avg_loss:.4f} | "
              f"Acc: {accuracy:5.2f}% | "
              f"keep_frac: {status['keep_frac']:.2f} | "
              f"η: {status['efficiency']:.6f} bits/J")

    # ========================================================================
    # Final Results - Comprehensive Report
    # ========================================================================

    results = auto.finish()

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)

    print("\nFinal Metrics:")
    print("-"*70)
    print(f"  Total time:              {results['total_time_s']:.2f}s")
    print(f"  Final accuracy:          {results['final_accuracy']:.2f}%")
    print(f"  Total energy:            {results['total_energy_j']:.2f} J")
    print(f"  Information gain:        {results['info_gain_bits']:.4f} bits")
    print(f"  Thermodynamic efficiency: {results['efficiency_bits_per_j']:.6f} bits/J")

    print("\nAutoH3 Configuration:")
    print("-"*70)
    print(f"  Mode:                    {MODE}")
    print(f"  Average keep_frac:       {results['avg_keep_frac']:.2f}")
    print(f"  Average uniform_mix:     {results['avg_uniform_mix']:.2f}")
    print(f"  Samples processed:       {results['samples_processed']}")

    print("\nPhase Breakdown:")
    print("-"*70)
    phase_stats = results.get('phase_stats', {})
    for phase, stats in phase_stats.items():
        print(f"  {phase.capitalize():15s}: {stats['epochs']} epochs, "
              f"η={stats['avg_efficiency']:.6f} bits/J")

    print("\n" + "="*70)
    print("Report saved to:", auto.get_log_path())
    print("="*70)


def compare_modes():
    """
    Quick comparison of different AutoH3 modes.

    Run this to see how different modes trade off accuracy vs efficiency.
    """
    print("\n" + "="*70)
    print("AutoH3 Mode Comparison Guide")
    print("="*70)

    modes = {
        "safe": {
            "desc": "Maximum accuracy, minimal speedup",
            "keep_frac": "~0.70-0.80",
            "speedup": "5-10%",
            "accuracy": "±0.5pp"
        },
        "balanced": {
            "desc": "Good balance (RECOMMENDED)",
            "keep_frac": "~0.55-0.65",
            "speedup": "10-20%",
            "accuracy": "±1.0pp"
        },
        "green": {
            "desc": "Maximum efficiency, minor accuracy trade-off",
            "keep_frac": "~0.45-0.55",
            "speedup": "20-30%",
            "accuracy": "-1 to -3pp"
        },
        "extreme": {
            "desc": "Extreme savings for edge devices",
            "keep_frac": "~0.35-0.45",
            "speedup": "30-40%",
            "accuracy": "-3 to -5pp"
        }
    }

    for mode, info in modes.items():
        print(f"\n{mode.upper()}:")
        print(f"  Description:  {info['desc']}")
        print(f"  Keep fraction: {info['keep_frac']}")
        print(f"  Expected speedup: {info['speedup']}")
        print(f"  Accuracy change: {info['accuracy']}")

    print("\n" + "="*70)
    print("Recommendation: Start with 'balanced' mode, then adjust based on your needs")
    print("="*70)


if __name__ == '__main__':
    # Uncomment to see mode comparison guide
    # compare_modes()

    # Run training
    main()
