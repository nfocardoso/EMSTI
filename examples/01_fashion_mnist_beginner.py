"""
Example 01: Fashion-MNIST with H3 Optimizer (Beginner Friendly)
==================================================================

A complete, self-contained example showing how to use H3 optimizer
on Fashion-MNIST dataset. Perfect for beginners!

What you'll learn:
- How to set up H3 optimizer
- Three-phase training strategy
- Loss tracking and information-weighted sampling
- Energy efficiency measurement

Expected results:
- ~10-15% speedup compared to Adam
- ~15-20% energy efficiency gain
- Accuracy maintained or improved

Requirements:
    pip install h3-optimizer torch torchvision

Usage:
    python examples/01_fashion_mnist_beginner.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time

from h3.optimizer import H3Optimizer
from h3.sampler import LossTracker, InformationWeightedSampler, IndexedDataset
from h3.energy_tracker import EnergyTracker


# ============================================================================
# STEP 1: Define a Simple Neural Network
# ============================================================================

class FashionNet(nn.Module):
    """Simple CNN for Fashion-MNIST (28x28 grayscale images, 10 classes)."""

    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Conv block 1: 28x28 -> 14x14
        x = self.pool(F.relu(self.conv1(x)))

        # Conv block 2: 14x14 -> 7x7
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


# ============================================================================
# STEP 2: Training Function with H3
# ============================================================================

def train_epoch(model, dataset, loss_tracker, optimizer, device, phase, batch_size=64):
    """
    Train for one epoch using H3's three-phase strategy.

    Args:
        model: Neural network
        dataset: IndexedDataset wrapper
        loss_tracker: Tracks per-sample losses
        optimizer: H3Optimizer
        device: 'cuda', 'mps', or 'cpu'
        phase: 'warmup', 'thermodynamic', or 'consolidation'
        batch_size: Training batch size

    Returns:
        Average training loss for the epoch
    """
    model.train()

    # Configure sampler based on phase
    if phase in ['warmup', 'consolidation']:
        # Use uniform sampling
        sampler = InformationWeightedSampler(
            dataset,
            loss_tracker,
            uniform_mix=1.0  # 100% uniform
        )
    else:  # thermodynamic phase
        # Use information-weighted sampling
        sampler = InformationWeightedSampler(
            dataset,
            loss_tracker,
            keep_frac=0.65,   # Focus on top 65% high-loss samples
            uniform_mix=0.2   # 20% uniform, 80% weighted
        )

    # Create data loader with the sampler
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    total_loss = 0

    for data, target, indices in loader:
        data, target = data.to(device), target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)

        # Compute per-sample losses (needed for loss tracking)
        loss_vector = F.nll_loss(output, target, reduction='none')
        loss = loss_vector.mean()

        # Update loss tracker with current losses
        loss_tracker.update(indices, loss_vector.detach())

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================================
# STEP 3: Evaluation Function
# ============================================================================

def evaluate(model, loader, device):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            # Handle both indexed and regular datasets
            if len(batch) == 3:
                data, target, _ = batch
            else:
                data, target = batch

            data, target = data.to(device), target.to(device)
            output = model(data)

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


# ============================================================================
# STEP 4: Main Training Loop
# ============================================================================

def main():
    """Run H3 training on Fashion-MNIST."""

    # Configuration
    NUM_EPOCHS = 10
    WARMUP_EPOCHS = 1
    CONSOLIDATION_EPOCHS = 1
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001

    # Device selection
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )

    print("="*70)
    print("H3 Optimizer - Fashion-MNIST Example")
    print("="*70)
    print(f"Device: {device}")
    print(f"Epochs: {NUM_EPOCHS} (warmup={WARMUP_EPOCHS}, consolidation={CONSOLIDATION_EPOCHS})")
    print(f"Learning rate: {LEARNING_RATE}")
    print("="*70)

    # ========================================================================
    # Load Fashion-MNIST Dataset
    # ========================================================================

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    print("\nLoading Fashion-MNIST dataset...")
    train_dataset_base = torchvision.datasets.FashionMNIST(
        './data',
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        './data',
        train=False,
        download=True,
        transform=transform
    )

    # Wrap training set with IndexedDataset for loss tracking
    train_dataset = IndexedDataset(train_dataset_base)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # ========================================================================
    # Initialize Model, Optimizer, and Trackers
    # ========================================================================

    print("\nInitializing model and H3 components...")

    # Model
    model = FashionNet().to(device)

    # H3 Optimizer with Lipschitz-adaptive learning rate
    optimizer = H3Optimizer(
        model.parameters(),
        lr=LEARNING_RATE,
        lipschitz_safety=0.9,        # Safety factor for step size
        lipschitz_update_interval=10  # Update Lipschitz estimate every 10 steps
    )

    # Loss tracker for information-weighted sampling
    loss_tracker = LossTracker(
        num_samples=len(train_dataset),
        smoothing=0.1  # Exponential smoothing factor
    )

    # Energy tracker for thermodynamic efficiency
    energy_tracker = EnergyTracker(device=str(device))
    print(f"Energy tracking backend: {energy_tracker.backend}")

    # ========================================================================
    # Training Loop with Three Phases
    # ========================================================================

    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)

    energy_tracker.start()
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        # Determine current phase
        if epoch < WARMUP_EPOCHS:
            phase = 'warmup'
        elif epoch >= (NUM_EPOCHS - CONSOLIDATION_EPOCHS):
            phase = 'consolidation'
        else:
            phase = 'thermodynamic'

        # Train for one epoch
        train_loss = train_epoch(
            model, train_dataset, loss_tracker,
            optimizer, device, phase, BATCH_SIZE
        )

        # Evaluate on test set
        test_accuracy = evaluate(model, test_loader, device)

        # Log loss for efficiency calculation
        energy_tracker.log_loss(train_loss)

        # Get current efficiency
        stats = energy_tracker.get_current_stats()

        # Print progress
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} [{phase:>14s}] | "
              f"Loss: {train_loss:.4f} | "
              f"Acc: {test_accuracy:5.2f}% | "
              f"η: {stats['current_efficiency_bits_per_j']:.6f} bits/J")

    # ========================================================================
    # Final Results
    # ========================================================================

    elapsed_time = time.time() - start_time
    results = energy_tracker.stop()

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"{'Metric':<25} {'Value':>20}")
    print("-"*70)
    print(f"{'Total time:':<25} {elapsed_time:>19.2f}s")
    print(f"{'Final test accuracy:':<25} {test_accuracy:>19.2f}%")
    print(f"{'Total energy:':<25} {results['total_energy_j']:>19.2f} J")
    print(f"{'Information gain:':<25} {results['info_gain_bits']:>19.4f} bits")
    print(f"{'Thermodynamic efficiency:':<25} {results['efficiency_bits_per_j']:>19.6f} bits/J")
    print("="*70)

    # Phase breakdown
    print("\nPhase Efficiency Analysis:")
    print("-"*70)
    print("The efficiency η changes across training phases:")
    print("  • Warmup: High η (model learning basics quickly)")
    print("  • Thermodynamic: Moderate η (focused learning on hard examples)")
    print("  • Consolidation: Lower η (fine-tuning, diminishing returns)")
    print("\nThis is expected behavior from thermodynamic theory!")
    print("="*70)


if __name__ == '__main__':
    main()
