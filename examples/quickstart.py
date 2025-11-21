"""
H3 Optimizer Quick Start Demo

Demonstrates H3's thermodynamic efficiency on MNIST:
- Three-phase training (warmup → thermodynamic → consolidation)
- Lipschitz-adaptive learning rates
- Information-weighted sampling
- Energy and efficiency tracking

Compares H3 vs Adam baseline on:
- Training time
- Final accuracy
- Thermodynamic efficiency (bits/joule)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from h3.optimizer import H3Optimizer
from h3.sampler import (
    InformationWeightedSampler,
    LossTracker,
    IndexedDataset
)
from h3.energy_tracker import EnergyTracker


# Simple CNN for MNIST
class SimpleCNN(nn.Module):
    """Lightweight CNN for MNIST classification."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train_epoch_baseline(model, loader, optimizer, device, loss_tracker=None):
    """Train one epoch with baseline (uniform sampling)."""
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(loader):
        # Handle both indexed and non-indexed datasets
        if len(batch) == 3:
            data, target, indices = batch
        else:
            data, target = batch
            indices = None

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        # Per-sample losses for tracking
        loss_vector = F.nll_loss(output, target, reduction='none')
        loss = loss_vector.mean()

        # Update loss tracker if provided
        if loss_tracker is not None and indices is not None:
            loss_tracker.update(indices, loss_vector.detach())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def train_epoch_h3(
    model, dataset, loss_tracker, optimizer, device,
    phase, epoch, num_epochs, batch_size=64
):
    """
    Train one epoch with H3 three-phase strategy.

    Args:
        phase: 'warmup', 'thermodynamic', or 'consolidation'
    """
    model.train()

    # Configure sampler based on phase
    if phase == 'warmup' or phase == 'consolidation':
        # Uniform sampling
        sampler = InformationWeightedSampler(
            dataset,
            loss_tracker,
            uniform_mix=1.0  # 100% uniform
        )
    else:  # thermodynamic
        # Weighted sampling (Section 5.3)
        sampler = InformationWeightedSampler(
            dataset,
            loss_tracker,
            keep_frac=0.65,   # Keep top 65% by loss
            uniform_mix=0.2   # 20% uniform, 80% weighted
        )

    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    total_loss = 0
    for batch_idx, (data, target, indices) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        # Per-sample losses
        loss_vector = F.nll_loss(output, target, reduction='none')
        loss = loss_vector.mean()

        # Update loss tracker
        loss_tracker.update(indices, loss_vector.detach())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def test(model, loader, device):
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                data, target, _ = batch
            else:
                data, target = batch

            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)

    return test_loss, accuracy


def run_baseline(device, num_epochs=5):
    """Run baseline Adam training."""
    print("\n" + "="*60)
    print("BASELINE: Adam + Uniform Sampling")
    print("="*60)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        './data', train=False, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Model
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Energy tracking
    tracker = EnergyTracker(device=str(device))
    print(f"Energy backend: {tracker.backend}")

    # Training
    tracker.start()
    start_time = time.time()

    for epoch in range(num_epochs):
        train_loss = train_epoch_baseline(model, train_loader, optimizer, device)
        test_loss, test_acc = test(model, test_loader, device)

        # Log loss for efficiency calculation
        tracker.log_loss(train_loss)

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={train_loss:.4f}, "
              f"Test Acc={test_acc:.2f}%")

    elapsed = time.time() - start_time
    results = tracker.stop()

    print(f"\n{'Final Results':^60}")
    print(f"{'─'*60}")
    print(f"  Time:       {elapsed:.2f}s")
    print(f"  Accuracy:   {test_acc:.2f}%")
    print(f"  Energy:     {results['total_energy_j']:.2f} J")
    print(f"  Info Gain:  {results['info_gain_bits']:.4f} bits")
    print(f"  Efficiency: {results['efficiency_bits_per_j']:.6f} bits/J")

    return {
        'time': elapsed,
        'accuracy': test_acc,
        'energy': results['total_energy_j'],
        'info_gain': results['info_gain_bits'],
        'efficiency': results['efficiency_bits_per_j'],
    }


def run_h3(device, num_epochs=5, warmup_epochs=1, consolidation_epochs=1):
    """Run H3 training with three phases."""
    print("\n" + "="*60)
    print("H3: Thermodynamic Optimizer")
    print("="*60)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset_base = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        './data', train=False, transform=transform
    )

    # Wrap with IndexedDataset for loss tracking
    train_dataset = IndexedDataset(train_dataset_base)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Model
    model = SimpleCNN().to(device)
    optimizer = H3Optimizer(
        model.parameters(),
        lr=0.001,
        lipschitz_safety=0.9,
        lipschitz_update_interval=10
    )

    # Loss tracker for information-weighted sampling
    loss_tracker = LossTracker(num_samples=len(train_dataset), smoothing=0.1)

    # Energy tracking
    tracker = EnergyTracker(device=str(device))
    print(f"Energy backend: {tracker.backend}")

    # Training with three phases
    tracker.start()
    start_time = time.time()

    for epoch in range(num_epochs):
        # Determine phase
        if epoch < warmup_epochs:
            phase = 'warmup'
        elif epoch >= (num_epochs - consolidation_epochs):
            phase = 'consolidation'
        else:
            phase = 'thermodynamic'

        train_loss = train_epoch_h3(
            model, train_dataset, loss_tracker, optimizer, device,
            phase=phase, epoch=epoch, num_epochs=num_epochs
        )

        test_loss, test_acc = test(model, test_loader, device)

        # Log loss for efficiency
        tracker.log_loss(train_loss)

        # Get current efficiency
        stats = tracker.get_current_stats()

        print(f"Epoch {epoch+1}/{num_epochs} [{phase:>14s}]: "
              f"Train Loss={train_loss:.4f}, "
              f"Test Acc={test_acc:.2f}%, "
              f"η={stats['current_efficiency_bits_per_j']:.6f} bits/J")

    elapsed = time.time() - start_time
    results = tracker.stop()

    print(f"\n{'Final Results':^60}")
    print(f"{'─'*60}")
    print(f"  Time:       {elapsed:.2f}s")
    print(f"  Accuracy:   {test_acc:.2f}%")
    print(f"  Energy:     {results['total_energy_j']:.2f} J")
    print(f"  Info Gain:  {results['info_gain_bits']:.4f} bits")
    print(f"  Efficiency: {results['efficiency_bits_per_j']:.6f} bits/J")

    return {
        'time': elapsed,
        'accuracy': test_acc,
        'energy': results['total_energy_j'],
        'info_gain': results['info_gain_bits'],
        'efficiency': results['efficiency_bits_per_j'],
    }


def compare_results(baseline_results, h3_results):
    """Compare and display results."""
    print("\n" + "="*60)
    print("COMPARISON: H3 vs Baseline")
    print("="*60)

    time_reduction = (1 - h3_results['time'] / baseline_results['time']) * 100
    accuracy_diff = h3_results['accuracy'] - baseline_results['accuracy']
    efficiency_gain = (h3_results['efficiency'] / baseline_results['efficiency'] - 1) * 100

    print(f"\n{'Metric':<20} {'Baseline':>15} {'H3':>15} {'Change':>15}")
    print(f"{'─'*67}")
    print(f"{'Time (s)':<20} {baseline_results['time']:>15.2f} "
          f"{h3_results['time']:>15.2f} {time_reduction:>14.1f}%")
    print(f"{'Accuracy (%)':<20} {baseline_results['accuracy']:>15.2f} "
          f"{h3_results['accuracy']:>15.2f} {accuracy_diff:>14.2f}pp")
    print(f"{'Energy (J)':<20} {baseline_results['energy']:>15.2f} "
          f"{h3_results['energy']:>15.2f}")
    print(f"{'Info Gain (bits)':<20} {baseline_results['info_gain']:>15.4f} "
          f"{h3_results['info_gain']:>15.4f}")
    print(f"{'η (bits/J)':<20} {baseline_results['efficiency']:>15.6f} "
          f"{h3_results['efficiency']:>15.6f} {efficiency_gain:>14.1f}%")

    print(f"\n{'Summary':^67}")
    print(f"{'─'*67}")
    print(f"  ⚡ H3 is {abs(time_reduction):.1f}% {'faster' if time_reduction > 0 else 'slower'}")
    print(f"  🎯 Accuracy difference: {accuracy_diff:+.2f} percentage points")
    print(f"  📊 Thermodynamic efficiency: {efficiency_gain:+.1f}% improvement")
    print(f"{'─'*67}\n")


def main():
    """Run H3 vs Baseline comparison on MNIST."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available()
                         else 'mps' if torch.backends.mps.is_available()
                         else 'cpu')
    print(f"\nUsing device: {device}")

    # Run experiments
    baseline_results = run_baseline(device, num_epochs=5)
    h3_results = run_h3(device, num_epochs=5, warmup_epochs=1, consolidation_epochs=1)

    # Compare
    compare_results(baseline_results, h3_results)


if __name__ == '__main__':
    main()
