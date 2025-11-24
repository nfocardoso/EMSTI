"""
Example 06: H3 with Custom Datasets (CSV, NumPy, Pandas)
==========================================================

This example shows how to use H3 optimizer with your own custom datasets:
- CSV files
- NumPy arrays
- Pandas DataFrames
- Any custom PyTorch Dataset

You'll learn:
- How to create custom datasets compatible with H3
- How to wrap them with IndexedDataset
- How to use H3 with non-image data (tabular, time series, etc.)
- Complete end-to-end example with synthetic data

This is perfect for real-world applications where you have your own data!

Requirements:
    pip install h3-optimizer torch numpy pandas scikit-learn

Usage:
    python examples/06_custom_datasets.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

from h3.optimizer import H3Optimizer
from h3.sampler import LossTracker, InformationWeightedSampler, IndexedDataset
from h3.energy_tracker import EnergyTracker


# ============================================================================
# Custom Dataset Classes
# ============================================================================

class NumpyDataset(Dataset):
    """Dataset from NumPy arrays."""

    def __init__(self, X, y):
        """
        Args:
            X: NumPy array of features (N, D)
            y: NumPy array of labels (N,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CSVDataset(Dataset):
    """Dataset from CSV file."""

    def __init__(self, csv_path, target_column, feature_columns=None):
        """
        Args:
            csv_path: Path to CSV file
            target_column: Name of target column
            feature_columns: List of feature column names (None = all except target)
        """
        df = pd.read_csv(csv_path)

        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]

        self.X = torch.FloatTensor(df[feature_columns].values)
        self.y = torch.LongTensor(df[target_column].values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PandasDataset(Dataset):
    """Dataset from Pandas DataFrame."""

    def __init__(self, dataframe, target_column, feature_columns=None):
        """
        Args:
            dataframe: Pandas DataFrame
            target_column: Name of target column
            feature_columns: List of feature column names (None = all except target)
        """
        if feature_columns is None:
            feature_columns = [col for col in dataframe.columns if col != target_column]

        self.X = torch.FloatTensor(dataframe[feature_columns].values)
        self.y = torch.LongTensor(dataframe[target_column].values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# Model for Tabular Data
# ============================================================================

class TabularNet(nn.Module):
    """Simple fully-connected network for tabular data."""

    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions [128, 64, 32]
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataset, loss_tracker, optimizer, criterion,
                device, phase, batch_size=64):
    """Train one epoch with H3."""
    model.train()

    # Configure sampler based on phase
    if phase in ['warmup', 'consolidation']:
        sampler = InformationWeightedSampler(
            dataset,
            loss_tracker,
            uniform_mix=1.0
        )
    else:  # thermodynamic
        sampler = InformationWeightedSampler(
            dataset,
            loss_tracker,
            keep_frac=0.60,
            uniform_mix=0.3
        )

    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    total_loss = 0
    correct = 0
    total = 0

    for data, target, indices in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss_vector = criterion(output, target)
        loss = loss_vector.mean()

        loss_tracker.update(indices, loss_vector.detach())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate(model, loader, device):
    """Evaluate model."""
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
# Example 1: NumPy Arrays
# ============================================================================

def demo_numpy_dataset():
    """Train H3 on synthetic classification data from NumPy."""

    print("\n" + "="*70)
    print("DEMO 1: H3 with NumPy Arrays")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate synthetic classification data
    print("\nGenerating synthetic dataset...")
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=5,
        random_state=42
    )

    # Split and normalize
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")

    # Create custom datasets
    train_dataset_base = NumpyDataset(X_train, y_train)
    test_dataset = NumpyDataset(X_test, y_test)

    # Wrap with IndexedDataset for H3
    train_dataset = IndexedDataset(train_dataset_base)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Model
    model = TabularNet(
        input_dim=X_train.shape[1],
        hidden_dims=[128, 64, 32],
        num_classes=5,
        dropout=0.3
    ).to(device)

    # H3 components
    optimizer = H3Optimizer(model.parameters(), lr=0.001)
    loss_tracker = LossTracker(len(train_dataset), smoothing=0.1)
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Energy tracking
    energy_tracker = EnergyTracker(device=str(device))
    energy_tracker.start()
    start_time = time.time()

    # Training
    NUM_EPOCHS = 10
    print("\nTraining with H3...")

    for epoch in range(NUM_EPOCHS):
        # Determine phase
        if epoch < 1:
            phase = 'warmup'
        elif epoch >= NUM_EPOCHS - 1:
            phase = 'consolidation'
        else:
            phase = 'thermodynamic'

        train_loss, train_acc = train_epoch(
            model, train_dataset, loss_tracker, optimizer,
            criterion, device, phase, batch_size=64
        )

        test_acc = evaluate(model, test_loader, device)

        energy_tracker.log_loss(train_loss)
        stats = energy_tracker.get_current_stats()

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} [{phase:>14s}] | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:5.2f}% | "
              f"Test Acc: {test_acc:5.2f}%")

    # Results
    elapsed = time.time() - start_time
    results = energy_tracker.stop()

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Final test accuracy: {test_acc:.2f}%")
    print(f"  Energy: {results['total_energy_j']:.2f} J")
    print(f"  Efficiency: {results['efficiency_bits_per_j']:.6f} bits/J")


# ============================================================================
# Example 2: Pandas DataFrame
# ============================================================================

def demo_pandas_dataset():
    """Train H3 on synthetic data from Pandas."""

    print("\n" + "="*70)
    print("DEMO 2: H3 with Pandas DataFrame")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate synthetic data
    print("\nGenerating synthetic dataset...")
    X, y = make_classification(
        n_samples=3000,
        n_features=15,
        n_informative=10,
        n_classes=3,
        random_state=42
    )

    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    print(f"DataFrame shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())

    # Split
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    # Normalize
    scaler = StandardScaler()
    train_df[feature_names] = scaler.fit_transform(train_df[feature_names])
    test_df[feature_names] = scaler.transform(test_df[feature_names])

    print(f"\nTrain samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    # Create datasets
    train_dataset_base = PandasDataset(train_df, target_column='target')
    test_dataset = PandasDataset(test_df, target_column='target')

    train_dataset = IndexedDataset(train_dataset_base)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Model and training (simplified)
    model = TabularNet(
        input_dim=len(feature_names),
        hidden_dims=[64, 32],
        num_classes=3
    ).to(device)

    optimizer = H3Optimizer(model.parameters(), lr=0.001)
    loss_tracker = LossTracker(len(train_dataset), smoothing=0.1)
    criterion = nn.CrossEntropyLoss(reduction='none')

    print("\nTraining with H3...")
    for epoch in range(8):
        phase = 'thermodynamic' if 1 <= epoch < 7 else 'warmup' if epoch == 0 else 'consolidation'

        train_loss, train_acc = train_epoch(
            model, train_dataset, loss_tracker, optimizer,
            criterion, device, phase, batch_size=64
        )

        test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1:2d}/8 [{phase:>14s}] | "
              f"Loss: {train_loss:.4f} | "
              f"Test Acc: {test_acc:5.2f}%")

    print(f"\nFinal test accuracy: {test_acc:.2f}%")


# ============================================================================
# Example 3: Creating Synthetic CSV for Demo
# ============================================================================

def demo_csv_dataset():
    """Train H3 on data loaded from CSV."""

    print("\n" + "="*70)
    print("DEMO 3: H3 with CSV Files")
    print("="*70)

    # Create synthetic CSV
    print("\nCreating synthetic CSV...")
    X, y = make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=8,
        n_classes=4,
        random_state=42
    )

    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y

    # Save to CSV
    csv_path = './data/synthetic_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")

    # Load from CSV
    print("\nLoading from CSV...")
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    # Save splits
    train_csv = './data/train.csv'
    test_csv = './data/test.csv'
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Create datasets from CSV
    train_dataset_base = CSVDataset(train_csv, target_column='label')
    test_dataset = CSVDataset(test_csv, target_column='label')

    print(f"Train samples: {len(train_dataset_base)}")
    print(f"Test samples: {len(test_dataset)}")

    train_dataset = IndexedDataset(train_dataset_base)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Quick training demo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TabularNet(
        input_dim=10,
        hidden_dims=[32, 16],
        num_classes=4
    ).to(device)

    optimizer = H3Optimizer(model.parameters(), lr=0.001)
    loss_tracker = LossTracker(len(train_dataset), smoothing=0.1)
    criterion = nn.CrossEntropyLoss(reduction='none')

    print("\nTraining with H3...")
    for epoch in range(6):
        phase = 'thermodynamic' if 1 <= epoch < 5 else 'warmup' if epoch == 0 else 'consolidation'

        train_loss, _ = train_epoch(
            model, train_dataset, loss_tracker, optimizer,
            criterion, device, phase, batch_size=64
        )

        test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1:2d}/6 [{phase:>14s}] | Test Acc: {test_acc:5.2f}%")

    print(f"\nSuccess! H3 works with CSV datasets.")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all custom dataset demos."""

    print("="*70)
    print("H3 with Custom Datasets - Complete Guide")
    print("="*70)
    print("\nThis example shows how to use H3 with your own data:")
    print("  • NumPy arrays")
    print("  • Pandas DataFrames")
    print("  • CSV files")
    print("  • Any custom PyTorch Dataset")
    print("="*70)

    # Run demos
    demo_numpy_dataset()
    demo_pandas_dataset()
    demo_csv_dataset()

    # Summary
    print("\n" + "="*70)
    print("Summary: Using H3 with Custom Datasets")
    print("="*70)
    print("""
Key Steps:
1. Create a PyTorch Dataset class (see NumpyDataset, PandasDataset examples)
2. Wrap it with IndexedDataset for H3 compatibility
3. Use H3Optimizer, LossTracker, and InformationWeightedSampler
4. Train with three-phase strategy

Your custom Dataset must:
  ✓ Inherit from torch.utils.data.Dataset
  ✓ Implement __len__() method
  ✓ Implement __getitem__() returning (features, label)
  ✓ Return tensors (not NumPy arrays)

Then just wrap with IndexedDataset and you're ready for H3!

Examples in this file show:
  • Tabular data (classification)
  • CSV loading
  • Pandas integration
  • NumPy arrays
  • Feature normalization

The same pattern works for:
  • Time series data
  • Text (after embedding)
  • Audio features
  • Any numerical data!
    """)
    print("="*70)


if __name__ == '__main__':
    main()
