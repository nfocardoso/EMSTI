"""
Quickstart example for H3 Optimizer.

Demonstrates basic usage with a simple neural network.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import H3 components
import sys
sys.path.append('..')
from h3.optimizer import H3Optimizer
from h3.sampler import InformationWeightedSampler
from h3.scheduler import LipschitzAdaptiveLR
from h3.energy_tracker import EnergyTracker


# Simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=2):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3

    # Create dummy dataset
    X = torch.randn(1000, 10)
    y = torch.randint(0, 2, (1000,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = SimpleNet().to(device)

    # Initialize H3 optimizer
    optimizer = H3Optimizer(model.parameters(), lr=learning_rate)

    # Initialize scheduler
    scheduler = LipschitzAdaptiveLR(optimizer)

    # Initialize energy tracker
    energy_tracker = EnergyTracker(device=str(device))

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("Starting training with H3 Optimizer...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        energy_tracker.start()

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Optimization step
            optimizer.step()

            total_loss += loss.item()

        energy_metrics = energy_tracker.stop()
        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Time: {energy_metrics.get('elapsed_time', 0):.2f}s")

    # Print energy summary
    print("\nEnergy Summary:")
    summary = energy_tracker.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
