"""
CIFAR-10 Optimization Experiments
Systematic evaluation of H3 configurations

Objective: Find optimal H3 configuration that matches/exceeds Adam accuracy
while maintaining efficiency gains.

Target: Accuracy ≥81.5%, Speedup ≥5%, Efficiency ≥+10%
"""
import time
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

import sys
sys.path.insert(0, '/home/user/EMSTI')

from h3 import H3Optimizer, LossTracker, InformationWeightedSampler, IndexedDataset, EnergyTracker
from h3.adaptive import cosine_annealing_schedule, linear_annealing_schedule

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIGS = [
    {
        "name": "baseline_adam",
        "optimizer": "adam",
        "description": "Standard Adam baseline"
    },
    {
        "name": "h3_fixed_conservative",
        "optimizer": "h3",
        "keep_frac": 0.75,  # Less aggressive
        "uniform_mix": 0.3,
        "lipschitz_safety": 0.9,
        "loss_smoothing": 0.1,
        "description": "H3 with conservative fixed sampling"
    },
    {
        "name": "h3_adaptive_cosine",
        "optimizer": "h3_adaptive",
        "schedule": "cosine",
        "min_keep_frac": 0.50,
        "warmup_epochs": 3,
        "consolidation_epochs": 3,
        "lipschitz_safety": 0.95,
        "loss_smoothing": 0.2,
        "description": "H3 with cosine annealing schedule"
    },
    {
        "name": "h3_adaptive_cosine_aggressive",
        "optimizer": "h3_adaptive",
        "schedule": "cosine",
        "min_keep_frac": 0.40,  # More aggressive
        "warmup_epochs": 4,
        "consolidation_epochs": 4,
        "lipschitz_safety": 0.95,
        "loss_smoothing": 0.2,
        "description": "H3 with aggressive cosine annealing"
    },
    {
        "name": "h3_adaptive_linear",
        "optimizer": "h3_adaptive",
        "schedule": "linear",
        "min_keep_frac": 0.50,
        "warmup_epochs": 3,
        "consolidation_epochs": 3,
        "lipschitz_safety": 0.9,
        "loss_smoothing": 0.15,
        "description": "H3 with linear annealing schedule"
    }
]

NUM_EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.001

device = torch.device('mps' if torch.backends.mps.is_available()
                     else 'cuda' if torch.cuda.is_available()
                     else 'cpu')

print(f"🖥️  Device: {device}")

# ============================================================================
# DATA LOADING
# ============================================================================
print("\n📊 Loading CIFAR-10...")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

print(f"✅ Loaded: {len(train_dataset)} train, {len(test_dataset)} test\n")

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================
def evaluate(model, loader, device):
    """Evaluate model on test set"""
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
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return 100. * correct / total

# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_model(config):
    """Train model with given configuration"""
    print("="*80)
    print(f"🔬 EXPERIMENT: {config['name']}")
    print(f"📝 {config['description']}")
    print("="*80)

    # Initialize model
    model = resnet18(num_classes=10).to(device)

    # Setup optimizer
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                 shuffle=True, num_workers=2)
        tracker = None

    else:  # H3 variants
        indexed_dataset = IndexedDataset(train_dataset)
        loss_tracker = LossTracker(
            num_samples=len(indexed_dataset),
            smoothing=config.get('loss_smoothing', 0.1)
        )

        optimizer = H3Optimizer(
            model.parameters(),
            lr=LEARNING_RATE,
            lipschitz_safety=config.get('lipschitz_safety', 0.9),
            lipschitz_update_interval=10
        )
        criterion = nn.CrossEntropyLoss(reduction='none')
        tracker = indexed_dataset, loss_tracker

    # Energy tracking
    energy_tracker = EnergyTracker(device=str(device))
    energy_tracker.start()
    start_time = time.time()

    # Training loop
    epoch_results = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        samples_seen = 0

        # Setup data loader for this epoch
        if config['optimizer'] == 'adam':
            loader = train_loader
            phase = "standard"
            keep_frac = 1.0

        elif config['optimizer'] == 'h3':
            # Fixed sampling
            indexed_dataset, loss_tracker = tracker
            sampler = InformationWeightedSampler(
                indexed_dataset,
                loss_tracker,
                keep_frac=config['keep_frac'],
                uniform_mix=config['uniform_mix']
            )
            loader = DataLoader(indexed_dataset, sampler=sampler,
                              batch_size=BATCH_SIZE, num_workers=2)
            phase = "thermodynamic" if epoch >= 2 else "warmup"
            keep_frac = config['keep_frac'] if epoch >= 2 else 1.0

        else:  # h3_adaptive
            indexed_dataset, loss_tracker = tracker

            # Get adaptive schedule
            if config['schedule'] == 'cosine':
                keep_frac, uniform_mix, phase = cosine_annealing_schedule(
                    epoch, NUM_EPOCHS,
                    min_keep_frac=config['min_keep_frac'],
                    warmup_epochs=config['warmup_epochs'],
                    consolidation_epochs=config['consolidation_epochs']
                )
            else:  # linear
                keep_frac, uniform_mix, phase = linear_annealing_schedule(
                    epoch, NUM_EPOCHS,
                    min_keep_frac=config['min_keep_frac'],
                    warmup_epochs=config['warmup_epochs'],
                    consolidation_epochs=config['consolidation_epochs']
                )

            sampler = InformationWeightedSampler(
                indexed_dataset,
                loss_tracker,
                keep_frac=keep_frac,
                uniform_mix=uniform_mix
            )
            loader = DataLoader(indexed_dataset, sampler=sampler,
                              batch_size=BATCH_SIZE, num_workers=2)

        # Training epoch
        for batch_data in loader:
            if len(batch_data) == 3:
                data, target, indices = batch_data
            else:
                data, target = batch_data
                indices = None

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(data)

            if config['optimizer'] == 'adam':
                loss = criterion(outputs, target)
            else:
                loss_vec = criterion(outputs, target)
                loss = loss_vec.mean()
                if indices is not None:
                    loss_tracker.update(indices, loss_vec.detach())

            energy_tracker.log_loss(loss.item())

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(data)
            samples_seen += len(data)

        # Evaluate
        test_acc = evaluate(model, test_loader, device)
        avg_loss = running_loss / samples_seen
        stats = energy_tracker.get_current_stats()

        epoch_results.append({
            'epoch': epoch + 1,
            'phase': phase,
            'keep_frac': keep_frac,
            'loss': avg_loss,
            'test_acc': test_acc,
            'efficiency': stats['current_efficiency_bits_per_j']
        })

        print(f"Epoch {epoch+1:2d} [{phase:>13} | Frac: {keep_frac*100:4.1f}%]: "
              f"Loss={avg_loss:.4f}, Acc={test_acc:.2f}%, "
              f"η={stats['current_efficiency_bits_per_j']:.6f} bits/J")

    # Final results
    elapsed = time.time() - start_time
    energy_results = energy_tracker.stop()

    results = {
        'config': config,
        'time_seconds': elapsed,
        'final_accuracy': test_acc,
        'final_loss': avg_loss,
        'total_energy_j': energy_results['total_energy_j'],
        'efficiency_bits_per_j': energy_results['efficiency_bits_per_j'],
        'epoch_history': epoch_results
    }

    print(f"\n{'Final Results':^80}")
    print(f"{'─'*80}")
    print(f"  Time:       {elapsed:.2f}s ({elapsed/60:.1f} min)")
    print(f"  Accuracy:   {test_acc:.2f}%")
    print(f"  Energy:     {energy_results['total_energy_j']:.2f} J")
    print(f"  Efficiency: {energy_results['efficiency_bits_per_j']:.6f} bits/J")
    print(f"{'─'*80}\n\n")

    return results

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================
print("\n" + "="*80)
print(f"🚀 STARTING {len(CONFIGS)} EXPERIMENTS")
print("="*80)
print(f"Configuration: {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}, lr {LEARNING_RATE}")
print(f"Expected runtime: ~{len(CONFIGS) * 20} minutes")
print("="*80 + "\n")

all_results = []

for i, config in enumerate(CONFIGS):
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT {i+1}/{len(CONFIGS)}")
    print(f"{'='*80}\n")

    result = train_model(config)
    all_results.append(result)

    # Save intermediate results
    Path('/home/user/EMSTI/experiments').mkdir(exist_ok=True)
    with open('/home/user/EMSTI/experiments/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"✅ Results saved to: experiments/results.json\n")

# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("📊 COMPARATIVE ANALYSIS")
print("="*80 + "\n")

# Find baseline
baseline = next(r for r in all_results if r['config']['name'] == 'baseline_adam')

print(f"{'Configuration':<35} {'Time (min)':<12} {'Acc (%)':<10} {'vs Baseline':<15} {'Efficiency':<15}")
print("─"*100)

for result in all_results:
    time_min = result['time_seconds'] / 60
    acc = result['final_accuracy']
    acc_diff = acc - baseline['final_accuracy']
    eff = result['efficiency_bits_per_j']

    if result['config']['name'] == 'baseline_adam':
        vs_baseline = "BASELINE"
    else:
        time_pct = (1 - result['time_seconds'] / baseline['time_seconds']) * 100
        vs_baseline = f"{time_pct:+.1f}% time"

    print(f"{result['config']['name']:<35} {time_min:<12.1f} {acc:<10.2f} {vs_baseline:<15} {eff:<.6f}")

print("\n" + "="*80)
print("🏆 BEST CONFIGURATION")
print("="*80 + "\n")

# Find best by accuracy
best_by_acc = max((r for r in all_results if r['config']['optimizer'] != 'adam'),
                  key=lambda x: x['final_accuracy'])

print(f"Best Accuracy: {best_by_acc['config']['name']}")
print(f"  Accuracy:   {best_by_acc['final_accuracy']:.2f}%")
print(f"  vs Baseline: {best_by_acc['final_accuracy'] - baseline['final_accuracy']:+.2f}pp")
print(f"  Speedup:    {(1 - best_by_acc['time_seconds']/baseline['time_seconds'])*100:+.1f}%")
print(f"  Efficiency: {(best_by_acc['efficiency_bits_per_j']/baseline['efficiency_bits_per_j']-1)*100:+.1f}%")

# Decision criteria
print("\n" + "="*80)
print("🎯 INTEGRATION DECISION")
print("="*80 + "\n")

acc_gap = best_by_acc['final_accuracy'] - baseline['final_accuracy']
speedup = (1 - best_by_acc['time_seconds']/baseline['time_seconds']) * 100
eff_gain = (best_by_acc['efficiency_bits_per_j']/baseline['efficiency_bits_per_j'] - 1) * 100

if acc_gap >= -0.5 and speedup >= 5 and eff_gain >= 10:
    decision = "✅ GREEN LIGHT"
    action = "INTEGRATE into main package and publish v0.2.0"
elif acc_gap >= -1.0 and speedup >= 3 and eff_gain >= 8:
    decision = "⚠️  YELLOW LIGHT"
    action = "Document as experimental feature, don't make default"
else:
    decision = "❌ RED LIGHT"
    action = "Keep as research code, don't integrate"

print(f"Decision: {decision}")
print(f"Action:   {action}")
print(f"\nCriteria:")
print(f"  Accuracy gap: {acc_gap:+.2f}pp (target: ≥-0.5pp) {'✅' if acc_gap >= -0.5 else '❌'}")
print(f"  Speedup:      {speedup:+.1f}% (target: ≥5%) {'✅' if speedup >= 5 else '❌'}")
print(f"  Eff gain:     {eff_gain:+.1f}% (target: ≥10%) {'✅' if eff_gain >= 10 else '❌'}")

print("\n✅ All results saved to: /home/user/EMSTI/experiments/results.json")
print("🎨 Run python experiments/analyze_results.py to generate visualizations")
