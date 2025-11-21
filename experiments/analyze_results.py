"""
Analyze H3 Optimization Experiment Results
Generates comprehensive visualizations and statistical analysis
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================================
# LOAD RESULTS
# ============================================================================
results_file = Path('/home/user/EMSTI/experiments/results.json')

if not results_file.exists():
    print("❌ Error: results.json not found!")
    print("Run cifar10_optimization.py first to generate results.")
    exit(1)

with open(results_file, 'r') as f:
    all_results = json.load(f)

print("=" * 80)
print("📊 H3 OPTIMIZATION ANALYSIS")
print("=" * 80)
print(f"\nLoaded {len(all_results)} experiment results\n")

# ============================================================================
# SETUP PLOTTING
# ============================================================================
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
fig = plt.figure(figsize=(16, 12))

# Color scheme
colors = {
    'baseline_adam': '#1f77b4',
    'h3_fixed_conservative': '#ff7f0e',
    'h3_adaptive_cosine': '#2ca02c',
    'h3_adaptive_cosine_aggressive': '#d62728',
    'h3_adaptive_linear': '#9467bd'
}

# ============================================================================
# PLOT 1: ACCURACY PROGRESSION
# ============================================================================
ax1 = plt.subplot(2, 2, 1)

for result in all_results:
    name = result['config']['name']
    epochs = [e['epoch'] for e in result['epoch_history']]
    accs = [e['test_acc'] for e in result['epoch_history']]

    ax1.plot(epochs, accs,
             label=name.replace('_', ' ').title(),
             color=colors.get(name, '#333333'),
             linewidth=2.5,
             marker='o' if name == 'baseline_adam' else 's',
             markersize=4)

ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Test Accuracy Progression', fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=9, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([75, 85])

# ============================================================================
# PLOT 2: EFFICIENCY OVER TIME
# ============================================================================
ax2 = plt.subplot(2, 2, 2)

for result in all_results:
    name = result['config']['name']
    epochs = [e['epoch'] for e in result['epoch_history']]
    effs = [e['efficiency'] for e in result['epoch_history']]

    ax2.plot(epochs, effs,
             label=name.replace('_', ' ').title(),
             color=colors.get(name, '#333333'),
             linewidth=2.5,
             marker='o' if name == 'baseline_adam' else 's',
             markersize=4)

ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Efficiency (bits/J)', fontsize=12, fontweight='bold')
ax2.set_title('Energy Efficiency Over Time', fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=9, loc='best')
ax2.grid(True, alpha=0.3)
ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# ============================================================================
# PLOT 3: FINAL COMPARISON BAR CHART
# ============================================================================
ax3 = plt.subplot(2, 2, 3)

names = [r['config']['name'].replace('_', '\n') for r in all_results]
accuracies = [r['final_accuracy'] for r in all_results]
bar_colors = [colors.get(r['config']['name'], '#333333') for r in all_results]

bars = ax3.bar(range(len(names)), accuracies, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.2f}%',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3.set_ylabel('Final Test Accuracy (%)', fontsize=12, fontweight='bold')
ax3.set_title('Final Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
ax3.set_xticks(range(len(names)))
ax3.set_xticklabels(names, fontsize=9, rotation=0)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim([75, 85])

# Add horizontal line at target (baseline - 0.5pp)
baseline_acc = next(r['final_accuracy'] for r in all_results if r['config']['name'] == 'baseline_adam')
target_acc = baseline_acc - 0.5
ax3.axhline(y=target_acc, color='red', linestyle='--', linewidth=2, label=f'Target: {target_acc:.2f}%', alpha=0.7)
ax3.axhline(y=baseline_acc, color='blue', linestyle='--', linewidth=2, label=f'Baseline: {baseline_acc:.2f}%', alpha=0.7)
ax3.legend(fontsize=9, loc='lower right')

# ============================================================================
# PLOT 4: EFFICIENCY VS ACCURACY SCATTER
# ============================================================================
ax4 = plt.subplot(2, 2, 4)

baseline = next(r for r in all_results if r['config']['name'] == 'baseline_adam')
baseline_time = baseline['time_seconds']

for result in all_results:
    name = result['config']['name']
    acc = result['final_accuracy']
    eff = result['efficiency_bits_per_j']
    speedup = (1 - result['time_seconds'] / baseline_time) * 100

    marker_size = 100 + speedup * 10  # Size represents speedup

    ax4.scatter(acc, eff,
                s=marker_size,
                color=colors.get(name, '#333333'),
                alpha=0.6,
                edgecolors='black',
                linewidths=2,
                label=f"{name.replace('_', ' ').title()}\n({speedup:+.1f}% time)")

ax4.set_xlabel('Final Accuracy (%)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Efficiency (bits/J)', fontsize=12, fontweight='bold')
ax4.set_title('Efficiency vs Accuracy Trade-off\n(marker size = speedup)',
              fontsize=14, fontweight='bold', pad=15)
ax4.legend(fontsize=8, loc='best', framealpha=0.9)
ax4.grid(True, alpha=0.3)
ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Add target regions
ax4.axvline(x=target_acc, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
ax4.axhline(y=baseline['efficiency_bits_per_j'] * 1.10, color='green', linestyle='--', linewidth=1.5, alpha=0.5)

plt.tight_layout()
plt.savefig('/home/user/EMSTI/experiments/optimization_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Saved visualization: experiments/optimization_analysis.png\n")

# ============================================================================
# STATISTICAL SUMMARY TABLE
# ============================================================================
print("=" * 80)
print("📊 DETAILED RESULTS SUMMARY")
print("=" * 80 + "\n")

# Find baseline
baseline = next(r for r in all_results if r['config']['name'] == 'baseline_adam')

print(f"{'Configuration':<35} {'Time':<10} {'Accuracy':<12} {'Δ Acc':<10} {'Speedup':<12} {'Eff Gain':<12}")
print("─" * 105)

for result in all_results:
    name = result['config']['name']
    time_min = result['time_seconds'] / 60
    acc = result['final_accuracy']
    acc_delta = acc - baseline['final_accuracy']

    if name == 'baseline_adam':
        speedup_str = "BASELINE"
        eff_gain_str = "BASELINE"
    else:
        speedup = (1 - result['time_seconds'] / baseline['time_seconds']) * 100
        eff_gain = (result['efficiency_bits_per_j'] / baseline['efficiency_bits_per_j'] - 1) * 100
        speedup_str = f"{speedup:+.1f}%"
        eff_gain_str = f"{eff_gain:+.1f}%"

    print(f"{name:<35} {time_min:<10.1f} {acc:<12.2f} {acc_delta:+10.2f}pp {speedup_str:<12} {eff_gain_str:<12}")

# ============================================================================
# BEST CONFIGURATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("🏆 BEST CONFIGURATION")
print("=" * 80 + "\n")

# Find best by accuracy (excluding baseline)
h3_results = [r for r in all_results if r['config']['optimizer'] != 'adam']
best_by_acc = max(h3_results, key=lambda x: x['final_accuracy'])

print(f"Best Configuration: {best_by_acc['config']['name']}")
print(f"  Description:  {best_by_acc['config']['description']}")
print(f"  Final Accuracy: {best_by_acc['final_accuracy']:.2f}%")
print(f"  vs Baseline:    {best_by_acc['final_accuracy'] - baseline['final_accuracy']:+.2f}pp")
print(f"  Training Time:  {best_by_acc['time_seconds']/60:.1f} min")
print(f"  Speedup:        {(1 - best_by_acc['time_seconds']/baseline['time_seconds'])*100:+.1f}%")
print(f"  Efficiency:     {best_by_acc['efficiency_bits_per_j']:.6f} bits/J")
print(f"  Eff Gain:       {(best_by_acc['efficiency_bits_per_j']/baseline['efficiency_bits_per_j']-1)*100:+.1f}%")

# ============================================================================
# INTEGRATION DECISION
# ============================================================================
print("\n" + "=" * 80)
print("🎯 INTEGRATION DECISION")
print("=" * 80 + "\n")

acc_gap = best_by_acc['final_accuracy'] - baseline['final_accuracy']
speedup = (1 - best_by_acc['time_seconds']/baseline['time_seconds']) * 100
eff_gain = (best_by_acc['efficiency_bits_per_j']/baseline['efficiency_bits_per_j'] - 1) * 100

# Decision criteria
if acc_gap >= -0.5 and speedup >= 5 and eff_gain >= 10:
    decision = "✅ GREEN LIGHT"
    action = "INTEGRATE into main package and publish v0.2.0"
    recommendation = """
    Recommended Actions:
    1. Update h3/__init__.py to export adaptive schedules
    2. Add example to examples/cifar10_adaptive.py
    3. Update README.md with new CIFAR-10 results
    4. Bump version to 0.2.0 in setup.py
    5. Rebuild and publish to PyPI: python setup.py sdist && twine upload dist/*
    6. Update docs/index.html with new results
    """
elif acc_gap >= -1.0 and speedup >= 3 and eff_gain >= 8:
    decision = "⚠️  YELLOW LIGHT"
    action = "Document as experimental feature, don't make default"
    recommendation = """
    Recommended Actions:
    1. Keep adaptive schedules in h3/adaptive.py
    2. Document in README as "Experimental: Adaptive Annealing"
    3. Add warning about accuracy trade-offs
    4. Consider further tuning of hyperparameters
    5. Run longer experiments (50+ epochs) to verify convergence
    """
else:
    decision = "❌ RED LIGHT"
    action = "Keep as research code, don't integrate"
    recommendation = """
    Recommended Actions:
    1. Keep code in experiments/ directory for research
    2. Do NOT integrate into main package
    3. Investigate alternative approaches:
       - Different annealing schedules
       - Hybrid fixed/adaptive strategies
       - Task-specific configurations
    4. Consider accuracy recovery mechanisms
    """

print(f"Decision: {decision}")
print(f"Action:   {action}\n")
print("Criteria Evaluation:")
print(f"  Accuracy gap: {acc_gap:+.2f}pp (target: ≥-0.5pp) {'✅' if acc_gap >= -0.5 else '❌'}")
print(f"  Speedup:      {speedup:+.1f}% (target: ≥5%) {'✅' if speedup >= 5 else '❌'}")
print(f"  Eff gain:     {eff_gain:+.1f}% (target: ≥10%) {'✅' if eff_gain >= 10 else '❌'}")
print(recommendation)

# ============================================================================
# PHASE ANALYSIS (for adaptive schedules)
# ============================================================================
print("\n" + "=" * 80)
print("📈 PHASE ANALYSIS (Adaptive Schedules)")
print("=" * 80 + "\n")

for result in all_results:
    if 'adaptive' in result['config']['name']:
        name = result['config']['name']
        print(f"\n{name}:")
        print(f"{'─' * 80}")
        print(f"{'Epoch':<8} {'Phase':<15} {'Keep Frac':<12} {'Loss':<10} {'Accuracy':<10}")
        print(f"{'─' * 80}")

        for epoch_data in result['epoch_history'][-10:]:  # Last 10 epochs
            print(f"{epoch_data['epoch']:<8} "
                  f"{epoch_data['phase']:<15} "
                  f"{epoch_data['keep_frac']*100:>10.1f}% "
                  f"{epoch_data['loss']:>9.4f} "
                  f"{epoch_data['test_acc']:>9.2f}%")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print("  📊 experiments/optimization_analysis.png")
print("  📄 experiments/results.json")
print("\nNext steps depend on integration decision above.")
