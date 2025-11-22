"""
H3Profiler: Thermodynamic Training Profiler
Pure measurement without intervention - works with ANY optimizer
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from .energy_tracker import EnergyTracker


class H3Profiler:
    """
    Non-invasive thermodynamic profiler for neural network training.

    Works with ANY optimizer (Adam, SGD, AdamW, etc.).
    Just wraps your training loop to measure energy and information flow.

    Example:
        >>> profiler = H3Profiler(device='cuda', name="mnist_adam")
        >>> profiler.start()
        >>>
        >>> # Your normal training loop
        >>> for epoch in range(20):
        ...     for batch in train_loader:
        ...         # ... normal training with Adam/SGD ...
        ...         profiler.log_batch(loss=loss.item())
        ...
        ...     acc = evaluate(model)
        ...     profiler.log_epoch(accuracy=acc)
        >>>
        >>> # Get insights
        >>> results = profiler.stop()
        >>> print(profiler.get_report())
        >>>
        >>> # Export for analysis
        >>> profiler.export_csv("./profiles/mnist_adam.csv")
    """

    def __init__(
        self,
        device: str = 'cuda',
        name: str = "training_run",
        baseline_accuracy: Optional[float] = None
    ):
        """
        Initialize profiler.

        Args:
            device: Device to monitor ('cuda', 'mps', 'cpu')
            name: Name for this profiling run
            baseline_accuracy: Expected final accuracy (for waste detection)
        """
        self.name = name
        self.device = device
        self.baseline_accuracy = baseline_accuracy

        # Energy tracking
        self.tracker = EnergyTracker(device=device)

        # State
        self.running = False
        self.epoch_data = []
        self.batch_losses = []
        self.initial_loss = None

        # Cumulative metrics
        self.total_energy_j = 0.0
        self.total_info_bits = 0.0
        self.start_time = None

    def start(self):
        """Start profiling session."""
        self.tracker.start()
        self.running = True
        self.start_time = time.time()
        print(f"🔬 H3Profiler started: {self.name}")
        print(f"   Device: {self.device}")
        print(f"   Monitoring: Energy, Information Gain, Efficiency η")
        print()

    def log_batch(self, loss: float):
        """
        Log batch-level metrics.

        Args:
            loss: Training loss for this batch
        """
        if not self.running:
            return

        self.batch_losses.append(loss)
        self.tracker.log_loss(loss)

        # Store initial loss (first batch)
        if self.initial_loss is None:
            self.initial_loss = loss

    def log_epoch(
        self,
        accuracy: float,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        extra_metrics: Optional[Dict] = None
    ):
        """
        Log end-of-epoch metrics.

        Args:
            accuracy: Validation accuracy (%)
            val_loss: Validation loss (optional)
            learning_rate: Current learning rate (optional)
            extra_metrics: Additional metrics to track (optional)
        """
        if not self.running:
            return

        # Get energy statistics
        stats = self.tracker.get_current_stats()

        # Calculate information gain for this epoch
        if len(self.batch_losses) > 0:
            avg_loss = np.mean(self.batch_losses)
            prev_loss = self.epoch_data[-1]['avg_train_loss'] if self.epoch_data else self.initial_loss

            # Information gain: reduction in uncertainty
            # ΔI ≈ log(L_prev / L_current)
            if prev_loss > 0 and avg_loss > 0:
                info_gain = np.log(prev_loss / avg_loss)
                # Clamp to reasonable range
                info_gain = max(0, min(info_gain, 10.0))
            else:
                info_gain = 0.0
        else:
            avg_loss = 0.0
            info_gain = 0.0

        # Calculate epoch energy
        epoch_energy = stats['total_energy_j'] - self.total_energy_j
        self.total_energy_j = stats['total_energy_j']
        self.total_info_bits += info_gain

        # Calculate efficiency
        efficiency = info_gain / epoch_energy if epoch_energy > 0 else 0.0

        # Build epoch record
        epoch_record = {
            'epoch': len(self.epoch_data) + 1,
            'accuracy': accuracy,
            'val_loss': val_loss,
            'avg_train_loss': avg_loss,
            'loss_std': np.std(self.batch_losses) if len(self.batch_losses) > 1 else 0.0,
            'energy_j': epoch_energy,
            'cumulative_energy_j': self.total_energy_j,
            'info_gain_bits': info_gain,
            'cumulative_info_bits': self.total_info_bits,
            'efficiency_bits_per_j': efficiency,
            'learning_rate': learning_rate,
        }

        # Add extra metrics
        if extra_metrics:
            epoch_record.update(extra_metrics)

        self.epoch_data.append(epoch_record)

        # Print progress
        print(f"Epoch {epoch_record['epoch']:2d}: "
              f"Acc={accuracy:5.2f}%, "
              f"Loss={avg_loss:.4f}, "
              f"Energy={epoch_energy/1000:5.2f}kJ, "
              f"η={efficiency:.6f}")

        # Reset batch losses
        self.batch_losses = []

    def stop(self) -> Dict:
        """
        Stop profiling and return results.

        Returns:
            Dict with summary statistics and full epoch data
        """
        if not self.running:
            return {}

        self.tracker.stop()
        self.running = False
        elapsed = time.time() - self.start_time

        # Calculate summary statistics
        avg_efficiency = self.total_info_bits / self.total_energy_j if self.total_energy_j > 0 else 0

        results = {
            'name': self.name,
            'device': self.device,
            'total_time_s': elapsed,
            'total_time_min': elapsed / 60,
            'total_energy_j': self.total_energy_j,
            'total_energy_kj': self.total_energy_j / 1000,
            'total_info_bits': self.total_info_bits,
            'avg_efficiency_bits_per_j': avg_efficiency,
            'epochs': len(self.epoch_data),
            'epoch_data': self.epoch_data,
            'final_accuracy': self.epoch_data[-1]['accuracy'] if self.epoch_data else 0,
            'initial_loss': self.initial_loss,
            'final_loss': self.epoch_data[-1]['avg_train_loss'] if self.epoch_data else 0,
        }

        print()
        print("="*70)
        print(f"🔬 H3Profiler stopped: {self.name}")
        print("="*70)
        print(f"Total time:      {elapsed/60:.1f} min")
        print(f"Total energy:    {self.total_energy_j/1000:.2f} kJ")
        print(f"Total info gain: {self.total_info_bits:.2f} bits")
        print(f"Avg efficiency:  {avg_efficiency:.6f} bits/J")
        print(f"Final accuracy:  {results['final_accuracy']:.2f}%")
        print("="*70)
        print()

        return results

    def get_report(self, verbose: bool = True) -> str:
        """
        Generate human-readable analysis report.

        Args:
            verbose: Include detailed phase-by-phase breakdown

        Returns:
            Formatted report string
        """
        if len(self.epoch_data) == 0:
            return "⚠️  No data logged yet. Call log_epoch() during training."

        lines = []
        lines.append("=" * 80)
        lines.append("📊 H3 THERMODYNAMIC TRAINING PROFILE")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Run:          {self.name}")
        lines.append(f"Device:       {self.device}")
        lines.append(f"Total Epochs: {len(self.epoch_data)}")
        lines.append("")

        # === ENERGY & EFFICIENCY ===
        lines.append("### 🔋 ENERGY & EFFICIENCY")
        lines.append(f"  Total Energy:       {self.total_energy_j/1000:8.2f} kJ")
        lines.append(f"  Total Info Gain:    {self.total_info_bits:8.2f} bits")

        avg_eta = self.total_info_bits / self.total_energy_j if self.total_energy_j > 0 else 0
        lines.append(f"  Average Efficiency: {avg_eta:8.6f} bits/J")
        lines.append("")

        # === EFFICIENCY BY PHASE ===
        if verbose:
            lines.append("### 📈 EFFICIENCY BY TRAINING PHASE")
            n = len(self.epoch_data)

            # Divide into thirds
            early_end = max(1, n // 3)
            mid_end = max(2, 2 * n // 3)

            early = self.epoch_data[:early_end]
            mid = self.epoch_data[early_end:mid_end]
            late = self.epoch_data[mid_end:]

            for phase_name, phase_data in [("Early", early), ("Middle", mid), ("Late", late)]:
                if len(phase_data) > 0:
                    avg_eta_phase = np.mean([e['efficiency_bits_per_j'] for e in phase_data])
                    avg_energy_phase = np.mean([e['energy_j'] for e in phase_data])
                    epoch_range = f"{phase_data[0]['epoch']}-{phase_data[-1]['epoch']}"

                    lines.append(f"  {phase_name:8s} (epochs {epoch_range:6s}): "
                               f"η = {avg_eta_phase:.6f} bits/J, "
                               f"avg energy = {avg_energy_phase/1000:.2f} kJ")
            lines.append("")

        # === DIMINISHING RETURNS ===
        optimal_epoch, waste_pct = self._find_optimal_stop()

        lines.append("### 📉 DIMINISHING RETURNS ANALYSIS")
        lines.append(f"  Optimal Stop Point:  Epoch {optimal_epoch}")

        if waste_pct > 5:
            wasted_energy = self.total_energy_j * (waste_pct / 100)
            lines.append(f"  Energy Wasted:       {wasted_energy/1000:.2f} kJ ({waste_pct:.1f}%)")
            lines.append(f"  💡 Training beyond epoch {optimal_epoch} provides minimal benefit")
        else:
            lines.append(f"  ✅ Training duration well-calibrated (< 5% waste)")
        lines.append("")

        # === CONVERGENCE ===
        lines.append("### 🎯 CONVERGENCE METRICS")

        if len(self.epoch_data) >= 5:
            # Recent accuracy improvement
            recent_acc_delta = (self.epoch_data[-1]['accuracy'] -
                              self.epoch_data[-5]['accuracy'])
            lines.append(f"  Last 5 epochs Δacc:  {recent_acc_delta:+.2f}pp")

            # Loss reduction
            initial_loss = self.epoch_data[0]['avg_train_loss']
            final_loss = self.epoch_data[-1]['avg_train_loss']
            total_reduction = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
            lines.append(f"  Total loss reduction: {total_reduction:.1f}%")

        lines.append("")

        # === RECOMMENDATIONS ===
        lines.append("### 💡 RECOMMENDATIONS")

        # Early stopping
        if waste_pct > 20:
            lines.append(f"  ⚠️  HIGH WASTE: Use early stopping at epoch {optimal_epoch}")
            lines.append(f"      → Save {waste_pct:.0f}% energy with minimal accuracy loss")
        elif waste_pct > 10:
            lines.append(f"  ℹ️  MODERATE WASTE: Consider stopping at epoch {optimal_epoch}")
        else:
            lines.append("  ✅ EFFICIENT: Training duration well-optimized")

        # H3 optimization suggestion
        if avg_eta < 0.0001:
            lines.append("")
            lines.append("  🚀 CONSIDER H3 OPTIMIZATION:")
            lines.append("      Your efficiency η is relatively low.")
            lines.append("      H3's thermodynamic sampling could improve this by 10-20%.")
            lines.append("")
            lines.append("      Try:")
            lines.append("      >>> from h3.presets import h3_mnist_fast")
            lines.append("      >>> optimizer, tracker, dataset = h3_mnist_fast(model.parameters(), train_dataset)")

        lines.append("")

        # === COMPARISON ===
        if self.baseline_accuracy is not None:
            lines.append("### 📊 VS BASELINE")
            final_acc = self.epoch_data[-1]['accuracy']
            acc_delta = final_acc - self.baseline_accuracy
            lines.append(f"  Target accuracy:  {self.baseline_accuracy:.2f}%")
            lines.append(f"  Achieved:         {final_acc:.2f}%")
            lines.append(f"  Delta:            {acc_delta:+.2f}pp")
            lines.append("")

        lines.append("=" * 80)
        lines.append(f"Profile saved: {self.name}")
        lines.append("Use export_csv() to save for further analysis")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _find_optimal_stop(self) -> Tuple[int, float]:
        """
        Find optimal stopping point based on diminishing returns.

        Returns:
            (optimal_epoch, waste_percentage)
        """
        if len(self.epoch_data) < 5:
            return len(self.epoch_data), 0.0

        # Strategy: Find where efficiency drops significantly AND accuracy plateaus
        accs = [e['accuracy'] for e in self.epoch_data]
        etas = [e['efficiency_bits_per_j'] for e in self.epoch_data]

        # Look for the "knee" in the efficiency curve
        for i in range(3, len(self.epoch_data) - 2):
            # Check recent accuracy improvement
            acc_window = accs[i-2:i+3]  # 5-epoch window
            acc_improvement = acc_window[-1] - acc_window[0]

            # Check efficiency drop
            eta_recent = np.mean(etas[i:min(i+3, len(etas))])
            eta_prev = np.mean(etas[max(0, i-3):i])

            if eta_prev > 0:
                eta_drop_pct = (eta_prev - eta_recent) / eta_prev * 100
            else:
                eta_drop_pct = 0

            # Criteria: low improvement + significant efficiency drop
            if acc_improvement < 0.5 and eta_drop_pct > 30:
                # Calculate waste
                wasted_energy = sum(e['energy_j'] for e in self.epoch_data[i:])
                waste_pct = (wasted_energy / self.total_energy_j) * 100
                return i + 1, waste_pct

        # No clear stopping point found
        return len(self.epoch_data), 0.0

    def export_csv(self, path: str):
        """
        Export epoch data to CSV for analysis.

        Args:
            path: Output CSV path
        """
        if len(self.epoch_data) == 0:
            print("⚠️  No data to export")
            return

        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.epoch_data)
        df.to_csv(path, index=False)
        print(f"✅ Exported profile to: {path}")

    def export_json(self, path: str):
        """
        Export full results to JSON.

        Args:
            path: Output JSON path
        """
        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        results = self.stop() if self.running else {
            'name': self.name,
            'epoch_data': self.epoch_data,
            'total_energy_j': self.total_energy_j,
            'total_info_bits': self.total_info_bits,
        }

        with open(path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✅ Exported full results to: {path}")

    def compare_to_baseline(self, baseline_log_path: str) -> str:
        """
        Compare this run to a baseline CSV log.

        Args:
            baseline_log_path: Path to baseline CSV

        Returns:
            Comparison report string
        """
        baseline_df = pd.read_csv(baseline_log_path)

        # Extract metrics
        baseline_energy = baseline_df['cumulative_energy_j'].iloc[-1]
        baseline_acc = baseline_df['accuracy'].iloc[-1]
        baseline_info = baseline_df['cumulative_info_bits'].iloc[-1]

        current_energy = self.total_energy_j
        current_acc = self.epoch_data[-1]['accuracy']
        current_info = self.total_info_bits

        # Calculate deltas
        energy_saved_pct = (1 - current_energy / baseline_energy) * 100 if baseline_energy > 0 else 0
        acc_delta = current_acc - baseline_acc
        info_delta = current_info - baseline_info

        green_score = baseline_energy / current_energy if current_energy > 0 else 1.0

        # Build report
        lines = []
        lines.append("=" * 70)
        lines.append("🔍 COMPARISON TO BASELINE")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Baseline: {Path(baseline_log_path).stem}")
        lines.append(f"Current:  {self.name}")
        lines.append("")
        lines.append("### METRICS")
        lines.append(f"  Accuracy:    {current_acc:6.2f}% vs {baseline_acc:6.2f}% "
                    f"({acc_delta:+.2f}pp)")
        lines.append(f"  Energy:      {current_energy/1000:6.2f} kJ vs {baseline_energy/1000:6.2f} kJ "
                    f"({energy_saved_pct:+.1f}%)")
        lines.append(f"  Info Gain:   {current_info:6.2f} bits vs {baseline_info:6.2f} bits "
                    f"({info_delta:+.2f})")
        lines.append("")
        lines.append(f"  🌱 Green Score: {green_score:.2f}x")
        lines.append("")

        # Verdict
        lines.append("### VERDICT")
        if acc_delta >= -0.5 and energy_saved_pct >= 5:
            lines.append("  ✅ SUCCESS: Comparable accuracy with energy savings")
        elif acc_delta >= -1.0 and energy_saved_pct >= 3:
            lines.append("  ⚠️  ACCEPTABLE: Small accuracy trade-off for efficiency")
        elif acc_delta < -1.5:
            lines.append("  ❌ ACCURACY LOSS: Consider tuning hyperparameters")
        else:
            lines.append("  ℹ️  MIXED: Evaluate trade-offs for your use case")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)
