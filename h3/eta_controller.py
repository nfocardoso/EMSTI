"""
η-Controller: Thermodynamic Coherence Control
EMSTI-inspired automatic hyperparameter adjustment

Physical Analog:
- η (efficiency) ~ temperature
- keep_frac ~ compression ratio
- uniform_mix ~ thermal noise
- System self-regulates to maintain target η
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class OperatingMode(Enum):
    """Operating modes with different trade-offs."""
    SAFE = "safe"           # Maximize accuracy (keep_frac ~0.70)
    BALANCED = "balanced"    # Balance speed/accuracy (keep_frac ~0.55)
    GREEN = "green"          # Maximize energy savings (keep_frac ~0.45)
    EXTREME = "extreme"      # Maximum savings, accept accuracy loss (keep_frac ~0.35)


@dataclass
class CoherenceTarget:
    """Target operating regime for η-controller."""
    mode: OperatingMode
    max_accuracy_loss_pp: float  # Maximum tolerable accuracy drop
    min_efficiency_gain_pct: float  # Minimum efficiency target

    # Automatically set bounds based on mode
    keep_frac_min: float = 0.35
    keep_frac_max: float = 0.85
    uniform_mix_min: float = 0.15
    uniform_mix_max: float = 0.60

    def __post_init__(self):
        """Set mode-specific bounds."""
        mode_config = {
            OperatingMode.SAFE: {
                'keep_frac_min': 0.65,
                'keep_frac_max': 0.85,
                'uniform_mix_min': 0.30,
                'uniform_mix_max': 0.50,
            },
            OperatingMode.BALANCED: {
                'keep_frac_min': 0.50,
                'keep_frac_max': 0.70,
                'uniform_mix_min': 0.25,
                'uniform_mix_max': 0.45,
            },
            OperatingMode.GREEN: {
                'keep_frac_min': 0.40,
                'keep_frac_max': 0.60,
                'uniform_mix_min': 0.35,
                'uniform_mix_max': 0.55,
            },
            OperatingMode.EXTREME: {
                'keep_frac_min': 0.30,
                'keep_frac_max': 0.50,
                'uniform_mix_min': 0.45,
                'uniform_mix_max': 0.60,
            }
        }

        if self.mode in mode_config:
            for key, value in mode_config[self.mode].items():
                setattr(self, key, value)


class EtaController:
    """
    Thermodynamic efficiency controller.

    Automatically adjusts H3 hyperparameters to maintain target efficiency
    while respecting accuracy constraints.

    Control Strategy:
    1. Monitor η, accuracy, loss volatility
    2. If accuracy drops too much → RELAX (increase keep_frac, uniform_mix)
    3. If efficiency declining → INTENSIFY (decrease keep_frac)
    4. If loss volatile (starvation) → ADD NOISE (increase uniform_mix)
    5. If plateaued → CONSOLIDATE (return to full data)

    Example:
        >>> target = CoherenceTarget(
        ...     mode=OperatingMode.BALANCED,
        ...     max_accuracy_loss_pp=1.0,
        ...     min_efficiency_gain_pct=10.0
        ... )
        >>> controller = EtaController(target)
        >>>
        >>> for epoch in range(20):
        ...     # Get control parameters
        ...     keep_frac, uniform_mix, phase = controller.control_step()
        ...
        ...     # ... training with these parameters ...
        ...
        ...     # Update controller
        ...     controller.observe(
        ...         eta=efficiency,
        ...         accuracy=val_acc,
        ...         train_loss=avg_loss,
        ...         energy_j=epoch_energy,
        ...         info_bits=info_gain
        ...     )
    """

    def __init__(
        self,
        target: CoherenceTarget,
        baseline_accuracy: Optional[float] = None,
        control_interval: int = 1,
        smoothing: float = 0.3,
        total_epochs: Optional[int] = None
    ):
        """
        Initialize controller.

        Args:
            target: Target operating regime
            baseline_accuracy: Expected baseline accuracy for comparison
            control_interval: Epochs between control adjustments
            smoothing: EMA smoothing factor (0-1)
            total_epochs: Total planned epochs (for phase timing)
        """
        self.target = target
        self.baseline_acc = baseline_accuracy
        self.control_interval = control_interval
        self.smoothing = smoothing
        self.total_epochs = total_epochs

        # State history
        self.history = {
            'eta': [],
            'accuracy': [],
            'loss': [],
            'energy': [],
            'info': [],
            'keep_frac': [],
            'uniform_mix': [],
            'phase': []
        }

        # Current control parameters
        self.keep_frac = self._initial_keep_frac()
        self.uniform_mix = self._initial_uniform_mix()
        self.phase = "warmup"

        # Control state
        self.epoch = 0
        self.warmup_complete = False
        self.thermodynamic_active = False
        self.consolidation_triggered = False

        # EMA trackers
        self.eta_ema = None
        self.acc_ema = None
        self.loss_ema = None

    def _initial_keep_frac(self) -> float:
        """Initial keep_frac based on mode."""
        mode_defaults = {
            OperatingMode.SAFE: 0.70,
            OperatingMode.BALANCED: 0.55,
            OperatingMode.GREEN: 0.45,
            OperatingMode.EXTREME: 0.35
        }
        return mode_defaults.get(self.target.mode, 0.55)

    def _initial_uniform_mix(self) -> float:
        """Initial uniform_mix based on mode."""
        mode_defaults = {
            OperatingMode.SAFE: 0.35,
            OperatingMode.BALANCED: 0.30,
            OperatingMode.GREEN: 0.45,
            OperatingMode.EXTREME: 0.55
        }
        return mode_defaults.get(self.target.mode, 0.30)

    def observe(
        self,
        eta: float,
        accuracy: float,
        train_loss: float,
        energy_j: float,
        info_bits: float
    ):
        """
        Record observations for current epoch.

        Args:
            eta: Efficiency (bits/J) for this epoch
            accuracy: Validation accuracy (%)
            train_loss: Average training loss
            energy_j: Energy consumed this epoch (J)
            info_bits: Information gained this epoch (bits)
        """
        # Update history
        self.history['eta'].append(eta)
        self.history['accuracy'].append(accuracy)
        self.history['loss'].append(train_loss)
        self.history['energy'].append(energy_j)
        self.history['info'].append(info_bits)
        self.history['keep_frac'].append(self.keep_frac)
        self.history['uniform_mix'].append(self.uniform_mix)
        self.history['phase'].append(self.phase)

        # Update EMAs
        if self.eta_ema is None:
            self.eta_ema = eta
            self.acc_ema = accuracy
            self.loss_ema = train_loss
        else:
            self.eta_ema = (1 - self.smoothing) * self.eta_ema + self.smoothing * eta
            self.acc_ema = (1 - self.smoothing) * self.acc_ema + self.smoothing * accuracy
            self.loss_ema = (1 - self.smoothing) * self.loss_ema + self.smoothing * train_loss

        self.epoch += 1

    def control_step(self) -> Tuple[float, float, str]:
        """
        Execute control step to get parameters for next epoch.

        Returns:
            (keep_frac, uniform_mix, phase) for next epoch
        """
        if self.epoch < 2:
            # Need warmup data
            return 1.0, 1.0, "warmup"

        # === PHASE TRANSITIONS ===

        # Check if warmup should end
        if not self.warmup_complete and self.epoch >= 2:
            if self._loss_stabilized():
                self.warmup_complete = True
                self.thermodynamic_active = True
                self.phase = "thermodynamic"
                print(f"   → Entering THERMODYNAMIC phase (epoch {self.epoch + 1})")

        # Check if should consolidate
        if self.thermodynamic_active and not self.consolidation_triggered:
            if self._should_consolidate():
                self.consolidation_triggered = True
                self.thermodynamic_active = False
                self.phase = "consolidation"
                print(f"   → Entering CONSOLIDATION phase (epoch {self.epoch + 1})")

        # === CONTROL LOGIC ===

        if self.phase == "warmup":
            # Full data, high exploration
            return 1.0, 1.0, "warmup"

        elif self.phase == "consolidation":
            # Full data, stabilization
            return 1.0, 1.0, "consolidation"

        else:  # thermodynamic
            # Dynamic adjustment
            self._adjust_parameters()
            return self.keep_frac, self.uniform_mix, "thermodynamic"

    def _adjust_parameters(self):
        """Adjust keep_frac and uniform_mix based on current state."""

        # Get recent measurements
        recent_eta = np.mean(self.history['eta'][-3:]) if len(self.history['eta']) >= 3 else self.eta_ema
        recent_acc = np.mean(self.history['accuracy'][-3:]) if len(self.history['accuracy']) >= 3 else self.acc_ema

        # === ACCURACY CONSTRAINT ===
        if self.baseline_acc is not None:
            acc_delta = recent_acc - self.baseline_acc

            if acc_delta < -self.target.max_accuracy_loss_pp:
                # Accuracy falling too much → RELAX
                adjustment = 0.05
                self.keep_frac = min(
                    self.keep_frac + adjustment,
                    self.target.keep_frac_max
                )
                self.uniform_mix = min(
                    self.uniform_mix + adjustment,
                    self.target.uniform_mix_max
                )
                print(f"   ⚠️  Accuracy constraint violated: Δacc={acc_delta:.2f}pp")
                print(f"      → Relaxing: keep_frac={self.keep_frac:.2f}, mix={self.uniform_mix:.2f}")

        # === EFFICIENCY TREND ===
        if len(self.history['eta']) >= 5:
            # Check if efficiency is declining
            recent_5 = self.history['eta'][-5:]
            eta_trend = np.polyfit(range(5), recent_5, 1)[0]

            if eta_trend < -0.00001:  # Declining efficiency
                # Try to increase efficiency by being more aggressive
                self.keep_frac = max(
                    self.keep_frac - 0.03,
                    self.target.keep_frac_min
                )
                print(f"   📉 Efficiency declining (slope={eta_trend:.6f})")
                print(f"      → Intensifying: keep_frac={self.keep_frac:.2f}")

        # === LOSS VOLATILITY (STARVATION INDICATOR) ===
        if len(self.history['loss']) >= 4:
            recent_losses = self.history['loss'][-4:]
            loss_volatility = np.std(recent_losses) / np.mean(recent_losses) if np.mean(recent_losses) > 0 else 0

            if loss_volatility > 0.2:  # High relative volatility
                # Add noise to prevent tunnel vision
                self.uniform_mix = min(
                    self.uniform_mix + 0.10,
                    self.target.uniform_mix_max
                )
                print(f"   ⚠️  High loss volatility detected: CV={loss_volatility:.2f}")
                print(f"      → Adding noise: mix={self.uniform_mix:.2f}")

        # === BOUNDS ENFORCEMENT ===
        self.keep_frac = np.clip(self.keep_frac, self.target.keep_frac_min, self.target.keep_frac_max)
        self.uniform_mix = np.clip(self.uniform_mix, self.target.uniform_mix_min, self.target.uniform_mix_max)

    def _loss_stabilized(self) -> bool:
        """Check if loss variance is low enough to exit warmup."""
        if len(self.history['loss']) < 3:
            return False

        recent_losses = self.history['loss'][-3:]
        cv = np.std(recent_losses) / np.mean(recent_losses) if np.mean(recent_losses) > 0 else 1.0

        return cv < 0.15  # Coefficient of variation < 15%

    def _should_consolidate(self) -> bool:
        """Check if should enter consolidation phase."""

        # If total_epochs specified, use fixed timing
        if self.total_epochs is not None:
            consolidation_start = max(self.total_epochs - 3, self.epoch + 1)
            if self.epoch >= consolidation_start:
                return True

        # Otherwise, use adaptive detection
        if len(self.history['accuracy']) < 5:
            return False

        # Check if accuracy plateaued
        recent_accs = self.history['accuracy'][-5:]
        acc_improvement = recent_accs[-1] - recent_accs[0]

        # If less than 0.5pp improvement in 5 epochs → consolidate
        if acc_improvement < 0.5:
            return True

        # Also check if we're close to baseline
        if self.baseline_acc is not None:
            recent_acc = recent_accs[-1]
            if recent_acc >= 0.98 * self.baseline_acc:
                return True

        return False

    def get_status(self) -> Dict:
        """Get current controller status."""
        if len(self.history['eta']) == 0:
            return {"status": "initializing"}

        recent_eta = np.mean(self.history['eta'][-3:]) if len(self.history['eta']) >= 3 else 0
        recent_acc = np.mean(self.history['accuracy'][-3:]) if len(self.history['accuracy']) >= 3 else 0

        status = {
            "epoch": self.epoch,
            "phase": self.phase,
            "keep_frac": self.keep_frac,
            "uniform_mix": self.uniform_mix,
            "recent_eta": recent_eta,
            "recent_accuracy": recent_acc,
            "eta_ema": self.eta_ema,
            "acc_ema": self.acc_ema,
            "loss_ema": self.loss_ema,
        }

        # Add constraint violations
        if self.baseline_acc is not None:
            acc_delta = recent_acc - self.baseline_acc
            status["accuracy_delta_pp"] = acc_delta
            status["accuracy_constraint_ok"] = acc_delta >= -self.target.max_accuracy_loss_pp

        return status

    def get_report(self) -> str:
        """Generate human-readable status report."""
        status = self.get_status()

        lines = []
        lines.append("=" * 60)
        lines.append("🎛️  η-CONTROLLER STATUS")
        lines.append("=" * 60)
        lines.append(f"Epoch:        {status['epoch']}")
        lines.append(f"Phase:        {status['phase']}")
        lines.append(f"Mode:         {self.target.mode.value}")
        lines.append("")
        lines.append("### CURRENT PARAMETERS")
        lines.append(f"  keep_frac:    {status['keep_frac']:.2f}")
        lines.append(f"  uniform_mix:  {status['uniform_mix']:.2f}")
        lines.append("")
        lines.append("### RECENT METRICS")
        lines.append(f"  Efficiency η: {status['recent_eta']:.6f} bits/J")
        lines.append(f"  Accuracy:     {status['recent_accuracy']:.2f}%")

        if 'accuracy_delta_pp' in status:
            acc_ok = "✅" if status['accuracy_constraint_ok'] else "❌"
            lines.append(f"  vs Baseline:  {status['accuracy_delta_pp']:+.2f}pp {acc_ok}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


# === CONVENIENCE FUNCTION ===
def create_controlled_h3(
    model_params,
    train_dataset,
    mode: str = "balanced",
    accuracy_tolerance: float = 1.0,
    lr: float = 1e-3,
    total_epochs: Optional[int] = None
):
    """
    Create H3 optimizer with automatic η-control.

    Args:
        model_params: Model parameters to optimize
        train_dataset: Training dataset
        mode: Operating mode ("safe", "balanced", "green", "extreme")
        accuracy_tolerance: Max acceptable accuracy loss (pp)
        lr: Learning rate
        total_epochs: Total planned epochs (for phase timing)

    Returns:
        (optimizer, controller, loss_tracker, indexed_dataset)

    Example:
        >>> opt, ctrl, tracker, dataset = create_controlled_h3(
        ...     model.parameters(),
        ...     train_dataset,
        ...     mode="balanced",
        ...     accuracy_tolerance=1.0,
        ...     total_epochs=20
        ... )
        >>>
        >>> for epoch in range(20):
        ...     # Get current control parameters
        ...     keep_frac, uniform_mix, phase = ctrl.control_step()
        ...
        ...     # Create sampler with current parameters
        ...     sampler = InformationWeightedSampler(
        ...         dataset, tracker,
        ...         keep_frac=keep_frac,
        ...         uniform_mix=uniform_mix
        ...     )
        ...     loader = DataLoader(dataset, sampler=sampler, batch_size=128)
        ...
        ...     # ... training loop ...
        ...
        ...     # Update controller
        ...     ctrl.observe(eta, acc, loss, energy, info)
        ...
        ...     # Print status
        ...     if epoch % 5 == 0:
        ...         print(ctrl.get_report())
    """
    from .optimizer import H3Optimizer
    from .sampler import LossTracker, IndexedDataset

    # Parse mode
    try:
        op_mode = OperatingMode(mode)
    except ValueError:
        raise ValueError(f"Invalid mode: {mode}. Must be one of: safe, balanced, green, extreme")

    # Create target
    target = CoherenceTarget(
        mode=op_mode,
        max_accuracy_loss_pp=accuracy_tolerance,
        min_efficiency_gain_pct=5.0 if op_mode == OperatingMode.SAFE else 10.0
    )

    # Create components
    indexed_dataset = IndexedDataset(train_dataset)
    loss_tracker = LossTracker(num_samples=len(indexed_dataset), smoothing=0.2)
    optimizer = H3Optimizer(model_params, lr=lr, lipschitz_safety=0.95)
    controller = EtaController(target, total_epochs=total_epochs)

    return optimizer, controller, loss_tracker, indexed_dataset
