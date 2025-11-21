"""
Thermo-Audit Logging System
Generates standardized CSV logs for training analysis and optimization
"""

import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
import json


class ThermoAuditLogger:
    """
    Standardized logger for H3 training runs.

    Generates CSV logs compatible with Training Cost Optimizer.
    Tracks: energy, information gain, efficiency, phase, hyperparameters.

    Example:
        >>> logger = ThermoAuditLogger("mnist_run")
        >>> for epoch in range(10):
        ...     # ... training code ...
        ...     logger.log_epoch(
        ...         epoch=epoch,
        ...         phase="thermodynamic",
        ...         keep_frac=0.65,
        ...         uniform_mix=0.2,
        ...         train_loss=0.234,
        ...         val_acc=98.5,
        ...         energy_stats=tracker.get_current_stats()
        ...     )
        >>> print(f"Log saved to: {logger.get_path()}")
    """

    def __init__(
        self,
        run_name: str,
        out_dir: str = "./thermo_logs",
        metadata: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Thermo-Audit logger.

        Args:
            run_name: Identifier for this training run
            out_dir: Directory to save logs (default: ./thermo_logs)
            metadata: Additional metadata (model, dataset, etc.)
            config: Training configuration (epochs, batch_size, lr, etc.)
        """
        self.run_name = run_name
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.csv_path = self.out_dir / f"{run_name}-{timestamp}.csv"
        self.meta_path = self.out_dir / f"{run_name}-{timestamp}.json"

        # Store metadata
        self.metadata = metadata or {}
        self.config = config or {}
        self._save_metadata()

        # Initialize CSV
        with self.csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "step",
                "phase",
                "keep_frac",
                "uniform_mix",
                "train_loss",
                "val_acc",
                "val_loss",
                "total_energy_j",
                "cumulative_energy_j",
                "info_gain_bits",
                "cumulative_info_bits",
                "efficiency_bits_per_j",
                "learning_rate",
                "samples_seen",
            ])

        self.epoch_count = 0
        self.cumulative_energy = 0.0
        self.cumulative_info = 0.0

    def _save_metadata(self):
        """Save metadata and config to JSON file."""
        meta = {
            "run_name": self.run_name,
            "timestamp": datetime.now().isoformat(),
            "metadata": self.metadata,
            "config": self.config,
        }
        with self.meta_path.open("w") as f:
            json.dump(meta, f, indent=2)

    def log_epoch(
        self,
        epoch: int,
        phase: str,
        keep_frac: float,
        uniform_mix: float,
        train_loss: float,
        val_acc: float,
        energy_stats: Dict[str, float],
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        samples_seen: Optional[int] = None,
    ):
        """
        Log metrics for one epoch.

        Args:
            epoch: Current epoch number
            phase: Training phase (warmup/thermodynamic/consolidation)
            keep_frac: Data retention fraction
            uniform_mix: Uniform sampling mix ratio
            train_loss: Training loss
            val_acc: Validation accuracy (%)
            energy_stats: Dict from EnergyTracker.get_current_stats()
            val_loss: Validation loss (optional)
            learning_rate: Current learning rate (optional)
            samples_seen: Number of samples processed (optional)
        """
        # Update cumulatives
        epoch_energy = float(energy_stats.get("total_energy_j", 0.0))
        epoch_info = float(energy_stats.get("info_gain_bits", 0.0))

        self.cumulative_energy += epoch_energy
        self.cumulative_info += epoch_info

        # Calculate step (for TCO compatibility)
        step = self.epoch_count

        with self.csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                step,
                phase,
                keep_frac,
                uniform_mix,
                train_loss,
                val_acc,
                val_loss if val_loss is not None else "",
                epoch_energy,
                self.cumulative_energy,
                epoch_info,
                self.cumulative_info,
                float(energy_stats.get("efficiency_bits_per_j", 0.0)),
                learning_rate if learning_rate is not None else "",
                samples_seen if samples_seen is not None else "",
            ])

        self.epoch_count += 1

    def get_path(self) -> str:
        """Get path to CSV log file."""
        return str(self.csv_path)

    def get_metadata_path(self) -> str:
        """Get path to metadata JSON file."""
        return str(self.meta_path)

    def summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dict with: epochs logged, total energy, total info gain, avg efficiency
        """
        return {
            "epochs_logged": self.epoch_count,
            "total_energy_j": self.cumulative_energy,
            "total_info_bits": self.cumulative_info,
            "avg_efficiency_bits_per_j": (
                self.cumulative_info / self.cumulative_energy
                if self.cumulative_energy > 0
                else 0.0
            ),
            "csv_path": str(self.csv_path),
            "metadata_path": str(self.meta_path),
        }


def create_baseline_logger(
    run_name: str,
    optimizer_name: str = "Adam",
    out_dir: str = "./thermo_logs"
) -> ThermoAuditLogger:
    """
    Create logger for baseline (non-H3) runs.

    For baseline runs, use:
    - phase: "standard"
    - keep_frac: 1.0
    - uniform_mix: 1.0

    Example:
        >>> logger = create_baseline_logger("mnist_adam")
        >>> logger.log_epoch(
        ...     epoch=1, phase="standard", keep_frac=1.0, uniform_mix=1.0,
        ...     train_loss=0.5, val_acc=95.0, energy_stats=tracker.get_current_stats()
        ... )
    """
    return ThermoAuditLogger(
        run_name=run_name,
        out_dir=out_dir,
        metadata={"optimizer": optimizer_name, "type": "baseline"}
    )
