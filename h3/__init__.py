# H3 Optimizer Package
from h3.optimizer import H3Optimizer
from h3.sampler import LossTracker, InformationWeightedSampler, IndexedDataset
from h3.energy_tracker import EnergyTracker
from h3.hooks import ThermoAuditLogger, create_baseline_logger
from h3.presets import (
    h3_cifar10_safe_adamw,
    h3_cifar10_balanced,
    h3_cifar10_green,
    h3_mnist_fast,
    load_preset,
)
from h3.explain import explain_thermo_log, green_score, compare_runs
from h3.profiler import H3Profiler
from h3.eta_controller import (
    EtaController,
    CoherenceTarget,
    OperatingMode,
    create_controlled_h3,
)
from h3.auto import AutoH3, auto_h3
from h3.cli import h3_report_cli

__version__ = "0.2.1"

__all__ = [
    "H3Optimizer",
    "LossTracker",
    "InformationWeightedSampler",
    "IndexedDataset",
    "EnergyTracker",
    "ThermoAuditLogger",
    "create_baseline_logger",
    "h3_cifar10_safe_adamw",
    "h3_cifar10_balanced",
    "h3_cifar10_green",
    "h3_mnist_fast",
    "load_preset",
    "explain_thermo_log",
    "green_score",
    "compare_runs",
    "H3Profiler",
    "EtaController",
    "CoherenceTarget",
    "OperatingMode",
    "create_controlled_h3",
    "AutoH3",
    "auto_h3",
    "h3_report_cli",
]
