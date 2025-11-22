# H3 Optimizer Package
from h3.optimizer import H3Optimizer
from h3.sampler import LossTracker, InformationWeightedSampler, IndexedDataset
from h3.energy_tracker import EnergyTracker
from h3.hooks import ThermoAuditLogger, create_baseline_logger
from h3.presets import (
    MNISTFast,
    CIFAR10Safe,
    EdgeDevice,
    h3_mnist_fast,
    h3_cifar_safe,
    h3_edge,
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

__version__ = "0.2.0-dev"

__all__ = [
    "H3Optimizer",
    "LossTracker",
    "InformationWeightedSampler",
    "IndexedDataset",
    "EnergyTracker",
    "ThermoAuditLogger",
    "create_baseline_logger",
    "MNISTFast",
    "CIFAR10Safe",
    "EdgeDevice",
    "h3_mnist_fast",
    "h3_cifar_safe",
    "h3_edge",
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
]
