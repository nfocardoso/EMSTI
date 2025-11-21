"""
Preset H3 Configurations
Pre-tuned settings for common datasets and use cases
"""

from typing import Tuple
from .optimizer import H3Optimizer
from .sampler import LossTracker, InformationWeightedSampler, IndexedDataset


class H3Preset:
    """Base class for H3 presets."""

    def __init__(
        self,
        model_params,
        train_dataset,
        lr: float = 1e-3,
        name: str = "custom"
    ):
        self.name = name
        self.lr = lr

        # Setup core components
        self.indexed_dataset = IndexedDataset(train_dataset)
        self.loss_tracker = LossTracker(
            num_samples=len(self.indexed_dataset),
            smoothing=self.get_smoothing()
        )
        self.optimizer = H3Optimizer(
            model_params,
            lr=lr,
            lipschitz_safety=self.get_lipschitz_safety(),
            lipschitz_update_interval=self.get_lipschitz_interval()
        )

    def get_smoothing(self) -> float:
        """Loss tracker smoothing factor."""
        return 0.1

    def get_lipschitz_safety(self) -> float:
        """Lipschitz safety margin."""
        return 0.9

    def get_lipschitz_interval(self) -> int:
        """Lipschitz update interval."""
        return 10

    def get_sampler(self, epoch: int, total_epochs: int) -> Tuple[str, InformationWeightedSampler]:
        """
        Get sampler for current epoch.

        Returns:
            (phase_name, sampler)
        """
        raise NotImplementedError

    def get_components(self):
        """Get all components (optimizer, loss_tracker, indexed_dataset)."""
        return self.optimizer, self.loss_tracker, self.indexed_dataset


class MNISTFast(H3Preset):
    """
    Aggressive preset for highly redundant datasets (MNIST, Fashion-MNIST).

    Strategy:
    - Deep cuts: keep_frac=0.65 (skip 35% of data)
    - Low uniform_mix: 0.2 (focus on hard samples)
    - Short warmup: 1 epoch
    - Short consolidation: 1 epoch

    Expected results:
    - 10-15% speedup
    - 15-20% efficiency gain
    - Accuracy maintained (within 0.2pp)

    Example:
        >>> from h3.presets import MNISTFast
        >>> preset = MNISTFast(model.parameters(), train_dataset)
        >>> optimizer, loss_tracker, indexed_dataset = preset.get_components()
        >>>
        >>> for epoch in range(10):
        ...     phase, sampler = preset.get_sampler(epoch, 10)
        ...     loader = DataLoader(indexed_dataset, sampler=sampler, batch_size=64)
        ...     # ... training loop ...
    """

    def __init__(self, model_params, train_dataset, lr: float = 1e-3):
        super().__init__(model_params, train_dataset, lr, name="mnist_fast")

    def get_smoothing(self) -> float:
        return 0.1

    def get_lipschitz_safety(self) -> float:
        return 0.9

    def get_sampler(self, epoch: int, total_epochs: int) -> Tuple[str, InformationWeightedSampler]:
        # Phase 1: Warmup (1 epoch)
        if epoch < 1:
            return "warmup", InformationWeightedSampler(
                self.indexed_dataset, self.loss_tracker, uniform_mix=1.0
            )

        # Phase 3: Consolidation (last 1 epoch)
        if epoch >= total_epochs - 1:
            return "consolidation", InformationWeightedSampler(
                self.indexed_dataset, self.loss_tracker, uniform_mix=1.0
            )

        # Phase 2: Thermodynamic (middle epochs)
        return "thermodynamic", InformationWeightedSampler(
            self.indexed_dataset,
            self.loss_tracker,
            keep_frac=0.65,
            uniform_mix=0.2
        )


class CIFAR10Safe(H3Preset):
    """
    Conservative preset for complex datasets (CIFAR-10, CIFAR-100).

    Strategy:
    - Moderate cuts: keep_frac=0.55 (skip 45% of data)
    - Higher uniform_mix: 0.3 (more exploration)
    - Longer warmup: 3 epochs
    - Longer consolidation: 3 epochs

    Expected results:
    - 5-8% speedup
    - 8-12% efficiency gain
    - Accuracy trade-off: -0.5 to -1.5pp

    Example:
        >>> from h3.presets import CIFAR10Safe
        >>> preset = CIFAR10Safe(model.parameters(), train_dataset, lr=1e-3)
        >>> optimizer, loss_tracker, indexed_dataset = preset.get_components()
        >>>
        >>> for epoch in range(20):
        ...     phase, sampler = preset.get_sampler(epoch, 20)
        ...     # ... training ...
    """

    def __init__(self, model_params, train_dataset, lr: float = 1e-3):
        super().__init__(model_params, train_dataset, lr, name="cifar10_safe")

    def get_smoothing(self) -> float:
        return 0.2

    def get_lipschitz_safety(self) -> float:
        return 0.95

    def get_lipschitz_interval(self) -> int:
        return 20

    def get_sampler(self, epoch: int, total_epochs: int) -> Tuple[str, InformationWeightedSampler]:
        # Phase 1: Warmup (3 epochs)
        if epoch < 3:
            return "warmup", InformationWeightedSampler(
                self.indexed_dataset, self.loss_tracker, uniform_mix=1.0
            )

        # Phase 3: Consolidation (last 3 epochs)
        if epoch >= total_epochs - 3:
            return "consolidation", InformationWeightedSampler(
                self.indexed_dataset, self.loss_tracker, uniform_mix=1.0
            )

        # Phase 2: Thermodynamic (middle epochs)
        return "thermodynamic", InformationWeightedSampler(
            self.indexed_dataset,
            self.loss_tracker,
            keep_frac=0.55,
            uniform_mix=0.3
        )


class EdgeDevice(H3Preset):
    """
    Ultra-aggressive preset for edge devices (IoT, mobile, embedded).

    Strategy:
    - Very deep cuts: keep_frac=0.40 (skip 60% of data)
    - High uniform_mix: 0.5 (prevent tunnel vision)
    - Accepts larger accuracy trade-off for maximum energy savings

    Expected results:
    - 15-25% speedup
    - 20-30% efficiency gain
    - Accuracy trade-off: -2 to -4pp (acceptable for edge use cases)

    Use when:
    - Energy is severely constrained
    - Slightly lower accuracy is acceptable
    - Fast inference/deployment is critical

    Example:
        >>> from h3.presets import EdgeDevice
        >>> preset = EdgeDevice(model.parameters(), train_dataset)
        >>> # ... training with aggressive energy savings ...
    """

    def __init__(self, model_params, train_dataset, lr: float = 1e-3):
        super().__init__(model_params, train_dataset, lr, name="edge_device")

    def get_smoothing(self) -> float:
        return 0.25  # More smoothing for stability

    def get_lipschitz_safety(self) -> float:
        return 0.95

    def get_sampler(self, epoch: int, total_epochs: int) -> Tuple[str, InformationWeightedSampler]:
        # Warmup (2 epochs)
        if epoch < 2:
            return "warmup", InformationWeightedSampler(
                self.indexed_dataset, self.loss_tracker, uniform_mix=1.0
            )

        # Consolidation (last 2 epochs)
        if epoch >= total_epochs - 2:
            return "consolidation", InformationWeightedSampler(
                self.indexed_dataset, self.loss_tracker, uniform_mix=1.0
            )

        # Thermodynamic (aggressive)
        return "thermodynamic", InformationWeightedSampler(
            self.indexed_dataset,
            self.loss_tracker,
            keep_frac=0.40,
            uniform_mix=0.5  # High mix to avoid starvation
        )


# Convenience factory functions
def h3_mnist_fast(model_params, train_dataset, lr: float = 1e-3) -> Tuple:
    """Quick setup for MNIST-like datasets."""
    preset = MNISTFast(model_params, train_dataset, lr)
    return preset.get_components()


def h3_cifar_safe(model_params, train_dataset, lr: float = 1e-3) -> Tuple:
    """Quick setup for CIFAR-10/100."""
    preset = CIFAR10Safe(model_params, train_dataset, lr)
    return preset.get_components()


def h3_edge(model_params, train_dataset, lr: float = 1e-3) -> Tuple:
    """Quick setup for edge devices (maximum energy savings)."""
    preset = EdgeDevice(model_params, train_dataset, lr)
    return preset.get_components()
