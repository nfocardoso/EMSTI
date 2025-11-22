"""
AutoH3: Zero-Config H3 Optimization
Complete automation with one-liner setup
"""

from typing import Optional, Dict, Tuple
import torch
from torch.utils.data import DataLoader

from .optimizer import H3Optimizer
from .sampler import LossTracker, InformationWeightedSampler, IndexedDataset
from .energy_tracker import EnergyTracker
from .profiler import H3Profiler
from .eta_controller import EtaController, CoherenceTarget, OperatingMode


class AutoH3:
    """
    Zero-configuration H3 optimization.

    Combines profiler + controller + energy tracking into one simple class.
    Just create and train - everything else is automatic.

    Example:
        >>> auto = AutoH3(
        ...     model.parameters(),
        ...     train_dataset,
        ...     mode="balanced",
        ...     name="mnist_experiment"
        ... )
        >>>
        >>> # Simple training loop
        >>> for epoch in range(20):
        ...     for batch_idx, (data, target) in enumerate(train_loader):
        ...         loss = auto.training_step(
        ...             model, data, target, criterion
        ...         )
        ...
        ...     accuracy = auto.evaluate_epoch(model, test_loader)
        >>>
        >>> # Get comprehensive report
        >>> auto.finish()
    """

    def __init__(
        self,
        model_params,
        train_dataset,
        mode: str = "balanced",
        name: str = "auto_h3_run",
        lr: float = 1e-3,
        device: str = 'cuda',
        total_epochs: Optional[int] = None,
        baseline_accuracy: Optional[float] = None,
        accuracy_tolerance: float = 1.0,
        enable_profiling: bool = True,
        enable_controller: bool = True
    ):
        """
        Initialize AutoH3.

        Args:
            model_params: Model parameters to optimize
            train_dataset: Training dataset
            mode: Operating mode ("safe", "balanced", "green", "extreme")
            name: Name for this run
            lr: Learning rate
            device: Device for energy tracking
            total_epochs: Total planned epochs (for phase timing)
            baseline_accuracy: Expected baseline accuracy
            accuracy_tolerance: Max acceptable accuracy loss (pp)
            enable_profiling: Enable H3Profiler for analysis
            enable_controller: Enable η-Controller for auto-tuning
        """
        self.name = name
        self.mode = mode
        self.device = device
        self.enable_profiling = enable_profiling
        self.enable_controller = enable_controller

        # Parse mode
        try:
            self.op_mode = OperatingMode(mode)
        except ValueError:
            raise ValueError(f"Invalid mode: {mode}. Must be one of: safe, balanced, green, extreme")

        # Create core components
        self.indexed_dataset = IndexedDataset(train_dataset)
        self.loss_tracker = LossTracker(
            num_samples=len(self.indexed_dataset),
            smoothing=0.2
        )
        self.optimizer = H3Optimizer(
            model_params,
            lr=lr,
            lipschitz_safety=0.95
        )

        # Energy tracker
        self.energy_tracker = EnergyTracker(device=device)

        # Controller (if enabled)
        self.controller = None
        if enable_controller:
            target = CoherenceTarget(
                mode=self.op_mode,
                max_accuracy_loss_pp=accuracy_tolerance,
                min_efficiency_gain_pct=5.0 if self.op_mode == OperatingMode.SAFE else 10.0
            )
            self.controller = EtaController(
                target,
                baseline_accuracy=baseline_accuracy,
                total_epochs=total_epochs
            )

        # Profiler (if enabled)
        self.profiler = None
        if enable_profiling:
            self.profiler = H3Profiler(
                device=device,
                name=name,
                baseline_accuracy=baseline_accuracy
            )

        # State
        self.epoch = 0
        self.started = False
        self.current_phase = "warmup"
        self.current_keep_frac = 1.0
        self.current_uniform_mix = 1.0

    def start(self):
        """Start training session."""
        self.energy_tracker.start()
        if self.profiler:
            self.profiler.start()
        self.started = True

        print(f"🚀 AutoH3 started: {self.name}")
        print(f"   Mode: {self.mode}")
        print(f"   Controller: {'✅ Enabled' if self.enable_controller else '❌ Disabled'}")
        print(f"   Profiler: {'✅ Enabled' if self.enable_profiling else '❌ Disabled'}")
        print()

    def get_sampler(self) -> InformationWeightedSampler:
        """
        Get sampler for current epoch.

        Returns:
            Configured InformationWeightedSampler
        """
        if not self.started:
            self.start()

        # Get control parameters
        if self.controller:
            self.current_keep_frac, self.current_uniform_mix, self.current_phase = \
                self.controller.control_step()
        else:
            # Use static parameters based on mode
            if self.epoch < 2:
                self.current_phase = "warmup"
                self.current_keep_frac = 1.0
                self.current_uniform_mix = 1.0
            else:
                self.current_phase = "thermodynamic"
                mode_defaults = {
                    OperatingMode.SAFE: (0.70, 0.35),
                    OperatingMode.BALANCED: (0.55, 0.30),
                    OperatingMode.GREEN: (0.45, 0.45),
                    OperatingMode.EXTREME: (0.35, 0.55)
                }
                self.current_keep_frac, self.current_uniform_mix = mode_defaults.get(
                    self.op_mode, (0.55, 0.30)
                )

        # Create sampler
        return InformationWeightedSampler(
            self.indexed_dataset,
            self.loss_tracker,
            keep_frac=self.current_keep_frac,
            uniform_mix=self.current_uniform_mix
        )

    def training_step(
        self,
        model: torch.nn.Module,
        data: torch.Tensor,
        target: torch.Tensor,
        criterion: torch.nn.Module,
        indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Execute one training step.

        Args:
            model: Neural network model
            data: Input batch
            target: Target labels
            criterion: Loss function (must use reduction='none')
            indices: Data indices (if using sampler)

        Returns:
            Loss tensor (scalar)
        """
        self.optimizer.zero_grad()
        output = model(data)

        # Compute loss
        if hasattr(criterion, 'reduction'):
            original_reduction = criterion.reduction
            criterion.reduction = 'none'
            loss_vec = criterion(output, target)
            criterion.reduction = original_reduction
        else:
            loss_vec = criterion(output, target)

        loss = loss_vec.mean()

        # Update loss tracker
        if indices is not None:
            self.loss_tracker.update(indices, loss_vec.detach())

        # Backprop
        loss.backward()
        self.optimizer.step()

        # Log to profiler
        if self.profiler:
            self.profiler.log_batch(loss.item())

        return loss

    def evaluate_epoch(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        device: Optional[str] = None
    ) -> float:
        """
        Evaluate model at end of epoch.

        Args:
            model: Neural network model
            test_loader: Test data loader
            device: Device for evaluation (default: same as init)

        Returns:
            Accuracy (%)
        """
        if device is None:
            device = self.device

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        accuracy = 100. * correct / total

        # Get energy statistics
        stats = self.energy_tracker.get_current_stats()
        epoch_eta = stats.get('current_efficiency_bits_per_j', 0.0)
        epoch_energy = stats.get('epoch_energy_j', 0.0)
        epoch_info = stats.get('epoch_info_bits', 0.0)

        # Estimate train loss (use last tracked value)
        train_loss = self.loss_tracker.losses.mean().item() if hasattr(self.loss_tracker.losses, 'mean') else 0.0

        # Update controller
        if self.controller:
            self.controller.observe(
                eta=epoch_eta,
                accuracy=accuracy,
                train_loss=train_loss,
                energy_j=epoch_energy,
                info_bits=epoch_info
            )

        # Update profiler
        if self.profiler:
            self.profiler.log_epoch(
                accuracy=accuracy,
                val_loss=None,
                learning_rate=self.optimizer.param_groups[0]['lr'] if hasattr(self.optimizer, 'param_groups') else None
            )

        # Print progress
        print(f"Epoch {self.epoch+1:2d} [{self.current_phase:>13}] "
              f"keep={self.current_keep_frac:.2f}, mix={self.current_uniform_mix:.2f} | "
              f"Acc={accuracy:5.2f}%, η={epoch_eta:.6f}")

        self.epoch += 1
        model.train()

        return accuracy

    def finish(self) -> Dict:
        """
        Finish training and generate reports.

        Returns:
            Dict with comprehensive results
        """
        # Stop trackers
        energy_results = self.energy_tracker.stop()

        profiler_results = {}
        if self.profiler:
            profiler_results = self.profiler.stop()

        controller_status = {}
        if self.controller:
            controller_status = self.controller.get_status()

        # Generate reports
        print("\n" + "="*80)
        print("📊 AUTOH3 FINAL REPORT")
        print("="*80 + "\n")

        if self.profiler:
            print(self.profiler.get_report(verbose=True))

        if self.controller:
            print("\n" + self.controller.get_report())

        # Combined results
        results = {
            'name': self.name,
            'mode': self.mode,
            'epochs': self.epoch,
            'energy': energy_results,
            'profiler': profiler_results,
            'controller': controller_status,
        }

        return results

    def get_loader(self, batch_size: int = 64, **kwargs) -> DataLoader:
        """
        Get configured data loader for current epoch.

        Args:
            batch_size: Batch size
            **kwargs: Additional DataLoader arguments

        Returns:
            DataLoader with appropriate sampler
        """
        sampler = self.get_sampler()
        return DataLoader(
            self.indexed_dataset,
            sampler=sampler,
            batch_size=batch_size,
            **kwargs
        )


# Convenience function
def auto_h3(
    model_params,
    train_dataset,
    mode: str = "balanced",
    **kwargs
) -> AutoH3:
    """
    Quick setup for AutoH3.

    Args:
        model_params: Model parameters
        train_dataset: Training dataset
        mode: Operating mode ("safe", "balanced", "green", "extreme")
        **kwargs: Additional AutoH3 arguments

    Returns:
        Configured AutoH3 instance

    Example:
        >>> auto = auto_h3(model.parameters(), train_dataset, mode="balanced")
        >>> auto.start()
        >>> # ... training ...
        >>> auto.finish()
    """
    return AutoH3(model_params, train_dataset, mode=mode, **kwargs)
