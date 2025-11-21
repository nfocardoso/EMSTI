"""
Information-weighted sampler for H3 optimizer.

Implements loss-based sample prioritization from H3 paper Section 5.3.
Samples with higher loss (more information potential) are prioritized to
focus computational resources where learning is most needed.

Reference: H3 paper, Section 5.3, Eq. 5.1
"""

from typing import Optional, Iterator
import torch
from torch.utils.data import Sampler, Dataset


class LossTracker:
    """
    Tracks per-sample losses with exponential smoothing.

    Maintains a smoothed loss estimate for each training sample, updated
    via exponential moving average (Eq. 5.1 from paper):
        L_i^(t) = (1 - β) * L_i^(t-1) + β * L_current

    Args:
        num_samples: Total number of samples in dataset.
        smoothing: Smoothing factor β ∈ [0, 1] (default: 0.1).
            Higher values give more weight to recent losses.
        initial_loss: Initial loss value for unseen samples (default: 1.0).

    Attributes:
        losses: Tensor of shape (num_samples,) containing smoothed losses.
        initialized: Boolean tensor indicating which samples have been seen.

    Example:
        >>> tracker = LossTracker(num_samples=50000, smoothing=0.1)
        >>> # After forward pass:
        >>> batch_losses = torch.tensor([0.5, 0.8, 0.3])
        >>> batch_indices = torch.tensor([0, 15, 42])
        >>> tracker.update(batch_indices, batch_losses)
        >>> weights = tracker.get_weights()
    """

    def __init__(
        self,
        num_samples: int,
        smoothing: float = 0.1,
        initial_loss: float = 1.0,
    ) -> None:
        """Initialize loss tracker."""
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if not 0.0 <= smoothing <= 1.0:
            raise ValueError(f"smoothing must be in [0, 1], got {smoothing}")
        if initial_loss <= 0.0:
            raise ValueError(f"initial_loss must be positive, got {initial_loss}")

        self.num_samples = num_samples
        self.smoothing = smoothing
        self.initial_loss = initial_loss

        # Initialize losses to initial_loss (neutral starting point)
        self.losses = torch.full((num_samples,), initial_loss, dtype=torch.float32)

        # Track which samples have been seen
        self.initialized = torch.zeros(num_samples, dtype=torch.bool)

    def update(self, indices: torch.Tensor, losses: torch.Tensor) -> None:
        """
        Update loss estimates for given samples.

        Applies exponential smoothing (Eq. 5.1):
        - For new samples: L_i = loss (direct assignment)
        - For seen samples: L_i = (1-β)*L_old + β*L_new

        Args:
            indices: Tensor of sample indices, shape (batch_size,).
            losses: Tensor of loss values, shape (batch_size,).
                Must be non-negative.

        Raises:
            ValueError: If indices or losses have invalid shapes or values.
        """
        if indices.dim() != 1:
            raise ValueError(f"indices must be 1D, got shape {indices.shape}")
        if losses.dim() != 1:
            raise ValueError(f"losses must be 1D, got shape {losses.shape}")
        if len(indices) != len(losses):
            raise ValueError(
                f"indices and losses must have same length, "
                f"got {len(indices)} and {len(losses)}"
            )
        if torch.any(losses < 0):
            raise ValueError("losses must be non-negative")
        if torch.any(indices < 0) or torch.any(indices >= self.num_samples):
            raise ValueError(
                f"indices must be in [0, {self.num_samples}), "
                f"got range [{indices.min()}, {indices.max()}]"
            )

        # Move to same device as internal state
        indices = indices.to(self.losses.device)
        losses = losses.to(self.losses.device)

        # Determine which samples are being seen for first time
        first_time = ~self.initialized[indices]

        # For first-time samples: direct assignment
        if first_time.any():
            first_indices = indices[first_time]
            first_losses = losses[first_time]
            self.losses[first_indices] = first_losses
            self.initialized[first_indices] = True

        # For previously seen samples: exponential smoothing (Eq. 5.1)
        seen_before = self.initialized[indices]
        if seen_before.any():
            seen_indices = indices[seen_before]
            seen_losses = losses[seen_before]
            old_losses = self.losses[seen_indices]

            # L_i^(t) = (1 - β) * L_i^(t-1) + β * L_current
            self.losses[seen_indices] = (
                (1 - self.smoothing) * old_losses + self.smoothing * seen_losses
            )

    def get_weights(
        self,
        cap_factor: float = 3.0,
        min_weight: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute normalized sampling weights from loss estimates.

        Applies capping to prevent outlier domination and ensures minimum weight
        for exploration.

        Args:
            cap_factor: Maximum weight as multiple of mean loss (default: 3.0).
                Weights are capped at cap_factor * mean(losses).
            min_weight: Minimum weight for any sample (default: 0.1).
                Ensures all samples have non-zero probability.

        Returns:
            weights: Normalized sampling weights, shape (num_samples,).
                Sum of weights equals num_samples (for compatibility with
                torch.utils.data.WeightedRandomSampler).

        Example:
            >>> tracker = LossTracker(1000)
            >>> # ... after training ...
            >>> weights = tracker.get_weights(cap_factor=3.0, min_weight=0.1)
            >>> sampler = WeightedRandomSampler(weights, num_samples=1000)
        """
        if cap_factor <= 0.0:
            raise ValueError(f"cap_factor must be positive, got {cap_factor}")
        if not 0.0 <= min_weight < 1.0:
            raise ValueError(f"min_weight must be in [0, 1), got {min_weight}")

        weights = self.losses.clone()

        # Apply capping: w_i = min(L_i, cap_factor * mean(L))
        mean_loss = weights.mean()
        cap_value = cap_factor * mean_loss
        weights = torch.clamp(weights, max=cap_value)

        # Apply minimum weight (for exploration)
        weights = torch.clamp(weights, min=min_weight)

        # Normalize so sum(weights) = num_samples
        # (standard convention for WeightedRandomSampler)
        weights = weights / weights.sum() * self.num_samples

        return weights

    def reset(self) -> None:
        """Reset all loss estimates to initial values."""
        self.losses.fill_(self.initial_loss)
        self.initialized.fill_(False)

    def to(self, device: torch.device) -> "LossTracker":
        """Move loss tracker to specified device."""
        self.losses = self.losses.to(device)
        self.initialized = self.initialized.to(device)
        return self

    def state_dict(self) -> dict:
        """Return state dictionary for checkpointing."""
        return {
            "losses": self.losses,
            "initialized": self.initialized,
            "num_samples": self.num_samples,
            "smoothing": self.smoothing,
            "initial_loss": self.initial_loss,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint."""
        self.losses = state_dict["losses"]
        self.initialized = state_dict["initialized"]
        self.num_samples = state_dict["num_samples"]
        self.smoothing = state_dict["smoothing"]
        self.initial_loss = state_dict["initial_loss"]


class InformationWeightedSampler(Sampler):
    """
    Sampler for three-phase H3 training with loss-based prioritization.

    Prioritizes samples with high loss (high information content) while
    maintaining exploration via uniform mixing. Supports three-phase training:
    - Warmup: uniform sampling (set uniform_mix=1.0)
    - Thermodynamic: weighted sampling (set uniform_mix=0.2, keep_frac=0.65)
    - Consolidation: uniform sampling (set uniform_mix=1.0)

    Args:
        data_source: Dataset to sample from. Must support len().
        loss_tracker: LossTracker instance containing per-sample loss estimates.
        keep_frac: Fraction of samples to keep per epoch (default: 1.0).
            If < 1.0, only top keep_frac*N samples by loss are considered.
            From paper Section 5.3: typically 0.65 in thermodynamic phase.
        uniform_mix: Fraction of uniform sampling vs weighted (default: 0.2).
            Final weights = (1-α)*loss_weights + α*uniform, where α=uniform_mix.
            Set to 1.0 for pure uniform sampling (warmup/consolidation).
        replacement: Whether to sample with replacement (default: False).
        num_samples: Number of samples to draw per epoch (default: len(data_source)).
        generator: Random generator for reproducibility (default: None).

    Example:
        >>> dataset = MyDataset()
        >>> tracker = LossTracker(len(dataset))
        >>>
        >>> # Warmup phase: uniform sampling
        >>> sampler = InformationWeightedSampler(
        ...     dataset, tracker, uniform_mix=1.0
        ... )
        >>>
        >>> # Thermodynamic phase: weighted sampling
        >>> sampler = InformationWeightedSampler(
        ...     dataset, tracker, keep_frac=0.65, uniform_mix=0.2
        ... )
        >>>
        >>> loader = DataLoader(dataset, sampler=sampler, batch_size=128)

    Note:
        Dataset must return (data, target, index) tuples for loss tracking.
        Use IndexedDataset wrapper if needed.
    """

    def __init__(
        self,
        data_source: Dataset,
        loss_tracker: LossTracker,
        keep_frac: float = 1.0,
        uniform_mix: float = 0.2,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Initialize information-weighted sampler."""
        # Validate inputs
        if not hasattr(data_source, "__len__"):
            raise ValueError("data_source must support len()")
        if len(data_source) != loss_tracker.num_samples:
            raise ValueError(
                f"data_source length ({len(data_source)}) must match "
                f"loss_tracker num_samples ({loss_tracker.num_samples})"
            )
        if not 0.0 < keep_frac <= 1.0:
            raise ValueError(f"keep_frac must be in (0, 1], got {keep_frac}")
        if not 0.0 <= uniform_mix <= 1.0:
            raise ValueError(f"uniform_mix must be in [0, 1], got {uniform_mix}")

        self.data_source = data_source
        self.loss_tracker = loss_tracker
        self.keep_frac = keep_frac
        self.uniform_mix = uniform_mix
        self.replacement = replacement
        self.generator = generator

        # Number of samples to draw per epoch
        if num_samples is None:
            self.num_samples = len(data_source)
        else:
            if num_samples <= 0:
                raise ValueError(f"num_samples must be positive, got {num_samples}")
            self.num_samples = num_samples

    def __iter__(self) -> Iterator[int]:
        """
        Generate sample indices for one epoch.

        Returns:
            iterator: Iterator over sample indices.
        """
        # Get loss-based weights from tracker
        loss_weights = self.loss_tracker.get_weights()

        # Create uniform weights (all equal)
        uniform_weights = torch.ones_like(loss_weights)

        # Mix weighted and uniform sampling
        # final_weights = (1 - α) * loss_weights + α * uniform_weights
        # where α = uniform_mix
        mixed_weights = (
            (1 - self.uniform_mix) * loss_weights
            + self.uniform_mix * uniform_weights
        )

        # Apply keep_frac: select top keep_frac*N samples
        if self.keep_frac < 1.0:
            num_keep = int(self.keep_frac * len(self.data_source))
            num_keep = max(1, num_keep)  # Keep at least 1 sample

            # Get indices of top-k samples by weight
            top_k_weights, top_k_indices = torch.topk(
                mixed_weights, k=num_keep, sorted=False
            )

            # Renormalize weights for selected subset
            sampling_weights = top_k_weights / top_k_weights.sum() * self.num_samples
            candidate_indices = top_k_indices
        else:
            # Use all samples
            sampling_weights = mixed_weights
            candidate_indices = torch.arange(len(self.data_source))

        # Determine actual number of samples to draw
        # When keep_frac < 1.0, we might have fewer candidates than num_samples
        actual_num_samples = min(self.num_samples, len(candidate_indices))

        # Sample indices using multinomial distribution
        if self.replacement:
            # Sample with replacement
            sampled_positions = torch.multinomial(
                sampling_weights,
                num_samples=actual_num_samples,
                replacement=True,
                generator=self.generator,
            )
        else:
            # Sample without replacement
            sampled_positions = torch.multinomial(
                sampling_weights,
                num_samples=actual_num_samples,
                replacement=False,
                generator=self.generator,
            )

        # Map sampled positions back to original dataset indices
        sampled_indices = candidate_indices[sampled_positions]

        return iter(sampled_indices.tolist())

    def __len__(self) -> int:
        """Return number of samples per epoch."""
        return self.num_samples


class IndexedDataset(Dataset):
    """
    Wrapper that adds indices to dataset samples.

    Wraps an existing dataset to return (data, target, index) tuples,
    required for InformationWeightedSampler.

    Args:
        dataset: Base dataset to wrap. Must support indexing and len().

    Example:
        >>> base_dataset = torchvision.datasets.CIFAR10(...)
        >>> indexed_dataset = IndexedDataset(base_dataset)
        >>> data, target, index = indexed_dataset[0]
    """

    def __init__(self, dataset: Dataset) -> None:
        """Initialize indexed dataset wrapper."""
        if not hasattr(dataset, "__getitem__"):
            raise ValueError("dataset must support indexing (__getitem__)")
        if not hasattr(dataset, "__len__"):
            raise ValueError("dataset must support len()")

        self.dataset = dataset

    def __getitem__(self, index: int):
        """
        Get item with index.

        Args:
            index: Sample index.

        Returns:
            tuple: (data, target, index) where index is added to original output.
        """
        item = self.dataset[index]

        # Handle different dataset output formats
        if isinstance(item, (tuple, list)):
            # Dataset returns (data, target) or similar
            return (*item, index)
        else:
            # Dataset returns single item (e.g., unsupervised)
            return (item, index)

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.dataset)
