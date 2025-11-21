"""
Adaptive H3 Training Utilities
Dynamic annealing schedules for optimal efficiency/accuracy trade-off
"""
import math
from typing import Tuple, Optional


def cosine_annealing_schedule(
    epoch: int,
    total_epochs: int,
    min_keep_frac: float = 0.50,
    warmup_epochs: int = 3,
    consolidation_epochs: int = 3
) -> Tuple[float, float, str]:
    """
    Calculate adaptive keep_frac and uniform_mix using cosine annealing.

    Phase 1 (Warmup): Full data, uniform sampling
    Phase 2 (Annealing): Cosine curve 100% → min_frac → 100%
    Phase 3 (Consolidation): Full data, uniform sampling

    Theory:
    - Warmup: Initialize with full dataset for stable gradients
    - Annealing: Focus on hard examples (sharp minimum) then relax (flat minimum)
    - Consolidation: Full dataset for final generalization

    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total training epochs
        min_keep_frac: Minimum data fraction (lowest point of curve)
        warmup_epochs: Number of warmup epochs
        consolidation_epochs: Number of consolidation epochs

    Returns:
        (keep_frac, uniform_mix, phase_name)

    Example:
        >>> for epoch in range(20):
        ...     frac, mix, phase = cosine_annealing_schedule(epoch, 20, 0.5, 3, 3)
        ...     print(f"Epoch {epoch}: {phase}, keep={frac:.2f}, mix={mix:.2f}")
    """
    # Phase 1: Warmup
    if epoch < warmup_epochs:
        return 1.0, 1.0, "warmup"

    # Phase 3: Consolidation
    if epoch >= (total_epochs - consolidation_epochs):
        return 1.0, 1.0, "consolidation"

    # Phase 2: Annealing (cosine curve)
    middle_epochs = total_epochs - warmup_epochs - consolidation_epochs
    progress = (epoch - warmup_epochs) / middle_epochs

    # Cosine interpolation: 1.0 → min_frac → 1.0
    # First half: descend, second half: ascend
    cosine_value = math.cos(2 * math.pi * progress)
    current_frac = min_keep_frac + (1.0 - min_keep_frac) * (0.5 * (1 + cosine_value))

    # Adaptive uniform_mix: lower during focused phase
    uniform_mix = 0.2 + 0.3 * abs(cosine_value)  # 0.2-0.5 range

    return current_frac, uniform_mix, "annealing"


def linear_annealing_schedule(
    epoch: int,
    total_epochs: int,
    min_keep_frac: float = 0.50,
    warmup_epochs: int = 3,
    consolidation_epochs: int = 3
) -> Tuple[float, float, str]:
    """
    Linear annealing: 100% → min → 100% (simpler alternative)

    Similar to cosine but with linear transitions. More predictable but
    less smooth than cosine annealing.

    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total training epochs
        min_keep_frac: Minimum data fraction
        warmup_epochs: Number of warmup epochs
        consolidation_epochs: Number of consolidation epochs

    Returns:
        (keep_frac, uniform_mix, phase_name)
    """
    if epoch < warmup_epochs:
        return 1.0, 1.0, "warmup"

    if epoch >= (total_epochs - consolidation_epochs):
        return 1.0, 1.0, "consolidation"

    middle_epochs = total_epochs - warmup_epochs - consolidation_epochs
    progress = (epoch - warmup_epochs) / middle_epochs

    # Triangle wave: 0 → 1 → 0
    if progress < 0.5:
        # Descending: 1.0 → min_frac
        current_frac = 1.0 - (1.0 - min_keep_frac) * (progress * 2)
    else:
        # Ascending: min_frac → 1.0
        current_frac = min_keep_frac + (1.0 - min_keep_frac) * ((progress - 0.5) * 2)

    uniform_mix = 0.25

    return current_frac, uniform_mix, "annealing"


def exponential_annealing_schedule(
    epoch: int,
    total_epochs: int,
    min_keep_frac: float = 0.50,
    warmup_epochs: int = 3,
    consolidation_epochs: int = 3
) -> Tuple[float, float, str]:
    """
    Exponential decay then growth (aggressive focus)

    Drops quickly to min_keep_frac then recovers quickly. More aggressive
    than cosine - focuses harder on difficult examples in middle phase.

    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total training epochs
        min_keep_frac: Minimum data fraction
        warmup_epochs: Number of warmup epochs
        consolidation_epochs: Number of consolidation epochs

    Returns:
        (keep_frac, uniform_mix, phase_name)
    """
    if epoch < warmup_epochs:
        return 1.0, 1.0, "warmup"

    if epoch >= (total_epochs - consolidation_epochs):
        return 1.0, 1.0, "consolidation"

    middle_epochs = total_epochs - warmup_epochs - consolidation_epochs
    progress = (epoch - warmup_epochs) / middle_epochs

    # Exponential curve
    if progress < 0.5:
        # Decay: fast drop
        t = progress * 2
        current_frac = 1.0 - (1.0 - min_keep_frac) * (1 - math.exp(-3 * t))
    else:
        # Growth: fast recovery
        t = (progress - 0.5) * 2
        current_frac = min_keep_frac + (1.0 - min_keep_frac) * (1 - math.exp(-3 * t))

    uniform_mix = 0.3

    return current_frac, uniform_mix, "annealing"


def visualize_schedule(schedule_fn, total_epochs=20, **kwargs):
    """
    Visualize annealing schedule (useful for debugging)

    Args:
        schedule_fn: One of the annealing schedule functions
        total_epochs: Number of epochs to visualize
        **kwargs: Additional arguments to pass to schedule_fn

    Returns:
        List of (epoch, keep_frac, uniform_mix, phase) tuples
    """
    results = []
    for epoch in range(total_epochs):
        keep_frac, uniform_mix, phase = schedule_fn(epoch, total_epochs, **kwargs)
        results.append((epoch, keep_frac, uniform_mix, phase))
    return results


if __name__ == "__main__":
    # Demo the schedules
    print("="*80)
    print("H3 Adaptive Annealing Schedules")
    print("="*80)

    for schedule_name, schedule_fn in [
        ("Cosine", cosine_annealing_schedule),
        ("Linear", linear_annealing_schedule),
        ("Exponential", exponential_annealing_schedule)
    ]:
        print(f"\n{schedule_name} Schedule (20 epochs, min_frac=0.5):")
        print("-"*80)
        print(f"{'Epoch':<8} {'Phase':<15} {'Keep Frac':<12} {'Uniform Mix':<12}")
        print("-"*80)

        for epoch in range(20):
            keep_frac, uniform_mix, phase = schedule_fn(epoch, 20, 0.5, 3, 3)
            print(f"{epoch:>5}    {phase:<15} {keep_frac:>10.2f}   {uniform_mix:>10.2f}")
