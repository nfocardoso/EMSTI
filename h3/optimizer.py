"""
H3Optimizer: Thermodynamically-efficient optimizer with Lipschitz-adaptive learning rates
and AdamW-style decoupled weight decay.

Key improvements in v0.2.1:
- Decoupled weight decay (AdamW-style) by default
- More aggressive defaults (lipschitz_safety=0.8)
- Better Lipschitz initialization
- Robust parameter validation

Reference: H3 paper, Section 4.4 and Appendix A
"""

import math
from typing import Iterable, Optional
import torch
from torch.optim import Optimizer


class H3Optimizer(Optimizer):
    """
    Thermodynamically-efficient optimizer with Lipschitz-adaptive learning rates
    and AdamW-style decoupled weight decay.

    Key improvements in v0.2.1:
    - Decoupled weight decay (AdamW-style) by default
    - More aggressive defaults (lipschitz_safety=0.8)
    - Better Lipschitz initialization
    - Robust parameter validation

    Args:
        params: Iterable of parameters to optimize
        lr: Base learning rate (default: 1e-3)
        weight_decay: Weight decay factor (default: 0.0)
        lipschitz_safety: Safety margin in (0,1]. Lower=more aggressive (default: 0.8)
        lipschitz_update_interval: Steps between Lipschitz updates (default: 20)
        lipschitz_ema_beta: EMA smoothing factor (default: 0.9)
        decoupled_weight_decay: Use AdamW-style weight decay if True (default: True)
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lipschitz_safety: float = 0.8,
        lipschitz_update_interval: int = 20,
        lipschitz_ema_beta: float = 0.9,
        decoupled_weight_decay: bool = True,
    ):
        # Validation
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not (0.0 < lipschitz_safety <= 1.0):
            raise ValueError(f"lipschitz_safety must be in (0,1], got {lipschitz_safety}")
        if lipschitz_update_interval < 1:
            raise ValueError("lipschitz_update_interval must be >= 1")
        if not (0.0 < lipschitz_ema_beta < 1.0):
            raise ValueError("lipschitz_ema_beta must be in (0,1)")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            lipschitz_safety=lipschitz_safety,
            decoupled_weight_decay=decoupled_weight_decay,
        )
        super().__init__(params, defaults)

        self.lipschitz_update_interval = lipschitz_update_interval
        self.lipschitz_ema_beta = lipschitz_ema_beta
        self.step_count: int = 0
        self.lipschitz_estimate: float = 1.0
        self._has_seen_grad: bool = False

    @torch.no_grad()
    def _update_lipschitz(self) -> None:
        """Estimate local Lipschitz constant from gradient and parameter deltas."""
        grad_norm_sq = 0.0
        param_diff_norm_sq = 0.0

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                grad_norm_sq += float(g.norm(2).item() ** 2)

                state = self.state[p]
                prev_param = state.get("prev_param", None)
                if prev_param is not None:
                    diff = p.data - prev_param
                    param_diff_norm_sq += float(diff.norm(2).item() ** 2)

                # Save for next iteration
                state["prev_param"] = p.data.clone()

        grad_norm = math.sqrt(grad_norm_sq) if grad_norm_sq > 0.0 else 0.0
        param_diff_norm = math.sqrt(param_diff_norm_sq) if param_diff_norm_sq > 0.0 else 0.0

        # Better initialization on first gradient
        if not self._has_seen_grad and grad_norm > 0.0:
            self.lipschitz_estimate = grad_norm
            self._has_seen_grad = True

        # Update estimate with EMA
        if param_diff_norm > 1e-10 and grad_norm > 0.0:
            L_local = grad_norm / param_diff_norm
            beta = self.lipschitz_ema_beta
            self.lipschitz_estimate = (
                beta * self.lipschitz_estimate + (1.0 - beta) * L_local
            )

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Periodic Lipschitz update
        if self.step_count % self.lipschitz_update_interval == 0:
            self._update_lipschitz()

        lipschitz_estimate = max(self.lipschitz_estimate, 1e-8)

        for group in self.param_groups:
            base_lr = group["lr"]
            safety = group["lipschitz_safety"]
            lr_eff = base_lr * (safety / lipschitz_estimate)

            weight_decay = group["weight_decay"]
            decoupled = group.get("decoupled_weight_decay", True)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # AdamW-style decoupled weight decay
                if decoupled and weight_decay != 0.0:
                    p.data.mul_(1.0 - lr_eff * weight_decay)
                else:
                    # Adam-style: add to gradient
                    if weight_decay != 0.0:
                        grad = grad.add(p.data, alpha=weight_decay)

                # Gradient descent step
                p.data.add_(grad, alpha=-lr_eff)

        self.step_count += 1
        return loss
