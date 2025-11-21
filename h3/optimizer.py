"""
H3Optimizer: Hierarchical Hessian-informed optimizer for efficient deep learning.

Implements the adaptive learning rate algorithm from the H3 paper, combining:
- Adam-style exponential moving averages
- Lipschitz-adaptive learning rates (Eq. 4.9, 4.10)
- Per-parameter-group adaptation

Reference: H3 paper, Section 4.4 and Appendix A
"""

from typing import Optional, Callable, Iterable, Tuple
import torch
from torch.optim import Optimizer


class H3Optimizer(Optimizer):
    """
    H3 (Hierarchical Hessian-informed Hyperparameter) Optimizer.

    Combines Adam-style momentum with Lipschitz-adaptive learning rates for
    efficient deep learning. The optimizer estimates local Lipschitz constants
    to dynamically adjust learning rates per parameter group.

    The Lipschitz constant is estimated as (Eq. 4.10):
        L̂_k = ||g_{k+1} - g_k|| / ||w_{k+1} - w_k||

    And the adaptive learning rate is computed as (Eq. 4.10):
        η_{k+1} = min(η_max, γ/(L̂_k + ε))

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for computing running averages of gradient and its square
            (default: (0.9, 0.999)).
        eps: Term added to denominator for numerical stability (default: 1e-8).
        weight_decay: Weight decay coefficient (L2 penalty) (default: 0).
        lipschitz_safety: Safety factor γ for Lipschitz-based LR adaptation (default: 0.9).
            From paper Section 4.4.
        lipschitz_update_interval: Number of steps between Lipschitz constant updates
            (default: 10). Computing Lipschitz every step is expensive.

    Example:
        >>> model = MyModel()
        >>> optimizer = H3Optimizer(model.parameters(), lr=1e-3)
        >>> for batch in dataloader:
        >>>     optimizer.zero_grad()
        >>>     loss = criterion(model(batch.x), batch.y)
        >>>     loss.backward()
        >>>     optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        lipschitz_safety: float = 0.9,
        lipschitz_update_interval: int = 10,
    ) -> None:
        """Initialize H3 optimizer."""
        # Validate inputs
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 < lipschitz_safety <= 1.0:
            raise ValueError(f"Invalid lipschitz_safety: {lipschitz_safety}")
        if not lipschitz_update_interval > 0:
            raise ValueError(f"Invalid lipschitz_update_interval: {lipschitz_update_interval}")

        defaults = dict(
            lr=lr,
            initial_lr=lr,  # Store initial LR for Lipschitz adaptation
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            lipschitz_safety=lipschitz_safety,
            lipschitz_update_interval=lipschitz_update_interval,
        )
        super(H3Optimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Perform a single optimization step.

        Args:
            closure: Optional closure that reevaluates the model and returns the loss.

        Returns:
            loss: Loss value if closure is provided, None otherwise.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Process each parameter group
        for group in self.param_groups:
            beta1, beta2 = group['betas']

            # Track global step count for this group (for Lipschitz interval)
            if 'global_step' not in group:
                group['global_step'] = 0

            group['global_step'] += 1
            global_step = group['global_step']

            # =========================================================================
            # LIPSCHITZ ADAPTATION (Section 4.4, Eq. 4.10)
            # Update learning rate every lipschitz_update_interval steps
            # =========================================================================
            should_update_lipschitz = (
                global_step % group['lipschitz_update_interval'] == 0
                and global_step > 0
            )

            if should_update_lipschitz:
                # Aggregate gradient and parameter differences across the group
                grad_diffs = []
                param_diffs = []

                for p in group['params']:
                    if p.grad is None:
                        continue

                    state = self.state[p]

                    # Check if we have previous gradients and parameters
                    if 'prev_grad' in state and 'prev_params' in state:
                        # g_{k+1} - g_k (current grad minus previous grad)
                        grad_diff = p.grad - state['prev_grad']
                        grad_diffs.append(grad_diff.flatten())

                        # w_{k+1} - w_k (current params PRE-update minus previous params POST-update)
                        param_diff = p.data - state['prev_params']
                        param_diffs.append(param_diff.flatten())

                # Compute Lipschitz constant for this parameter group
                if len(grad_diffs) > 0:
                    # Concatenate all tensors in group into single vectors
                    grad_diff_vec = torch.cat(grad_diffs)
                    param_diff_vec = torch.cat(param_diffs)

                    # L̂_k = ||g_{k+1} - g_k|| / ||w_{k+1} - w_k||
                    grad_diff_norm = torch.norm(grad_diff_vec)  # L2 norm
                    param_diff_norm = torch.norm(param_diff_vec)  # L2 norm

                    # Avoid division by zero
                    if param_diff_norm > group['eps']:
                        L_hat = grad_diff_norm / param_diff_norm

                        # η_{k+1} = min(η_max, γ/(L̂_k + ε))
                        # where γ is lipschitz_safety factor (default: 0.9)
                        adapted_lr = group['lipschitz_safety'] / (L_hat + group['eps'])
                        group['lr'] = min(group['initial_lr'], adapted_lr.item())

            # =========================================================================
            # PARAMETER UPDATES (Adam-style with adaptive LR)
            # =========================================================================
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state on first step
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values (first moment)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values (second moment)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Store previous gradient for Lipschitz estimation
                    state['prev_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Store previous parameters for Lipschitz estimation
                    state['prev_params'] = p.data.clone()

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1

                # Apply weight decay (L2 regularization)
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Update biased first moment estimate: m_t = β1 * m_{t-1} + (1 - β1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate: v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                # m̂_t = m_t / (1 - β1^t)
                bias_correction1 = 1 - beta1 ** state['step']
                # v̂_t = v_t / (1 - β2^t)
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute step size (with bias correction)
                step_size = group['lr'] / bias_correction1

                # Compute bias-corrected second moment
                bias_correction2_sqrt = bias_correction2 ** 0.5

                # Denominator: sqrt(v̂_t) + ε
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group['eps'])

                # Update parameters: θ_t = θ_{t-1} - η_t * m̂_t / (sqrt(v̂_t) + ε)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # =========================================================================
                # STORE STATE FOR NEXT LIPSCHITZ UPDATE
                # Save current gradient and parameters (POST-update) for next iteration
                # =========================================================================
                state['prev_grad'] = grad.clone()
                state['prev_params'] = p.data.clone()

        return loss

    def __repr__(self) -> str:
        """String representation of optimizer."""
        return f"{self.__class__.__name__} ({id(self)})"
