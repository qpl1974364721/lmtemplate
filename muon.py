"""Muon optimizer (MomentUm Orthogonalized by Newton-Schulz).

Reference implementation: Keller Jordan's
https://github.com/KellerJordan/Muon (the same recipe used by modded-nanogpt
and by the reference DeltaNet training code).

Muon is used for the 2-D hidden weight matrices only.  Embeddings, the LM head,
norm gains and the short-convolution kernels are 1-D/3-D and are optimised with
AdamW instead -- see ``train.py::build_optimizers`` for how the three learning
rates from ``delta_net.md`` are assigned.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate ``U V^T`` of ``G = U S V^T`` with a quintic Newton-Schulz iteration.

    The coefficients ``(3.4445, -4.7750, 2.0315)`` make the iteration converge
    to the orthogonal factor while keeping the intermediate Frobenius norm
    bounded, so it can run in bf16.
    """
    if G.ndim != 2:
        raise ValueError(f"Newton-Schulz expects a 2-D tensor, got shape {tuple(G.shape)}")

    a, b, c = 3.4445, -4.7750, 2.0315

    X = G.to(torch.bfloat16)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    X = X / (X.norm() + eps)

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X


class Muon(Optimizer):
    """SGD-with-momentum whose update is the orthogonalised gradient."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        "Muon only accepts 2-D parameters; "
                        f"got shape {tuple(p.shape)}. Put 1-D/3-D params in AdamW."
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]
                # buf = momentum * buf + (1 - momentum) * grad
                buf.lerp_(grad, 1.0 - momentum)

                if group["nesterov"]:
                    g = grad.lerp(buf, momentum)
                else:
                    g = buf

                update = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                # Keep the update scale comparable to Adam's ~1 per element.
                update = update * max(1.0, update.size(0) / update.size(1)) ** 0.5

                if weight_decay:
                    p.mul_(1.0 - lr * weight_decay)

                p.add_(update.to(p.dtype), alpha=-lr)

        return loss
