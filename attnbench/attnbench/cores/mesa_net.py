"""MesaNet — fla `chunk_mesa_net` / `mesa_net_decoding_one_step`.

Reference: fla-org/flash-linear-attention `fla/ops/mesa_net` (MesaNet: Sequence
Modeling by Locally Optimal Test-Time Training).  MesaNet keeps two states
``H_t`` / ``U_t`` and at each token solves ``(H_t + diag(lamb)) x_t = q_t`` by
conjugate-gradient iteration, reading out ``U_t x_t``.

Mounted under the uniform scaffold.  Core-internal preprocessing mirrors the
reference layer: q/k L2-normalization, ``g = logsigmoid(a)`` (log-space decay,
per head), ``beta = sigmoid(b)`` (per head), and a per-(head, key-dim) learnable
``lamb = softplus(lambda_params) + lower_bound``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from attnbench.cores.base import AttnCore
from attnbench.registry import register_core
from fla.ops.mesa_net import chunk_mesa_net, mesa_net_decoding_one_step


@dataclass
class MesaNetConfig:
    lambda_lower_bound: float = 0.25
    max_cg_iteration: int = 30   # CG steps for the linear solve
    gate_reduce: str = "mean"    # how the scaffold's [B,T,H,D] gate logits become [B,T,H]


@register_core(
    "mesa_net",
    config_cls=MesaNetConfig,
    description="MesaNet via fla chunk_mesa_net / mesa_net_decoding_one_step",
)
class MesaNetCore(AttnCore):
    def __init__(self, config: MesaNetConfig, num_heads: int, head_dim: int):
        super().__init__(config, num_heads, head_dim)
        assert config.gate_reduce in ("mean", "sum", "max")

        # per-(head, key-dim) lambda parameter, initialized so that
        # softplus(lambda_params) + lower_bound ~= 1.0 (reference layer)
        lambda_initial_value = 1.0
        init_lamb = torch.log(
            torch.exp(torch.tensor(lambda_initial_value - config.lambda_lower_bound, dtype=torch.float32)) - 1.0
        )
        self.lambda_params = nn.Parameter(
            torch.empty(num_heads * head_dim, dtype=torch.float32).fill_(init_lamb.item())
        )
        self.lambda_params._no_weight_decay = True

    def _reduce_gate(self, g_raw: torch.Tensor) -> torch.Tensor:
        if self.config.gate_reduce == "mean":
            return g_raw.mean(dim=-1)
        if self.config.gate_reduce == "sum":
            return g_raw.sum(dim=-1)
        return g_raw.max(dim=-1).values

    def _prep(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, aux: Dict[str, Any]):
        eps = 1e-6
        q = q / (torch.linalg.vector_norm(q, dim=-1, keepdim=True) + eps)
        k = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + eps)
        g = F.logsigmoid(self._reduce_gate(aux["g_raw"]).float())       # [B, T, H] log-space decay
        beta = aux["beta_raw"].float().sigmoid()                         # [B, T, H]
        lamb = F.softplus(self.lambda_params.float()) + self.config.lambda_lower_bound
        lamb = lamb.reshape(self.num_heads, -1)                          # [H, K]
        return q, k, v, g, beta, lamb

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        aux: Dict[str, Any],
        state: Optional[Any] = None,
        mode: str = "chunk",
        output_final_state: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        q, k, v, g, beta, lamb = self._prep(q, k, v, aux)

        if mode == "chunk":
            h_kk_init, h_kv_init = state if state is not None else (None, None)
            o, h_kk, h_kv = chunk_mesa_net(
                q=q, k=k, v=v, g=g, beta=beta, lamb=lamb,
                h_kk_init=h_kk_init, h_kv_init=h_kv_init,
                output_final_state=output_final_state,
                max_CG_iteration=self.config.max_cg_iteration,
                use_qk_l2norm_in_kernel=False,
            )
            new_state = (h_kk, h_kv) if output_final_state else None
        else:
            if state is None:
                # fresh single-step decode == a 1-token chunk pass (produces states)
                o, h_kk, h_kv = chunk_mesa_net(
                    q=q, k=k, v=v, g=g, beta=beta, lamb=lamb,
                    h_kk_init=None, h_kv_init=None,
                    output_final_state=True,
                    max_CG_iteration=self.config.max_cg_iteration,
                    use_qk_l2norm_in_kernel=False,
                )
            else:
                prev_h_kk, prev_h_kv = state
                o, h_kk, h_kv = mesa_net_decoding_one_step(
                    q=q.squeeze(0), k=k.squeeze(0), v=v.squeeze(0),
                    g=g.squeeze(0), beta=beta.squeeze(0), lamb=lamb,
                    prev_h_kk=prev_h_kk, prev_h_kv=prev_h_kv,
                    max_CG_iteration=self.config.max_cg_iteration,
                )
                o = o.unsqueeze(0).to(q)
            new_state = (h_kk, h_kv)

        return o, new_state
