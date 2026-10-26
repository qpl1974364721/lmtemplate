"""0.4B DeltaNet language model (lmtemplate-style backbone).

Architecture references
-----------------------
* DeltaNet: "Parallelizing Linear Transformers with the Delta Rule over
  Sequence Length" (Yang et al., arXiv:2406.06484).  The chunked delta-rule
  kernel itself is reused from ``fla`` (``fla.layers.delta_net.DeltaNet``),
  which is the paper authors' own implementation, so this file only wires the
  kernel into a full network.
* Backbone: lmtemplate's pre-norm residual block
  (RMSNorm -> token mixer -> residual -> RMSNorm -> GatedMLP -> residual),
  with untied input embedding and LM head.

Unlike the eval branches of this repo, this module has no softmax-attention
fallback, no KV-cache / generation code and no fused cross-entropy: it is the
training-time definition of the model only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from fla.modules.mlp import GatedMLP


@dataclass
class ModelConfig:
    """Model hyper-parameters (0.4B, 24 layers, hidden 1024, 8 heads)."""

    vocab_size: int = 32000
    n_layers: int = 24
    dim: int = 1024
    n_heads: int = 8
    expand_ratio: int = 4
    norm_eps: float = 1e-5

    # Input embedding and lm_head are intentionally NOT tied (same choice as
    # the other branches of this repo / the checkpoints already on the cluster).
    tie_word_embeddings: bool = False

    # Token mixer. Only "delta_net" is implemented in this branch.
    attn_impl: str = "delta_net"

    # --- DeltaNet hyper-parameters (fla.layers.delta_net.DeltaNet) ----------
    dn_mode: str = "chunk"          # "chunk" (parallel, for training) | "fused_recurrent"
    dn_expand_k: float = 1.0
    dn_expand_v: float = 1.0
    dn_use_gate: bool = False       # output gate (the paper's gated variant)
    dn_use_short_conv: bool = True  # short conv on q/k/v, important for quality
    dn_conv_size: int = 4
    dn_allow_neg_eigval: bool = False
    dn_qk_activation: str = "silu"
    dn_qk_norm: str = "l2"

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def max_position_embeddings(self) -> int:
        """Only for HF-style tooling (lm-eval reads it); the model has no RoPE."""
        return 4096

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ModelOutput:
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


def _build_token_mixer(config: ModelConfig, layer_idx: int) -> nn.Module:
    """Instantiate the token mixer described by ``config.attn_impl``."""
    if config.attn_impl != "delta_net":
        raise ValueError(
            f"Unknown attn_impl: {config.attn_impl!r}. This branch only ships "
            "'delta_net'."
        )

    # Imported lazily so that importing this module does not require triton/fla
    # kernels to be available (e.g. for parameter counting or config checks).
    from fla.layers.delta_net import DeltaNet

    return DeltaNet(
        mode=config.dn_mode,
        hidden_size=config.dim,
        expand_k=config.dn_expand_k,
        expand_v=config.dn_expand_v,
        num_heads=config.n_heads,
        use_beta=True,
        use_gate=config.dn_use_gate,
        use_short_conv=config.dn_use_short_conv,
        conv_size=config.dn_conv_size,
        conv_bias=False,
        allow_neg_eigval=config.dn_allow_neg_eigval,
        layer_idx=layer_idx,
        qk_activation=config.dn_qk_activation,
        qk_norm=config.dn_qk_norm,
        norm_eps=config.norm_eps,
    )


class Layer(nn.Module):
    """Pre-norm residual block: mixer + gated MLP."""

    def __init__(self, layer_idx: int, config: ModelConfig):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.post_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.mixer = _build_token_mixer(config, layer_idx)
        self.mlp = GatedMLP(hidden_size=config.dim, hidden_ratio=config.expand_ratio)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pre-norm: the mixer must see the normalised residual stream.
        # DeltaNet ignores attention masks for packed (unpadded) sequences, so
        # we never pass one during training.
        mixer_out = self.mixer(self.input_norm(hidden_states), attention_mask=None, use_cache=False)[0]
        hidden_states = hidden_states + mixer_out

        hidden_states = hidden_states + self.mlp(self.post_norm(hidden_states))
        return hidden_states


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [Layer(i, config) for i in range(config.n_layers)]
        )
        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        use_gradient_checkpoint: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.emb(input_ids)

        for layer in self.layers:
            if use_gradient_checkpoint and self.training:
                hidden_states = checkpoint(layer, hidden_states, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states)

        return self.norm(hidden_states)


def chunked_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Cross-entropy without materialising the full ``[B*T, vocab]`` logits.

    ``labels`` are already shifted (``labels[:, t] == input_ids[:, t + 1]``).
    The vocabulary is 32k, so for a 0.5M-token batch the full logits tensor
    would be tens of GB; slicing over the flattened batch/time axis keeps the
    peak at ``chunk_size * vocab`` floats.
    """
    hidden = hidden_states.reshape(-1, hidden_states.size(-1))
    target = labels.reshape(-1)

    total = hidden.new_zeros((), dtype=torch.float32)
    count = torch.zeros((), dtype=torch.float32, device=hidden.device)
    for start in range(0, hidden.size(0), chunk_size):
        h = hidden[start:start + chunk_size]
        t = target[start:start + chunk_size]
        logits = F.linear(h, weight).float()
        total = total + F.cross_entropy(
            logits, t, ignore_index=ignore_index, reduction="sum"
        )
        count = count + (t != ignore_index).sum()

    return total / count.clamp(min=1.0)


class ModelForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = LanguageModel(config)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    # -- helpers used by eval / lm-eval harness wrappers ---------------------
    def tie_weights(self) -> None:
        """No-op: embedding and LM head are independent matrices."""
        return

    def get_input_embeddings(self):
        return self.model.emb

    def get_output_embeddings(self):
        return self.lm_head

    def get_output_norm(self):
        return self.model.norm

    @property
    def dtype(self) -> torch.dtype:
        """HF-style convenience (lm-eval reads ``model.dtype``)."""
        try:
            return next(self.parameters()).dtype
        except StopIteration:  # pragma: no cover
            return torch.float32

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.model.emb.weight.numel() + self.lm_head.weight.numel()
        return n

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_gradient_checkpoint: bool = False,
        **kwargs,
    ) -> ModelOutput:
        """Causal-LM forward.

        ``attention_mask`` is accepted so lm-eval's ``HFLM`` can drive this
        module (it always passes one).  The backbone itself is a packed-sequence
        recurrent model with no padding mask support, so evaluation must use
        ``batch_size=1``; the argument is otherwise ignored.
        """
        hidden_states = self.model(
            input_ids, use_gradient_checkpoint=use_gradient_checkpoint
        )

        loss = None
        if labels is not None:
            loss = chunked_cross_entropy(hidden_states, self.lm_head.weight, labels)

        logits = None
        if not self.training:
            logits = self.lm_head(hidden_states)

        return ModelOutput(logits=logits, loss=loss)
