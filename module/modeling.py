from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from fla.modules.mlp import GatedMLP
from jaxtyping import Float, Int
from torch.utils.checkpoint import checkpoint

from .attention import Attention as AttentionBlock
from .cache_utils import Cache


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    n_layers: int = 24
    dim: int = 1024
    rotary_dim: int = 128
    n_heads: int = 8
    expand_ratio: int = 4
    rotary_base: int = 10000
    key_expand: int = 2
    # Input embeddings and lm_head are intentionally NOT tied. lm-eval's HFLM
    # calls `model.tie_weights()` on any nn.Module; the default behaviour would
    # otherwise overwrite lm_head.weight with emb.weight (shapes match).
    tie_word_embeddings: bool = False
    # Which attention/recurrence core each Layer uses:
    #   "attention" -> the template's FlashAttention block (module/attention.py)
    #   "gla"       -> fla.layers.simple_gla.SimpleGatedLinearAttention
    #   "gdn"       -> fla.layers.gated_deltanet.GatedDeltaNet
    #   "pgdn"      -> fla.layers.precond_gated_deltanet.PrecondGatedDeltaNet
    #   "kda"       -> fla.layers.kda.KimiDeltaAttention
    #   "gka"       -> Gated KalmaNet (implementation not present in this repo)
    attn_impl: str = "attention"

@dataclass
class ModelOutput:
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[dict] = None
    loss: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None

class RotaryEmbedding(nn.Module):
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.rotary_dim = config.rotary_dim
        self.rotary_base = config.rotary_base

        assert self.rotary_dim % 2 == 0

        inv_freq = 1.0 / (self.rotary_base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)


    def forward(
        self, 
        hidden_state: Float[torch.Tensor, 'b t d'],     # RoPE actually does not use hidden states. Only to obtain the device
        position_ids: Int[torch.Tensor, 'b t']
    ) -> Tuple[
        Float[torch.Tensor, 'b t rd'], 
        Float[torch.Tensor, 'b t rd']
    ]:
        self.inv_freq = self.inv_freq.to(position_ids)
        freqs = torch.einsum('b t, d->b t d', position_ids, self.inv_freq)
        freqs = torch.cat([freqs, freqs], dim=-1)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos.to(hidden_state), sin.to(hidden_state)

def _build_attention(config: "ModelConfig", layer_idx: int) -> nn.Module:
    """Instantiate the attention/recurrence core selected by ``config.attn_impl``."""
    impl = config.attn_impl
    head_dim = config.dim // config.n_heads
    if impl == "attention":
        return AttentionBlock(
            layer_idx=layer_idx,
            dim=config.dim,
            n_heads=config.n_heads,
            rotary_emb_dim=config.rotary_dim,
        )
    if impl == "gla":
        from fla.layers.simple_gla import SimpleGatedLinearAttention
        return SimpleGatedLinearAttention(
            hidden_size=config.dim,
            expand_k=1,
            expand_v=1,
            num_heads=config.n_heads,
            layer_idx=layer_idx,
        )
    if impl == "gdn":
        from fla.layers.gated_deltanet import GatedDeltaNet
        return GatedDeltaNet(
            hidden_size=config.dim,
            expand_v=1,
            head_dim=head_dim,
            num_heads=config.n_heads,
            layer_idx=layer_idx,
        )
    if impl == "pgdn":
        try:
            from fla.layers.precond_gated_deltanet import PrecondGatedDeltaNet
        except ImportError as exc:
            raise ImportError(
                "PGDN requires an FLA build containing "
                "fla.layers.precond_gated_deltanet.PrecondGatedDeltaNet."
            ) from exc
        # The 20BT checkpoint contains g_proj/o_norm plus ATK-specific
        # parameters, matching the gated short-conv PGDN layer used here.
        return PrecondGatedDeltaNet(
            hidden_size=config.dim,
            expand_v=1.0,
            head_dim=head_dim,
            num_heads=config.n_heads,
            mode="chunk",
            use_gate=True,
            use_short_conv=True,
            allow_neg_eigval=False,
            conv_size=4,
            conv_bias=False,
            layer_idx=layer_idx,
            norm_eps=1e-6,
            squash_x=1.5,
            squash_eps=1e-6,
            log_atk_scale_init=-0.2,
        )
    if impl == "kda":
        from fla.layers.kda import KimiDeltaAttention
        return KimiDeltaAttention(
            hidden_size=config.dim,
            expand_v=1,
            head_dim=head_dim,
            num_heads=config.n_heads,
            layer_idx=layer_idx,
        )
    if impl == "gka":
        # Gated KalmaNet lives in Undermyth's flash-linear-attention fork, not in
        # upstream fla.  Two fork versions are supported:
        #   * `fix/mesa-cg` (old): q/k normalization is forced to float16, i.e.
        #     the model is bf16 but the gka core runs in local fp16.
        #   * commit 4541223 (new): q/k keep the model dtype (bf16 core).
        # In the old version the chunk/prefill path can return a float16 `o`
        # that mismatches the bf16 `o_proj`.  Cast o_proj's input to its weight
        # dtype (no-op when they already match), without touching /home/csy/pkgs.
        from fla.layers.gka import GatedKalmaNet
        gka = GatedKalmaNet(
            hidden_size=config.dim,
            head_dim=config.dim // config.n_heads,
            num_q_heads=config.n_heads,
            layer_idx=layer_idx,
            ridge_strength=0.01,
        )

        def _cast_input_to_weight_dtype(module, args):
            x = args[0]
            if x.dtype != module.weight.dtype:
                x = x.to(module.weight.dtype)
            return (x,)

        gka.o_proj.register_forward_pre_hook(_cast_input_to_weight_dtype)
        return gka
    raise ValueError(f"Unknown attn_impl: {impl!r}")


class Layer(nn.Module):

    def __init__(self, layer_idx: int, config: ModelConfig):
        super().__init__()
        self.layer_idx = layer_idx
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.rotary_dim = config.rotary_dim

        self.input_norm = nn.RMSNorm(self.dim)
        self.post_norm = nn.RMSNorm(self.dim)

        self.attn = _build_attention(config, self.layer_idx)
        self.mlp = GatedMLP(hidden_size=config.dim, hidden_ratio=config.expand_ratio)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[Int[torch.Tensor, 'b t']] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
    ):
        x = self.input_norm(hidden_states)
        attn_output = self.attn(
            x, 
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values, 
            use_cache=use_cache
        )

        aux_loss = None
        if len(attn_output) == 3:
            x, _, past_key_values = attn_output
        else:
            x, _, past_key_values, aux_loss = attn_output

        hidden_states = hidden_states + x   # mark attn

        x = self.post_norm(hidden_states)
        x = self.mlp(x)
        hidden_states = hidden_states + x   # mark mlp

        return hidden_states, past_key_values, aux_loss
        
class LanguageModel(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_layers % 2 == 0
        self.config = config
        self.n_layers = config.n_layers

        self.rotary_emb = RotaryEmbedding(config)
        self.emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            Layer(i, config) for i in range(self.n_layers)
        ])
        self.norm = nn.RMSNorm(config.dim)

    def forward(
        self, 
        input_ids: Int[torch.Tensor, 'b t'], 
        position_ids: Optional[Int[torch.Tensor, 'b t']] = None,      # decided by generate(). when decoding, will contain only decoded part
        attention_mask: Optional[Int[torch.Tensor, 'b t']] = None,    # decided by generate(). when decoding, will contain both prefill and generated part
        use_cache: bool = False, 
        past_key_values: Optional[Cache] = None,
        use_gradient_checkpoint: bool = False
    ):

        if use_cache and past_key_values is None:
            past_key_values = Cache()

        hidden_states = self.emb(input_ids)
       
        if position_ids is None:
            # past_seen_tokens = layer_cache.get_sequence_length()   # [NOTE] not really past tokens. used only for training to decide position ids
            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)    # maximum length in the batch
        position_ids = position_ids.to(hidden_states.device)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        total_aux_loss = None
        for layer in self.layers:
            if use_gradient_checkpoint:
                hidden_states, past_key_values, aux_loss = checkpoint(
                    layer,
                    hidden_states,
                    attention_mask=attention_mask, 
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values, 
                    use_cache=use_cache, 
                    use_reentrant=False
                )
            else:
                hidden_states, past_key_values, aux_loss = layer(
                    hidden_states, 
                    attention_mask=attention_mask, 
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values, 
                    use_cache=use_cache, 
                )
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss if total_aux_loss is not None else aux_loss
            
        hidden_states = self.norm(hidden_states)
        # return ModelOutput(last_hidden_state=hidden_states, past_key_values=past_key_values)
        return hidden_states, past_key_values, total_aux_loss


def _extract_stopping_text_config(stopping_criteria):
    """Extract lm-eval's stop strings/tokenizer from its HF stopping criteria.

    lm-eval builds a ``StoppingCriteriaList`` of ``MultiTokenEOSCriteria``.
    Calling that criteria every generated token decodes tokens and forces a
    GPU synchronization, so our generate loop checks them only every few
    tokens.  The returned strings reproduce the same stop condition without
    depending on the exact tokenization of each stop sequence.
    """
    if stopping_criteria is None:
        return [], None, 0
    try:
        criteria = list(stopping_criteria)
    except TypeError:
        return [], None, 0

    sequences = []
    tokenizer = None
    lookback = 0
    for criterion in criteria:
        sequence = getattr(criterion, "sequence", None)
        criterion_tokenizer = getattr(criterion, "tokenizer", None)
        if not isinstance(sequence, str) or not sequence or criterion_tokenizer is None:
            return [], None, 0
        sequences.append(sequence)
        tokenizer = tokenizer or criterion_tokenizer
        sequence_len = getattr(criterion, "sequence_id_len", None)
        if not isinstance(sequence_len, int):
            sequence_len = len(criterion_tokenizer.encode(sequence, add_special_tokens=False)) + 2
        lookback = max(lookback, sequence_len)
    return list(dict.fromkeys(sequences)), tokenizer, lookback


def _generation_finished(
    generated,
    prompt_len,
    generated_len,
    eos_token_id,
    stop_sequences,
    stop_tokenizer,
    stop_window,
    stopping_criteria,
):
    """Check EOS and HF/lm-eval stop sequences with one host sync."""
    if stop_sequences and stop_tokenizer is not None:
        window_start = max(prompt_len, generated_len - stop_window)
        suffix = generated[:, window_start:generated_len].detach().cpu()
        decoded = stop_tokenizer.batch_decode(suffix)
        stop_hit = all(
            any(sequence in text for sequence in stop_sequences)
            for text in decoded
        )
        last_tokens = suffix[:, -1].tolist()
    else:
        # Fallback for a newer/unknown criteria implementation.
        stop_hit = (
            bool(stopping_criteria(generated[:, :generated_len], None))
            if stopping_criteria is not None
            else False
        )
        last_tokens = generated[:, generated_len - 1].detach().cpu().tolist()

    eos_hit = (
        eos_token_id is not None
        and all(token == eos_token_id for token in last_tokens)
    )
    return stop_hit or eos_hit


class ModelForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = LanguageModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.criteria = FusedLinearCrossEntropyLoss()

    def tie_weights(self) -> None:
        """No-op override.

        lm-eval's HFLM unconditionally calls ``model.tie_weights()`` on any
        ``nn.Module`` it is handed. The default PyTorch/transformers behaviour
        ties the input embedding with the output ``lm_head`` when their shapes
        match, which is *not* how this model was trained (the head is a
        separate linear layer). Keeping this as a no-op preserves the two
        independent weight matrices.
        """
        return

    def get_input_embeddings(self):
        return self.model.emb
    
    def get_output_embeddings(self):
        return self.lm_head

    def get_output_norm(self):
        return self.model.norm

    def get_decoder(self):
        return self.model.layers

    def forward(
        self, 
        input_ids: Int[torch.Tensor, 'b t'], 
        position_ids: Optional[Int[torch.Tensor, 'b t']] = None,      # decided by generate(). when decoding, will contain only decoded part
        attention_mask: Optional[Int[torch.Tensor, 'b t']] = None,    # decided by generate(). when decoding, will contain both prefill and generated part
        label: Optional[Int[torch.Tensor, 'b t']] = None,             # already shifted labels
        use_cache: bool = False, 
        past_key_values: Optional[Cache] = None,
        use_gradient_checkpoint: bool = False
    ):
        outputs = self.model(
            input_ids, 
            position_ids=position_ids,
            attention_mask=attention_mask, 
            use_cache=use_cache, 
            past_key_values=past_key_values,
            use_gradient_checkpoint=use_gradient_checkpoint
        )
        hidden_states = outputs[0]
        past_key_values = outputs[1]
        aux_loss = outputs[2]
        
        logits = None
        if not self.training:
            logits = self.lm_head(hidden_states)
            
        loss = None
        if label is not None:
            # label = torch.cat((label[..., 1:], torch.full_like(label[:, :1], self.criterion.ignore_index)), dim=1)
            loss = self.criteria(hidden_states, label, self.lm_head.weight, self.lm_head.bias)
            
        return ModelOutput(
            logits=logits,
            past_key_values=past_key_values,
            loss=loss,
            aux_loss=aux_loss
        )
    
    def generate(
        self,
        input_ids: Int[torch.Tensor, 'b t'],
        attention_mask: Optional[Int[torch.Tensor, 'b t']] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        use_cache: bool = True,
        past_key_values: Optional[Cache] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = False,
        eos_token_id: int = 2,
        stopping_criteria=None,
        stop_check_interval: int = 8,
        **kwargs
    ):
        # batch_size=1、无 padding，mask 全 1 无用；置 None 让 fla 层跳过 mask 处理，
        # 避免 fla simple_gla 里 v.mul_ 对 conv 输出 view 做 inplace 的报错。
        attention_mask = None

        stop_sequences, stop_tokenizer, stop_lookback = _extract_stopping_text_config(
            stopping_criteria
        )
        stop_check_interval = max(1, int(stop_check_interval))
        stop_window = stop_check_interval + stop_lookback

        # 推理阶段关闭 autograd，避免长上下文下保留全部中间激活导致显存暴涨
        with torch.inference_mode():
            # Determine stopping condition
            if max_new_tokens is None and max_length is None:
                max_new_tokens = 100

            if max_new_tokens is not None:
                max_new_tokens = int(max_new_tokens)
                target_length = input_ids.shape[1] + max_new_tokens
            else:
                target_length = int(max_length)
                max_new_tokens = max_length - input_ids.shape[1]

            batch_size, seq_len = input_ids.shape
            device = input_ids.device
            max_new_tokens = max(0, max_new_tokens)
            if max_new_tokens == 0:
                return input_ids

            # Prefill phase
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=device
            ).unsqueeze(0).expand(batch_size, -1)
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_values=past_key_values
            )
            hidden_states = outputs[0]
            past_key_values = outputs[1]

            # Get logits for next token prediction
            logits = self.lm_head(hidden_states)
            next_token_logits = logits[:, -1, :]

            # Preallocate the output sequence. Growing it with torch.cat on
            # every token adds allocator/copy overhead and prevents the CPU
            # from queuing decode work efficiently.
            generated = input_ids.new_empty((batch_size, target_length))
            generated[:, :seq_len].copy_(input_ids)
            generated_len = seq_len
            next_position_ids = torch.full(
                (batch_size, 1), seq_len, dtype=torch.long, device=device
            )

            # Decode loop
            for step in range(max_new_tokens):
                # Apply sampling logic
                if do_sample:
                    # Temperature
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature

                    # Top-k
                    if top_k > 0:
                        kth_vals = torch.kthvalue(next_token_logits, logits.size(-1) - top_k + 1, dim=-1).values
                        next_token_logits = torch.where(next_token_logits < kth_vals.unsqueeze(-1), torch.full_like(next_token_logits, float('-inf')), next_token_logits)

                    # Top-p (nucleus)
                    if top_p < 1.0:
                        sorted_logits, indices = torch.sort(next_token_logits, descending=True)
                        probs = torch.softmax(sorted_logits, dim=-1)
                        cumsum = torch.cumsum(probs, dim=-1)
                        mask = cumsum > top_p
                        mask[:, 1:] = mask[:, :-1].clone()
                        mask[:, 0] = False
                        sorted_logits = sorted_logits.masked_fill(mask, float('-inf'))
                        next_token_logits = torch.gather(sorted_logits, dim=-1, index=indices)

                    # Sample
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated[:, generated_len:generated_len + 1] = next_token
                generated_len += 1

                # Decoding stop strings on every token is a host-device sync
                # per token. Check every N tokens (and at the final token);
                # lm-eval strips anything after the first stop string before
                # scoring, so at most N-1 harmless extra tokens are generated.
                check_now = (
                    generated_len >= target_length
                    or (step + 1) % stop_check_interval == 0
                )
                if check_now and _generation_finished(
                    generated,
                    seq_len,
                    generated_len,
                    eos_token_id,
                    stop_sequences,
                    stop_tokenizer,
                    stop_window,
                    stopping_criteria,
                ):
                    break
                if generated_len >= target_length:
                    break

                # Prepare for next iteration
                next_token_id = next_token

                # Forward pass with cached states
                outputs = self.model(
                    input_ids=next_token_id,
                    position_ids=next_position_ids,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    past_key_values=past_key_values
                )
                hidden_states = outputs[0]
                past_key_values = outputs[1]

                # Get logits
                logits = self.lm_head(hidden_states)
                next_token_logits = logits[:, -1, :]
                next_position_ids.add_(1)

            return generated[:, :generated_len]
