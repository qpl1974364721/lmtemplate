"""Numerical correctness: chunk vs recurrent equivalence per core, softmax vs
reference formula, and state continuity across the chunk->recurrent boundary."""

import pytest
import torch

from attnbench.registry import build_core

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

VARIANTS = ["softmax", "gla", "kda", "gdn", "gka"]
H, D, T, B = 2, 32, 16, 2


def _make_core(variant):
    from attnbench.registry import get_spec
    cfg = get_spec(variant).config_cls()
    if hasattr(cfg, "chunk_size"):
        # GKA's chunk kernels need seq_len divisible by chunk_size
        cfg.chunk_size = T
    core = build_core(variant, cfg, num_heads=H, head_dim=D).to("cuda", dtype=torch.float32)
    core.eval()
    return core


def _make_inputs(seed=0, scale_gate=0.05):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", dtype=torch.float32, generator=gen) * scale
    q, k, v = randn(B, T, H, D), randn(B, T, H, D), randn(B, T, H, D)
    aux = {
        "g_raw": randn(B, T, H, D, scale=scale_gate),
        "beta_raw": randn(B, T, H, scale=scale_gate),
        "position_ids": torch.arange(T, device="cuda").unsqueeze(0).expand(B, -1).contiguous(),
    }
    return q, k, v, aux


@pytest.mark.parametrize("variant", VARIANTS)
def test_chunk_matches_recurrent_loop(variant):
    """chunk over T tokens == feeding recurrent T times with state threading."""
    core = _make_core(variant)
    q, k, v, aux = _make_inputs()

    with torch.no_grad():
        o_chunk, state_chunk = core(q, k, v, aux, state=None, mode="chunk",
                                    output_final_state=True)
        outs, state = [], None
        for t in range(T):
            q_t = q[:, t:t + 1]
            k_t = k[:, t:t + 1]
            v_t = v[:, t:t + 1]
            aux_t = {
                "g_raw": aux["g_raw"][:, t:t + 1],
                "beta_raw": aux["beta_raw"][:, t:t + 1],
                "position_ids": aux["position_ids"][:, t:t + 1],
            }
            o_t, state = core(q_t, k_t, v_t, aux_t, state=state, mode="recurrent",
                              output_final_state=True)
            outs.append(o_t)
        o_recurrent = torch.cat(outs, dim=1)

    assert o_chunk.shape == o_recurrent.shape
    assert torch.isfinite(o_chunk).all(), f"{variant}: non-finite chunk output"
    # fla's chunk and fused_recurrent implement the same math with different
    # scan algorithms; they agree to ~1e-2 in fp32 on these inputs (fla's own
    # consistency tests use a similar tolerance). softmax is exact (see below).
    tol = 1e-4 if variant == "softmax" else 2e-2
    torch.testing.assert_close(o_chunk, o_recurrent, atol=tol, rtol=tol)


@pytest.mark.parametrize("variant", VARIANTS)
def test_state_continuity(variant):
    """The state produced by chunk equals the state accumulated step-by-step."""
    core = _make_core(variant)
    q, k, v, aux = _make_inputs(seed=1)

    with torch.no_grad():
        _, state_chunk = core(q, k, v, aux, state=None, mode="chunk",
                              output_final_state=True)
        state = None
        for t in range(T):
            _, state = core(q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1],
                            {"g_raw": aux["g_raw"][:, t:t + 1],
                             "beta_raw": aux["beta_raw"][:, t:t + 1],
                             "position_ids": aux["position_ids"][:, t:t + 1]},
                            state=state, mode="recurrent", output_final_state=True)

    if variant == "softmax":
        kbuf_c, vbuf_c, cur_c = state_chunk
        kbuf_r, vbuf_r, cur_r = state
        assert cur_c == cur_r == T
        torch.testing.assert_close(kbuf_c[:, :, :T], kbuf_r[:, :, :T], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(vbuf_c[:, :, :T], vbuf_r[:, :, :T], atol=1e-4, rtol=1e-4)
    else:
        # GKA's chunk-vs-step difference includes different solver call paths
        # (chunk kernel vs one-step kernel); fla scan algorithms differ ~1e-2
        tol = 2e-2
        torch.testing.assert_close(state_chunk, state, atol=tol, rtol=tol)


def test_softmax_matches_reference():
    """SDPA chunk output matches the naive softmax(QK^T/sqrt(d))V formula."""
    core = _make_core("softmax")
    q, k, v, aux = _make_inputs(seed=2)

    with torch.no_grad():
        o, _ = core(q, k, v, aux, state=None, mode="chunk", output_final_state=False)

    # undo RoPE in reference by applying rope manually to plain inputs
    from attnbench.common.rope import RotaryCache, apply_rope
    rope = RotaryCache(D, 10000.0)
    cos, sin = rope.cos_sin(aux["position_ids"], "cuda", torch.float32)
    qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    qb, kb, vb = qr.transpose(1, 2), kr.transpose(1, 2), v.transpose(1, 2)
    scores = qb @ kb.transpose(-1, -2) / (D ** 0.5)
    mask = torch.tril(torch.ones(T, T, device="cuda", dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))
    ref = torch.softmax(scores, dim=-1) @ vb  # [B, H, T, D]
    ref = ref.transpose(1, 2)

    torch.testing.assert_close(o, ref, atol=1e-4, rtol=1e-4)


def test_chunk_on_top_of_cache():
    """Chunked prefill with an existing KV state (shifted causal mask)."""
    core = _make_core("softmax")
    q, k, v, aux = _make_inputs(seed=3)
    S1, S2 = T // 2, T - T // 2

    with torch.no_grad():
        # one-shot full chunk
        o_full, state_full = core(q, k, v, aux, state=None, mode="chunk",
                                  output_final_state=True)
        # two-shot: first half (fresh), second half on top of cache
        o1, state = core(q[:, :S1], k[:, :S1], v[:, :S1],
                         {key: val[:, :S1] for key, val in aux.items()},
                         state=None, mode="chunk", output_final_state=True)
        o2, state2 = core(q[:, S1:], k[:, S1:], v[:, S1:],
                          {key: val[:, S1:] for key, val in aux.items()},
                          state=state, mode="chunk", output_final_state=False)
        o_two = torch.cat([o1, o2], dim=1)

    torch.testing.assert_close(o_two, o_full, atol=1e-4, rtol=1e-4)
    assert state2[2] == state_full[2]
