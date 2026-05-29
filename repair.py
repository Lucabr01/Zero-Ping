"""
LatentRepairTransformer: packet-loss concealment in the RVQ latent domain.

Receives z_q_masked [B, T, D] and frame_mask [B, T] (1 = received, 0 = missing),
returns z_q_repaired [B, T, D] where missing frames are reconstructed by
attending only to valid frames within a local window [t-past, t+future].

Selective substitution (replacing only missing frames in the output) is handled
upstream in ZPCodec._apply_repair (model.py), not here. this module always produces a
full-length output tensor.

Why operate in the latent domain rather than on tokens?
    Tokens are discrete: a small error in the codec's estimate produces a
    completely different codebook entry, with no gradient signal. Latent vectors
    are continuous, so the transformer can produce soft interpolations between
    neighbouring frames and the repair loss (L1 on z) provides a smooth gradient.
    In a real RTP deployment this makes no difference to the transmitted bitstream.
"""

import math
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_local_attention_mask(
    T: int,
    past: int,
    future: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns a boolean mask [T, T] where True means the position is allowed.
    Query t can attend to key s if -past <= s - t <= future.
    This enforces a fixed-size local receptive field and a bounded lookahead.
    """
    idx = torch.arange(T, device=device)
    delta = idx.unsqueeze(0) - idx.unsqueeze(1)  # delta[t, s] = s - t
    return (delta >= -past) & (delta <= future)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Position Embedding (RoPE) to queries or keys.
    x:   [B, H, T, D_head]
    cos: [T, D_head]
    sin: [T, D_head]
    RoPE encodes relative positions directly in the dot product, making it
    compatible with the local attention mask without requiring learned positional embeddings.
    This suits perfectly in our case! We care about teh relative position rather than the absolute.
    """
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin tables up to max_seq_len."""
    def __init__(self, dim_head: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        assert dim_head % 2 == 0, "RoPE requires even dim_head"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim_head, 2).float() / dim_head))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('t,d->td', t, inv_freq)  # [T, dim_head/2]
        emb = torch.cat([freqs, freqs], dim=-1)        # [T, dim_head]
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, T: int) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:T], self.sin_cached[:T]


class MaskedLocalAttention(nn.Module):
    """
    Multi-head self-attention with two simultaneous masks:
      1) Local window mask: query t can only attend within [t-past, t+future].
         Keeps the receptive field bounded and latency predictable.
      2) Validity mask: keys from missing frames (frame_mask=0) are excluded,
         so the transformer cannot "cheat" by attending to frames it doesn't have.

    The combination forces the model to reconstruct missing frames exclusively
    from neighbouring received frames — the same information available at inference.
    """

    def __init__(self, dim: int, num_heads: int = 4, past: int = 8, future: int = 2):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5
        self.past = past
        self.future = future

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.dim_head)

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        """
        x:          [B, T, D]
        frame_mask: [B, T]  1 = received, 0 = missing
        """
        B, T, D = x.shape
        H, Dh = self.num_heads, self.dim_head

        qkv = self.to_qkv(x).reshape(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, H, T, Dh]

        cos, sin = self.rope(T)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, T]

        # Local window mask [T, T]: True where attention is allowed
        local_allowed = _build_local_attention_mask(T, self.past, self.future, x.device)

        # Validity mask: exclude keys from missing frames
        # frame_mask [B, T] -> key_valid [B, 1, 1, T]
        key_valid = frame_mask.bool().unsqueeze(1).unsqueeze(1)

        allowed = local_allowed.unsqueeze(0).unsqueeze(0) & key_valid  # [B, 1, T, T]

        # Failsafe: if a query has no valid key in its window (e.g. a long burst
        # of losses covers the entire local range), re-enable the diagonal so
        # softmax(-inf, ...) doesn't produce NaN. The query then attends to itself
        # (its learned missing_frame_embedding), which is a reasonable fallback.
        any_valid = allowed.any(dim=-1, keepdim=True)
        diag = torch.eye(T, dtype=torch.bool, device=x.device).unsqueeze(0).unsqueeze(0)
        allowed = allowed | (~any_valid & diag)

        scores = scores.masked_fill(~allowed, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)              # [B, H, T, Dh]
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block (attention + FFN) with masked local attention."""
    def __init__(self, dim: int, num_heads: int, ffn_mult: int, past: int, future: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MaskedLocalAttention(dim, num_heads, past, future)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), frame_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class LatentRepairTransformer(nn.Module):
    """
    Local inpainting of z_q after simulated packet loss on RVQ tokens.

    Architecture:
        missing_frame_embedding  — learned placeholder substituted for lost frames
                                   before the transformer sees the sequence
        mask_embedding           — additive token telling the model which frames
                                   are received (1) vs missing (0)
        in_proj                  — projects latent_dim -> hidden_dim
        blocks                   — stack of MaskedLocalAttention + FFN layers
        out_norm + out_proj      — projects hidden_dim -> latent_dim

    Args:
        latent_dim:  D of the RVQ latent (128 for ZPCodec).
        hidden_dim:  internal transformer width.
        num_layers:  number of transformer blocks.
        num_heads:   attention heads.
        ffn_mult:    FFN hidden size = hidden_dim * ffn_mult.
        past:        past frames in the local receptive field.
        future:      future frames (lookahead; each frame = 15ms latency cost).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        ffn_mult: int = 4,
        past: int = 8,
        future: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.past = past
        self.future = future

        # Learned placeholder for missing frames.
        # Substituted into z_q where frame_mask == 0 before the forward pass.
        self.missing_frame_embedding = nn.Parameter(torch.zeros(latent_dim))
        nn.init.normal_(self.missing_frame_embedding, std=0.02)

        # Additive embedding that signals received (1) vs missing (0) to the model.
        # Lets the transformer distinguish genuine latent vectors from placeholders.
        self.mask_embedding = nn.Embedding(2, hidden_dim)

        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ffn_mult, past, future)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def fill_missing(self, z_q: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        """
        Replace frames where frame_mask == 0 with the learned missing_frame_embedding.
        z_q:        [B, T, D]
        frame_mask: [B, T]
        """
        B, T, D = z_q.shape
        emb = self.missing_frame_embedding.view(1, 1, D).expand(B, T, D)
        m = frame_mask.unsqueeze(-1).to(z_q.dtype)
        return z_q * m + emb * (1.0 - m)

    def forward(
        self,
        z_q_masked: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        z_q_masked: [B, T, D] — missing frames must already contain
                    missing_frame_embedding; call fill_missing() first if needed.
        frame_mask: [B, T]    — 1 = received, 0 = missing.
        Returns:    [B, T, D] — full reconstructed sequence.
                    Selective substitution (only replacing missing frames in z_q)
                    is done upstream in ZPCodec._apply_repair.
        """
        x = self.in_proj(z_q_masked)
        x = x + self.mask_embedding(frame_mask.long())  # inject received/missing signal

        for block in self.blocks:
            x = block(x, frame_mask)

        x = self.out_norm(x)
        return self.out_proj(x)

    def forward_two_pass(
        self,
        z_q: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Two-pass forward that mimics streaming deployment behaviour.

        In real-time streaming, when estimating a lost frame at time t, the
        estimates for previously lost frames (s < t) are already in the buffer —
        not the original missing_frame_embedding. Training with a single pass
        creates a train/inference mismatch because the model never sees its own
        estimates as context. Two passes close that gap.

        Pass 1: standard forward with missing_emb as placeholder for all lost frames.
                Produces initial rough estimates z_pass1.

        Pass 2: update the buffer — lost frames now contain z_pass1 instead of
                missing_emb. Re-run the transformer on the updated buffer to
                produce refined estimates z_pass2.

        z_q:        [B, T, D]  original quantized latent (NOT yet masked).
        frame_mask: [B, T]
        Returns:    [B, T, D]  refined estimates for missing frames.
                    Values at received-frame positions are arbitrary —
                    selective substitution happens upstream in ZPCodec._apply_repair.

        I keep frame_mask unchanged between passes: lost frames stay 0 in the mask
        even after pass-1 estimates fill the buffer. This way the transformer in
        pass 2 still knows those frames were lost, limiting error accumulation in long bursts.
        
        """
        # Pass 1: fill missing with placeholder, run transformer
        z_masked_1 = self.fill_missing(z_q, frame_mask)
        z_pass1 = self.forward(z_masked_1, frame_mask)

        # Update buffer: received frames keep z_q, lost frames get pass-1 estimates
        m = frame_mask.unsqueeze(-1).to(z_q.dtype)
        z_buffer_updated = z_q * m + z_pass1 * (1.0 - m)

        # Pass 2: re-run on updated buffer.
        # Do NOT call fill_missing again — lost frames already contain z_pass1,
        # which is exactly the buffer state we want to simulate.
        z_pass2 = self.forward(z_buffer_updated, frame_mask)

        return z_pass2
