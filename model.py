"""
ZPCodec: full codec model combining encoder, RVQ, optional repair, and decoder.

Data flow:
    waveform [B, 1, T]
    -> ZPEncoder          -> latent z [B, D, T']
    -> ResidualVQ         -> quantized z_q [B, D, T'], indices, commit_loss
    -> (GE simulator)     -> frame_mask [B, T']   (training only, if use_repair=True)
    -> LatentRepairTransformer -> z_q_post [B, D, T']  (missing frames concealed)
    -> ZPDecoder          -> waveform [B, 1, T_out]

T' = T / hop_length  (hop_length = prod(ratios) = 240 for ratios=[8,5,3,2] -> 15ms/frame)

The repair module is optional (use_repair=False for stage 1 codec-only training).
The GE simulator is optional too: if no GilbertElliottConfig is provided, no
packet loss is simulated and frame_mask is never generated automatically.
"""

import typing as tp
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ

from components import ZPEncoder, ZPDecoder
from repair import LatentRepairTransformer
from GilbertElliot import GilbertElliottConfig, GilbertElliottSimulator


@contextmanager
def temporarily_set(obj, attr: str, value):
    """Context manager that sets obj.attr = value for the duration of the block,
    then restores the original value. Used to toggle quantize_dropout per-batch."""
    original = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, original)


class ZPCodec(nn.Module):
    """
    Full codec: encoder -> RVQ -> (repair) -> decoder.

    Stage 1 (codec pre-training): use_repair=False, no GilbertElliottConfig.
    Stage 2 (repair training):    use_repair=True, GilbertElliottConfig provided.
    Stage 3 (joint fine-tuning):  use_repair=True, GE curriculum via set_gilbert_elliott_config().
    """
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        ratios: tp.List[int] = [8, 5, 3, 2],
        norm: str = 'weight_norm',
        causal: bool = True,
        num_quantizers: int = 9,
        codebook_size: int = 1024,
        sample_rate: int = 16000,
        # --- Repair module ---
        use_repair: bool = False,
        repair_hidden_dim: int = 256,
        repair_num_layers: int = 4,
        repair_num_heads: int = 4,
        repair_ffn_mult: int = 2,
        repair_past: int = 8,
        repair_future: int = 2,
        repair_two_pass: bool = True,
        # --- Packet loss simulation ---
        gilbert_elliott_config: tp.Optional[GilbertElliottConfig] = None,
    ):
        super().__init__()
        self.encoder = ZPEncoder(
            channels=channels,
            dimension=dimension,
            n_filters=n_filters,
            ratios=ratios,
            norm=norm,
            causal=causal,
        )
        self.rvq = ResidualVQ(
            dim=dimension,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            kmeans_init=True,
            kmeans_iters=10, 
            use_cosine_sim=True, # prop to improved RVQGAN's paper 
            threshold_ema_dead_code=2,
            quantize_dropout=True,
            quantize_dropout_cutoff_index=5,   # first 5 quantizers are always active -
            # theoretically with 5 quant active we can switch to 3kbps. But this was not my focus for that project...
            quantize_dropout_multiple_of=1,
        )
        self.decoder = ZPDecoder(
            channels=channels,
            dimension=dimension,
            n_filters=n_filters,
            ratios=ratios,
            norm=norm,
            causal=causal,
        )

        self.sample_rate = sample_rate
        self.hop_length = int(np.prod(ratios))   # 240 for ratios=[8,5,3,2]

        self.use_repair = use_repair
        self.repair_two_pass = repair_two_pass
        if use_repair:
            self.repair = LatentRepairTransformer(
                latent_dim=dimension,
                hidden_dim=repair_hidden_dim,
                num_layers=repair_num_layers,
                num_heads=repair_num_heads,
                ffn_mult=repair_ffn_mult,
                past=repair_past,
                future=repair_future,
            )
        else:
            self.repair = None

        self.ge_simulator: tp.Optional[GilbertElliottSimulator] = None
        if gilbert_elliott_config is not None:
            self.set_gilbert_elliott_config(gilbert_elliott_config)

    
    # Runtime configuration of the packet-loss simulator
    def set_gilbert_elliott_config(self, config: GilbertElliottConfig) -> None:
        """Replace the GE simulator at runtime. Called between training stages
        to apply a harder loss curriculum without reloading the model."""
        self.ge_simulator = GilbertElliottSimulator(
            config=config,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
        )

    def sample_frame_mask(
        self,
        batch_size: int,
        num_frames: int,
        device: tp.Optional[torch.device] = None,
        seed: tp.Optional[int] = None,
    ) -> torch.Tensor:
        """Expose the GE simulator directly. Useful when the same mask needs to
        be reused across multiple points (e.g. logging, loss weighting)."""
        assert self.ge_simulator is not None, (
            "No GilbertElliottConfig configured. Call set_gilbert_elliott_config() first."
        )
        return self.ge_simulator.sample_frame_mask(
            batch_size, num_frames, device=device, seed=seed
        )

    # Encoding
    def _encode_raw(self, x: torch.Tensor):
        """Encode waveform to quantized latent. Returns (z, z_q, indices, commit_loss).
        quantize_dropout is randomly toggled per-call during training to teach
        the decoder to handle a variable number of active quantizers (bitrate scalability)."""
        z = self.encoder(x)                  # [B, D, T']
        z_seq = z.permute(0, 2, 1)           # [B, T', D]  — RVQ expects (B, T, D)
        use_dropout = self.training and (torch.rand(1).item() < 0.5) # dropout applied only 50% of the time, this improve the 
        # quality at full kbps. Citing the improved RVQGAN paper.
        with temporarily_set(self.rvq, 'quantize_dropout', use_dropout):
            z_q, indices, commit_loss = self.rvq(z_seq)
        z_q = z_q.permute(0, 2, 1)          # [B, D, T']
        return z, z_q, indices, commit_loss

    
    # Repair
    def _apply_repair(
        self,
        z_q: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the repair transformer and selectively substitute only missing frames.

        z_q:        [B, D, T']
        frame_mask: [B, T']   1 = received, 0 = missing

        The transformer outputs a full [B, D, T'] tensor, but received frames are
        kept as-is from z_q — only positions where frame_mask == 0 are replaced.
        This means z_q_post == z_q on received frames by construction, which is
        important for latent_repair_loss (the mask isolates the useful gradient).

        Two-pass mode (repair_two_pass=True): mimics streaming deployment where
        previous repair estimates are already in the buffer when estimating frame t.
        See LatentRepairTransformer.forward_two_pass for the full explanation.
        """
        assert self.repair is not None, "use_repair=False, repair not initialised"

        z_seq = z_q.permute(0, 2, 1)                      # [B, T', D]

        if self.repair_two_pass:
            z_repaired = self.repair.forward_two_pass(z_seq, frame_mask)
        else:
            # Single-pass fallback
            z_seq_filled = self.repair.fill_missing(z_seq, frame_mask)
            z_repaired = self.repair(z_seq_filled, frame_mask)

        # Selective substitution: keep received frames from z_q, replace missing ones
        m = frame_mask.unsqueeze(-1).to(z_seq.dtype)      # [B, T', 1]
        z_out = z_seq * m + z_repaired * (1.0 - m)
        return z_out.permute(0, 2, 1)                     # [B, D, T']

    def _get_frame_mask(
        self,
        z_q: torch.Tensor,
        frame_mask: tp.Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return the provided frame_mask, or sample one from the GE simulator."""
        if frame_mask is not None:
            return frame_mask
        assert self.ge_simulator is not None, (
            "use_repair=True but no GilbertElliottConfig configured. "
            "Call set_gilbert_elliott_config() before training."
        )
        B, _, T_prime = z_q.shape
        return self.ge_simulator.sample_frame_mask(B, T_prime, device=z_q.device)

    # Public encode / decode API
    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Encode waveform to (z_q, indices). x: [B, 1, T]"""
        _, z_q, indices, _ = self._encode_raw(x)
        return z_q, indices

    def decode(
        self,
        z_q: torch.Tensor,
        frame_mask: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode quantized latent to waveform.
        z_q:        [B, D, T']
        frame_mask: [B, T']  optional; if provided and use_repair=True, runs repair first.
        """
        if self.use_repair and frame_mask is not None:
            z_q = self._apply_repair(z_q, frame_mask)
        return self.decoder(z_q)

    # Training forward
    def forward(
        self,
        x: torch.Tensor,
        frame_mask: tp.Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ):
        """
        x:                    [B, 1, T]
        frame_mask:           [B, T'] optional. If use_repair=True and None,
                              sampled automatically from the GE simulator.
        return_intermediates: if True, also returns z_q pre/post repair and the
                              effective frame_mask — required by latent_repair_loss
                              and ZPCodecTrainer.forward_codec during training.

        Returns:
          return_intermediates=False:  (x_hat, commit_loss)
          return_intermediates=True:   (x_hat, commit_loss, z_q_pre, z_q_post, frame_mask)
              When use_repair=False: z_q_pre == z_q_post and frame_mask == None.
        """
        _, z_q_pre, _, commit_loss = self._encode_raw(x)
        commit_loss = commit_loss.mean()

        if self.use_repair:
            frame_mask = self._get_frame_mask(z_q_pre, frame_mask)
            z_q_post = self._apply_repair(z_q_pre, frame_mask)
        else:
            z_q_post = z_q_pre
            frame_mask = None

        x_hat = self.decoder(z_q_post)

        if return_intermediates:
            return x_hat, commit_loss, z_q_pre, z_q_post, frame_mask
        return x_hat, commit_loss
