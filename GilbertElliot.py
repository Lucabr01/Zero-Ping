"""
Gilbert-Elliott packet-loss simulator in the latent frame domain.
I could have done it directly on the tokens, but it would have been more difficult for the transformer to learn
how to repair the latent vectors. In a real implementation scenario (with RTP packets) this doesn't change anything.

The classic model (Hasslinger & Hohlfeld 2008) has two states G/B with:
    p = P(G -> B)        probability of entering the Bad state
    r = P(B -> G)        probability of leaving the Bad state
    k = P(no loss | G)   typically ~1 (Good channel, nearly clean)
    h = P(no loss | B)   typically ~0.5 (Bad channel, lossy)

Stationary probabilities:
    pi_G = r / (p + r)
    pi_B = p / (p + r)

Expected error rate:
    p_E = (1-k) * pi_G + (1-h) * pi_B

Difference from the baseline implementation (which operates on the waveform):
here we directly generate a frame_mask [B, T'] over latent frames.
Packetization over latent frames is derived from packet_ms, sample_rate, and hop_length.
"""

import math
from dataclasses import dataclass, field
import typing as tp

import numpy as np
import torch


@dataclass
class GilbertElliottConfig:
    """
    Simulator parameters. Defaults:
    channel degrades rarely, recovers quickly, short bursts.
    """
    p: float = 0.005          # G -> B transition probability
    r: float = 0.5            # B -> G transition probability
    k: float = 0.999          # P(no loss | Good state)
    h: float = 0.5            # P(no loss | Bad state)
    packet_ms: float = 15.0   # packet duration in milliseconds

    def stationary_loss_rate(self) -> float:
        pi_G = self.r / (self.p + self.r)
        pi_B = self.p / (self.p + self.r)
        return (1.0 - self.k) * pi_G + (1.0 - self.h) * pi_B

    def __post_init__(self):
        assert 0.0 < self.p < 1.0, "p must be in (0, 1)"
        assert 0.0 < self.r < 1.0, "r must be in (0, 1)"
        assert 0.0 <= self.k <= 1.0, "k must be in [0, 1]"
        assert 0.0 <= self.h <= 1.0, "h must be in [0, 1]"
        assert self.packet_ms > 0, "packet_ms must be positive"


class GilbertElliottSimulator:
    """
    Generates a frame_mask [B, T'] (1 = received, 0 = missing) where T' is the
    number of latent frames produced by the codec. If a packet spans multiple
    consecutive frames (packet_ms > frame_ms), all frames in that packet are
    dropped together.

    Args:
        config:        Gilbert-Elliott model parameters.
        sample_rate:   audio sample rate (Hz).
        hop_length:    total encoder downsampling factor (1 latent frame
                       = hop_length samples). For Zero-Ping with
                       ratios=[8,5,3,2] this is 240 -> 15ms at 16kHz.
    """

    def __init__(
        self,
        config: GilbertElliottConfig,
        sample_rate: int = 16000,
        hop_length: int = 240, #as said before, 15ms (i.e. thats whats carryed in a single packet)
    ):
        self.config = config
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        frame_ms = 1000.0 * hop_length / sample_rate
        self.frames_per_packet = max(1, round(config.packet_ms / frame_ms))

    def _sample_one(self, T: int, rng: np.random.Generator) -> np.ndarray:
        """Generate a single frame_mask of length T."""
        cfg = self.config
        n_packets = math.ceil(T / self.frames_per_packet)

        # Initialize state by sampling from the stationary distribution
        pi_G = cfg.r / (cfg.p + cfg.r)
        state_good = bool(rng.random() < pi_G)

        mask = np.ones(T, dtype=np.float32)

        for i in range(n_packets):
            no_loss_prob = cfg.k if state_good else cfg.h
            if rng.random() >= no_loss_prob:
                # Drop all frames belonging to this packet
                start = i * self.frames_per_packet
                end = min(start + self.frames_per_packet, T)
                mask[start:end] = 0.0

            # Markov state transition
            if state_good:
                state_good = (rng.random() >= cfg.p)
            else:
                state_good = (rng.random() < cfg.r)

        return mask

    def sample_frame_mask(
        self,
        batch_size: int,
        num_frames: int,
        device: tp.Optional[torch.device] = None,
        seed: tp.Optional[int] = None,
    ) -> torch.Tensor:
        """
        Returns a float tensor [B, T'] with values in {0, 1}.
        where 1 = frame received, 0 = frame lost.
        Sequences in the batch are sampled independently.
        """
        rng = np.random.default_rng(seed)
        masks = np.stack([
            self._sample_one(num_frames, rng) for _ in range(batch_size)
        ], axis=0)
        out = torch.from_numpy(masks)
        if device is not None:
            out = out.to(device)
        return out
