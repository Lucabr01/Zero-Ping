"""
trainer.py — Training wrapper.

Bundles together:
  - ZPCodec                              (encoder + RVQ + repairTransformer + decoder)
  - SpeechMultiScaleSTFTDiscriminator    
  - CodecLosses                           

Exposes three atomic operations, to be called in order every step:
  1. forward_codec(x)                          -> CodecOutputs   (single forward)
  2. discriminator_loss(x, x_fake.detach())    -> scalar         (for opt_D)
  3. generator_losses(x, outputs, use_adv=...) -> dict           (for opt_G)

Intended training loop pattern:

    outputs = trainer.forward_codec(x)

    # --- D-step ---
    loss_D = trainer.discriminator_loss(x, outputs.x_fake.detach())
    opt_D.zero_grad(); loss_D.backward(); opt_D.step()

    # --- G-step ---
    losses_G = trainer.generator_losses(x, outputs,
                                        use_adversarial=(step >= warmup))
    opt_G.zero_grad(); losses_G['total'].backward(); opt_G.step()

Why a single forward_codec call?
    x_fake must be the same tensor for both the D-step and the G-step.
    If we called the codec twice we'd get two different outputs (dropout,
    GE mask sampling), and the D would be trained on a different sample
    than the G breaking the adversarial game.

Default loss weights:
  - lambda_time    = 1.0     waveform L1
  - lambda_mel     = 15.0    multi-scale mel (main perceptual driver)
  - lambda_commit  = 0.25    RVQ commit loss
  - lambda_repair  = 0.0     latent repair on missing frames (only when use_repair=True)
  - lambda_adv     = 1.0     hinge GAN adversarial loss
  - lambda_fm      = 2.0     feature matching loss

Default mel scales: [64, 128, 256, 512, 1024, 2048] with n_mels [10, 20, 40, 64, 80, 80].
Starting from 64 instead of 32 (RVQGAN default).
"""

from dataclasses import dataclass
import typing as tp

import torch
import torch.nn as nn

from model import ZPCodec
from losses import CodecLoss
from msstftd_16k_speech import (
    SpeechMultiScaleSTFTDiscriminator,
    discriminator_hinge_loss,
    generator_hinge_loss,
    feature_matching_loss,
    freeze_discriminator,
    unfreeze_discriminator,
)


@dataclass
class CodecOutputs:
    """
    All intermediates produced by a single codec forward pass.
    Storing them here avoids a second forward call: both the D-step and the
    G-step consume the same x_fake, z_pre, z_post, and frame_mask.

    Attributes:
        x_fake:      [B, 1, T_out] reconstructed audio (T_out may differ from T
                     due to asymmetric causal convolution padding)
        commit_loss: scalar, already averaged across quantizers by ResidualVQ
        z_pre:       [B, D, T'] quantized latent before repair
        z_post:      [B, D, T'] quantized latent after repair (== z_pre if use_repair=False)
        frame_mask:  [B, T'] or None if use_repair=False
    """
    x_fake: torch.Tensor
    commit_loss: torch.Tensor
    z_pre: torch.Tensor
    z_post: torch.Tensor
    frame_mask: tp.Optional[torch.Tensor]


class ZPCodecTrainer(nn.Module):
    def __init__(
        self,
        codec: ZPCodec,
        # --- discriminator ---
        disc_filters: int = 32,
        disc_n_ffts: tp.Tuple[int, ...] = (128, 256, 512, 1024),
        # --- loss weights ---
        lambda_time: float = 1.0,
        lambda_mel: float = 15.0,
        lambda_commit: float = 0.25,
        lambda_repair: float = 0.0,
        lambda_adv: float = 1.0,
        lambda_fm: float = 2.0,
        lambda_stft: float = 0.0,
        # --- mel loss params ---
        mel_window_lengths: tp.List[int] = [64, 128, 256, 512, 1024, 2048],
        mel_n_mels: tp.List[int] = [10, 20, 40, 64, 80, 80],
        # --- stft loss params ---
        stft_window_lengths: tp.List[int] = [160, 320, 640, 1280],
    ):
        super().__init__()
        self.codec = codec
        self.discriminator = SpeechMultiScaleSTFTDiscriminator(
            filters=disc_filters,
            n_ffts=disc_n_ffts,
        )
        self.codec_loss = CodecLoss(
            sample_rate=codec.sample_rate,
            lambda_time=lambda_time,
            lambda_mel=lambda_mel,
            lambda_commit=lambda_commit,
            lambda_repair=lambda_repair,
            lambda_stft=lambda_stft,
            mel_window_lengths=mel_window_lengths,
            mel_n_mels=mel_n_mels,
            stft_window_lengths=stft_window_lengths,
        )
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm

    # Parameter groups for separate optimizers
    def generator_parameters(self):
        """Codec parameters (encoder, RVQ, repair, decoder) — optimized by opt_G."""
        return self.codec.parameters()

    def discriminator_parameters(self):
        """MS-STFT discriminator parameters — optimized by opt_D."""
        return self.discriminator.parameters()

    # 1) Single codec forward (call once per step, reuse for both D and G)
    def forward_codec(
        self,
        x: torch.Tensor,
        frame_mask: tp.Optional[torch.Tensor] = None,
    ) -> CodecOutputs:
        """
        Run one codec forward and collect all intermediates needed for both steps.
        Must be called BEFORE discriminator_loss and generator_losses so that
        both steps operate on the same x_fake (no double forward).

        If use_repair=True and frame_mask is None, the GE simulator inside the
        codec samples a new mask automatically.
        """
        x_fake, commit_loss, z_pre, z_post, fmask = self.codec(
            x, frame_mask=frame_mask, return_intermediates=True
        )
        return CodecOutputs(
            x_fake=x_fake,
            commit_loss=commit_loss,
            z_pre=z_pre,
            z_post=z_post,
            frame_mask=fmask,
        )

    # 2) D-step: discriminator loss
    def discriminator_loss(
        self,
        x_real: torch.Tensor,
        x_fake_detached: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hinge loss across all STFT scales of the discriminator.

        x_fake_detached MUST be outputs.x_fake.detach() — passing a live tensor
        would propagate gradients into the codec during the D-step, coupling
        the two optimizers incorrectly.

        Auto-crops to minimum length: the causal decoder may produce a few extra
        samples due to asymmetric convolution padding.
        """
        unfreeze_discriminator(self.discriminator)
        T = min(x_real.shape[-1], x_fake_detached.shape[-1])
        x_real = x_real[..., :T]
        x_fake_detached = x_fake_detached[..., :T]
        real_logits, _ = self.discriminator(x_real)
        fake_logits, _ = self.discriminator(x_fake_detached)
        return discriminator_hinge_loss(real_logits, fake_logits)

    # 3) G-step: generator loss (weighted sum of all components)
    def generator_losses(
        self,
        x_real: torch.Tensor,
        outputs: CodecOutputs,
        use_adversarial: bool = True,
    ) -> tp.Dict[str, torch.Tensor]:
        """
        Compute all generator loss components and aggregate them.

        Args:
            x_real:          [B, 1, T] original audio
            outputs:         result of forward_codec — x_fake must still have grad
            use_adversarial: if False, skip adv and fm losses. Typically False
                             during the initial GAN warmup phase.

        Returns:
            dict with keys:
              - 'time', 'mel', 'commit', ['repair'], ['adv'], ['fm'], ['stft']:
                detached scalars for logging
              - 'total': live tensor (with grad) — call .backward() on this
        """
        repair_active = outputs.frame_mask is not None
        loss_recon, recon_components = self.codec_loss(
            x=x_real,
            x_hat=outputs.x_fake,
            commit_loss=outputs.commit_loss,
            z_pre=outputs.z_pre if repair_active else None,
            z_post=outputs.z_post if repair_active else None,
            frame_mask=outputs.frame_mask,
        )

        # Drop the detached 'total' from codec_loss — we recompute it below
        # after adding the adversarial terms.
        recon_components.pop('total', None)
        components: tp.Dict[str, torch.Tensor] = dict(recon_components)

        if use_adversarial:
            # Freeze discriminator so its weights don't receive gradients
            # from the generator loss path.
            freeze_discriminator(self.discriminator)
            T = min(x_real.shape[-1], outputs.x_fake.shape[-1])
            x_real_c = x_real[..., :T]
            x_fake_c = outputs.x_fake[..., :T]

            _, real_fmaps = self.discriminator(x_real_c)
            fake_logits, fake_fmaps = self.discriminator(x_fake_c)

            l_adv = generator_hinge_loss(fake_logits)
            l_fm = feature_matching_loss(real_fmaps, fake_fmaps)

            loss_total = (
                loss_recon
                + self.lambda_adv * l_adv
                + self.lambda_fm * l_fm
            )
            components['adv'] = l_adv.detach()
            components['fm'] = l_fm.detach()
        else:
            loss_total = loss_recon

        # 'total' is the only live tensor — all others are detached for logging
        components['total'] = loss_total
        return components
