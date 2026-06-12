"""
This code defines a multi-scale STFT discriminator architecture and its losses.
Prop to Meta's EnCodec for the real/imaginary Conv2D design and multi-scale STFT approach and implementation.

Default STFT scales are short/medium speech-oriented windows:
    n_ffts = [128, 256, 512, 1024]

At 16 kHz, these correspond to:
    128 samples  = 8 ms   -> very short glitches, clicks, burst/token-loss artifacts
    256 samples  = 16 ms  -> fast consonants, short packet/frame artifacts
    512 samples  = 32 ms  -> local phonetic and formant structure
    1024 samples = 64 ms  -> pitch harmonics and voiced speech stability

Hop lengths are n_fft / 4, matching the common multi-resolution STFT recipe.

The discriminator receives waveform audio, not spectrograms.
Each sub-discriminator computes its own complex STFT internally and represents it
with real/imaginary channels before applying Conv2D patch discrimination.

Note — this is NOT the same as MultiScaleSTFTLoss in losses.py:
  - MultiScaleSTFTLoss  : non-learned L1 reconstruction loss on STFT magnitudes,
                          windows [160, 320, 640, 1280], active from step 0.
  - SpeechMultiScaleSTFTDiscriminator (this file): learned GAN discriminator with
                          Conv2D on complex STFT, windows [128, 256, 512, 1024],
                          active only after GAN warmup. Outputs logits + feature maps
                          for hinge loss and feature matching.
"""

import typing as tp

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange

from Utils import NormConv2d, get_2d_padding


FeatureMapType = tp.List[torch.Tensor]
LogitsType = torch.Tensor
DiscriminatorOutput = tp.Tuple[tp.List[LogitsType], tp.List[FeatureMapType]]


class SpeechSTFTDiscriminator(nn.Module):
    """Single STFT patch discriminator for 16 kHz mono speech.

    Input:
        x: waveform tensor [B, 1, T]

    Output:
        logits:
            Patch logits [B, 1, T_stft, F_reduced].
            These are raw discriminator scores, not probabilities.

        fmap:
            Intermediate feature maps used for feature matching loss.

    Design:
        waveform
        -> complex STFT
        -> concatenate real/imag as channels
        -> Conv2D stack over [time, frequency]
        -> patch logits

    Why real/imag instead of only magnitude:
        For waveform codecs, phase-sensitive artifacts matter. Token loss,
        quantization, and packet-like corruption can produce transient smear,
        metallic artifacts, and phase instability. Real/imag keeps more
        information than magnitude-only STFT.
    """

    def __init__(
        self,
        filters: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        max_filters: int = 256,
        filters_scale: int = 1,
        kernel_size: tp.Tuple[int, int] = (3, 9),
        dilations: tp.Sequence[int] = (1, 2, 4),
        stride: tp.Tuple[int, int] = (1, 2),
        normalized: bool = True,
        norm: str = "weight_norm",
        activation: str = "LeakyReLU",
        activation_params: tp.Optional[dict] = None,
        center: bool = False,
    ) -> None:
        super().__init__()

        if activation_params is None:
            activation_params = {"negative_slope": 0.2}

        if in_channels != 1:
            raise ValueError(
                "This adapted discriminator is intentionally mono-first for 16 kHz speech. "
                f"Expected in_channels=1, got {in_channels}."
            )

        if len(kernel_size) != 2:
            raise ValueError(f"kernel_size must have length 2, got {kernel_size}.")

        if len(stride) != 2:
            raise ValueError(f"stride must have length 2, got {stride}.")

        if n_fft <= 0 or hop_length <= 0 or win_length <= 0:
            raise ValueError(
                f"n_fft, hop_length, and win_length must be positive. "
                f"Got n_fft={n_fft}, hop_length={hop_length}, win_length={win_length}."
            )

        if win_length > n_fft:
            raise ValueError(
                f"win_length must be <= n_fft. Got win_length={win_length}, n_fft={n_fft}."
            )

        if filters <= 0 or max_filters <= 0:
            raise ValueError(
                f"filters and max_filters must be positive. "
                f"Got filters={filters}, max_filters={max_filters}."
            )

        if filters_scale < 1:
            raise ValueError(f"filters_scale must be >= 1, got {filters_scale}.")

        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.center = center

        try:
            self.activation = getattr(nn, activation)(**activation_params)
        except AttributeError as exc:
            raise ValueError(f"Unknown activation function: {activation}") from exc

        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window_fn=torch.hann_window,
            normalized=normalized,
            center=center,
            pad_mode="reflect" if center else "constant",
            power=None,
        )

        # Mono complex STFT represented as real + imaginary channels.
        # [B, 1, F, T] complex -> [B, 2, F, T] real-valued.
        spec_channels = 2 * in_channels

        self.convs = nn.ModuleList()

        # Initial local time-frequency feature extractor.
        self.convs.append(
            NormConv2d(
                spec_channels,
                filters,
                kernel_size=kernel_size,
                padding=get_2d_padding(kernel_size),
                norm=norm,
            )
        )

        in_chs = filters

        # Dilated Conv2D blocks.
        # Dilation grows over time only: dilation=(dilation, 1).
        # Stride=(1, 2) preserves temporal resolution and compresses frequency.
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * filters, max_filters)

            self.convs.append(
                NormConv2d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=(int(dilation), 1),
                    padding=get_2d_padding(kernel_size, (int(dilation), 1)),
                    norm=norm,
                )
            )

            in_chs = out_chs

        # Final refinement convolution before logits.
        final_kernel = (kernel_size[0], kernel_size[0])
        out_chs = min((filters_scale ** (len(dilations) + 1)) * filters, max_filters)

        self.convs.append(
            NormConv2d(
                in_chs,
                out_chs,
                kernel_size=final_kernel,
                padding=get_2d_padding(final_kernel),
                norm=norm,
            )
        )

        # Final patch-classification head.
        self.conv_post = NormConv2d(
            out_chs,
            out_channels,
            kernel_size=final_kernel,
            padding=get_2d_padding(final_kernel),
            norm=norm,
        )

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, FeatureMapType]:
        """Run one STFT discriminator.

        Args:
            x:
                Waveform tensor with shape [B, 1, T].

        Returns:
            logits:
                Patch logits with shape [B, 1, T_stft, F_reduced].

            fmap:
                List of intermediate feature maps.
        """

        if x.ndim != 3:
            raise ValueError(f"Expected waveform [B, 1, T], got shape {tuple(x.shape)}.")

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected mono channel dimension={self.in_channels}, got {x.shape[1]}."
            )

        if x.shape[-1] < self.win_length:
            raise ValueError(
                f"Input waveform is shorter than win_length. "
                f"Got T={x.shape[-1]}, win_length={self.win_length}."
            )

        fmap: FeatureMapType = []

        # Complex STFT:
        # [B, 1, T] -> [B, 1, Freq, Frames], complex dtype.
        z = self.spec_transform(x)

        # Represent complex STFT as real-valued channels:
        # [B, 1, Freq, Frames] complex -> [B, 2, Freq, Frames].
        z = torch.cat([z.real, z.imag], dim=1)

        # Conv2D operates over [time, frequency]:
        # [B, 2, Freq, Frames] -> [B, 2, Frames, Freq].
        z = rearrange(z, "b c f t -> b c t f")

        for layer in self.convs:
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)

        logits = self.conv_post(z)
        return logits, fmap


class SpeechMultiScaleSTFTDiscriminator(nn.Module):
    """Multi-scale STFT discriminator for 16 kHz mono speech.

    Default scales:
        128 samples  = 8 ms   -> very short glitches, clicks, burst/token-loss artifacts
        256 samples  = 16 ms  -> fast consonants, short packet/frame artifacts
        512 samples  = 32 ms  -> local phonetic and formant structure
        1024 samples = 64 ms  -> pitch harmonics and voiced speech stability


    Input:
        x: waveform tensor [B, 1, T]

    Output:
        logits:
            List of logits, one tensor per STFT scale.

        fmaps:
            List of feature-map lists, one list per STFT scale.
    """

    def __init__(
        self,
        filters: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
        n_ffts: tp.Sequence[int] = (128, 256, 512, 1024),
        hop_lengths: tp.Optional[tp.Sequence[int]] = None,
        win_lengths: tp.Optional[tp.Sequence[int]] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        n_ffts = tuple(int(n_fft) for n_fft in n_ffts)

        if len(n_ffts) == 0:
            raise ValueError("n_ffts must contain at least one STFT scale.")

        if hop_lengths is None:
            hop_lengths = tuple(n_fft // 4 for n_fft in n_ffts)
        else:
            hop_lengths = tuple(int(hop) for hop in hop_lengths)

        if win_lengths is None:
            win_lengths = tuple(n_ffts)
        else:
            win_lengths = tuple(int(win) for win in win_lengths)

        if not (len(n_ffts) == len(hop_lengths) == len(win_lengths)):
            raise ValueError(
                "n_ffts, hop_lengths, and win_lengths must have the same length. "
                f"Got len(n_ffts)={len(n_ffts)}, "
                f"len(hop_lengths)={len(hop_lengths)}, "
                f"len(win_lengths)={len(win_lengths)}."
            )

        for n_fft, hop, win in zip(n_ffts, hop_lengths, win_lengths):
            if n_fft <= 0 or hop <= 0 or win <= 0:
                raise ValueError(
                    f"Invalid STFT scale: n_fft={n_fft}, hop_length={hop}, win_length={win}."
                )
            if win > n_fft:
                raise ValueError(
                    f"win_length must be <= n_fft. Got win_length={win}, n_fft={n_fft}."
                )

        self.n_ffts = n_ffts
        self.hop_lengths = hop_lengths
        self.win_lengths = win_lengths

        self.discriminators = nn.ModuleList(
            [
                SpeechSTFTDiscriminator(
                    filters=filters,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_fft=n_ffts[i],
                    hop_length=hop_lengths[i],
                    win_length=win_lengths[i],
                    **kwargs,
                )
                for i in range(len(n_ffts))
            ]
        )

        self.num_discriminators = len(self.discriminators)

    def forward(self, x: torch.Tensor) -> DiscriminatorOutput:
        """Run all STFT-scale discriminators.

        Args:
            x:
                Waveform tensor [B, 1, T].

        Returns:
            logits:
                List of patch-logit tensors, one per scale.

            fmaps:
                List of feature-map lists, one per scale.
        """

        logits: tp.List[LogitsType] = []
        fmaps: tp.List[FeatureMapType] = []

        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)

        return logits, fmaps



"""
 GAN training uses three distinct loss terms, each targeting a different set
 of parameters. The training step order matters:

   1. discriminator_hinge_loss  -> updates discriminator (opt_D)
                                   x_fake must be detached to block grad flow
                                   into the codec/generator

   2. generator_hinge_loss      -> updates generator/decoder (opt_G)
                                   x_fake must NOT be detached here

   3. feature_matching_loss     -> updates generator/decoder (opt_G)
                                   real fmaps must be detached so the generator
                                   moves toward real features without
                                   backpropagating through the discriminator

 (2) and (3) are summed into a single generator loss and optimized together.
"""

def discriminator_hinge_loss(
    real_logits: tp.Sequence[torch.Tensor],
    fake_logits: tp.Sequence[torch.Tensor],
) -> torch.Tensor:
    """Hinge loss for the D-step: trains the discriminator to score real audio
    high (>=1) and reconstructed codec audio low (<=-1).

    Updates: discriminator only.
    Call with: fake_logits from disc(x_fake.detach()) — detach is mandatory,
    otherwise gradients flow back into the codec during the D-step.

    Hinge formulation (per scale, averaged):
        L_D = E[relu(1 - real)] + E[relu(1 + fake)]

    The relu clips gradients when the discriminator is already correct by a
    margin of 1, making training more stable than a vanilla BCE or LSGAN loss.
    Averaged across STFT scales so lambda_adv doesn't depend on scale count.
    """
    if len(real_logits) != len(fake_logits):
        raise ValueError(
            f"real_logits and fake_logits must have the same length. "
            f"Got {len(real_logits)} and {len(fake_logits)}."
        )
    if len(real_logits) == 0:
        raise ValueError("Expected at least one discriminator logit tensor.")

    loss = real_logits[0].new_tensor(0.0)
    for real, fake in zip(real_logits, fake_logits):
        loss = loss + torch.mean(F.relu(1.0 - real))
        loss = loss + torch.mean(F.relu(1.0 + fake))

    return loss / len(real_logits)


def generator_hinge_loss(
    fake_logits: tp.Sequence[torch.Tensor],
) -> torch.Tensor:
    """Adversarial loss for the G-step: trains the codec/decoder to fool the
    discriminator, i.e. push reconstructed audio logits as high as possible.

    Updates: generator/decoder only (discriminator frozen via freeze_discriminator).
    Call with: fake_logits from disc(x_fake) — do NOT detach x_fake here,
    gradients must flow back into the codec.

    Hinge formulation (per scale, averaged):
        L_G_adv = -E[fake]

    Simply maximizes the discriminator score on fake audio. No margin clipping
    on the generator side — the generator should always want higher scores.
    """
    if len(fake_logits) == 0:
        raise ValueError("Expected at least one fake logit tensor.")

    loss = fake_logits[0].new_tensor(0.0)
    for fake in fake_logits:
        loss = loss - torch.mean(fake)

    return loss / len(fake_logits)


def feature_matching_loss(
    real_fmaps: tp.Sequence[FeatureMapType],
    fake_fmaps: tp.Sequence[FeatureMapType],
) -> torch.Tensor:
    """Feature matching loss for the G-step: encourages the codec to produce
    audio whose discriminator-internal representations match those of real audio.

    Updates: generator/decoder only.
    This is a perceptual-style loss computed inside the discriminator's Conv2D
    stack — each intermediate activation is a learned time-frequency feature
    detector. Matching these features provides a denser training signal than
    the scalar adversarial logit alone, and typically stabilizes GAN training
    by reducing mode collapse and gradient variance.

    real_fmaps must be detached: the generator should move toward the real
    features, not update the discriminator through this term.

    L_fm = mean over all scales and layers of L1(fake_fmap, real_fmap.detach())

    Averaged across all (scale, layer) pairs so lambda_fm is scale-invariant.
    """
    if len(real_fmaps) != len(fake_fmaps):
        raise ValueError(
            f"real_fmaps and fake_fmaps must have the same number of scales. "
            f"Got {len(real_fmaps)} and {len(fake_fmaps)}."
        )
    if len(real_fmaps) == 0:
        raise ValueError("Expected at least one feature-map scale.")

    loss = real_fmaps[0][0].new_tensor(0.0)
    n = 0
    for real_scale, fake_scale in zip(real_fmaps, fake_fmaps):
        if len(real_scale) != len(fake_scale):
            raise ValueError(
                f"Feature-map layer count mismatch: "
                f"got {len(real_scale)} real layers and {len(fake_scale)} fake layers."
            )
        for real_fmap, fake_fmap in zip(real_scale, fake_scale):
            loss = loss + F.l1_loss(fake_fmap, real_fmap.detach())
            n += 1

    if n == 0:
        raise ValueError("No feature maps found for feature matching loss.")

    return loss / n


def freeze_discriminator(discriminator: nn.Module) -> None:
    """Freeze discriminator weights during the Generator-step to prevent unintended updates."""
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)


def unfreeze_discriminator(discriminator: nn.Module) -> None:
    """Unfreeze discriminator weights before the Discriminator-step."""
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
