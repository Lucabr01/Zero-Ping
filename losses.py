"""
All the loss functions for the codec (generator).

Main references:
  - SoundStream  (Zeghidour et al., 2021)
  - EnCodec      (Défossez et al., 2022)
  - RVQGAN/DAC   (Kumar et al., 2023)

Discriminators, feature matching, and the balancer are handled elsewhere.

Components:
  - MultiScaleMelLoss   : L1 on log-mel at multiple STFT scales (RVQGAN-style,
                          but with bins adapted to the 0–8 kHz band).
  - time_domain_loss    : L1 on the waveform (with auto-crop to minimum length).
  - latent_repair_loss  : L1 in the RVQ latent domain, masked to missing frames
                          only (only transformer in the second stage).
  - MultiScaleSTFTLoss  : L1 on linear STFT magnitudes at multiple scales,
                          windows [160, 320, 640, 1280]. Non-learned reconstruction loss —
                          NOT the same as SpeechMultiScaleSTFTDiscriminator in
                          msstftd_16k_speech.py, which is a learned GAN discriminator
                          with Conv2D on complex STFT. This was added later on, why? Because the Mel loss,
                          while perceptually relevant, does not penalize HF distortion enough: with only the 
                          Mel loss the model was producing very noisy outputs with good formants but bad HF content (sibilants, plosives).

                          CHECK pics/MSSTFT_comparison.png for a visual comparison to better understand the difference.

  - CodecLoss           : aggregator with configurable weights and a component
                          dict for logging.

Default weights chosen for 16 kHz voice chat without balancer (EnCodec used one):
    λ_time   = 1.0    (phase / waveform fidelity, secondary driver)
    λ_mel    = 15.0   (main perceptual driver, RVQGAN-style)
    λ_commit = 0.25   (RVQ commit loss, RVQGAN-style)
    λ_repair = 0.0    (direct supervision on missing frames - used only in the second stage as the main driver) - 0 cuz added later, applyed only during the transformer initialization! balanced with mel at 50-50 
    λ_stft   = 0.0    (direct penalty on linear STFT magnitudes) - added later! wanna see why? check the picture in the \pics directory.
"""

import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio



class MultiScaleMelLoss(nn.Module):
    """
    L1 on log-mel spectrograms at multiple STFT scales.

    Pattern: window_lengths with hop = win/4, variable n_mels per scale.
    Adapted for this setup (16 kHz, voice):
      - capped at ~80 mel bins (RVQGAN uses 320 at 44.1 kHz, overkill here)
      - small windows (32, 64) kept: capture plosives and transients
      - large windows (1024, 2048) for spectral envelope and prosody

    For each scale, computes both L1 on linear mel and L1 on log-mel:
      - linear: weights energetic peaks more (formants, plosives)
      - log:    also weights weak components (fricatives)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_lengths: tp.List[int] = [64, 128, 256, 512, 1024, 2048],
        n_mels: tp.List[int] = [10, 20, 40, 64, 80, 80],
        log_eps: float = 1e-5,
        f_min: float = 0.0,
        f_max: tp.Optional[float] = None,
    ):
        super().__init__()
        assert len(window_lengths) == len(n_mels), (
            "window_lengths and n_mels must have the same length"
        )

        self.window_lengths = window_lengths
        self.n_mels = n_mels
        self.log_eps = log_eps

        # one MelSpectrogram transform per scale
        self.mel_transforms = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=w,
                win_length=w,
                hop_length=w // 4,
                n_mels=n,
                f_min=f_min,
                f_max=f_max,
                power=1.0,            # amplitude, not power
                center=True,
                norm='slaney',
                mel_scale='slaney',
            )
            for w, n in zip(window_lengths, n_mels)
        ])

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        x, x_hat: [B, 1, T] or [B, T]. Different lengths OK: min-crop applied.
        Returns a scalar averaged across scales.
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        if x_hat.dim() == 3:
            x_hat = x_hat.squeeze(1)

        T = min(x.shape[-1], x_hat.shape[-1])
        x = x[..., :T]
        x_hat = x_hat[..., :T]

        loss = x.new_zeros(())
        for mel_t in self.mel_transforms:
            mel_x = mel_t(x)
            mel_xhat = mel_t(x_hat)

            loss = loss + F.l1_loss(mel_x, mel_xhat)
            loss = loss + F.l1_loss(
                torch.log(mel_x + self.log_eps),
                torch.log(mel_xhat + self.log_eps),
            )

        # average across scales so the external weight doesn't depend on scale count
        return loss / len(self.mel_transforms)



class MultiScaleSTFTLoss(nn.Module):
    """
    L1 on linear STFT magnitude spectrograms at multiple scales.

    Unlike MultiScaleMelLoss (perceptual, log-spaced mel bins), frequency bins
    here are LINEAR, so high frequencies receive more bins proportionally and
    contribute MORE directly to the loss.

    Distortion comparison at +15dB @ 5kHz (16kHz audio):
      - Mel:  ~3-5 bins out of 80  → ~5% of the loss signal
      - STFT: ~200 bins out of 640 → ~30% of the loss signal

    For each scale computes:
      - L1 on linear magnitude : weights energetic peaks more
      - L1 on log magnitude    : also weights weak components

    Default windows are speech-friendly for 16kHz voice:
      win=160  (10ms) → bin=50Hz, HF transients (fricatives, plosives)
      win=320  (20ms) → bin=50Hz, standard speech frame
      win=640  (40ms) → bin=25Hz, formants
      win=1280 (80ms) → bin=12.5Hz, prosody/envelope
    """

    def __init__(
        self,
        window_lengths: tp.List[int] = [160, 320, 640, 1280],
        log_eps: float = 1e-7,
    ):
        super().__init__()
        self.window_lengths = window_lengths
        self.log_eps = log_eps
        # pre-register Hann windows as buffers to avoid recreating them each forward
        for w in window_lengths:
            self.register_buffer(
                f"win_{w}", torch.hann_window(w), persistent=False
            )

    def _stft_mag(self, x: torch.Tensor, win_len: int) -> torch.Tensor:
        """STFT magnitude on [B, T] or [T] -> [B, F, T'] with F=win_len/2+1."""
        window = getattr(self, f"win_{win_len}")
        spec = torch.stft(
            x,
            n_fft=win_len,
            hop_length=win_len // 4,
            win_length=win_len,
            window=window,
            center=True,
            return_complex=True,
            pad_mode="reflect",
        )
        return spec.abs()

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        x, x_hat: [B, 1, T] or [B, T]. Different lengths OK: min-crop applied.
        Returns a scalar averaged across scales.
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        if x_hat.dim() == 3:
            x_hat = x_hat.squeeze(1)

        T = min(x.shape[-1], x_hat.shape[-1])
        x = x[..., :T]
        x_hat = x_hat[..., :T]

        loss = x.new_zeros(())
        for w in self.window_lengths:
            # cast to float32: bf16 autocast does not support torch.stft
            with torch.autocast(device_type=x.device.type, enabled=False):
                mag_x = self._stft_mag(x.float(), w)
                mag_xhat = self._stft_mag(x_hat.float(), w)

            loss = loss + F.l1_loss(mag_x, mag_xhat)
            loss = loss + F.l1_loss(
                torch.log(mag_x + self.log_eps),
                torch.log(mag_xhat + self.log_eps),
            )

        return loss / len(self.window_lengths)


def time_domain_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    L1 on the waveform with auto-crop to minimum length.
    Needed because the decoder may produce a few extra samples due to
    asymmetric convolution padding.
    """
    T = min(x.shape[-1], x_hat.shape[-1])
    return F.l1_loss(x[..., :T], x_hat[..., :T])



def latent_repair_loss(
    z_pre: torch.Tensor,
    z_post: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    """
    L1 between the clean RVQ latent (target) and the latent AFTER _apply_repair,
    masked to frames where frame_mask == 0 (missing).

    On received frames the two tensors are identical by construction (selective
    substitution in ZPCodec._apply_repair), so the mask isolates the useful
    supervision signal to missing frames only.

    Args:
        z_pre:      [B, D, T'] original quantized latent (pre-repair)
        z_post:     [B, D, T'] latent after _apply_repair
        frame_mask: [B, T']    1 = received, 0 = missing

    Returns:
        scalar: L1 averaged over missing positions.
        If no frame is missing in the batch, returns 0 (no-op gradient).
    """
    assert z_pre.shape == z_post.shape, (
        f"shape mismatch: z_pre {tuple(z_pre.shape)} vs z_post {tuple(z_post.shape)}"
    )
    B, D, T_prime = z_pre.shape
    assert frame_mask.shape == (B, T_prime), (
        f"frame_mask must be [B, T']; got {tuple(frame_mask.shape)}"
    )

    missing = (1.0 - frame_mask).unsqueeze(1).to(z_pre.dtype)   # [B, 1, T']
    n_missing = missing.sum()

    if n_missing.item() < 1.0:
        # no missing frames in the batch: zero gradient, zero scalar
        return z_pre.new_zeros(())

    diff = (z_pre - z_post).abs()                                # [B, D, T']
    return (diff * missing).sum() / (n_missing * D)



class CodecLoss(nn.Module):
    """
    Aggregates generator losses with configurable weights.

    Returns (total_loss, components_dict) where components are detached for
    logging. Repair loss is optional: included only when z_pre, z_post, and
    frame_mask are all provided.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        lambda_time: float = 1.0,
        lambda_mel: float = 15.0,
        lambda_commit: float = 0.25,
        lambda_repair: float = 0.0,
        lambda_stft: float = 0.0,
        mel_window_lengths: tp.List[int] = [64, 128, 256, 512, 1024, 2048],
        mel_n_mels: tp.List[int] = [10, 20, 40, 64, 80, 80],
        stft_window_lengths: tp.List[int] = [160, 320, 640, 1280],
    ):
        super().__init__()
        self.mel_loss = MultiScaleMelLoss(
            sample_rate=sample_rate,
            window_lengths=mel_window_lengths,
            n_mels=mel_n_mels,
        )
        self.lambda_time = lambda_time
        self.lambda_mel = lambda_mel
        self.lambda_commit = lambda_commit
        self.lambda_repair = lambda_repair
        self.lambda_stft = lambda_stft

        # instantiate STFT loss only if needed to avoid unnecessary compute
        if lambda_stft > 0.0:
            self.stft_loss = MultiScaleSTFTLoss(
                window_lengths=stft_window_lengths,
            )
        else:
            self.stft_loss = None

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        commit_loss: torch.Tensor,
        z_pre: tp.Optional[torch.Tensor] = None,
        z_post: tp.Optional[torch.Tensor] = None,
        frame_mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, torch.Tensor]]:
        """
        Args:
            x:           [B, 1, T]  original audio
            x_hat:       [B, 1, T]  reconstructed audio (T may differ)
            commit_loss: scalar     already averaged by the RVQ
            z_pre:       [B, D, T'] RVQ latent pre-repair  (optional)
            z_post:      [B, D, T'] latent post-repair      (optional)
            frame_mask:  [B, T']    GE mask                 (optional)

        Returns:
            total_loss, {'time', 'mel', 'commit', ('repair'), 'total'}
        """
        l_time = time_domain_loss(x, x_hat)
        l_mel = self.mel_loss(x, x_hat)
        l_commit = commit_loss

        loss = (
            self.lambda_time * l_time
            + self.lambda_mel * l_mel
            + self.lambda_commit * l_commit
        )

        components = {
            'time': l_time.detach(),
            'mel': l_mel.detach(),
            'commit': l_commit.detach(),
        }

        # STFT loss: linear bins -> directly penalizes HF distortion
        if self.stft_loss is not None and self.lambda_stft > 0.0:
            l_stft = self.stft_loss(x, x_hat)
            loss = loss + self.lambda_stft * l_stft
            components['stft'] = l_stft.detach()

        repair_provided = (
            z_pre is not None and z_post is not None and frame_mask is not None
        )
        if repair_provided:
            l_repair = latent_repair_loss(z_pre, z_post, frame_mask)
            loss = loss + self.lambda_repair * l_repair
            components['repair'] = l_repair.detach()

        components['total'] = loss.detach()
        return loss, components
