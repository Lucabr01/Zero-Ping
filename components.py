
'''
Here you can find the encoder and decoder architecture. The encoder takes in the raw waveform and produces a sequence of latent frames, which are then quantized with RVQ and fed to the transformer.
The decoder takes in the (possibly corrupted) latent frames and reconstructs the waveform.
'''

import typing as tp

import numpy as np
import torch.nn as nn
from Utils import Snake, SLSTM, SConv1d, SConvTranspose1d


class ZPEncoderResnetBlock(nn.Module):
    """Residual block with a bottleneck conv stack; used in both encoder and decoder. Yeah, wrong name :) """
    """props to the code implementation to Meta's EnCodec"""
    def __init__(self, dim: int, kernel_sizes: tp.List[int] = [3, 1], dilations: tp.List[int] = [1, 1],
                 norm: str = 'weight_norm', causal: bool = False, compress: int = 2, make_act=None):
        super().__init__()
        assert len(kernel_sizes) == len(dilations)
        if make_act is None:
            make_act = lambda _: nn.ELU()
        pad_mode = 'constant' if causal else 'reflect'
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                make_act(in_chs),
                SConv1d(in_chs, out_chs, kernel_size=kernel_size, dilation=dilation,
                        norm=norm, causal=causal, pad_mode=pad_mode),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut = nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class ZPEncoder(nn.Module):
    """
    Causal convolutional encoder: waveform -> latent frames.
    Architecture: stem conv -> (resnet blocks + strided conv) x len(ratios) -> LSTM -> projection.
    hop_length = prod(ratios); default 240 samples = 15ms at 16kHz.
    """
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        ratios: tp.List[int] = [8, 5, 3, 2],
        norm: str = 'weight_norm',
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 7,
        residual_dilations: tp.List[int] = [1, 3, 9],
        causal: bool = True,
        compress: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        self.hop_length = np.prod(self.ratios)
        self.residual_dilations = residual_dilations

        pad_mode = 'constant' if causal else 'reflect'
        mult = 1

        model: tp.List[nn.Module] = [
            SConv1d(channels, mult * n_filters, kernel_size, norm=norm, causal=causal, pad_mode=pad_mode)
        ]

        for ratio in self.ratios:
            # dilated residual stack before each downsampling step (like soundstream) to catch long-term dependencies early on.
            # useful for our use case...
            for dilation in self.residual_dilations:
                model += [
                    ZPEncoderResnetBlock(
                        mult * n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation, 1],
                        norm=norm,
                        causal=causal,
                        compress=compress,
                    )
                ]
            # strided conv doubles channels while halving time resolution
            model += [
                nn.ELU(),
                SConv1d(mult * n_filters, mult * n_filters * 2, kernel_size=ratio * 2,
                        stride=ratio, norm=norm, causal=causal, pad_mode=pad_mode),
            ]
            mult *= 2
        ''' Only one LSTM layer while EnCodec has 2, this is because (counting up the transformer and the 30ms lookahead)
            with 2 the total latency was too high to be realtime. Tested with RTF...
        '''
        model += [SLSTM(mult * n_filters, num_layers=1)]
        model += [
            nn.ELU(),
            SConv1d(mult * n_filters, dimension, last_kernel_size, norm=norm, causal=causal, pad_mode=pad_mode),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ZPDecoder(nn.Module):
    """
    Causal convolutional decoder: latent frames -> waveform.
    Mirror of ZPEncoder: projection -> LSTM -> (transposed conv + resnet blocks) x len(ratios) -> Tanh output.
    Uses Snake activations in residual blocks (better for audio periodicity), props to the improved RVQGAN paper.
    In the encoder the snake activations didn't improve much, so I left ELU there. For waveform reconstruction they are a game changer.
    """
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        ratios: tp.List[int] = [8, 5, 3, 2],
        norm: str = 'weight_norm',
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 7,
        residual_dilations: tp.List[int] = [1, 3, 9],
        causal: bool = True,
        compress: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = ratios
        self.hop_length = np.prod(self.ratios)
        self.residual_dilations = residual_dilations

        pad_mode = 'constant' if causal else 'reflect'
        mult = 2 ** len(ratios)  # start at maximum channel width

        model: tp.List[nn.Module] = [
            SConv1d(dimension, mult * n_filters, kernel_size, norm=norm, causal=causal, pad_mode=pad_mode)
        ]
        model += [SLSTM(mult * n_filters, num_layers=1)]

        make_act = lambda ch: Snake(ch)

        for ratio in self.ratios:
            # transposed conv halves channels while doubling time resolution
            model += [
                nn.ELU(),
                SConvTranspose1d(mult * n_filters, mult * n_filters // 2, kernel_size=ratio * 2,
                                 stride=ratio, norm=norm, causal=causal),
            ]
            mult //= 2
            for dilation in self.residual_dilations:
                model += [
                    ZPEncoderResnetBlock(
                        mult * n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation, 1],
                        norm=norm,
                        causal=causal,
                        compress=compress,
                        make_act=make_act,
                    )
                ]

        model += [
            nn.ELU(),
            SConv1d(mult * n_filters, channels, last_kernel_size, norm=norm, causal=causal, pad_mode=pad_mode),
            nn.Tanh(),  # clamp output to [-1, 1]
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
