from __future__ import annotations

from logging import getLogger
from typing import Sequence

from vspreview.plugins import MappedNodesViewPlugin, PluginConfig
from vstools import DitherType, depth, initialize_input, split, stack_clips, vs

__all__ = ["DFTViewPlugin"]


__version__ = "1.0.0"


class DFTViewPlugin(MappedNodesViewPlugin):
    """A plugin that displays the Discrete Fourier Transform (DFT) of all planes of a video node."""

    _autofit = True

    @initialize_input(bits=32, dither_type=DitherType.NONE)
    def get_node(self, node: vs.VideoNode) -> vs.VideoNode:
        if not hasattr(vs.core, "fftspectrum_rs"):
            message = (
                'The required plugin "fftspectrum_rs" is not installed. '
                'Please install it from: "https://github.com/sgt0/vapoursynth-fftspectrum-rs"'
            )
            getLogger().error(message)
            return node.std.BlankClip().text.Text(message, 5, 4)

        assert node.format
        planes = split(node)

        if len(planes) == 1:
            return planes[0].fftspectrum_rs.FFTSpectrum()

        planes = [p.fftspectrum_rs.FFTSpectrum().text.Text(text=k) for k, p in zip(node.format.name, planes)]

        org: Sequence[vs.VideoNode | Sequence[vs.VideoNode]]

        if node.format.subsampling_w == 2:
            middle = [(blank := planes[1].std.BlankClip(keep=True)), *planes[1:], blank]
            org = [planes[0], middle]

        elif node.format.subsampling_w == 1:
            org = [planes[0], planes[1:]]
        else:
            org = planes

        return depth(stack_clips([org]), 8, dither_type=DitherType.NONE)
