from abc import abstractmethod

from jetpytools import complex_hash
from vskernels import Bicubic, BicubicSharp, Bilinear, Catrom, Kernel, KernelLike, Lanczos, Mitchell
from vstools import DynamicClipsCache, Matrix, VSObjectABC, core, vs

common_kernels: list[Kernel] = [
    # Bicubic-based
    Catrom(),
    Mitchell(),
    BicubicSharp(),
    # Bilinear-based
    Bilinear(),
    # Lanczos-based
    Lanczos(taps=2),
    Lanczos(taps=3),
    Lanczos(taps=4),
]


def get_kernel_name(kernel: KernelLike) -> str:
    name = kernel.__class__.__name__

    if name.lower().endswith("bicubic"):
        name = name[: -len("bicubic")]

    if isinstance(kernel, Bicubic):

        def _e(x: float) -> str:
            return str(int(x)) if float(x).is_integer() else f"{x:.2f}"

        name += f" (b={_e(kernel.b)}, c={_e(kernel.c)})"
    elif isinstance(kernel, Lanczos):
        name += f" (taps={kernel.taps})"

    return name


class RescaleWorkClip(DynamicClipsCache[vs.VideoNode]):
    def get_clip(self, key: vs.VideoNode) -> vs.VideoNode:
        return core.resize.Bilinear(key, format=vs.GRAYS, matrix=Matrix.BT709, matrix_in=Matrix.from_video(key))


class DynamicDataCache[T, R](dict[T, R], VSObjectABC):
    def __init__(self, cache_size: int = 2) -> None:
        self.cache_size = cache_size

    @abstractmethod
    def get_data(self, key: T) -> R:
        raise NotImplementedError

    def __getitem__(self, args: T, /) -> R:
        __key = complex_hash.hash(args)

        if __key not in self:
            self[__key] = self.get_data(args)  # type: ignore

            if len(self) > self.cache_size:
                del self[next(iter(self.keys()))]

        return super().__getitem__(__key)  # type: ignore
