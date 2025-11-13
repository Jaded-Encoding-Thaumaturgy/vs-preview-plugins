from __future__ import annotations

from typing import Any, Callable

import numpy as np
from jetpytools import mod2
from numpy.typing import NDArray
from PyQt6.QtWidgets import QFrame, QLabel
from scipy.fft import dct
from scipy.signal import argrelextrema
from vspreview.core import (
    CroppingInfo,
    DoubleSpinBox,
    ExtendedWidget,
    Frame,
    HBoxLayout,
    PlotMouseEvent,
    PlottingCanvasDefaultFrame,
    SpinBox,
    Switch,
    VBoxLayout,
)
from vspreview.main import MainWindow
from vstools import ClipFramesCache, ClipsCache, depth, padder, plane, vs


class FrequencyCanvas(PlottingCanvasDefaultFrame):
    def __init__(self, plugin: FrequencyAnalyzer) -> None:
        super().__init__(plugin.main, True, False, True, 5)

        self.plugin = plugin

    def on_mouse_moved(self, event: PlotMouseEvent) -> None:
        width_check = self.plugin.dimension_switch.isChecked()

        assert event.xdata is not None

        self.plugin.position.setText(
            f"{'Width' if width_check else 'Height'}: {int(event.xdata)}, "
            f"Error: {self.axes.lines[0].get_ydata()[int(event.xdata)]:.3f}"  # type: ignore
        )

    def _render(self, frame: Frame) -> None:
        dct, dct_cross, dct_up = self.plugin.get_dct_value(int(frame))

        inter_norm = self.plugin.get_inter(dct, dct_up, *self.plugin.check_range)
        inter_cross = self.plugin.get_inter(dct_cross, None, *self.plugin.check_range)

        current_dimension = "Width" if self.plugin.dimension_switch.isChecked() else "Height"
        opposite_dimension = "Height" if self.plugin.dimension_switch.isChecked() else "Width"

        results_text = "Spikes: "
        if len(inter_norm):
            results_text += ", ".join(map(str, sorted(set(inter_norm))))
        else:
            results_text += f"None found. ({len(inter_cross)} found on {opposite_dimension})"

        self.plugin.results.setText(results_text)

        self.axes.plot(dct, label=current_dimension)
        self.axes.plot(dct_cross, label=opposite_dimension)

        for inter, dct_base, color in [(inter_norm, dct, "r"), (inter_cross, dct_cross, "b")]:
            if len(inter):
                self.axes.scatter(inter, dct_base[inter], linewidth=0.3, s=self.plugin.check_radius, c=color)


class FrequencyAnalyzer(ExtendedWidget):
    def __init__(self, main: MainWindow) -> None:
        super().__init__()

        self.main = main

        self.clip_cache = ClipsCache()
        self.cropped_clip_cache = ClipsCache()
        self.transp_clip_cache = ClipsCache()
        self.padded_clip_cache = ClipsCache()
        self.upscaled_clip_cache = ClipsCache()

        self.frames_cache = ClipFramesCache()

        self.minmax_values = dict[tuple[int, int], tuple[int, int, int, int]]()

        self.first_paint = True

    def setup_ui(self) -> None:
        self.results = QLabel()
        self.position = QLabel()

        font = self.results.font()
        font.setPixelSize(15)

        self.results.setFont(font)
        self.position.setFont(font)

        self.results.setMaximumHeight(25)
        self.position.setMaximumHeight(25)

        self.canvas = FrequencyCanvas(self)

        self.dimension_switch = Switch(
            12, 40, 1, ("Height", "Width", 2), checked=False, clicked=self.on_dimension_to_check_change
        )

        on_update_cb = self.on_update_values(False, False, True, True, True, True)

        self.min_max_label = QLabel()

        self.height_min_spinbox = SpinBox(None, 0, None, "px", valueChanged=on_update_cb)
        self.height_max_spinbox = SpinBox(None, 0, None, "px", valueChanged=on_update_cb)

        self.width_min_spinbox = SpinBox(None, 0, None, "px", valueChanged=on_update_cb)
        self.width_max_spinbox = SpinBox(None, 0, None, "px", valueChanged=on_update_cb)

        self.filter_rate_spinbox = SpinBox(None, 0, 20, value=0, valueChanged=self.filter_rate_on_change)

        self.cull_rate_spinbox = DoubleSpinBox(value=3.0, valueChanged=self.filter_rate_on_change)
        self.cull_rate_spinbox.setRange(0.0, 10.0)
        self.cull_rate_spinbox.setDecimals(1)
        self.cull_rate_spinbox.setSingleStep(0.1)
        self.cull_rate_spinbox.setSuffix("x")

        self.check_radius_spinbox = SpinBox(None, 10, 100, value=50, valueChanged=on_update_cb)

        left_controls_frame = QFrame()
        HBoxLayout(
            left_controls_frame,
            [
                VBoxLayout([self.dimension_switch]),
                VBoxLayout(
                    [
                        self.min_max_label,
                        HBoxLayout(
                            [
                                self.height_min_spinbox,
                                self.height_max_spinbox,
                                self.width_min_spinbox,
                                self.width_max_spinbox,
                            ]
                        ),
                    ]
                ),
            ],
        )

        controls_frame = QFrame()
        HBoxLayout(
            controls_frame,
            [
                VBoxLayout([QLabel("Filter rate"), self.filter_rate_spinbox]),
                VBoxLayout([QLabel("Cull rate"), self.cull_rate_spinbox]),
                VBoxLayout([QLabel("Check radius"), self.check_radius_spinbox]),
            ],
        )

        self.on_dimension_to_check_change(False)

        VBoxLayout(
            self,
            [
                HBoxLayout([left_controls_frame, controls_frame, VBoxLayout([self.results, self.position])]),
                self.canvas,
                self.canvas.controls,
            ],
        )

        self.main.cropValuesChanged.connect(self.on_crop_changed)

    @property
    def curr_clip(self) -> vs.VideoNode:
        assert self.main.outputs
        return self.main.outputs._items[self.main.current_output.index].source.clip

    @property
    def crop_clip(self) -> vs.VideoNode:
        if self.curr_clip not in self.clip_cache:
            self.clip_cache[self.curr_clip] = depth(plane(self.curr_clip, 0), 32)

        clip = self.clip_cache[self.curr_clip]

        crop_info = self.main.current_output.crop_values

        if crop_info.active:
            if self.curr_clip not in self.cropped_clip_cache:
                self.cropped_clip_cache[self.curr_clip] = clip.std.CropAbs(
                    min(crop_info.width, clip.width - crop_info.left),
                    min(crop_info.height, clip.height - crop_info.top),
                    crop_info.left,
                    crop_info.top,
                )
        else:
            self.cropped_clip_cache[self.curr_clip] = clip

        return self.cropped_clip_cache[self.curr_clip]

    @property
    def check_range(self) -> tuple[int, int]:
        if self.dimension_switch.isChecked():
            return self.width_min_spinbox.value(), self.width_max_spinbox.value()

        return self.height_min_spinbox.value(), self.height_max_spinbox.value()

    @property
    def check_radius(self) -> int:
        return self.check_radius_spinbox.value()

    def on_current_output_changed(self, index: int, prev_index: int) -> None:
        assert self.main.outputs

        old_node = self.main.outputs[prev_index].source.clip

        old_key = (old_node.width, old_node.height)
        new_key = (self.curr_clip.width, self.curr_clip.height)

        if old_key in self.minmax_values:
            self.minmax_values[old_key] = (
                self.height_min_spinbox.value(),
                self.height_max_spinbox.value(),
                self.width_min_spinbox.value(),
                self.width_max_spinbox.value(),
            )

        if new_key not in self.minmax_values:
            if self.curr_clip.width < 1400:
                lrate, hrate = 0.0, 0.0
            else:
                lrate, hrate = 0.465, 0.925

            self.minmax_values[new_key] = (
                int(self.curr_clip.height * lrate),
                int(self.curr_clip.height * hrate),
                int(self.curr_clip.width * lrate),
                int(self.curr_clip.width * hrate),
            )

        self.height_min_spinbox.setMaximum(int(self.curr_clip.height * 0.9))
        self.height_max_spinbox.setMaximum(self.curr_clip.height)
        self.width_min_spinbox.setMaximum(int(self.curr_clip.width * 0.9))
        self.width_max_spinbox.setMaximum(self.curr_clip.width)

        hmin, hmax, wmin, wmax = self.minmax_values[new_key]

        self.height_min_spinbox.setValue(hmin)
        self.height_max_spinbox.setValue(hmax)
        self.width_min_spinbox.setValue(wmin)
        self.width_max_spinbox.setValue(wmax)

        self.clean_cache(True, True, True, True, True, True)

    def on_current_frame_changed(self, frame: Frame) -> None:
        if self.first_paint:
            self.on_current_output_changed(self.main.current_output.index, self.main.current_output.index)
            self.first_paint = False

        self.canvas.render(frame, True)

    def on_crop_changed(self, crop_info: CroppingInfo) -> None:
        self.clean_cache(True, True)

        self.canvas.render()

    def filter_rate_on_change(self, value: int) -> None:
        self.clean_cache(True, True, True, True, True, True)
        self.canvas.render()

    def on_dimension_to_check_change(self, check_width: bool) -> None:
        if check_width:
            self.width_min_spinbox.show()
            self.width_max_spinbox.show()

            self.height_min_spinbox.hide()
            self.height_max_spinbox.hide()

            self.min_max_label.setText("Min/Max width checked")
        else:
            self.width_min_spinbox.hide()
            self.width_max_spinbox.hide()

            self.height_min_spinbox.show()
            self.height_max_spinbox.show()

            self.min_max_label.setText("Min/Max height checked")

        self.canvas.render()

    def on_update_values(self, *args: bool) -> Callable[..., None]:
        def _cb(_: Any = ...) -> None:
            self.clean_cache(*args)
            self.canvas.render()

        return _cb

    def clean_cache(
        self,
        generic: bool = False,
        crop: bool = False,
        transpose: bool = False,
        pad: bool = False,
        upscale: bool = False,
        frames: bool = False,
    ) -> None:
        if frames:
            self.frames_cache.clear()

        if self.curr_clip in self.cropped_clip_cache and self.crop_clip in self.upscaled_clip_cache:
            if upscale:
                del self.upscaled_clip_cache[self.crop_clip]

            if pad:
                del self.padded_clip_cache[self.crop_clip]

                if self.crop_clip in self.transp_clip_cache:
                    del self.padded_clip_cache[self.transp_clip_cache[self.crop_clip]]

            if transpose:
                self.transp_clip_cache[self.crop_clip]

        if crop:
            del self.cropped_clip_cache[self.curr_clip]

        if generic:
            del self.clip_cache[self.curr_clip]

    def get_dct_value(self, frame_num: int) -> tuple[Any, Any, Any | None]:
        cut_clip = self.crop_clip
        up_scale = self.filter_rate_spinbox.value()
        cut_rate = self.cull_rate_spinbox.value()

        def _get_arr(clip: vs.VideoNode) -> Any:
            if clip not in self.padded_clip_cache:
                top_cut = 20 if clip.height > 720 else 10
                side_cut = 20
                if cut_rate:
                    side_cut = mod2(clip.width / (2 + cut_rate))

                self.padded_clip_cache[clip] = padder.MIRROR(
                    clip.std.Crop(side_cut, side_cut, top_cut, top_cut), side_cut // 2, side_cut // 2, top_cut, top_cut
                ).std.Transpose()

            rows = np.asarray(self.frames_cache[self.padded_clip_cache[clip]][frame_num][0], np.float32)

            return sum(abs(dct(row)) for row in rows) / len(rows)

        if cut_clip not in self.transp_clip_cache:
            self.transp_clip_cache[cut_clip] = cut_clip.std.Transpose()

        dct_v, dct_h = _get_arr(cut_clip), _get_arr(self.transp_clip_cache[cut_clip])

        if check_width := self.dimension_switch.isChecked():
            dct_vv, dct_hh, rate = (dct_v, dct_h, len(dct_v) / len(dct_h))
        else:
            dct_vv, dct_hh, rate = (dct_h, dct_v, len(dct_h) / len(dct_v))

        if up_scale:
            if cut_clip not in self.upscaled_clip_cache:
                self.upscaled_clip_cache[cut_clip] = cut_clip.resize.Point(
                    cut_clip.width * up_scale, cut_clip.height * up_scale
                )

            up_clip = self.upscaled_clip_cache[cut_clip]

            if check_width:
                dct_up = _get_arr(up_clip)
            else:
                if up_clip not in self.transp_clip_cache:
                    self.transp_clip_cache[up_clip] = up_clip.std.Transpose()
                dct_up = _get_arr(self.transp_clip_cache[up_clip])
        else:
            dct_up = None

        dct_cross = np.array([dct_vv[int(i * rate)] for i in range(len(dct_hh))])

        if check_width:
            return dct_h, dct_cross, dct_up

        return dct_v, dct_cross, dct_up

    def get_inter(self, dct: NDArray[Any], dct_up: NDArray[Any] | None, min_val: int, max_val: int) -> NDArray[Any]:
        max_index = argrelextrema(dct, np.less, order=self.check_radius)[0]
        min_index = argrelextrema(dct, np.greater, order=self.check_radius)[0]

        if dct_up is not None:
            max_index_up = argrelextrema(dct_up[: dct.shape[0]], np.less, order=self.check_radius)[0]
            min_index_up = argrelextrema(dct_up[: dct.shape[0]], np.greater, order=self.check_radius)[0]
        else:
            max_index_up = min_index_up = []  # type: ignore

        return np.array(
            [
                x
                for x in (*max_index, *min_index)
                if (min_val < x < max_val) and x not in max_index_up and x not in min_index_up
            ]
        )

    def init_outputs(self) -> None:
        assert self.main.outputs
