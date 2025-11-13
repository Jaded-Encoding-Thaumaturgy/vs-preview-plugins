from __future__ import annotations

from math import ceil, floor
from threading import Timer
from typing import Any, Callable, Sequence, cast

import numpy as np
from jetpytools import complex_hash, mod2
from PyQt6.QtCore import QObject, Qt, QThreadPool, pyqtSignal
from PyQt6.QtWidgets import QLabel, QWidget
from scipy.signal import argrelextrema
from vskernels import Kernel
from vspreview.core import (
    ComboBox,
    DoubleSpinBox,
    ExtendedWidget,
    Frame,
    HBoxLayout,
    PlotMouseEvent,
    PlottingCanvasDefaultFrame,
    ProgressBar,
    PushButton,
    SpinBox,
    Stretch,
    Switch,
    VBoxLayout,
)
from vspreview.core.vsenv import Runner
from vspreview.main import MainWindow
from vspreview.models import GeneralModel
from vstools import FieldBased, clip_data_gather, core, get_h, get_w, vs

from .utils import DynamicDataCache, RescaleWorkClip, common_kernels, get_kernel_name


class RescaleErrorRunner(QObject):
    error_step = pyqtSignal()
    update = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(
        self,
        analyzer: RescaleErrorAnalyzer,
        clip: vs.VideoNode,
        frame: int,
        is_integer: bool,
        attempts: Sequence[tuple[float, float]],
        mode: str,
        bs_parity: int,
        metric_mode: int,
        kernel: Kernel,
        field_based: FieldBased,
    ) -> None:
        super().__init__()

        self.analyzer = analyzer
        self.clip = clip
        self.frame = frame
        self.is_integer = is_integer
        self.attempts = attempts
        self.mode = mode
        self.bs_parity = bs_parity
        self.metric_mode = metric_mode
        self.kernel = kernel
        self.field_based = field_based

    def run(self) -> None:
        rescaled_clips = self.analyzer.clips_cache[
            (self.clip, self.attempts, self.kernel, self.is_integer, self.mode, self.bs_parity, self.field_based)
        ]

        rescaled_clip = self.clip.std.BlankClip(length=len(self.attempts)).std.FrameEval(
            lambda n: rescaled_clips[n][self.frame]
        )

        if self.metric_mode == 0:
            metric_expr = "x y - abs"
        elif self.metric_mode == 1:
            metric_expr = "x y - 2 pow"
        else:
            raise RuntimeError

        rescaled_clip = core.std.Expr([self.clip[self.frame] * len(self.attempts), rescaled_clip], metric_expr)
        rescaled_clip = rescaled_clip.std.CropRel(10, 10, 10, 10).std.PlaneStats()

        self.error_step.emit()

        errors = clip_data_gather(
            rescaled_clip,
            lambda i, n: self.update.emit(int(i / n * 100)),
            lambda n, f: cast(float, f.props.PlaneStatsAverage),
        )

        attempt_idx = 0 if "h" not in self.mode else 1

        self.data = ([x[attempt_idx] for x in self.attempts], errors)

        self.finished.emit()
        self.deleteLater()


class RescaleErrorCanvas(PlottingCanvasDefaultFrame):
    def __init__(self, plugin: RescaleErrorAnalyzer) -> None:
        self.plugin = plugin

        super().__init__(plugin.main, True, False, True, 5)

    def on_mouse_moved(self, event: PlotMouseEvent) -> None:
        attempts = self.axes.lines[0].get_xdata()
        errors = self.axes.lines[0].get_ydata()

        width_check = "h" not in self.plugin.rescale_mode

        assert event.xdata is not None

        close_error = [(i, abs(x - event.xdata)) for i, x in enumerate(attempts)]  # type: ignore
        close_error.sort(key=lambda v: v[1])

        error_idx = close_error[0][0]

        self.plugin.position.setText(
            f"{'Width' if width_check else 'Height'}: "
            f"{attempts[error_idx]:.{self.plugin.step_decimals}f}, Error: {errors[error_idx]:.16f}"  # type: ignore
        )

    def _render(self, frame: Frame) -> None:
        self.plugin.set_visibility(False, False, True)

        data = self.plugin.current_data[self.plugin.main.current_output.index][int(frame)]
        assert data

        attempts, errors = map(np.array, data)

        order = 10 * floor(1 / self.plugin.step_spinbox.value())

        min_indices = argrelextrema(errors, np.less, order=order)[0]

        self.axes.plot(attempts, errors, label="Error")

        if len(min_indices):
            self.axes.scatter(attempts[min_indices], errors[min_indices], linewidth=0.3, s=50, c="r")

        best_arr = [(float(attempts[i]), float(errors[i])) for i in min_indices]
        best_arr.sort(key=lambda v: v[1])
        best_arr = best_arr[:5]

        best_indices = ", ".join(map(str, (x[0] for x in best_arr))) or "None"

        self.plugin.results.setText(f"Spikes: {best_indices}")


type KeyCache = tuple[vs.VideoNode, Sequence[tuple[float, float]], Kernel, bool, str, int, FieldBased]


class RescaleErrorClipCache(DynamicDataCache[KeyCache, list[vs.VideoNode]]):
    def get_data(self, key: KeyCache) -> list[vs.VideoNode]:
        clip, attempts, kernel, is_integer, mode, bs_parity, field_based = key

        if is_integer:

            def _gen_clip_from_attempt(attempt: tuple[Any, Any]) -> vs.VideoNode:
                return kernel.scale(kernel.descale(clip, *attempt, field_based=field_based), clip.width, clip.height)
        else:
            clip_ar = clip.width / clip.height
            do_h, do_w = "h" in mode, "w" in mode

            def _gen_clip_from_attempt(attempt: tuple[Any, Any]) -> vs.VideoNode:
                src_width, src_height = attempt

                base_height = mod2(ceil(src_height)) + bs_parity
                base_width = mod2(base_height * clip_ar) + bs_parity

                width = (base_width - 2 * floor((base_width - src_width) / 2)) if do_w else clip.width
                height = (base_height - 2 * floor((base_height - src_height) / 2)) if do_h else clip.height

                de_args = dict[str, Any]()

                if do_h:
                    de_args.update(src_height=src_height, src_top=(height - src_height) / 2)

                if do_w:
                    de_args.update(src_width=src_width, src_left=(width - src_width) / 2)

                return kernel.scale(
                    kernel.descale(clip, width, height, field_based=field_based, **de_args),
                    clip.width,
                    clip.height,
                    **de_args,
                )

        return list(map(_gen_clip_from_attempt, attempts))


class RescaleErrorAnalyzer(ExtendedWidget):
    def __init__(self, main: MainWindow) -> None:
        super().__init__()

        self.main = main

        self.workers_lookup = dict[int, dict[Frame, RescaleErrorRunner]]()
        self.workclips = RescaleWorkClip()
        self.clips_cache = RescaleErrorClipCache(10)
        self.minmax_values = dict[tuple[int, int], tuple[int, int, int, int]]()
        self.current_datas = dict[int, list[dict[int, tuple[list[float], list[float]] | None]]]()

        self.last_edit_timer: Timer | None = None

        self.render_lock = False

    def setup_ui(self) -> None:
        # Controls
        self.controls = QWidget()

        self.dimension_switch = Switch(
            12, 40, 1, ("Height", "Width", 2), checked=False, clicked=self.on_dimension_to_check_change
        )
        self.kernels_combobox = ComboBox[str](
            model=GeneralModel[str](list(map(get_kernel_name, common_kernels)), False),
            currentIndex=0,
            sizeAdjustPolicy=ComboBox.SizeAdjustPolicy.AdjustToContents,
        )

        self.bs_parity_label = QLabel()
        self.bs_parity_switch = Switch(12, 40, 1, ("Even", "Odd", 2), checked=False)

        self.min_max_label = QLabel()

        self.height_min_spinbox = SpinBox(None, 0, None, "px")
        self.height_max_spinbox = SpinBox(None, 0, None, "px")

        self.width_min_spinbox = SpinBox(None, 0, None, "px")
        self.width_max_spinbox = SpinBox(None, 0, None, "px")

        self.step_spinbox = DoubleSpinBox(
            value=1.0, minimum=0.001, maximum=1.0, decimals=3, stepType=DoubleSpinBox.StepType.AdaptiveDecimalStepType
        )

        self.metric_type_combobox = ComboBox[str](
            model=GeneralModel[str](["MAE", "MSE"], False),
            currentIndex=1,
            sizeAdjustPolicy=ComboBox.SizeAdjustPolicy.AdjustToContents,
        )

        self.update_button = PushButton("Update", clicked=lambda _: self.render_canvas())

        HBoxLayout(
            self.controls,
            [
                VBoxLayout(
                    [
                        QLabel("Dimension to check"),
                        self.dimension_switch,
                    ]
                ),
                VBoxLayout(
                    [
                        self.bs_parity_label,
                        self.bs_parity_switch,
                    ]
                ),
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
                VBoxLayout([QLabel("Step"), self.step_spinbox]),
                VBoxLayout([QLabel("Kernel"), self.kernels_combobox]),
                VBoxLayout([QLabel("Metric"), self.metric_type_combobox]),
                Stretch(),
                self.update_button,
            ],
        )

        self.on_dimension_to_check_change()

        self.controls.setFixedHeight(80)

        self.bs_parity_switch.clicked.connect(self.render_callback)
        self.dimension_switch.clicked.connect(self.render_callback)
        self.kernels_combobox.currentIndexChanged.connect(self.render_callback)
        self.metric_type_combobox.currentIndexChanged.connect(self.render_callback)

        # Loading
        self.errors_loading = QWidget()

        self.errors_progress = ProgressBar(self, value=0)
        self.errors_progress.setGeometry(200, 80, 250, 20)

        VBoxLayout(self.errors_loading, [Stretch(), QLabel("Calculating errors..."), self.errors_progress, Stretch()])

        # Rescale Clip Loading
        self.rescale_loading = QWidget()

        self.rescale_progress = ProgressBar(self, value=0, minimum=0, maximum=0)
        self.rescale_progress.setGeometry(200, 80, 250, 20)

        VBoxLayout(
            self.rescale_loading, [Stretch(), QLabel("Creating rescaled clip..."), self.rescale_progress, Stretch()]
        )

        # Display
        self.display = QWidget()

        self.results = QLabel()
        self.position = QLabel()

        font = self.results.font()
        font.setPixelSize(15)

        self.results.setFont(font)
        self.position.setFont(font)

        self.results.setMaximumHeight(25)
        self.position.setMaximumHeight(25)

        self.canvas = RescaleErrorCanvas(self)

        VBoxLayout(self.display, [VBoxLayout([self.results, self.position]), self.canvas, self.canvas.controls])

        # Final
        self.vlayout = VBoxLayout(self, [self.controls, self.rescale_loading, self.errors_loading, self.display])

        self.set_visibility(False, False, False)

    def set_visibility(self, rescale: bool, errors: bool, display: bool) -> None:
        if rescale or errors or display:
            if self.vlayout.alignment():
                self.vlayout.setAlignment(Qt.AlignmentFlag(0))
        else:
            self.vlayout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.rescale_loading.setVisible(rescale)
        self.errors_loading.setVisible(errors)
        self.display.setVisible(display)

    def on_dimension_to_check_change(self) -> None:
        if self.rescale_mode == "w":
            if self.height_min_spinbox.isVisible():
                self.width_min_spinbox.setValue(get_w(self.height_min_spinbox.value(), self.curr_clip))
                self.width_max_spinbox.setValue(get_w(self.height_max_spinbox.value(), self.curr_clip))

            self.width_min_spinbox.show()
            self.width_max_spinbox.show()

            self.height_min_spinbox.hide()
            self.height_max_spinbox.hide()

            self.bs_parity_label.setText("Base width parity")
            self.min_max_label.setText("Min/Max width checked")
        else:
            if self.width_min_spinbox.isVisible():
                self.height_min_spinbox.setValue(get_h(self.width_min_spinbox.value(), self.curr_clip))
                self.height_max_spinbox.setValue(get_h(self.width_max_spinbox.value(), self.curr_clip))

            self.width_min_spinbox.hide()
            self.width_max_spinbox.hide()

            self.height_min_spinbox.show()
            self.height_max_spinbox.show()

            self.bs_parity_label.setText("Base height parity")
            self.min_max_label.setText("Min/Max height checked")

    @property
    def curr_clip(self) -> vs.VideoNode:
        assert self.main.outputs
        return self.main.outputs._items[self.main.current_output.index].source.clip

    def on_current_output_changed(self, index: int, prev_index: int) -> None:
        assert self.main.outputs

        self.render_lock = True

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

        self.render_lock = False

    def render_callback(self) -> None:
        self.safe_render()

    def render_canvas(self, frame: Frame | None = None) -> None:
        if self.render_lock:
            return

        if frame is None:
            frame = self.main.current_output.last_showed_frame

        if self.current_data[self.main.current_output.index][int(frame)] is not None:
            self.canvas.render(frame, True)
            return

        if frame not in self.workers:
            self.set_visibility(True, False, False)

            rescale_mode, bs_parity, metric_mode, start, stop, step, decimals = (
                *self._get_values(),
                self.step_decimals,
            )

            wclip = self.workclips[self.main.current_output.source.clip]

            attempts = [
                ((x * wclip.width / wclip.height, x) if "h" in rescale_mode else (x, x / wclip.width * wclip.height))
                for x in [
                    round(cast(float, val), decimals)
                    for val in np.arange(start, stop + step, step)
                    if round(val, decimals) <= stop
                ]
            ]

            if not decimals:
                attempts = [(round(w, 0), round(h, 0)) for w, h in attempts]

            self.workers[frame] = RescaleErrorRunner(
                self,
                wclip,
                int(frame),
                decimals == 0,
                attempts,
                rescale_mode,
                bs_parity,
                metric_mode,
                self.kernel,
                FieldBased.PROGRESSIVE,
            )

            self.workers[frame].error_step.connect(self.error_step_hook(frame, self.main.current_output.index))
            self.workers[frame].update.connect(self.progress_hook(frame, self.main.current_output.index))
            self.workers[frame].finished.connect(self.render_hook(frame, self.main.current_output.index))

            pool = QThreadPool.globalInstance()

            assert pool

            pool.start(Runner(self.workers[frame].run))

    def safe_render(self, frame: Frame | None = None) -> None:
        if not (self.errors_loading.isVisible() or self.display.isVisible()):
            return

        self.render_canvas(frame)

    def on_current_frame_changed(self, frame: Frame | None = None) -> None:
        self.safe_render()

    def error_step_hook(self, frame: Frame, out_idx: int) -> Callable[[], None]:
        def _callback() -> None:
            if (frame, out_idx) == (self.main.current_output.last_showed_frame, self.main.current_output.index):
                self.set_visibility(False, True, False)

        return _callback

    def render_hook(self, frame: Frame, out_idx: int) -> Callable[[], None]:
        def _callback() -> None:
            if frame not in self.workers:
                return

            self.current_data[out_idx][int(frame)] = self.workers[frame].data

            if (frame, out_idx) == (self.main.current_output.last_showed_frame, self.main.current_output.index):
                self.canvas.render(frame, True)

            if frame in self.workers:
                del self.workers[frame]

        return _callback

    def progress_hook(self, frame: Frame, out_idx: int) -> Callable[[int], None]:
        def _callback(update: int) -> None:
            if (frame, out_idx) == (self.main.current_output.last_showed_frame, self.main.current_output.index):
                if not self.errors_progress.isVisible():
                    self.set_visibility(False, True, False)

                self.errors_progress.setValue(update)

        return _callback

    @property
    def rescale_mode(self) -> str:
        return "w" if self.dimension_switch.isChecked() else "h"

    @property
    def step_decimals(self) -> int:
        *_, step = self._get_values()
        if step.is_integer():
            return 0
        return max(str(step)[::-1].find("."), 0)

    def _get_values(self) -> tuple[str, int, int, float, float, float]:
        if "h" in self.rescale_mode:
            start = self.height_min_spinbox.value()
            stop = self.height_max_spinbox.value()
        else:
            start = self.width_min_spinbox.value()
            stop = self.width_max_spinbox.value()

        step = round(self.step_spinbox.value(), max(str(self.step_spinbox.minimum())[::-1].find("."), 0))
        step = min(max(step, self.step_spinbox.minimum()), self.step_spinbox.maximum())

        bs_parity = 1 if self.bs_parity_switch.isChecked() else 0

        return self.rescale_mode, bs_parity, self.metric_type_combobox.currentIndex(), start, stop, step

    @property
    def kernel(self) -> Kernel:
        return common_kernels[self.kernels_combobox.currentIndex()]

    @property
    def workers(self) -> dict[Frame, RescaleErrorRunner]:
        key = complex_hash.hash(*self._get_values(), self.kernel)

        if key not in self.workers_lookup:
            self.workers_lookup[key] = dict[Frame, RescaleErrorRunner]()

        return self.workers_lookup[key]

    @property
    def current_data(self) -> list[dict[int, tuple[list[float], list[float]] | None]]:
        assert self.main.outputs

        key = complex_hash.hash(*self._get_values(), self.kernel)

        if key not in self.current_datas:
            self.current_datas[key] = [dict.fromkeys(range(out.source.clip.num_frames)) for out in self.main.outputs]

        return self.current_datas[key]

    def init_outputs(self) -> None:
        self.current_datas.clear()
