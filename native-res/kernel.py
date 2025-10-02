from __future__ import annotations

import logging
from typing import Callable, Sequence, cast

from PyQt6.QtCore import QObject, Qt, QThreadPool, pyqtSignal
from PyQt6.QtWidgets import QHeaderView, QLabel, QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
from vsexprtools import norm_expr
from vskernels import (
    BicubicSharp,
    Bilinear,
    BorderHandling,
    Catrom,
    Kernel,
    KernelLike,
    Lanczos,
    Mitchell,
    SampleGridModel,
)
from vspreview.core import (
    ComboBox,
    DoubleSpinBox,
    ExtendedWidget,
    Frame,
    HBoxLayout,
    ProgressBar,
    PushButton,
    Stretch,
    Switch,
    VBoxLayout,
)
from vspreview.core.vsenv import Runner
from vspreview.main import MainWindow
from vspreview.models import GeneralModel
from vsscale import Rescale
from vstools import (
    DynamicClipsCache,
    FieldBased,
    Matrix,
    R,
    T,
    clip_data_gather,
    complex_hash,
    core,
    get_y,
    vs,
    vs_object,
)

__all__ = ["KernelAnalyzer"]

CONTROL_HEIGHT: int = 26

common_kernels: list[KernelLike] = [
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


class KernelWorkClip(DynamicClipsCache[vs.VideoNode, vs.VideoNode]):
    def get_clip(self, key: vs.VideoNode) -> vs.VideoNode:
        return core.resize.Bilinear(key, format=vs.GRAYS, matrix=Matrix.BT709, matrix_in=Matrix.from_video(key))


class KernelRunner(QObject):
    error_step = pyqtSignal()
    update = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(
        self,
        analyzer: KernelAnalyzer,
        clip: vs.VideoNode,
        frame: int,
        kernels: Sequence[Kernel],
        target_width: float | None,
        target_height: float | None,
        src_top: float,
        src_left: float,
        border_handling: BorderHandling,
        sample_grid_model: SampleGridModel,
        field_based: FieldBased,
    ) -> None:
        super().__init__()

        self.analyzer = analyzer
        self.clip = clip
        self.frame = frame
        self.kernels = kernels
        self.target_width = target_width
        self.target_height = target_height
        self.src_top = src_top
        self.src_left = src_left
        self.border_handling = border_handling
        self.sample_grid_model = sample_grid_model
        self.field_based = field_based
        self.data = []
        self._should_stop = False

    def stop(self) -> None:
        """Stop the worker gracefully."""
        self._should_stop = True

    def run(self) -> None:
        if self.analyzer.dimension_switch.isChecked():
            target_w = int(self.target_width) if self.target_width is not None else int(self.clip.width)
            target_h = int(self.clip.height)
        else:
            target_w = int(self.clip.width)
            target_h = int(self.target_height) if self.target_height is not None else int(self.clip.height)

        logging.debug(
            f"Target dimensions: {target_w}x{target_h}, "
            f"shifts: ({self.src_top}, {self.src_left}), "
            f"border handling: {self.border_handling}, "
            f"sample grid model: {self.sample_grid_model}, "
            f"field based: {self.field_based}, "
        )

        wclip = self.analyzer.curr_clip
        wclip = self.field_based.apply(wclip)
        wclip = get_y(wclip)[self.frame]

        results = []

        for i, kernel in enumerate(self.kernels):
            if self._should_stop:
                logging.debug("Worker stopped by user request")
                return

            prefix = f"{i + 1} / {len(self.kernels)}"

            logging.debug(f"{prefix} Analyzing kernel {kernel.__class__.__name__} @ {self.frame}")

            try:
                rs = Rescale(
                    wclip,
                    target_h,
                    kernel,
                    upscaler=kernel,
                    downscaler=kernel,
                    width=target_w,
                    shift=(self.src_top, self.src_left),
                    field_based=self.field_based,
                    border_handling=self.border_handling,
                )

                rs.descale_args.mode = "hw"[self.analyzer.dimension_switch.isChecked()]

                upscaled = rs.rescale

                mae_clip = norm_expr([wclip, upscaled], "x y - abs")
                mae_clip = mae_clip.std.PlaneStats()
                mae_errors = clip_data_gather(
                    mae_clip,
                    lambda i, n: None,
                    lambda n, f: cast(float, f.props.PlaneStatsAverage),
                )

                mse_clip = norm_expr([wclip, upscaled], "x y - 2 pow")
                mse_clip = mse_clip.std.PlaneStats()
                mse_errors = clip_data_gather(
                    mse_clip,
                    lambda i, n: None,
                    lambda n, f: cast(float, f.props.PlaneStatsAverage),
                )

                mae_value = mae_errors[0] if mae_errors else 0.0
                mse_value = mse_errors[0] if mse_errors else 0.0

                results.append((kernel, mae_value, mse_value))

                self.update.emit(int((i + 1) / len(self.kernels) * 100))

            except Exception as e:
                logging.error(f"Error analyzing kernel {kernel.__class__.__name__}: {e}")
                results.append((kernel, float("inf"), float("inf")))
                self.update.emit(int((i + 1) / len(self.kernels) * 100))

        logging.debug([f"{result[0].__class__.__name__}: {result[1]}\n" for result in results])

        if not self._should_stop:
            self.data = results
            self.finished.emit()

        self.deleteLater()


class KernelResultsDisplay(QWidget):
    def __init__(self, plugin: KernelAnalyzer) -> None:
        super().__init__()
        self.plugin = plugin
        self.setup_ui()

    def setup_ui(self) -> None:
        self.kernels_table = QTableWidget(self)
        self.kernels_table.setColumnCount(4)
        self.kernels_table.setHorizontalHeaderLabels(["Kernel", "Error%", "Mean Absolute Error", "Mean Squared Error"])

        if (header := self.kernels_table.horizontalHeader()) is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        self.kernels_table.setAlternatingRowColors(True)
        self.kernels_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.kernels_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.kernels_table.setSortingEnabled(False)

        self.kernels_table.setColumnWidth(0, 300)
        self.kernels_table.setColumnWidth(1, 80)
        self.kernels_table.setColumnWidth(2, 120)
        self.kernels_table.setColumnWidth(3, 120)

        font = self.kernels_table.font()
        font.setPixelSize(12)
        self.kernels_table.setFont(font)

        layout = QVBoxLayout()
        layout.addWidget(self.kernels_table)
        self.setLayout(layout)

    def update_results(self, results: list[tuple[Kernel, float, float]]) -> None:
        if not results:
            self.kernels_table.setRowCount(0)
            return

        results.sort(key=lambda x: x[1])

        _, best_mae, _ = results[0]

        valid_results = [(k, mae, mse) for k, mae, mse in results if mae != float("inf")]

        self.kernels_table.setRowCount(len(valid_results))

        for row, (kernel, mae, mse) in enumerate(valid_results):
            kernel_name = _format_kernel_name(kernel)
            error_percent = (mae / best_mae) * 100 if best_mae > 0 else 100.0

            kernel_item = QTableWidgetItem(kernel_name)
            error_percent_item = QTableWidgetItem(f"{error_percent:.1f}%")
            mae_item = QTableWidgetItem(f"{mae:.10f}")
            mse_item = QTableWidgetItem(f"{mse:.10f}")

            self.kernels_table.setItem(row, 0, kernel_item)
            self.kernels_table.setItem(row, 1, error_percent_item)
            self.kernels_table.setItem(row, 2, mae_item)
            self.kernels_table.setItem(row, 3, mse_item)

        self.kernels_table.resizeColumnsToContents()


class DynamicDataCache(vs_object, dict[T, R]):
    def __init__(self, cache_size: int = 2) -> None:
        self.cache_size = cache_size

    def get_data(self, key: T) -> R:
        raise NotImplementedError

    def __getitem__(self, args: T, /) -> R:
        __key = complex_hash.hash(args)

        if __key not in self:
            self[__key] = self.get_data(args)  # type: ignore

            if len(self) > self.cache_size:
                del self[next(iter(self.keys()))]

        return super().__getitem__(__key)  # type: ignore

    def __vs_del__(self, core_id: int) -> None:
        self.clear()


class KernelAnalyzer(ExtendedWidget):
    def __init__(self, main: MainWindow) -> None:
        super().__init__()

        self.main = main

        self.workers_lookup = dict[int, dict[Frame, KernelRunner]]()
        self.workclips = KernelWorkClip()
        self.current_data = dict[int, dict[int, list[tuple[Kernel, float, float]] | None]]()

        self.render_lock = False

    def setup_ui(self) -> None:
        self.controls = QWidget()

        self.dimension_switch = Switch(
            12, 40, 1, ("Height", "Width", 2), checked=False, clicked=self.on_dimension_change
        )

        self.width_label = QLabel("Width")
        self.width_spinbox = DoubleSpinBox(
            value=1920.0,
            minimum=1.0,
            maximum=None,
            decimals=3,
            stepType=DoubleSpinBox.StepType.AdaptiveDecimalStepType,
        )
        self.width_spinbox.setSingleStep(1.0)
        self.width_spinbox.setMinimumHeight(CONTROL_HEIGHT)
        self.width_spinbox.setMaximumHeight(CONTROL_HEIGHT)

        self.height_label = QLabel("Height")
        self.height_spinbox = DoubleSpinBox(
            value=1080.0,
            minimum=1.0,
            maximum=None,
            decimals=3,
            stepType=DoubleSpinBox.StepType.AdaptiveDecimalStepType,
        )
        self.height_spinbox.setSingleStep(1.0)
        self.height_spinbox.setMinimumHeight(CONTROL_HEIGHT)
        self.height_spinbox.setMaximumHeight(CONTROL_HEIGHT)

        self.src_top_spinbox = DoubleSpinBox(
            value=0.0,
            minimum=-100.0,
            maximum=100.0,
            decimals=6,
        )
        self.src_top_spinbox.setSingleStep(0.000025)
        self.src_top_spinbox.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.src_top_spinbox.setFixedHeight(CONTROL_HEIGHT)

        self.src_left_spinbox = DoubleSpinBox(
            value=0.0,
            minimum=-100.0,
            maximum=100.0,
        )
        self.src_left_spinbox.setSingleStep(0.000025)
        self.src_left_spinbox.setDecimals(6)
        self.src_left_spinbox.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.src_left_spinbox.setFixedHeight(CONTROL_HEIGHT)

        self.border_handling_combobox = ComboBox[str](
            model=GeneralModel[str]([member.name.replace("_", " ").title() for member in BorderHandling], False),
            currentIndex=0,
            sizeAdjustPolicy=ComboBox.SizeAdjustPolicy.AdjustToContents,
        )
        self.border_handling_combobox.setMaximumHeight(CONTROL_HEIGHT)

        self.sample_grid_model_switch = Switch(12, 48, 1, ("Edges", "Centers", 2), checked=False)
        self.sample_grid_model_switch.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        self.field_based_combobox = ComboBox[str](
            model=GeneralModel[str]([member.pretty_string for member in FieldBased], False),
            currentIndex=0,
            sizeAdjustPolicy=ComboBox.SizeAdjustPolicy.AdjustToContents,
        )
        self.field_based_combobox.setMaximumHeight(CONTROL_HEIGHT)

        self.uncommon_kernels_switch = Switch(12, 48, 1, ("Common", "Extensive", 2), checked=False)
        self.uncommon_kernels_switch.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        self.update_button = PushButton("Analyze", clicked=lambda _: self.render_canvas())

        HBoxLayout(
            self.controls,
            [
                VBoxLayout(
                    [
                        QLabel("Dimension to check"),
                        self.dimension_switch,
                    ]
                ),
                VBoxLayout([self.width_label, self.width_spinbox]),
                VBoxLayout([self.height_label, self.height_spinbox]),
                VBoxLayout(
                    [
                        QLabel("Shift"),
                        HBoxLayout(
                            [
                                self.src_top_spinbox,
                                self.src_left_spinbox,
                            ]
                        ),
                    ]
                ),
                VBoxLayout([QLabel("Border handling"), self.border_handling_combobox]),
                VBoxLayout([QLabel("Grid model"), self.sample_grid_model_switch]),
                VBoxLayout([QLabel("Field order"), self.field_based_combobox]),
                VBoxLayout([QLabel("Kernels to check"), self.uncommon_kernels_switch]),
                Stretch(),
                self.update_button,
            ],
        )

        self.controls.setFixedHeight(80)

        self.dimension_switch.clicked.connect(self.on_dimension_switch_changed)
        self.uncommon_kernels_switch.clicked.connect(self.on_kernels_switch_changed)

        self.on_dimension_change()
        self.on_dimension_to_check_change()
        self.set_field_based_default()

        self.errors_loading = QWidget()
        self.errors_progress = ProgressBar(self, value=0)
        self.errors_progress.setGeometry(200, 80, 250, 20)
        VBoxLayout(self.errors_loading, [Stretch(), QLabel("Analyzing kernels..."), self.errors_progress, Stretch()])

        self.display = QWidget()
        self.results_display = KernelResultsDisplay(self)
        VBoxLayout(self.display, [self.results_display])

        self.vlayout = VBoxLayout(self, [self.controls, self.errors_loading, self.display])
        self.set_visibility(False, False)

        self.on_dimension_change()

    def set_visibility(self, errors: bool, display: bool) -> None:
        if errors or display:
            if self.vlayout.alignment():
                self.vlayout.setAlignment(Qt.AlignmentFlag(0))
        else:
            self.vlayout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.errors_loading.setVisible(errors)
        self.display.setVisible(display)

    @property
    def curr_clip(self) -> vs.VideoNode:
        assert self.main.outputs
        return self.main.outputs._items[self.main.current_output.index].source.clip

    def on_current_output_changed(self, index: int, prev_index: int) -> None:
        assert self.main.outputs

        self.render_lock = True

        self.current_data.clear()

        self.on_dimension_change()
        self.on_dimension_to_check_change()
        self.set_field_based_default()

        self.render_lock = False

    def on_dimension_switch_changed(self) -> None:
        self.on_dimension_change()
        self.on_dimension_to_check_change()

    def on_kernels_switch_changed(self) -> None:
        self.workers_lookup.clear()
        self.current_data.clear()

        if (
            hasattr(self, "main")
            and self.main.current_output
            and (frame := self.main.current_output.last_showed_frame) in self.workers
        ):
            del self.workers[frame]

    def on_dimension_change(self) -> None:
        if self.dimension_switch.isChecked():
            self.width_label.show()
            self.width_spinbox.show()

            self.height_label.hide()
            self.height_spinbox.hide()
        else:
            self.width_label.hide()
            self.width_spinbox.hide()

            self.height_label.show()
            self.height_spinbox.show()

    def on_dimension_to_check_change(self) -> None:
        if self.dimension_switch.isChecked():
            default_width = self.curr_clip.width // 1.5
            self.width_spinbox.setMaximum(self.curr_clip.width)
            self.width_spinbox.setValue(float(default_width))
        else:
            default_height = self.curr_clip.height // 1.5
            self.height_spinbox.setMaximum(self.curr_clip.height)
            self.height_spinbox.setValue(float(default_height))

    def set_field_based_default(self) -> None:
        try:
            field_based = FieldBased.from_video(self.curr_clip)
            self.field_based_combobox.setCurrentIndex(field_based.value)
        except Exception:
            self.field_based_combobox.setCurrentIndex(0)

    def render_callback(self) -> None:
        self.safe_render()

    def render_canvas(self, frame: Frame | None = None) -> None:
        if self.render_lock:
            return

        if frame is None:
            frame = self.main.current_output.last_showed_frame

        if (
            self.main.current_output.index in self.current_data
            and int(frame) in self.current_data[self.main.current_output.index]
        ):
            del self.current_data[self.main.current_output.index][int(frame)]

        if frame in self.workers:
            if hasattr((worker := self.workers[frame]), "stop") and callable(worker.stop):
                worker.stop()

            del self.workers[frame]

        if frame not in self.workers:
            if self.uncommon_kernels_switch.isChecked():
                kernels = common_kernels.copy()
                uncommon_kernels = get_uncommon_kernels()
                kernels.extend(uncommon_kernels)
            else:
                kernels = common_kernels.copy()

            self.set_visibility(True, False)

            self.results_display.kernels_table.setRowCount(0)

            target_width: float | None = None
            target_height: float | None = None

            if self.dimension_switch.isChecked():
                target_width = self.width_spinbox.value()
            else:
                target_height = self.height_spinbox.value()

            src_top = self.src_top_spinbox.value()
            src_left = self.src_left_spinbox.value()

            border_handling = BorderHandling(self.border_handling_combobox.currentIndex())
            sample_grid_model = SampleGridModel(0 if self.sample_grid_model_switch.isChecked() else 1)

            field_based_idx = self.field_based_combobox.currentIndex()
            field_based = FieldBased(field_based_idx)

            self.workers[frame] = KernelRunner(
                self,
                self.curr_clip,
                int(frame),
                kernels,
                target_width,
                target_height,
                src_top,
                src_left,
                border_handling,
                sample_grid_model,
                field_based,
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
                self.set_visibility(True, False)

        return _callback

    def render_hook(self, frame: Frame, out_idx: int) -> Callable[[], None]:
        def _callback() -> None:
            if frame not in self.workers:
                return

            if out_idx not in self.current_data:
                self.current_data[out_idx] = {}

            self.current_data[out_idx][int(frame)] = self.workers[frame].data

            if (frame, out_idx) == (self.main.current_output.last_showed_frame, self.main.current_output.index):
                self.set_visibility(False, True)
                self.results_display.update_results(self.workers[frame].data)

            if frame in self.workers:
                del self.workers[frame]

        return _callback

    def progress_hook(self, frame: Frame, out_idx: int) -> Callable[[int], None]:
        def _callback(update: int) -> None:
            if (frame, out_idx) == (self.main.current_output.last_showed_frame, self.main.current_output.index):
                if not self.errors_progress.isVisible():
                    self.set_visibility(True, False)

                self.errors_progress.setValue(update)

        return _callback

    @property
    def workers(self) -> dict[Frame, KernelRunner]:
        key = complex_hash.hash(
            self.dimension_switch.isChecked(),
            self.width_spinbox.value(),
            self.height_spinbox.value(),
            self.src_top_spinbox.value(),
            self.src_left_spinbox.value(),
            self.border_handling_combobox.currentIndex(),
            self.sample_grid_model_switch.isChecked(),
            self.field_based_combobox.currentIndex(),
            self.uncommon_kernels_switch.isChecked(),
        )

        if key not in self.workers_lookup:
            self.workers_lookup[key] = dict[Frame, KernelRunner]()

        return self.workers_lookup[key]

    def init_outputs(self) -> None:
        self.current_data.clear()


def get_uncommon_kernels() -> list[KernelLike]:
    import importlib
    import inspect

    def kernel_key(kernel):
        attrs = []

        for attr in dir(kernel):
            if attr.startswith("_"):
                continue

            value = getattr(kernel, attr)

            if callable(value):
                continue

            try:
                hash(value)
                attrs.append((attr, value))
            except TypeError:
                continue

        return tuple(sorted(attrs))

    common_instances = {kernel_key(kernel) for kernel in common_kernels}
    added_instances = set()
    uncommon = []

    try:
        from vskernels.abstract.base import abstract_kernels, partial_abstract_kernels  # noqa: F401
    except ImportError as e:
        logging.error(f"Failed to import vskernels registries: {e}")
        return uncommon

    vskernel_modules = [
        "vskernels.kernels.zimg.bicubic",
        "vskernels.kernels.zimg.spline",
        "vskernels.kernels.zimg.various",
        "vskernels.kernels.custom.various",
        "vskernels.kernels.placebo",
    ]

    all_kernel_classes = set()

    for module_name in vskernel_modules:
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            logging.debug(f"Failed to import {module_name}: {e}")
            continue

        if hasattr(module, "__all__"):
            class_names = module.__all__
        else:
            class_names = [
                name for name, obj in inspect.getmembers(module, inspect.isclass) if not name.startswith("_")
            ]

        for class_name in class_names:
            if not (kernel_class := getattr(module, class_name, None)):
                continue

            if not any("Kernel" in base.__name__ for base in kernel_class.__mro__):
                continue

            if class_name.startswith("Custom") or class_name == "BicubicAuto":
                continue

            all_kernel_classes.add(kernel_class)

    def add_kernel_instance(kernel: KernelLike) -> None:
        if (key := kernel_key(kernel)) not in common_instances and key not in added_instances:
            uncommon.append(kernel)
            added_instances.add(key)

    for kernel_class in all_kernel_classes:
        try:
            sig = inspect.signature(kernel_class.__init__)
            params = list(sig.parameters.values())[1:]
            param_sets = []

            if (class_name := kernel_class.__name__) in ["Lanczos", "CustomLanczos"]:
                for taps in range(5, 13):
                    param_sets.append({"taps": taps})

            elif class_name in ["BlackMan", "BlackManMinLobe", "Hann", "Hamming", "Welch", "Cosine", "Bohman", "Sinc"]:
                continue

            elif class_name == "Gaussian":
                for sigma in [0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    taps = 2 if sigma <= 0.4 else 3 if sigma <= 0.7 else 4 if sigma <= 0.9 else 5
                    param_sets.append({"sigma": sigma, "taps": taps})

            elif class_name == "Bicubic":
                variants = [
                    (round(b, 2), round(c, 2))
                    for b in [i * 0.1 for i in range(11)]
                    for c in [j * 0.1 for j in range(11)]
                ]

                for b, c in variants:
                    param_sets.append({"b": b, "c": c})

            else:
                param_sets.append({})

                if any(param.name == "radius" for param in params):
                    param_sets.append({"radius": 3.0})

            for param_set in param_sets:
                try:
                    kernel = kernel_class(**param_set)
                    add_kernel_instance(kernel)
                except Exception:
                    continue

        except Exception as e:
            logging.error(f"Error processing kernel class {class_name}: {e}")
            continue

    return uncommon


def _format_kernel_name(kernel: KernelLike) -> str:
    def _e(x: float) -> str:
        return str(int(x)) if float(x).is_integer() else f"{x:.2f}"

    name = kernel.__class__.__name__

    attrs = []

    if hasattr(kernel, "b") and hasattr(kernel, "c"):
        attrs.append(f"b={_e(getattr(kernel, 'b'))}")
        attrs.append(f"c={_e(getattr(kernel, 'c'))}")

    if hasattr(kernel, "taps"):
        attrs.append(f"taps={getattr(kernel, 'taps')}")

    if hasattr(kernel, "sigma") and (sigma := getattr(kernel, "sigma")) is not None:
        attrs.append(f"sigma={sigma:.1f}")

    if hasattr(kernel, "radius") and (radius := getattr(kernel, "radius")) is not None:
        attrs.append(f"radius={radius:.1f}")

    if attrs:
        name += " (" + ", ".join(attrs) + ")"

    return name
