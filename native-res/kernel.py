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

CONTROL_HEIGHT = 24
DEFAULT_TAPS_RANGE = (5, 16)
DEFAULT_SIGMA_RANGE = (1.0, 1.0)
DEFAULT_BICUBIC_RANGE = (0.0, 1.0)
DEFAULT_BLUR_RANGE = (1.0, 1.0)
DEFAULT_SHIFT_RANGE = (0.0, 0.0)
DEFAULT_STEP_VALUES = {
    "bicubic": 0.05,
    "taps": 1.0,
    "sigma": 0.1,
    "blur": 0.1,
    "shift": 0.00125,
}


class ParameterRange:
    def __init__(self, min_val: float, max_val: float, step: float, decimals: int = 3):
        self.min_val, self.max_val = self._normalize_range(min_val, max_val)
        self.step = step
        self.decimals = decimals

    @classmethod
    def _normalize_range(cls, min_val: float, max_val: float) -> tuple[float, float]:
        if min_val > max_val:
            logging.debug(f"Swapped min/max values: min={max_val}, max={min_val}")
            return max_val, min_val

        return min_val, max_val

    def generate_values(self) -> list[float]:
        if self.min_val == self.max_val:
            return [self.min_val]

        num_steps = int((self.max_val - self.min_val) / self.step) + 1
        return [round(self.min_val + i * self.step, self.decimals) for i in range(num_steps)]


class KernelParameterGenerator:
    @classmethod
    def _get_range(
        cls,
        extensive_options: dict[str, float],
        min_key: str,
        max_key: str,
        step_key: str,
        default_range: tuple[float, float],
        default_step: float,
        decimals: int = 3,
        cast_int: bool = False,
    ) -> ParameterRange:
        min_val = extensive_options.get(min_key, default_range[0])
        max_val = extensive_options.get(max_key, default_range[1])
        step = extensive_options.get(step_key, default_step)

        if cast_int:
            min_val = int(min_val)
            max_val = int(max_val)

        return ParameterRange(min_val, max_val, step, decimals)

    @classmethod
    def get_taps_range(cls, extensive_options: dict[str, float]) -> ParameterRange:
        return cls._get_range(
            extensive_options,
            "taps_min",
            "taps_max",
            "taps_step",
            DEFAULT_TAPS_RANGE,
            DEFAULT_STEP_VALUES["taps"],
            decimals=0,
            cast_int=True,
        )

    @classmethod
    def get_sigma_range(cls, extensive_options: dict[str, float]) -> ParameterRange:
        return cls._get_range(
            extensive_options, "sigma_min", "sigma_max", "sigma_step", DEFAULT_SIGMA_RANGE, DEFAULT_STEP_VALUES["sigma"]
        )

    @classmethod
    def get_bicubic_range(cls, extensive_options: dict[str, float]) -> ParameterRange:
        return cls._get_range(
            extensive_options,
            "bicubic_min",
            "bicubic_max",
            "bicubic_step",
            DEFAULT_BICUBIC_RANGE,
            DEFAULT_STEP_VALUES["bicubic"],
        )

    @classmethod
    def get_steps_range(cls, extensive_options: dict[str, float], decimals: int = 3) -> ParameterRange:
        return cls._get_range(
            extensive_options,
            "blur_steps_min",
            "blur_steps_max",
            "blur_steps_step",
            DEFAULT_BLUR_RANGE,
            DEFAULT_STEP_VALUES["blur"],
            decimals=decimals,
        )

    @classmethod
    def get_shift_range(cls, extensive_options: dict[str, float]) -> ParameterRange:
        return cls._get_range(
            extensive_options,
            "shift_min",
            "shift_max",
            "shift_step",
            DEFAULT_SHIFT_RANGE,
            DEFAULT_STEP_VALUES["shift"],
            decimals=6,
        )


common_kernels: list[KernelLike] = [
    Catrom(),
    Mitchell(),
    BicubicSharp(),
    Bilinear(),
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
        shift_variations: list[tuple[float, float]] | None = None,
        blur_steps_variations: list[float] | None = None,
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
        self.shift_variations = shift_variations or [(src_top, src_left)]
        self.blur_steps_variations = blur_steps_variations or [1.0]
        self.data = []
        self._should_stop = False

    def stop(self) -> None:
        self._should_stop = True

    def _check_should_stop(self) -> bool:
        if self._should_stop:
            logging.debug("Worker stopped by user request")

        return self._should_stop

    def _log_kernel_error(self, operation: str, kernel_name: str, error: Exception) -> None:
        logging.error(f"Error {operation} kernel {kernel_name}: {error}")

    def _calculate_total_kernels(self) -> int:
        return len(self.kernels) * len(self.shift_variations) * len(self.blur_steps_variations)

    def _calculate_kernel_index(self, kernel_index: int, shift_index: int, blur_index: int) -> int:
        return (
            kernel_index
            + (shift_index * len(self.kernels) * len(self.blur_steps_variations))
            + (blur_index * len(self.kernels))
        )

    def _analyze_kernel(
        self,
        kernel: KernelLike,
        shift_top: float,
        shift_left: float,
        blur_steps: float,
        wclip: vs.VideoNode,
        target_w: int,
        target_h: int,
    ) -> tuple[float, float]:
        rs = Rescale(
            wclip,
            target_h,
            kernel,
            upscaler=kernel,
            downscaler=kernel,
            width=target_w,
            shift=(shift_top, shift_left),
            blur_steps=blur_steps,
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

        return mae_value, mse_value

    def _process_kernel_analysis(
        self,
        kernel: KernelLike,
        shift_top: float,
        shift_left: float,
        blur_steps: float,
        wclip: vs.VideoNode,
        target_w: int,
        target_h: int,
        kernel_index: int,
        shift_index: int,
        blur_index: int,
        total_kernels: int,
        results: list,
    ) -> None:
        current_kernel_index = self._calculate_kernel_index(kernel_index, shift_index, blur_index)
        prefix = f"{current_kernel_index + 1} / {total_kernels}"

        logging.debug(
            f"{prefix} Analyzing kernel {kernel.__class__.__name__} "
            f"with shift ({shift_top}, {shift_left}), blur_steps {blur_steps} @ {self.frame}"
        )

        try:
            mae_value, mse_value = self._analyze_kernel(
                kernel, shift_top, shift_left, blur_steps, wclip, target_w, target_h
            )
            sigma = getattr(kernel, "sigma", 1.0) if kernel.__class__.__name__ == "Gaussian" else 1.0
            results.append((kernel, mae_value, mse_value, shift_top, shift_left, blur_steps, sigma))
        except Exception as e:
            self._log_kernel_error("analyzing", kernel.__class__.__name__, e)
            sigma = getattr(kernel, "sigma", 1.0) if kernel.__class__.__name__ == "Gaussian" else 1.0
            results.append((kernel, float("inf"), float("inf"), shift_top, shift_left, blur_steps, sigma))

        self.update.emit(int((current_kernel_index + 1) / total_kernels * 100))

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
        total_kernels = self._calculate_total_kernels()

        for shift_index, (shift_top, shift_left) in enumerate(self.shift_variations):
            if self._check_should_stop():
                return

            for blur_index, blur_steps in enumerate(self.blur_steps_variations):
                if self._check_should_stop():
                    return

                for kernel_index, kernel in enumerate(self.kernels):
                    if self._check_should_stop():
                        return

                    self._process_kernel_analysis(
                        kernel,
                        shift_top,
                        shift_left,
                        blur_steps,
                        wclip,
                        target_w,
                        target_h,
                        kernel_index,
                        shift_index,
                        blur_index,
                        total_kernels,
                        results,
                    )

        logging.debug([f"{result[0].__class__.__name__}: {result[1]}\n" for result in results])

        if not self._check_should_stop():
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

    def update_results(self, results: list[tuple[Kernel, float, float, float, float, float, float]]) -> None:
        if not results:
            self.kernels_table.setRowCount(0)
            return

        results.sort(key=lambda x: x[1])
        _, best_mae, _, _, _, _, _ = results[0]

        valid_results = [
            (k, mae, mse, src_top, src_left, blur_steps, sigma)
            for k, mae, mse, src_top, src_left, blur_steps, sigma in results
            if mae != float("inf")
        ]

        self.kernels_table.setRowCount(len(valid_results))

        for row, (kernel, mae, mse, src_top, src_left, blur_steps, sigma) in enumerate(valid_results):
            kernel_name = _format_kernel_name(kernel, src_top, src_left, blur_steps, sigma)
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
        self.current_data = dict[int, dict[int, list[tuple[Kernel, float, float, float, float, float, float]] | None]]()

        self.render_lock = False

    def setup_ui(self) -> None:
        self.controls = QWidget()

        self.dimension_switch = Switch(
            12, 40, 1, ("Height", "Width", 2), checked=False, clicked=self.on_dimension_change
        )

        self.width_label = QLabel("Width")
        self.width_spinbox = self._create_spinbox(1920.0, 1.0, 9999.0, 3, 1.0)

        self.height_label = QLabel("Height")
        self.height_spinbox = self._create_spinbox(1080.0, 1.0, 9999.0, 3, 1.0)

        self.src_top_spinbox = self._create_spinbox(0.0, -100.0, 100.0, 6, 0.000025)
        self.src_left_spinbox = self._create_spinbox(0.0, -100.0, 100.0, 6, 0.000025)

        # Video processing options
        self.border_handling_combobox = ComboBox[str](
            model=GeneralModel[str]([member.name.replace("_", " ").title() for member in BorderHandling], False),
            currentIndex=0,
            sizeAdjustPolicy=ComboBox.SizeAdjustPolicy.AdjustToContents,
        )
        self._setup_combobox_height(self.border_handling_combobox)

        self.sample_grid_model_switch = Switch(12, 48, 1, ("Edges", "Centers", 2), checked=False)
        self._setup_control_size_policy(self.sample_grid_model_switch)

        self.field_based_combobox = ComboBox[str](
            model=GeneralModel[str]([member.pretty_string for member in FieldBased], False),
            currentIndex=0,
            sizeAdjustPolicy=ComboBox.SizeAdjustPolicy.AdjustToContents,
        )
        self._setup_combobox_height(self.field_based_combobox)

        # Kernel search options
        self.uncommon_kernels_switch = Switch(12, 56, 1, ("Default", "Advanced", 2), checked=False)
        self._setup_control_size_policy(self.uncommon_kernels_switch)

        self.update_button = PushButton("Analyze", clicked=lambda _: self.render_canvas())

        self.kernel_type_bar = QWidget()
        self.setup_kernel_type_bar()
        self.windowed_parameter_controls = QWidget()
        self.setup_windowed_parameter_controls()

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
                self._create_labeled_control("Border handling", self.border_handling_combobox),
                self._create_labeled_control("Sample grid", self.sample_grid_model_switch),
                self._create_labeled_control("Field order", self.field_based_combobox),
                self._create_labeled_control("Extensive kernel search", self.uncommon_kernels_switch),
                Stretch(),
                self.update_button,
            ],
        )

        self.controls.setFixedHeight(80)

        self.dimension_switch.clicked.connect(self.on_dimension_switch_changed)
        self.uncommon_kernels_switch.clicked.connect(self.on_windowed_kernels_switch_changed)

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

        self.vlayout = VBoxLayout(
            self,
            [
                self.controls,
                self.kernel_type_bar,
                self.windowed_parameter_controls,
                self.errors_loading,
                self.display,
            ],
        )
        self.set_visibility(False, False)

        self.on_dimension_change()

    def setup_kernel_type_bar(self) -> None:
        self.bicubic_switch = Switch(12, 48, 1, ("Disabled", "Enabled", 2), checked=True)
        self.lanczos_switch = Switch(12, 48, 1, ("Disabled", "Enabled", 2), checked=True)
        self.gaussian_switch = Switch(12, 48, 1, ("Disabled", "Enabled", 2), checked=True)
        self.spline_switch = Switch(12, 48, 1, ("Disabled", "Enabled", 2), checked=True)

        for switch in [self.bicubic_switch, self.lanczos_switch, self.gaussian_switch, self.spline_switch]:
            self._setup_control_size_policy(switch)

        self.bicubic_switch.clicked.connect(self.update_windowed_parameter_visibility)
        self.lanczos_switch.clicked.connect(self.update_windowed_parameter_visibility)
        self.gaussian_switch.clicked.connect(self.update_windowed_parameter_visibility)
        self.spline_switch.clicked.connect(self.update_windowed_parameter_visibility)

        self.global_blur_steps_min_spinbox = self._create_spinbox(1.0, 0.1, 5.0, 3)
        self.global_blur_steps_max_spinbox = self._create_spinbox(1.0, 0.1, 5.0, 3)
        self.global_blur_steps_step_spinbox = self._create_spinbox(
            DEFAULT_STEP_VALUES["blur"], 0.01, 2.0, 3, DEFAULT_STEP_VALUES["blur"]
        )
        self.global_shift_min_spinbox = self._create_spinbox(0.0, -10.0, 10.0, 6)
        self.global_shift_max_spinbox = self._create_spinbox(0.0, -10.0, 10.0, 6)
        self.global_shift_step_spinbox = self._create_spinbox(
            DEFAULT_STEP_VALUES["shift"], -10.0, 10.0, 6, DEFAULT_STEP_VALUES["shift"]
        )

        HBoxLayout(
            self.kernel_type_bar,
            [
                self._create_labeled_control("Search Bicubic", self.bicubic_switch),
                self._create_labeled_control("Search Lanczos", self.lanczos_switch),
                self._create_labeled_control("Search Gaussian", self.gaussian_switch),
                self._create_labeled_control("Search Spline", self.spline_switch),
                Stretch(),
                VBoxLayout(
                    [
                        QLabel("Blur min/max"),
                        HBoxLayout(
                            [
                                self.global_blur_steps_min_spinbox,
                                self.global_blur_steps_max_spinbox,
                            ]
                        ),
                    ]
                ),
                self._create_labeled_control("Blur steps", self.global_blur_steps_step_spinbox),
                VBoxLayout(
                    [
                        QLabel("Shift min/max"),
                        HBoxLayout(
                            [
                                self.global_shift_min_spinbox,
                                self.global_shift_max_spinbox,
                            ]
                        ),
                    ]
                ),
                self._create_labeled_control("Shift steps", self.global_shift_step_spinbox),
            ],
        )

        self.kernel_type_bar.setVisible(False)

    def setup_windowed_parameter_controls(self) -> None:
        self.bicubic_min_spinbox = self._create_spinbox(0.0, -100.0, 100.0, 3)
        self.bicubic_max_spinbox = self._create_spinbox(1.0, -100.0, 100.0, 3)
        self.bicubic_step_spinbox = self._create_spinbox(
            DEFAULT_STEP_VALUES["bicubic"], 0.01, 10.0, 3, DEFAULT_STEP_VALUES["bicubic"]
        )

        self.taps_min_spinbox = self._create_spinbox(5.0, 0.0, 100.0, 3)
        self.taps_max_spinbox = self._create_spinbox(12.0, 0.0, 100.0, 3)
        self.taps_step_spinbox = self._create_spinbox(
            DEFAULT_STEP_VALUES["taps"], 0.1, 10.0, 3, DEFAULT_STEP_VALUES["taps"]
        )

        self.sigma_min_spinbox = self._create_spinbox(0.1, 0.1, 10.0, 3)
        self.sigma_max_spinbox = self._create_spinbox(10.0, 0.1, 10.0, 3)
        self.sigma_step_spinbox = self._create_spinbox(
            DEFAULT_STEP_VALUES["sigma"], 0.01, 10.0, 3, DEFAULT_STEP_VALUES["sigma"]
        )

        self.bicubic_minmax_label = QLabel("b/c min/max")
        self.bicubic_step_label = QLabel("b/c step")
        self.taps_minmax_label = QLabel("Taps min/max")
        self.taps_step_label = QLabel("Taps step")
        self.sigma_minmax_label = QLabel("Sigma min/max")
        self.sigma_step_label = QLabel("Sigma step")

        main_layout = VBoxLayout(self.windowed_parameter_controls)
        hbox_layout = HBoxLayout()
        hbox_layout.addWidget(
            self._create_group_widget(self.bicubic_minmax_label, [self.bicubic_min_spinbox, self.bicubic_max_spinbox]),
            1,
        )

        hbox_layout.addWidget(self._create_group_widget(self.bicubic_step_label, [self.bicubic_step_spinbox]), 1)

        hbox_layout.addWidget(
            self._create_group_widget(self.taps_minmax_label, [self.taps_min_spinbox, self.taps_max_spinbox]), 1
        )

        hbox_layout.addWidget(self._create_group_widget(self.taps_step_label, [self.taps_step_spinbox]), 1)

        hbox_layout.addWidget(
            self._create_group_widget(self.sigma_minmax_label, [self.sigma_min_spinbox, self.sigma_max_spinbox]), 1
        )

        hbox_layout.addWidget(self._create_group_widget(self.sigma_step_label, [self.sigma_step_spinbox]), 1)

        main_layout.addLayout(hbox_layout)

        self.windowed_parameter_controls.setVisible(False)

    def _create_group_widget(self, label: QLabel, spinboxes: list[DoubleSpinBox]) -> QWidget:
        group_widget = QWidget()
        group_layout = VBoxLayout(group_widget)
        group_layout.addWidget(label)
        spinbox_layout = HBoxLayout()
        for spinbox in spinboxes:
            spinbox_layout.addWidget(spinbox)
        group_layout.addLayout(spinbox_layout)
        group_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        return group_widget

    def _create_spinbox(
        self, value: float, minimum: float, maximum: float, decimals: int, step: float = 0.1
    ) -> DoubleSpinBox:
        spinbox = DoubleSpinBox(
            value=value,
            minimum=minimum,
            maximum=maximum,
            decimals=decimals,
            stepType=DoubleSpinBox.StepType.AdaptiveDecimalStepType,
        )
        spinbox.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        spinbox.setSingleStep(step)
        spinbox.setMaximumHeight(CONTROL_HEIGHT)
        return spinbox

    def _setup_control_size_policy(self, control: QWidget) -> None:
        control.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    def _setup_combobox_height(self, combobox: ComboBox) -> None:
        combobox.setMaximumHeight(CONTROL_HEIGHT)

    def _create_labeled_control(self, label_text: str, control: QWidget) -> VBoxLayout:
        return VBoxLayout([QLabel(label_text), control])

    def _safe_kernel_creation(self, kernel_class: type, **kwargs) -> KernelLike | None:
        try:
            return kernel_class(**kwargs)
        except Exception:
            return None

    def _log_kernel_error(self, operation: str, kernel_name: str, error: Exception) -> None:
        logging.error(f"Error {operation} kernel {kernel_name}: {error}")

    def update_windowed_parameter_visibility(self) -> None:
        bicubic_enabled = self.bicubic_switch.isChecked()
        self.bicubic_minmax_label.setVisible(bicubic_enabled)
        self.bicubic_min_spinbox.setVisible(bicubic_enabled)
        self.bicubic_max_spinbox.setVisible(bicubic_enabled)
        self.bicubic_step_label.setVisible(bicubic_enabled)
        self.bicubic_step_spinbox.setVisible(bicubic_enabled)

        taps_enabled = self.lanczos_switch.isChecked() or self.spline_switch.isChecked()
        self.taps_minmax_label.setVisible(taps_enabled)
        self.taps_min_spinbox.setVisible(taps_enabled)
        self.taps_max_spinbox.setVisible(taps_enabled)
        self.taps_step_label.setVisible(taps_enabled)
        self.taps_step_spinbox.setVisible(taps_enabled)

        gaussian_enabled = self.gaussian_switch.isChecked()
        self.sigma_minmax_label.setVisible(gaussian_enabled)
        self.sigma_min_spinbox.setVisible(gaussian_enabled)
        self.sigma_max_spinbox.setVisible(gaussian_enabled)
        self.sigma_step_label.setVisible(gaussian_enabled)
        self.sigma_step_spinbox.setVisible(gaussian_enabled)

        any_options_visible = bicubic_enabled or taps_enabled or gaussian_enabled
        self.windowed_parameter_controls.setVisible(any_options_visible)

    def set_visibility(self, errors: bool, display: bool) -> None:
        if errors or display:
            if self.vlayout.alignment():
                self.vlayout.setAlignment(Qt.AlignmentFlag(0))
        else:
            self.vlayout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.errors_loading.setVisible(errors)
        self.display.setVisible(display)

        self.kernel_type_bar.setVisible(self.uncommon_kernels_switch.isChecked())
        self.windowed_parameter_controls.setVisible(self.uncommon_kernels_switch.isChecked())
        if self.uncommon_kernels_switch.isChecked():
            self.update_windowed_parameter_visibility()

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

    def on_windowed_kernels_switch_changed(self) -> None:
        self.workers_lookup.clear()
        self.current_data.clear()

        if (
            hasattr(self, "main")
            and self.main.current_output
            and (frame := self.main.current_output.last_showed_frame) in self.workers
        ):
            del self.workers[frame]

        self.windowed_parameter_controls.setVisible(self.uncommon_kernels_switch.isChecked())
        self.kernel_type_bar.setVisible(self.uncommon_kernels_switch.isChecked())

        if self.uncommon_kernels_switch.isChecked():
            self.update_windowed_parameter_visibility()

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

        kernels = list(common_kernels.copy() if frame not in self.workers else self.workers[frame].kernels)

        if self.uncommon_kernels_switch.isChecked():
            windowed_options = self.get_windowed_options()
            enabled_kernels = self.get_enabled_kernel_types()
            analysis_kernels = get_analysis_kernels(windowed_options, enabled_kernels)
            kernels.extend(analysis_kernels)

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

        shift_variations = [(src_top, src_left)]
        blur_steps_variations = [1.0]

        if self.uncommon_kernels_switch.isChecked():
            windowed_options = self.get_windowed_options()
            enabled_kernels = self.get_enabled_kernel_types()

            # shift variations
            shift_range = KernelParameterGenerator.get_shift_range(windowed_options)
            shift_values = shift_range.generate_values()

            if len(shift_values) > 1:
                shift_variations = [(shift, shift) for shift in shift_values]
                logging.debug(f"Generated {len(shift_variations)} shift variations: {shift_variations}")
            else:
                logging.debug("Shift min equals max, using default shift")

            # blur steps variations
            steps_range = KernelParameterGenerator.get_steps_range(windowed_options)
            blur_steps_values = steps_range.generate_values()

            if len(blur_steps_values) > 1:
                blur_steps_variations = blur_steps_values
                logging.debug(f"Generated {len(blur_steps_variations)} blur steps variations: {blur_steps_variations}")
            else:
                logging.debug("Blur steps min equals max, using default blur steps")

        border_handling = BorderHandling(self.border_handling_combobox.currentIndex())
        sample_grid_model = SampleGridModel(0 if self.sample_grid_model_switch.isChecked() else 1)

        field_based_idx = self.field_based_combobox.currentIndex()
        field_based = FieldBased(field_based_idx)

        kernel_list = [k for k in kernels if isinstance(k, Kernel)]
        self.workers[frame] = KernelRunner(
            self,
            self.curr_clip,
            int(frame),
            kernel_list,
            target_width,
            target_height,
            src_top,
            src_left,
            border_handling,
            sample_grid_model,
            field_based,
            shift_variations,
            blur_steps_variations,
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

    def get_enabled_kernel_types(self) -> dict[str, bool]:
        return {
            "bicubic": self.bicubic_switch.isChecked(),
            "lanczos": self.lanczos_switch.isChecked(),
            "gaussian": self.gaussian_switch.isChecked(),
            "spline": self.spline_switch.isChecked(),
        }

    def get_windowed_options(self) -> dict[str, float]:
        options = {
            "bicubic_min": self.bicubic_min_spinbox.value(),
            "bicubic_max": self.bicubic_max_spinbox.value(),
            "bicubic_step": self.bicubic_step_spinbox.value(),
            "taps_min": self.taps_min_spinbox.value(),
            "taps_max": self.taps_max_spinbox.value(),
            "taps_step": self.taps_step_spinbox.value(),
            "sigma_min": self.sigma_min_spinbox.value(),
            "sigma_max": self.sigma_max_spinbox.value(),
            "sigma_step": self.sigma_step_spinbox.value(),
            "blur_steps_min": self.global_blur_steps_min_spinbox.value(),
            "blur_steps_max": self.global_blur_steps_max_spinbox.value(),
            "blur_steps_step": self.global_blur_steps_step_spinbox.value(),
            "shift_min": self.global_shift_min_spinbox.value(),
            "shift_max": self.global_shift_max_spinbox.value(),
            "shift_step": self.global_shift_step_spinbox.value(),
        }
        logging.debug(f"Windowed options: {options}")
        return options

    def set_windowed_options(self, options: dict[str, float] = {}) -> None:
        spinbox_mapping = {
            "bicubic_min": self.bicubic_min_spinbox,
            "bicubic_max": self.bicubic_max_spinbox,
            "bicubic_step": self.bicubic_step_spinbox,
            "taps_min": self.taps_min_spinbox,
            "taps_max": self.taps_max_spinbox,
            "taps_step": self.taps_step_spinbox,
            "sigma_min": self.sigma_min_spinbox,
            "sigma_max": self.sigma_max_spinbox,
            "sigma_step": self.sigma_step_spinbox,
            "blur_steps_min": self.global_blur_steps_min_spinbox,
            "blur_steps_max": self.global_blur_steps_max_spinbox,
            "blur_steps_step": self.global_blur_steps_step_spinbox,
            "shift_min": self.global_shift_min_spinbox,
            "shift_max": self.global_shift_max_spinbox,
            "shift_step": self.global_shift_step_spinbox,
        }

        for key, value in options.items():
            if key in spinbox_mapping:
                spinbox_mapping[key].setValue(value)

    def validate_windowed_options(self) -> bool:
        if self.bicubic_min_spinbox.value() >= self.bicubic_max_spinbox.value():
            return False

        if self.taps_min_spinbox.value() >= self.taps_max_spinbox.value():
            return False

        if self.sigma_min_spinbox.value() >= self.sigma_max_spinbox.value():
            return False

        if self.global_blur_steps_min_spinbox.value() >= self.global_blur_steps_max_spinbox.value():
            return False

        return not (self.global_shift_min_spinbox.value() >= self.global_shift_max_spinbox.value())

    def init_outputs(self) -> None:
        self.current_data.clear()


def _safe_kernel_creation(kernel_class: type, **kwargs) -> KernelLike | None:
    try:
        return kernel_class(**kwargs)
    except Exception:
        return None


def _apply_global_options_to_common_kernels(
    windowed_options: dict[str, float], add_kernel_instance: Callable[[KernelLike], None]
) -> None:
    has_global_shift = windowed_options.get("shift_min", 0.0) != 0.0 or windowed_options.get("shift_max", 0.0) != 0.0
    has_global_steps = (
        windowed_options.get("blur_steps_min", 1.0) != 1.0 or windowed_options.get("blur_steps_max", 3.0) != 3.0
    )

    if not (has_global_shift or has_global_steps):
        return

    for kernel in common_kernels:
        add_kernel_instance(kernel)

        if has_global_steps and hasattr(kernel, "steps"):
            steps_min = windowed_options.get("blur_steps_min", 1.0)
            steps_max = windowed_options.get("blur_steps_max", 3.0)
            steps_step = windowed_options.get("blur_steps_step", 1.0)

            if steps_min > steps_max:
                steps_min, steps_max = steps_max, steps_min
                logging.debug(f"Swapped steps min/max values: min={steps_min}, max={steps_max}")

            steps_values = [
                round(steps_min + i * steps_step, 3) for i in range(int((steps_max - steps_min) / steps_step) + 1)
            ]

            for steps in steps_values:
                if steps_min <= steps <= steps_max and steps != getattr(kernel, "steps", 1.0):
                    try:
                        kernel_class = kernel.__class__
                        kernel_params = {k: v for k, v in kernel.__dict__.items() if not k.startswith("_")}
                        kernel_params["steps"] = steps
                        new_kernel = kernel_class(**kernel_params)
                        add_kernel_instance(new_kernel)
                    except Exception:
                        continue


def _generate_lanczos_kernels(
    windowed_options: dict[str, float], add_kernel_instance: Callable[[KernelLike], None]
) -> None:
    taps_range = KernelParameterGenerator.get_taps_range(windowed_options)
    taps_values = taps_range.generate_values()

    for taps in taps_values:
        if taps_range.min_val <= taps <= taps_range.max_val and (
            kernel := _safe_kernel_creation(Lanczos, taps=int(taps))
        ):
            add_kernel_instance(kernel)


def _generate_gaussian_kernels(
    windowed_options: dict[str, float], add_kernel_instance: Callable[[KernelLike], None]
) -> None:
    sigma_range = KernelParameterGenerator.get_sigma_range(windowed_options)
    sigma_values = sigma_range.generate_values()

    for sigma in sigma_values:
        if sigma_range.min_val <= sigma <= sigma_range.max_val:
            taps = 2 if sigma <= 0.4 else 3 if sigma <= 0.7 else 4 if sigma <= 0.9 else 5
            from vskernels import Gaussian

            if kernel := _safe_kernel_creation(Gaussian, sigma=sigma, taps=taps):
                add_kernel_instance(kernel)


def _generate_bicubic_kernels(
    windowed_options: dict[str, float], add_kernel_instance: Callable[[KernelLike], None]
) -> None:
    bc_range = KernelParameterGenerator.get_bicubic_range(windowed_options)
    bc_values = bc_range.generate_values()

    variants = [
        (round(b, 3), round(c, 3))
        for b in bc_values
        if bc_range.min_val <= b <= bc_range.max_val
        for c in bc_values
        if bc_range.min_val <= c <= bc_range.max_val
    ]

    for b, c in variants:
        from vskernels import Bicubic

        if kernel := _safe_kernel_creation(Bicubic, b=b, c=c):
            add_kernel_instance(kernel)


def _generate_spline_kernels(
    windowed_options: dict[str, float], add_kernel_instance: Callable[[KernelLike], None]
) -> None:
    taps_range = KernelParameterGenerator.get_taps_range(windowed_options)
    taps_values = taps_range.generate_values()

    for taps in taps_values:
        if taps_range.min_val <= taps <= taps_range.max_val:
            from vskernels import Spline, Spline16, Spline36, Spline64

            if taps == 2:
                if kernel := _safe_kernel_creation(Spline16):
                    add_kernel_instance(kernel)
            elif taps == 3:
                if kernel := _safe_kernel_creation(Spline36):
                    add_kernel_instance(kernel)
            elif taps == 4:
                if kernel := _safe_kernel_creation(Spline64):
                    add_kernel_instance(kernel)
            else:
                if kernel := _safe_kernel_creation(Spline, taps=taps):
                    add_kernel_instance(kernel)


def get_analysis_kernels(
    windowed_options: dict[str, float] = {},
    check_kernels: dict[str, bool] = {},
) -> list[KernelLike]:
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

    def add_kernel_instance(kernel: KernelLike) -> None:
        if (key := kernel_key(kernel)) not in common_instances and key not in added_instances:
            uncommon.append(kernel)
            added_instances.add(key)

    for kernel in common_kernels:
        add_kernel_instance(kernel)

    any_kernel_type_enabled = any(check_kernels.values())

    if not any_kernel_type_enabled:
        _apply_global_options_to_common_kernels(windowed_options, add_kernel_instance)

    if check_kernels.get("lanczos", False):
        _generate_lanczos_kernels(windowed_options, add_kernel_instance)

    if check_kernels.get("gaussian", False):
        _generate_gaussian_kernels(windowed_options, add_kernel_instance)

    if check_kernels.get("bicubic", False):
        _generate_bicubic_kernels(windowed_options, add_kernel_instance)

    if check_kernels.get("spline", False):
        _generate_spline_kernels(windowed_options, add_kernel_instance)

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

    for kernel_class in all_kernel_classes:
        try:
            sig = inspect.signature(kernel_class.__init__)
            params = list(sig.parameters.values())[1:]
            param_sets = []

            class_name = kernel_class.__name__

            if class_name in [
                "Lanczos",
                "CustomLanczos",
                "Gaussian",
                "Bicubic",
                "Spline",
                "Spline16",
                "Spline36",
                "Spline64",
            ]:
                continue

            param_sets.append({})

            if any(param.name == "steps" for param in params) and any_kernel_type_enabled:
                steps_range = KernelParameterGenerator.get_steps_range(windowed_options)
                steps_values = steps_range.generate_values()

                for steps in steps_values:
                    param_sets.append({"steps": steps})

            for param_set in param_sets:
                if kernel := _safe_kernel_creation(kernel_class, **param_set):
                    add_kernel_instance(kernel)

        except Exception as e:
            logging.error(f"Error processing kernel class: {e}")
            continue

    return _deduplicate_kernels(uncommon)


def _deduplicate_kernels(kernels: list[KernelLike]) -> list[KernelLike]:
    dedup_keys = set()
    deduped_kernels = []

    for kernel in kernels:
        key_parts = [kernel.__class__.__name__]

        for attr in ("b", "c", "taps", "sigma"):
            if hasattr(kernel, attr):
                val = getattr(kernel, attr)

                if isinstance(val, float):
                    val = round(val, 6)

                key_parts.append(f"{attr}={val}")

        dedup_key = tuple(key_parts)

        if dedup_key not in dedup_keys:
            dedup_keys.add(dedup_key)
            deduped_kernels.append(kernel)

    if len(deduped_kernels) >= 500:
        logging.warning(
            f"You are about to generate a LOT of kernels ({len(deduped_kernels)}). "
            "This may take a while and may not be worth it!"
        )

    return deduped_kernels


def _format_kernel_name(
    kernel: KernelLike, src_top: float = 0.0, src_left: float = 0.0, blur_steps: float = 1.0, sigma: float = 1.0
) -> str:
    def _e(x: float) -> str:
        return str(int(x)) if float(x).is_integer() else f"{x:.2f}"

    name = kernel.__class__.__name__

    attrs = []

    if hasattr(kernel, "b") and hasattr(kernel, "c"):
        attrs.append(f"b={_e(getattr(kernel, 'b'))}")
        attrs.append(f"c={_e(getattr(kernel, 'c'))}")

    if hasattr(kernel, "taps") and (taps := getattr(kernel, "taps")) is not None:
        if isinstance(taps, float) and not taps.is_integer():
            attrs.append(f"taps={taps:.3f}")
        else:
            attrs.append(f"taps={taps}")

    if hasattr(kernel, "sigma") and (sigma := getattr(kernel, "sigma")) is not None:
        attrs.append(f"sigma={sigma:.3f}")

    if src_top != 0.0 or src_left != 0.0:
        attrs.append(f"shift=({src_top:.6f}, {src_left:.6f})")

    if blur_steps != 1.0:
        attrs.append(f"blur={blur_steps:.3f}")

    if attrs:
        name += " (" + ", ".join(attrs) + ")"

    return name
