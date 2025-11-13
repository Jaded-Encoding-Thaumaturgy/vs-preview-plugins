from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import QTabWidget
from vspreview.core import Frame
from vspreview.main import MainWindow
from vspreview.plugins import AbstractPlugin, PluginConfig

from .frequency import FrequencyAnalyzer
from .kernel import KernelAnalyzer
from .rescale import RescaleErrorAnalyzer

__all__ = ["NativeResolutionPlugin"]


class NativeResolutionPlugin(AbstractPlugin, QTabWidget):
    _config = PluginConfig("dev.setsugen.native_resolution", "Native Resolution")

    def __init__(self, main: MainWindow) -> None:
        super().__init__(main)

        self.rescale = RescaleErrorAnalyzer(main)
        self.dft = FrequencyAnalyzer(main)
        self.kernel = KernelAnalyzer(main)

        self.addTab(self.rescale, "Rescale Error Analyze")
        self.addTab(self.dft, "Frequency Analyze")
        self.addTab(self.kernel, "Kernel Analyze")

        self.modes = (self.rescale, self.dft, self.kernel)
        self.cache = [(-1, -1)] * len(self.modes)

        self.currentChanged.connect(self.on_tab_change)

    def setup_ui(self) -> None:
        for mode in self.modes:
            mode.setup_ui()

    def add_shortcuts(self) -> None:
        def reset_zoom() -> None:
            if self.underMouse():
                current_mode = self.modes[self.currentIndex()]
                if hasattr(current_mode, "canvas"):
                    current_mode.canvas.render(None, False)
                elif hasattr(current_mode, "render_canvas"):
                    current_mode.render_canvas()

        def render_canvas_on_rescale_error() -> None:
            if self.underMouse() and self.currentIndex() == 0:
                self.rescale.render_canvas()

        def render_canvas_on_kernel() -> None:
            if self.underMouse() and self.currentIndex() == 2:
                self.kernel.render_canvas()

        self.add_shortcut("Reset zoom", self, reset_zoom, QKeySequence(Qt.Key.Key_Escape))
        self.add_shortcut("Render canvas", self, render_canvas_on_rescale_error, QKeySequence(Qt.Key.Key_Enter))
        self.add_shortcut("Render canvas", self, render_canvas_on_rescale_error, QKeySequence(Qt.Key.Key_Return))
        self.add_shortcut("Render kernel analyze", self, render_canvas_on_kernel, QKeySequence(Qt.Key.Key_G))

    def init_outputs(self) -> None:
        assert self.main.outputs

        for mode in self.modes:
            mode.init_outputs()

    def on_tab_change(self) -> None:
        idx, key = (
            self.currentIndex(),
            (int(self.main.current_output.last_showed_frame), self.main.current_output.index),
        )
        if self.cache[idx] != key:
            if self.cache[idx][1] != key[1]:
                self.modes[idx].on_current_output_changed(key[1], self.cache[idx][1])
            self.modes[idx].on_current_frame_changed(self.main.current_output.last_showed_frame)

    def on_current_frame_changed(self, frame: Frame) -> None:
        idx = self.currentIndex()
        self.modes[idx].on_current_frame_changed(frame)
        self.cache[idx] = (int(frame), self.main.current_output.index)

    def on_current_output_changed(self, index: int, prev_index: int) -> None:
        idx = self.currentIndex()
        self.modes[idx].on_current_output_changed(index, prev_index)
        self.cache[idx] = (int(self.main.current_output.last_showed_frame), index)
