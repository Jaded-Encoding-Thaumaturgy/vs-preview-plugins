from __future__ import annotations

from PyQt6.QtWidgets import QTabWidget
from vspreview.main import MainWindow
from vspreview.plugins import AbstractPlugin, PluginConfig

from .components.trace import TraceTab

__all__ = ["TraceMoePlugin"]


class TraceMoePlugin(AbstractPlugin, QTabWidget):
    """VSPreview plugin for Trace.moe anime tracing."""

    _config = PluginConfig("dev.lightarrowsexe.trace_moe", "Trace.moe")

    def __init__(self, main: MainWindow) -> None:
        super().__init__(main)

        self.trace_tab = TraceTab(self)

    def setup_ui(self) -> None:
        """Set up the plugin UI."""

        self.addTab(self.trace_tab, "Trace")
