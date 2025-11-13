from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING
from uuid import uuid4

from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QIntValidator, QPixmap, QResizeEvent
from PyQt6.QtWidgets import (
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from vspreview.core import ExtendedWidget, HBoxLayout, VBoxLayout

from .models import MatchInfo, SearchResult
from .worker import Worker, WorkerConfiguration

if TYPE_CHECKING:
    from ...trace_moe import TraceMoePlugin


class TraceTab(ExtendedWidget):
    __slots__ = (
        "anilist_id_edit",
        "min_similarity_edit",
        "plugin",
        "start_button",
        "status_label",
        "worker",
        "worker_thread",
    )

    def __init__(self, plugin: TraceMoePlugin) -> None:
        super().__init__()
        self.plugin = plugin
        self.worker_thread: QThread | None = None
        self.worker: Worker | None = None

        self.setup_ui()

    def setup_ui(self) -> None:
        self.start_button = QPushButton("Start trace")
        self.start_button.clicked.connect(self.start_tracing)
        self.start_button.setMaximumWidth(100)

        self.anilist_id_edit = QLineEdit()
        self.anilist_id_edit.setValidator(QIntValidator())
        self.anilist_id_edit.setPlaceholderText("Anilist ID")
        self.anilist_id_edit.setMaximumWidth(64)

        self.min_similarity_edit = QSpinBox()
        self.min_similarity_edit.setRange(0, 99)
        self.min_similarity_edit.setValue(85)
        self.min_similarity_edit.setMaximumWidth(54)
        self.min_similarity_edit.setSuffix("%")

        self.min_similarity_label = QLabel("Min similarity:")

        self.min_similarity_hbox = HBoxLayout([], [])
        self.min_similarity_hbox.setSpacing(0)
        self.min_similarity_hbox.addWidget(self.min_similarity_label)
        self.min_similarity_hbox.addWidget(self.min_similarity_edit)

        self.status_label = QLabel("Waiting...")

        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.results_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.results_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.results_scroll.setStyleSheet(
            """
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #2a2a2a;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666;
            }
        """
        )

        self.results_container = QWidget()
        self.results_container.setContentsMargins(0, 0, 0, 0)

        self.results_layout = VBoxLayout(self.results_container, [])
        self.results_layout.setSpacing(6)
        self.results_layout.setContentsMargins(0, 0, 0, 0)

        self.results_scroll.setWidget(self.results_container)

        main_layout = VBoxLayout(self, [])
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.match_widgets = list[tuple[MatchInfo, QWidget]]()

        header_widget = QWidget()
        header_widget.setContentsMargins(0, 0, 0, 0)

        header_layout = HBoxLayout(
            header_widget,
            [self.start_button, self.anilist_id_edit, self.min_similarity_hbox],
        )
        header_layout.addStretch()
        header_layout.addWidget(self.status_label)

        main_layout.addWidget(header_widget)
        main_layout.addWidget(self.results_scroll)

    def start_tracing(self) -> None:
        self.clear_results()

        anilist_id = None

        anilist_text = self.anilist_id_edit.text().strip()
        if anilist_text:
            try:
                anilist_id = int(anilist_text)
            except ValueError:
                self.status_label.setText("Invalid Anilist ID")
                return

        try:
            current_frame = self.plugin.main.current_output.last_showed_frame
            frame_number = int(current_frame)
        except Exception as e:
            self.status_label.setText(f"No frame available: {e}")
            return

        self.start_button.setEnabled(False)
        self.status_label.setText("Working...")

        plugin_path = self.plugin.main.current_config_dir / "trace-moe"
        min_similarity = self.min_similarity_edit.value() / 100.0

        conf = WorkerConfiguration(
            uuid=str(uuid4()),
            node=self.plugin.main.current_output,
            path=plugin_path,
            main=self.plugin.main,
            frame_num=frame_number,
            api_key="",
            anilist_id=anilist_id,
            cut_black_borders=True,
            min_similarity=min_similarity,
        )

        self.worker_thread = QThread()
        self.worker = Worker()
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(partial(self.worker.run, conf))
        self.worker.finished.connect(partial(self.on_worker_finished, conf=conf))

        self.worker_thread.start()

    def on_worker_finished(self, result_path: str, search_result: SearchResult, *, conf: WorkerConfiguration) -> None:
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread.deleteLater()

        if self.worker:
            self.worker.deleteLater()

        self.start_button.setEnabled(True)
        self.status_label.setText("Done!")

        self.show_results(search_result)

    def clear_results(self) -> None:
        while self.results_layout.count() > 0:
            item = self.results_layout.takeAt(0)
            if item:
                widget = item.widget()
                if widget:
                    widget.setParent(None)
                else:
                    del item

        self.match_widgets = list[tuple[MatchInfo, QWidget]]()
        self.status_label.setText("Waiting...")

    def show_results(self, search_result: SearchResult) -> None:
        self.clear_results()

        if not search_result.matches:
            min_similarity_percent = self.min_similarity_edit.value()
            self.status_label.setText(f"No matches above {min_similarity_percent}%")
            return

        self.match_widgets = list[tuple[MatchInfo, QWidget]]()

        for i, match in enumerate(search_result.matches):
            match_widget = self.create_match_widget(match, i)
            self.match_widgets.append((match, match_widget))
            self.load_thumbnail(match, match_widget)

        first_match = search_result.matches[0]
        title = first_match.anilist.title_native or "Unknown Title"
        episode = f"Episode {first_match.episode}" if first_match.episode else ""
        timestamp = first_match.format_timestamp()
        similarity = first_match.similarity_as_percentage

        result_text = f"{title} - {episode} at {timestamp} ({similarity})"
        logging.debug(f"Trace result: {result_text}")

        self.status_label.setText("Done!")
        self.results_layout.addStretch(1)

    def create_match_widget(self, match: MatchInfo, index: int) -> QWidget:
        match_widget = QWidget()
        match_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        match_widget.setMaximumHeight(154)
        match_widget.setContentsMargins(0, 0, 0, 0)
        match_widget.setStyleSheet(
            "QWidget {background-color: transparent; border: 0.5px solid #333; margin: 0px; padding: 0px;}"
        )

        match_layout = HBoxLayout(match_widget, [])
        match_layout.setSpacing(12)
        match_layout.setContentsMargins(0, 0, 0, 0)
        match_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        image_container = QWidget()
        image_container.setStyleSheet(
            "QWidget {background-color: #0f0f0f; min-height: 144px; max-height: 144px; margin: 0px;}"
        )
        image_container.setMinimumHeight(144)
        image_container.setMaximumHeight(144)
        image_container.setMinimumWidth(180)
        image_container.setContentsMargins(0, 0, 0, 0)

        image_container_layout = QVBoxLayout(image_container)
        image_container_layout.setContentsMargins(0, 0, 0, 0)
        image_container_layout.setSpacing(0)

        image_label = QLabel("Loading...")
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setScaledContents(True)
        image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        image_label.setStyleSheet("QLabel {background-color: transparent; margin: 0px; border: 0px solid #444;}")

        image_container_layout.addWidget(image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        image_container_layout.addStretch()

        match_layout.addWidget(image_container)
        setattr(match_widget, "image_label", image_label)

        info_widget = QWidget()
        info_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        info_widget.setContentsMargins(0, 8, 0, 0)
        info_widget.setStyleSheet("QWidget { border: 0; }")
        info_widget.setMaximumHeight(154)

        info_layout = VBoxLayout(info_widget, [])
        info_layout.setSpacing(4)
        info_layout.setContentsMargins(0, 0, 0, 0)

        native_title = match.anilist.title_native or "Unknown Title"
        romaji_title = match.anilist.title_romaji or ""
        english_title = match.anilist.title_english or ""
        n_l, r_l, e_l = native_title.lower(), romaji_title.lower(), english_title.lower()

        title_color = "#4a9eff" if index == 0 else "#ffffff"

        subtitles = (
            [
                " | ".join(
                    [
                        s
                        for cond, s in [
                            (romaji_title and r_l != n_l, romaji_title),
                            (english_title and e_l != n_l and e_l != r_l, english_title),
                        ]
                        if cond
                    ]
                )
            ]
            if not (n_l == r_l == e_l and n_l)
            else []
        )
        subtitle_text = subtitles[0] if subtitles and subtitles[0] else ""

        title_label = QLabel(native_title)
        title_label.setContentsMargins(0, 0, 0, 0)
        title_label.setStyleSheet(
            "QLabel {{color: "
            + f"{title_color}"
            + "; font-weight: bold; font-size: 24px; margin: 0px; padding: 0px; border: none;}}"
        )
        title_label.setWordWrap(True)
        info_layout.addWidget(title_label)

        if subtitle_text:
            subtitle_label = QLabel(subtitle_text)
            subtitle_label.setContentsMargins(0, 0, 0, 0)
            subtitle_label.setStyleSheet(
                "QLabel {color: #cccccc; font-size: 12px; margin: 0px; padding: 0px; border: none;}"
            )
            subtitle_label.setWordWrap(True)
            info_layout.addWidget(subtitle_label)

        info_layout.addSpacing(8)

        episode_num = str(match.episode) if match.episode else ""
        timestamp = match.format_timestamp()
        episode_ts_html = (
            f'<span style="color: #ff6b6b; font-weight: bold;">Episode {episode_num}</span>: ' if episode_num else ""
        ) + f'<span style="color: #51cf66; font-weight: bold;">{timestamp}</span>'

        composite_label = QLabel()
        composite_label.setContentsMargins(0, 0, 0, 0)
        composite_label.setTextFormat(Qt.TextFormat.RichText)
        composite_label.setText(episode_ts_html)
        composite_label.setStyleSheet("QLabel {font-size: 16px; margin: 0px; padding: 0px; border: none;}")
        composite_label.setWordWrap(False)
        info_layout.addWidget(composite_label)

        similarity_label = QLabel(f"~{match.similarity_as_percentage} Similarity")
        similarity_label.setContentsMargins(0, 0, 0, 0)
        similarity_label.setStyleSheet(
            "QLabel {color: #ccc; font-size: 13px; margin: 0px; padding: 0px; border: none;}"
        )
        similarity_label.setWordWrap(False)
        info_layout.addWidget(similarity_label)

        match_layout.addWidget(info_widget, alignment=Qt.AlignmentFlag.AlignTop)
        self.results_layout.addWidget(match_widget)

        return match_widget

    def load_thumbnail(self, match: MatchInfo, match_widget: QWidget) -> None:
        try:
            thumbnail_path = match.download_image()
            label = match_widget.image_label

            if thumbnail_path.exists():
                pixmap = QPixmap(str(thumbnail_path))
                if pixmap.isNull():
                    return label.setText("Failed to load")

                match_widget.original_pixmap = pixmap

                if (size := label.size()) and size.width() and size.height():
                    label.setPixmap(
                        pixmap.scaled(
                            size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                        )
                    )
                else:
                    label.setPixmap(pixmap)
            else:
                label.setText("No image")
        except Exception as e:
            print(f"Failed to load thumbnail: {e}")
            match_widget.image_label.setText("Error loading")

    def resizeEvent(self, a0: QResizeEvent | None) -> None:  # noqa: N802
        super().resizeEvent(a0)

        for _, w in getattr(self, "match_widgets", []):
            label = getattr(w, "image_label", None)
            pixmap = getattr(w, "original_pixmap", None)

            if label is None or pixmap is None or pixmap.isNull():
                continue

            size = label.size()
            if size.width() <= 0 or size.height() <= 0:
                continue

            label.setPixmap(
                pixmap.scaled(
                    size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
