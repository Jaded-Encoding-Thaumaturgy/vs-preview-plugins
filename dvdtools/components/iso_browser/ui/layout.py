from typing import Any
from PyQt6.QtWidgets import QHBoxLayout, QWidget

__all__ = [
    "setup_layout",
]


def setup_layout(parent: Any) -> None:
    """Set up widget layout."""

    file_widget = QWidget()
    file_layout = QHBoxLayout(file_widget)

    file_layout.addWidget(parent.file_label)
    file_layout.addWidget(parent.load_button)
    file_layout.addWidget(parent.dump_title_button)
    file_layout.addWidget(parent.dump_all_titles_button)
    file_layout.addWidget(parent.copy_script_button)
    file_layout.addWidget(parent.info_button)

    # Create chapter controls widget
    chapter_widget = QWidget()
    chapter_widget.setVisible(False)  # Start hidden
    chapter_layout = QHBoxLayout(chapter_widget)
    chapter_layout.addWidget(parent.chapter_label)
    chapter_layout.addWidget(parent.chapter_start_spin)
    chapter_layout.addWidget(parent.chapter_to_label)
    chapter_layout.addWidget(parent.chapter_end_spin)
    chapter_layout.addWidget(parent.chapter_dump_label)
    chapter_layout.addStretch()

    # Store reference to chapter widget for visibility control
    parent.chapter_widget = chapter_widget

    parent.vlayout.addWidget(file_widget)
    parent.vlayout.addWidget(parent.tree_manager.tree)
    parent.vlayout.addWidget(parent.info_label)
    parent.vlayout.addSpacing(10)
    parent.vlayout.addWidget(parent.tree_manager.chapters_tree)
    parent.vlayout.addWidget(chapter_widget)
