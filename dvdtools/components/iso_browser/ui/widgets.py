from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QLabel, QSpinBox
from vspreview.core.abstracts import PushButton

__all__ = [
    'create_widgets',
]


def create_widgets(parent) -> None:
    """Create and initialize all widgets."""

    parent.file_label = QLabel('No DVD loaded')

    parent.load_button = PushButton('Load ISO/IFO', parent, clicked=parent._on_load_iso)
    parent.load_button.setFixedWidth(150)
    parent.load_button.setToolTip('Load a DVD ISO or IFO file')
    parent.load_button.acceptDrops()

    parent.dump_title_button = PushButton('Dump Title', parent, clicked=parent.ffmpeg_handler.dump_title)
    parent.dump_title_button.setFixedWidth(150)
    parent.dump_title_button.setEnabled(False)
    parent.dump_title_button.setToolTip('Extract the selected title and angle to a file')

    parent.dump_all_titles_button = PushButton('Dump Entire Disc', parent, clicked=parent.ffmpeg_handler.dump_all_titles)
    parent.dump_all_titles_button.setFixedWidth(150)
    parent.dump_all_titles_button.setEnabled(False)
    parent.dump_all_titles_button.setToolTip('Extract all titles from the DVD to separate files')

    parent.copy_script_button = PushButton(unicode_icons['clipboard'], parent, clicked=parent._on_copy_script)
    parent.copy_script_button.setFixedWidth(20)
    parent.copy_script_button.setEnabled(False)
    parent.copy_script_button.setToolTip('Copy an IsoFile code snippet to clipboard')

    parent.info_button = PushButton(unicode_icons['info'], parent)
    parent.info_button.setFixedWidth(20)
    parent.info_button.setToolTip('Click to learn more about remuxing DVDs (opens in browser)')
    parent.info_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(
        'https://jaded-encoding-thaumaturgy.github.io/JET-guide/master/sources/dvd/remuxing/'
    )))

    parent.info_label = QLabel('Select a title to view details')

    # Add chapter trimming widgets
    parent.chapter_label = QLabel('Trim Chapters')
    parent.chapter_start_spin = QSpinBox()
    parent.chapter_to_label = QLabel('to')
    parent.chapter_end_spin = QSpinBox()
    parent.chapter_dump_label = QLabel('when dumping this title (inclusive)')

    # Configure spinboxes
    parent.chapter_start_spin.setMinimum(1)
    parent.chapter_end_spin.setMinimum(1)
    parent.chapter_start_spin.setValue(1)

    for widget in (
        parent.chapter_label,
        parent.chapter_start_spin,
        parent.chapter_end_spin,
        parent.chapter_to_label,
        parent.chapter_dump_label
    ):
        widget.setEnabled(False)


unicode_icons = {
    'clipboard': '⎘',
    'info': '🛈'
}
