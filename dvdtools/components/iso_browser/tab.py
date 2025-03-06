from __future__ import annotations

from logging import debug, error
from traceback import format_exc
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QFileDialog, QMessageBox,
                             QProgressDialog)
from vspreview.core import Frame
from vspreview.core.abstracts import AbstractSettingsWidget
from vspreview.plugins import AbstractPlugin
from vssource import IsoFile, Title
from vstools import SPath, vs

from .ffmpeg_handler import FFmpegHandler
from .tree_manager import ISOTreeManager
from .types import TitleInfo
from .ui import create_widgets, setup_layout

__all__ = [
    'IsoBrowserTab',
]


class IsoBrowserTab(AbstractSettingsWidget):
    """Tab for browsing DVD/BD ISO files and their titles/angles."""

    __slots__ = (
        'file_label', 'load_button', 'info_label', 'dump_title_button', 'tree_manager', 'ffmpeg_handler',
        'copy_script_button', 'chapter_start_spin', 'chapter_end_spin'
    )

    def __init__(self, plugin: AbstractPlugin) -> None:
        self.plugin = plugin

        self.tree_manager = ISOTreeManager(self)
        self.ffmpeg_handler = FFmpegHandler(self)

        super().__init__()
        self._init_state()

    def _init_state(self) -> None:
        """Initialize internal state variables."""

        self.iso_path: SPath | None = None
        self.iso_file: IsoFile | None = None  # type:ignore

        self.title_info: dict[tuple[int, int | None], TitleInfo] = {}

        self.current_node: vs.VideoNode | None = None
        self.current_title: Title | None = None
        self._last_source_path = SPath('.')

    def setup_ui(self) -> None:
        """Set up the user interface."""

        super().setup_ui()
        create_widgets(self)
        setup_layout(self)

        # Connect chapter spinbox signals
        self.chapter_start_spin.valueChanged.connect(self._on_chapter_range_changed)
        self.chapter_end_spin.valueChanged.connect(self._on_chapter_range_changed)

    def _on_load_iso(self) -> None:
        """Handle ISO file loading."""

        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Select ISO/IFO file', self.last_source_path.to_str(),
            'DVD files (*.iso *.ifo);;All files (*.*)'
        )

        if not file_path:
            debug(debug_mapping['no_file_selected'])
            return

        self._last_source_path = SPath(file_path)

        try:
            debug(debug_mapping['loading_iso'].format(file_path))
            self.iso_path = SPath(file_path)

            # Create progress dialog
            suffix = self.iso_path.suffix.lower()
            dialog_texts = dialog_text_map.get(suffix, dialog_text_map['.iso'])

            progress = QProgressDialog(dialog_texts['loading'], 'Cancel', 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setWindowTitle(dialog_texts['loading'])
            progress.setValue(0)
            progress.show()

            QApplication.processEvents()
            progress.setLabelText(dialog_texts['opening'])

            # For IFO files, use the DVD root directory (2 levels up)
            # For ISO files, use the ISO file path directly
            self.iso_file = IsoFile(self.iso_path.parent.parent if suffix == '.ifo' else self.iso_path)

            if progress.wasCanceled():
                raise Exception(error_mapping['operation_cancelled'])

            progress.setLabelText(dialog_texts['loading'])
            self.file_label.setText(self.iso_path.name)
            self.info_label.setText(debug_mapping['select_title'])

            if progress.wasCanceled():
                raise Exception(error_mapping['operation_cancelled'])

            total_titles = self.iso_file.title_count
            total_items = 0

            for title_idx in range(1, total_titles + 1):
                tt_srpt = self.iso_file.ifo0.tt_srpt[title_idx - 1]
                total_items += max(1, tt_srpt.nr_of_angles)

            items_processed = 0
            progress_per_item = 100 / total_items

            # Clear existing data
            self.tree_manager.clear()
            self.title_info.clear()

            # Process each title
            for title_idx in range(1, total_titles + 1):
                if progress.wasCanceled():
                    raise Exception(error_mapping['operation_cancelled'])

                progress.setLabelText(debug_mapping['loading_title'].format(title_idx, total_titles))

                try:
                    self.tree_manager._add_title_to_tree(title_idx)
                except Exception as e:
                    error(error_mapping['title_add_failed'].format(title_idx, e, format_exc()))
                    continue

                # Account for angles in progress calculation
                tt_srpt = self.iso_file.ifo0.tt_srpt[title_idx - 1]
                angle_count = max(1, tt_srpt.nr_of_angles)

                for angle in range(angle_count):
                    items_processed += 1

                    if angle_count > 1:
                        progress.setLabelText(debug_mapping['loading_title_angle'].\
                            format(title_idx, total_titles, angle + 1, angle_count))
                    else:
                        progress.setLabelText(debug_mapping['loading_title'].format(title_idx, total_titles))

                    progress.setValue(min(int(items_processed * progress_per_item), 100))
                    QApplication.processEvents()

            # Finish up
            self.tree_manager.tree.expandAll()

            if not self.iso_path.suffix.lower() == '.ifo':
                self.dump_all_titles_button.setEnabled(total_titles > 0)

            progress.setValue(100)

        except Exception as e:
            error(error_mapping['load_failed'].format(dialog_texts['error'], e, format_exc()))
            self._reset_iso_state()
            QMessageBox.critical(self, 'Error', error_mapping['load_failed_dialog'].format(dialog_texts['error'], str(e)))
        finally:
            if 'progress' in locals():
                progress.close()

    def _on_copy_script(self) -> None:
        """Copy an IsoFile script snippet to clipboard."""

        if not self.iso_path:
            return

        selected_item = self.tree_manager.tree.currentItem()

        if not selected_item or not (data := selected_item.data(0, Qt.ItemDataRole.UserRole)):
            return

        title_num = data['title']
        angle = data.get('angle')

        debug(debug_mapping['copying_snippet'].format(title_num, angle))

        script = self._generate_script(title_num, angle)

        QApplication.clipboard().setText(script)
        self.plugin.main.show_message(debug_mapping['snippet_copied'])

    def _generate_script(self, title_num: int, angle: int | None) -> str:
        """Generate a VapourSynth script for the given title and angle, as well as other relevant methods."""

        iso_path = self.iso_path.parent.parent if self.iso_path.suffix == '.ifo' else self.iso_path
        iso_path = iso_path.as_posix().replace('"', '\\"').replace("'", "\\'")

        # Build core code snippet
        script = [
            'from vssource import IsoFile', '',
            f'iso = IsoFile(\"{iso_path}\")',
        ]

        title_args = [str(title_num)]

        if angle is not None:
            title_args += [f'angle={angle}']

        script += [f'title = iso.get_title({", ".join(title_args)})']

        # Add chapter splitting if needed
        start_chapter = self.chapter_start_spin.value()
        end_chapter = self.chapter_end_spin.value()

        title_info = self.title_info.get((title_num, angle), {})
        chapter_count = title_info.get('chapter_count', end_chapter)

        if start_chapter > 1 or end_chapter < chapter_count:
            script += [f'title = title.split_range({start_chapter}, {end_chapter})']

        # Assign video node
        script += ['', 'src = title.video']

        return '\n'.join(script)

    def _reset_iso_state(self) -> None:
        """Reset the ISO state and UI elements."""

        debug(debug_mapping['resetting_state'])

        self.file_label.setText(debug_mapping['no_dvd_loaded'])
        self._init_state()
        self.tree_manager.clear()
        self.dump_title_button.setEnabled(False)
        self.dump_all_titles_button.setEnabled(False)
        self.copy_script_button.setEnabled(False)
        self.info_label.setText(debug_mapping['select_title'])

    def _check_current_title(self) -> bool:
        """Check if current title exists and is valid."""

        if not hasattr(self, 'current_title'):
            return False

        return self.current_title is not None

    def on_current_frame_changed(self, frame: Frame) -> None:
        """Handle frame change events."""

        if not self._check_current_title():
            return

        try:
            self.current_title.frame = frame.value
        except AttributeError as e:
            debug(debug_mapping['frame_update_failed'].format(e, format_exc()))
            pass

    def on_current_output_changed(self, index: int, prev_index: int) -> None:
        """Handle output change events."""

        if not self._check_current_title():
            return

        try:
            current_frame = self.plugin.main.current_frame
            if current_frame is None:
                return

            if self.plugin.main.current_output.node is not self.current_title.video:
                main = self.plugin.main
                new_output = main.outputs[index].with_node(self.current_title.video)
                new_output.name = (
                    f'Title {self.current_title._title}'
                    + (f' Angle {self.current_title.angle}'
                       if hasattr(self.current_title, 'angle') else '')
                )
                new_output.index = index
                main.outputs.items[index] = new_output
                main.refresh_video_outputs()
                return

            self.current_title.frame = current_frame.value
        except AttributeError as e:
            debug(debug_mapping['frame_update_failed'].format(e, format_exc()))
            pass

    def _on_chapter_range_changed(self) -> None:
        """Handle chapter range spinbox value changes."""

        if not self._check_current_title():
            return

        self.ffmpeg_handler.chapter_start = self.chapter_start_spin.value()
        self.ffmpeg_handler.chapter_end = self.chapter_end_spin.value()

    def __getstate__(self) -> dict[str, Any]:
        """Get state for serialization."""

        return {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state from serialization."""

        self._reset_iso_state()

        if (iso_path := state.get('iso_path')) is None:
            return

        if not (iso_path := SPath(iso_path)).exists():
            debug(debug_mapping['iso_not_found'].format(iso_path))
            return

        try:
            debug(debug_mapping['loading_saved_state'].format(iso_path))

            file_ext = iso_path.suffix.lower()
            dialog_texts = dialog_text_map.get(file_ext, dialog_text_map['.iso'])

            progress_dialog = QProgressDialog(dialog_texts['opening'], None, 0, 0, self)
            progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.show()

            self.iso_path = iso_path
            self.iso_file = IsoFile(self.iso_path)

            progress_dialog.setLabelText(dialog_texts['loading'])
            self.file_label.setText(self.iso_path.name)
            self.tree_manager.populate_tree()

            progress_dialog.close()
        except Exception as e:
            error(error_mapping['saved_state_load_failed'].format(file_ext.upper(), e, format_exc()))
            self._reset_iso_state()
            QMessageBox.critical(self, 'Error', error_mapping['load_failed_dialog'].format(dialog_texts['error'], str(e)))

    @property
    def last_source_path(self) -> SPath:
        """Last source path used."""

        if not hasattr(self, '_last_source_path'):
            self._last_source_path = SPath('.')
        elif not self._last_source_path.exists():
            debug(debug_mapping['source_path_not_found'].format(self._last_source_path))
            self._last_source_path = SPath('.')

        return self._last_source_path

    @last_source_path.setter
    def last_source_path(self, path: SPath) -> None:
        """Set the last source path."""

        if path != self._last_source_path:
            self._last_source_path = path
            debug(debug_mapping['source_path_updated'].format(path))


dialog_text_map = {
    '.iso': {
        'opening': 'Opening ISO file...',
        'loading': 'Loading titles from ISO...',
        'error': 'Failed to load ISO file'
    },
    '.ifo': {
        'opening': 'Opening IFO file...',
        'loading': 'Loading titles from DVD structure...',
        'error': 'Failed to load IFO file'
    }
}


error_mapping: dict[str, str] = {
    'operation_cancelled': 'Operation cancelled by user',
    'title_add_failed': 'Failed to add title {}: {}\n{}',
    'load_failed': '{}: {}\n{}',
    'load_failed_dialog': '{}: {}',
    'saved_state_load_failed': 'Failed to load saved {} state: {}\n{}'
}


debug_mapping: dict[str, str] = {
    'no_file_selected': 'No file selected',
    'loading_iso': 'Loading ISO file: {}',
    'select_title': 'Select a title to view details',
    'loading_title': 'Loading title {}/{}...',
    'loading_title_angle': 'Loading title {}/{} angle {}/{}...',
    'copying_snippet': 'Copying code snippet for title {} (angle={}) to clipboard',
    'snippet_copied': 'IsoFile code snippet copied to clipboard!',
    'resetting_state': 'Resetting ISO state',
    'no_dvd_loaded': 'No DVD loaded',
    'frame_update_failed': 'Failed to update frame: {}\n{}',
    'iso_not_found': 'Previously saved ISO file no longer exists: {}',
    'loading_saved_state': 'Loading saved ISO state: {}',
    'last_source_path': 'Last source path: {}',
    'source_path_not_found': 'Last source path no longer exists: {}',
    'source_path_updated': 'Source path updated to: {}',
}
