from logging import debug, error
from traceback import format_exc

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem
from vspreview import set_timecodes
from vspreview.core import Frame
from vssource import Title

__all__ = [
    'ISOTreeManager',
]


class ISOTreeManager:
    """Manages the tree widget for ISO browsing."""

    def __init__(self, parent) -> None:
        self.parent = parent
        self.tree = QTreeWidget()
        self.chapters_tree = QTreeWidget()
        self._setup_tree()
        self._setup_chapters_tree()

    def _setup_tree(self) -> None:
        """Configure tree widget."""

        self.tree.setHeaderLabels(['Titles'])
        self.tree.itemClicked.connect(self._on_tree_item_selected)

    def _setup_chapters_tree(self) -> None:
        """Configure chapters tree widget."""

        self.chapters_tree.setHeaderLabels(['Chapters'])
        self.chapters_tree.itemClicked.connect(self._on_chapter_selected)
        self.chapters_tree.setVisible(False)
        self.chapters_tree.setMinimumHeight(300)
        self.chapters_tree.setContentsMargins(0, 50, 0, 50)

    def populate_tree(self) -> None:
        """Populate the tree widget with titles and angles."""

        self.tree.clear()
        self.chapters_tree.clear()
        self.chapters_tree.setVisible(False)
        self.parent.title_info.clear()

        if not self.parent.iso_file:
            debug(debug_mapping['no_dvd'])
            return

        try:
            self._add_titles_to_tree()
            self.tree.expandAll()

            if not self.parent.iso_path.suffix.lower() == '.ifo':
                self.parent.dump_all_titles_button.setEnabled(self.parent.iso_file.title_count > 0)
        except Exception as e:
            error(error_mapping['populate_tree_failed'].format(e, format_exc()))
            self.parent._reset_iso_state()
            raise

    def _add_titles_to_tree(self) -> None:
        """Add all titles to the tree."""

        debug(debug_mapping['populating_tree'].format(self.parent.iso_file.title_count))

        for title_idx in range(1, self.parent.iso_file.title_count + 1):
            debug(debug_mapping['adding_title'].format(title_idx))

            try:
                self._add_title_to_tree(title_idx)
            except Exception as e:
                error(error_mapping['add_title_failed'].format(title_idx, e, format_exc()))
                continue

    def _add_title_to_tree(self, title_idx: int) -> None:
        """Add a title and its details to the tree widget."""

        tt_srpt = self.parent.iso_file.ifo0.tt_srpt[title_idx - 1]
        angle_count = tt_srpt.nr_of_angles

        debug(debug_mapping['title_angles'].format(title_idx, angle_count))

        if angle_count == 1:
            # Single angle title
            if self._load_title(title_idx, None) is None:
                error(error_mapping['load_base_title_failed'].format(title_idx))
                return None

            debug(debug_mapping['getting_title_info'].format(title_idx))
            title_info = self.parent.title_info.get((title_idx, None))
            if title_info is None:
                error(error_mapping['title_info_not_found'].format(title_idx))
                return None
            debug(debug_mapping['title_info_found'].format(title_info is not None))

            duration_str = self._format_duration(title_info['duration']) if title_info else ''
            debug(debug_mapping['formatted_duration'].format(duration_str))

            debug(debug_mapping['creating_tree_item'].format(title_idx))
            title_item = QTreeWidgetItem(self.tree, [f'Title {title_idx} ({duration_str})'])
            title_item.setData(0, Qt.ItemDataRole.UserRole, {'title': title_idx, 'angle': None})
            debug(debug_mapping['tree_item_created'].format(title_idx, duration_str))
            return

        # Multi-angle title
        title_item = QTreeWidgetItem(self.tree, [f'Title {title_idx} (Multi-Angle)'])
        title_item.setFlags(title_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)

        # Add all angles as sub-items
        for angle in range(1, angle_count + 1):
            debug(debug_mapping['loading_angle'].format(angle, title_idx))
            if self._load_title(title_idx, angle) is None:
                continue

            angle_info = self.parent.title_info.get((title_idx, angle))
            angle_duration = self._format_duration(angle_info['duration']) if angle_info else ''
            angle_item = QTreeWidgetItem(title_item, [f'Angle {angle} ({angle_duration})'])
            angle_item.setData(0, Qt.ItemDataRole.UserRole, {'title': title_idx, 'angle': angle})

    def _load_title(self, title_idx: int, angle: int | None) -> Title | None:
        """Load title and store its info."""

        try:
            debug(debug_mapping['attempting_title'].format(title_idx, angle))

            if not (title := self.parent.iso_file.get_title(title_idx, angle)):
                debug(debug_mapping['title_not_found'].format(title_idx, angle))
                return None

            debug(debug_mapping['getting_video_stream'].format(title_idx, angle))
            video = title.video

            if not video:
                debug(debug_mapping['no_video_stream'].format(title_idx, angle))
                return None

            debug(debug_mapping['getting_title_info_tt_srpt'].format(title_idx))
            tt_srpt = self.parent.iso_file.ifo0.tt_srpt[title_idx - 1]

            # Get audio tracks
            debug(debug_mapping['getting_audio_tracks'].format(title_idx, angle))
            audio_tracks = self._get_audio_tracks(title, title_idx, angle)

            debug(debug_mapping['creating_title_info'].format(title_idx, angle))

            title_info = {
                'title_idx': title_idx,
                'angle': angle,
                'chapter_count': len(title.chapters),
                'chapters': title.chapters,
                'audio_tracks': audio_tracks,
                'angle_count': tt_srpt.nr_of_angles,
                'width': video.width,
                'height': video.height,
                'fps': float(video.fps),
                'duration': float(video.num_frames / video.fps),
                'frame_count': video.num_frames,
                'vts': title._vts
            }

            debug(debug_mapping['storing_title_info'].format(title_idx, angle))
            self.parent.title_info[(title_idx, angle)] = title_info

            debug(debug_mapping['title_loaded'].format(title_idx, angle))

            self.parent.chapter_start_spin.setValue(1)
            self.parent.chapter_end_spin.setValue(title_info['chapter_count'])

            return title

        except Exception as e:
            error(error_mapping['load_title_failed'].format(title_idx, angle, e, format_exc()))
            return None

    def _get_audio_tracks(self, title: Title, title_idx: int, angle: int | None) -> list[str]:
        """Get audio tracks safely."""

        audio_tracks = []

        try:
            debug(debug_mapping['getting_audio_tracks'].format(title_idx, angle))

            try:
                debug(debug_mapping['getting_audio_track_info'])

                for track in title._audios:
                    if track and track.lower() != 'none':
                        audio_tracks.append(track)

            except Exception:
                debug(debug_mapping['audio_track_fallback'])
                audio_tracks.append("Unknown Codec")

        except Exception as e:
            debug(debug_mapping['audio_tracks_failed'].format(title_idx, angle, e))

        debug(debug_mapping['audio_tracks_found'].format(len(audio_tracks)))

        return audio_tracks

    def _on_tree_item_selected(self, item: QTreeWidgetItem) -> None:
        """Handle tree item selection."""

        if not item:
            self.chapters_tree.setVisible(False)
            return

        data = item.data(0, Qt.ItemDataRole.UserRole)

        if not data:
            self.chapters_tree.setVisible(False)
            return

        title_idx = data['title']
        angle = data['angle']

        try:
            debug(debug_mapping['loading_title'].format(title_idx, angle))
            self._load_selected_title(title_idx, angle)
            self.parent.copy_script_button.setEnabled(True)
        except Exception as e:
            error(error_mapping['load_title_failed'].format(title_idx, angle, e, format_exc()))
            raise

    def _load_selected_title(self, title_idx: int, angle: int | None) -> None:
        """Load and display the selected title."""

        if not (info := self.parent.title_info.get((title_idx, angle))):
            debug(debug_mapping['no_title_info'].format(title_idx, angle))
            return

        self.parent.current_title = self.parent.iso_file.get_title(title_idx, angle)
        self.parent.current_node = self.parent.current_title.video

        # Update chapter spinboxes
        chapter_count = info['chapter_count']
        has_chapters = chapter_count > 0

        self.parent.chapter_label.setEnabled(has_chapters)
        self.parent.chapter_start_spin.setEnabled(has_chapters)
        self.parent.chapter_end_spin.setEnabled(has_chapters)
        self.parent.chapter_to_label.setEnabled(has_chapters)
        self.parent.chapter_dump_label.setEnabled(has_chapters)

        if has_chapters:
            self.parent.chapter_start_spin.setMaximum(chapter_count)
            self.parent.chapter_end_spin.setMaximum(chapter_count)
            self.parent.chapter_end_spin.setValue(chapter_count)

        # Update FFmpeg handler chapter values
        self.parent.ffmpeg_handler.chapter_start = self.parent.chapter_start_spin.value() if has_chapters else None
        self.parent.ffmpeg_handler.chapter_end = self.parent.chapter_end_spin.value() if has_chapters else None
        self.parent.dump_title_button.setEnabled(bool(info['audio_tracks']))

        self._update_info_label(info)
        self._update_outputs(title_idx, angle)
        self._populate_chapters_tree(info)

    def _update_outputs(self, title_idx: int, angle: int | None) -> None:
        """Update the outputs with the new video and audio nodes and load chapters as scenechanges."""

        main = self.parent.plugin.main
        current_output_idx = main.current_output.index

        video_output = main.outputs[current_output_idx].with_node(self.parent.current_node)
        video_output.name = f"Title {title_idx}" + (f" Angle {angle}" if angle is not None else "") + " (Video)"

        # TODO: Add audio outputs

        if (info := self.parent.title_info.get((title_idx, angle))) and info['chapters']:
            debug(debug_mapping['loading_chapters'].format(len(info["chapters"])))

            try:
                chapter_frames = [
                    int(frame) for frame in info['chapters']
                    if frame > 0
                ]

                set_timecodes(video_output, chapter_frames)
                debug(debug_mapping['added_chapter_frames'].format(len(chapter_frames)))
            except Exception as e:
                error(error_mapping['load_chapters_failed'].format(e, format_exc()))

        main.outputs.items.clear()
        main.outputs.items.append(video_output)
        video_output.index = -1

        main.refresh_video_outputs()
        main.switch_output(0)

    def _update_info_label(self, info: dict) -> None:
        """Update info label with title details."""

        info_text = [
            f'Angle: {info['angle'] if info['angle'] is not None else 1}/{info['angle_count']}',
            f'Duration: {self._format_duration(info['duration'])}',
            f'Resolution: {info['width']}x{info['height']}',
            f'Chapters: {info['chapter_count']}',
            f'Frame Count: {info['frame_count']}',
            f'VTS: {info['vts']}'
        ]

        if info.get('audio_tracks', None):
            info_text += ['Audio Track(s):']

            for i, track in enumerate(info['audio_tracks'], 1):
                info_text += [f'  {i}. {track}']
        else:
            info_text += ['Audio Track(s): None']

        self.parent.info_label.setText('\n'.join(info_text))

    def _format_duration(self, duration_secs: float) -> str:
        """Format duration in seconds to HH:MM:SS format."""

        hours = int(duration_secs // 3600)
        minutes = int((duration_secs % 3600) // 60)
        seconds = int(duration_secs % 60)

        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

    def _populate_chapters_tree(self, info: dict) -> None:
        """Populate chapters tree with chapter information."""

        self.chapters_tree.clear()
        chapters = info.get('chapters', [])

        # Update chapter spinbox visibility and values based on chapter availability
        has_chapters = bool(chapters)

        # Show/hide chapter controls
        self.chapters_tree.setVisible(has_chapters)
        self.parent.chapter_widget.setVisible(has_chapters)

        if not has_chapters:
            return

        # Set spinbox ranges
        chapter_count = len(chapters)
        self.parent.chapter_start_spin.setMaximum(chapter_count)
        self.parent.chapter_end_spin.setMaximum(chapter_count)
        self.parent.chapter_end_spin.setValue(chapter_count)

        # Connect value changed signals to ensure valid ranges
        self.parent.chapter_start_spin.valueChanged.connect(self._on_chapter_start_changed)
        self.parent.chapter_end_spin.valueChanged.connect(self._on_chapter_end_changed)

        # Update FFmpeg handler chapter values
        self.parent.ffmpeg_handler.chapter_start = self.parent.chapter_start_spin.value()
        self.parent.ffmpeg_handler.chapter_end = self.parent.chapter_end_spin.value()

        # Populate chapter tree
        fps = info['fps']
        for i, frame in enumerate(chapters, 1):
            timestamp = self._format_duration(frame / fps)
            chapter_item = QTreeWidgetItem(self.chapters_tree, [f'Chapter {i} - Frame {frame} - {timestamp}'])
            chapter_item.setData(0, Qt.ItemDataRole.UserRole, {'frame': frame})

    def _on_chapter_start_changed(self, value: int) -> None:
        """Ensure start chapter doesn't exceed end chapter."""

        if value > self.parent.chapter_end_spin.value():
            self.parent.chapter_end_spin.setValue(value)

        # Store chapter values in title info
        title_idx = self.parent.current_title._title + 1
        angle = getattr(self.parent.current_title, 'angle', None)
        title_key = (title_idx, angle)

        debug(debug_mapping['updating_chapter_start'].format(title_key, value))
        if title_key in self.parent.title_info:
            debug(debug_mapping['current_title_info'].format(self.parent.title_info[title_key]))
            self.parent.title_info[title_key]['chapter_start'] = value
            self.parent.ffmpeg_handler.chapter_start = value
            debug(debug_mapping['updated_title_info'].format(self.parent.title_info[title_key]))

    def _on_chapter_end_changed(self, value: int) -> None:
        """Ensure end chapter isn't less than start chapter."""

        if value < self.parent.chapter_start_spin.value():
            self.parent.chapter_start_spin.setValue(value)

        # Store chapter values in title info
        title_idx = self.parent.current_title._title + 1
        angle = getattr(self.parent.current_title, 'angle', None)
        title_key = (title_idx, angle)

        debug(debug_mapping['updating_chapter_end'].format(title_key, value))

        if title_key in self.parent.title_info:
            debug(debug_mapping['current_title_info'].format(self.parent.title_info[title_key]))

            self.parent.title_info[title_key]['chapter_end'] = value
            self.parent.ffmpeg_handler.chapter_end = value

            debug(debug_mapping['updated_title_info'].format(self.parent.title_info[title_key]))

    def _on_chapter_selected(self, item: QTreeWidgetItem) -> None:
        """Handle chapter selection by jumping to the chapter frame."""

        if not item:
            return

        if not (data := item.data(0, Qt.ItemDataRole.UserRole)) or 'frame' not in data:
            return

        debug(debug_mapping['switching_frame'].format(data['frame']))
        self.parent.plugin.main.switch_frame(Frame(data['frame']))

    def clear(self) -> None:
        """Clear the tree widgets."""

        self.tree.clear()
        self.chapters_tree.clear()
        self.chapters_tree.setVisible(False)
        self.parent.chapter_widget.setVisible(False)  # Hide chapter controls when clearing


error_mapping: dict[str, str] = {
    'chapter_start_exceeds_end': 'Start chapter exceeds end chapter.',
    'end_chapter_less_than_start': 'End chapter is less than start chapter.',
    'populate_tree_failed': 'Failed to populate tree',
    'add_title_failed': 'Failed to add title {}',
    'load_base_title_failed': 'Failed to load base title {}',
    'title_info_not_found': 'Title info not found for title {}',
    'load_title_failed': 'Failed to load title {} angle {}',
    'load_chapters_failed': 'Failed to load chapters as scenechanges'
}


debug_mapping: dict[str, str] = {
    'no_dvd': 'No DVD loaded',
    'populating_tree': 'Populating tree with {} titles',
    'adding_title': 'Adding title {}',
    'title_angles': 'Title {} has {} angle(s)',
    'getting_title_info': 'Getting title info for title {} (no angle)',
    'title_info_found': 'Title info found: {}',
    'formatted_duration': 'Formatted duration: {}',
    'creating_tree_item': 'Creating tree item for title {}',
    'tree_item_created': 'Tree item created with label: Title {} ({})',
    'loading_angle': 'Loading angle {} for title {}',
    'attempting_title': 'Attempting to get title {} angle {}',
    'title_not_found': 'Title {} angle {} not found',
    'getting_video_stream': 'Getting video stream for title {} angle {}',
    'no_video_stream': 'No video stream found for title {} angle {}',
    'getting_title_info_tt_srpt': 'Getting title info from tt_srpt for title {}',
    'getting_audio_tracks': 'Getting audio tracks for title {} angle {}',
    'creating_title_info': 'Creating title info dict for title {} angle {}',
    'storing_title_info': 'Storing title info for title {} angle {}',
    'title_loaded': 'Successfully loaded title {} angle {}',
    'getting_audio_track_info': 'Getting info for audio track',
    'audio_track_fallback': 'Failed to get info for audio track, using fallback name',
    'audio_tracks_failed': 'Failed to get audio tracks for title {} angle {}: {}',
    'audio_tracks_found': 'Found {} audio tracks',
    'loading_title': 'Loading title {} (angle={})',
    'no_title_info': 'No info found for title {} angle {}',
    'loading_chapters': 'Loading {} chapters as scenechanges',
    'added_chapter_frames': 'Added {} chapter frames as scenechanges',
    'updating_chapter_start': 'Updating chapter start: title_key={}, value={}',
    'current_title_info': 'Current title_info before update: {}',
    'updated_title_info': 'Updated title_info: {}',
    'updating_chapter_end': 'Updating chapter end: title_key={}, value={}',
    'switching_frame': 'Switching to chapter frame {}',
}
