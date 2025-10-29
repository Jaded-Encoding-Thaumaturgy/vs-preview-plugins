import subprocess
from logging import debug, error, getLogger, warning
from traceback import format_exc
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox, QProgressDialog
from vstools import SPath

__all__ = [
    "FFmpegHandler",
]


class FFmpegHandler:
    """Handles FFmpeg-related operations for ISO dumping."""

    def __init__(self, parent: Any) -> None:
        self.parent = parent
        self.log_level = getLogger().getEffectiveLevel()

        # Add default chapter values
        self.chapter_start = None
        self.chapter_end = None

        # Initialize last dump path
        self._last_dump_path = SPath(".")

    def dump_all_titles(self) -> None:
        """Dump all titles from the ISO."""

        if not self._check_ffmpeg():
            return

        # Get save directory
        save_dir = QFileDialog.getExistingDirectory(
            self.parent, "Select output directory", self.last_dump_path.to_str()
        )

        if not save_dir:
            return

        save_dir = SPath(save_dir)
        self.last_dump_path = save_dir

        # Get all unique titles and their info
        unique_titles = {}

        for i, ((title_idx, _), info) in enumerate(self.parent.title_info.items()):
            if title_idx not in {k[0] for k in list(self.parent.title_info.keys())[:i]}:
                unique_titles[title_idx] = info

        total_titles = len(unique_titles)
        debug(debug_mapping["unique_titles_found"].format(total_titles))

        # Create progress dialog
        progress = QProgressDialog("Dumping titles...", "Cancel", 0, total_titles, self.parent)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setWindowTitle("Dumping Titles")

        titles_processed = 0

        try:
            for title_idx, info in sorted(unique_titles.items()):
                if progress.wasCanceled():
                    break

                # Set full chapter range for this title
                chapter_count = info.get("chapter_count", 1)
                self.chapter_start = 1
                self.chapter_end = chapter_count

                angle_count = info.get("angle_count", 1)
                debug(debug_mapping["processing_title_angles"].format(title_idx, angle_count))

                try:
                    output_path = save_dir / self._get_suggested_filename(title_idx, info)

                    if angle_count == 1:
                        progress.setLabelText(f"Processing title {title_idx}/{total_titles}")
                        debug(debug_mapping["dumping_title"].format(title_idx, output_path))
                        self._dump_title(title_idx, output_path.to_str())
                        continue

                    # Process titles with multiple angles
                    for angle in range(1, angle_count + 1):
                        # Replace the last digits in the filename with the current angle
                        # TODO: Use regular substitution here, but that requires rewriting a bunch of other stuff.
                        output_path = output_path.with_name(output_path.name.rsplit("_", 1)[0] + f"_{angle:02d}.mkv")

                        progress.setLabelText(
                            f"Processing title {title_idx}/{total_titles} - Angle {angle}/{angle_count}"
                        )

                        debug(debug_mapping["dumping_title_angle"].format(title_idx, angle, output_path))

                        try:
                            self._dump_title(title_idx, output_path.to_str(), angle)
                        except RuntimeError as e:
                            warning(error_mapping["dump_title_angle_failed"].format(title_idx, angle, str(e)))
                            continue

                except RuntimeError as e:
                    warning(error_mapping["dump_title_failed"].format(title_idx, str(e)))
                    continue
                except Exception as e:
                    error(error_mapping["unexpected_dump_error"].format(title_idx, str(e), format_exc()))
                    continue

                titles_processed += 1
                progress.setValue(titles_processed)
                QApplication.processEvents()

        finally:
            progress.close()

    def dump_title(self) -> None:
        """Dump currently selected title."""

        debug(debug_mapping["dumping_title_start"])

        if not self._check_ffmpeg():
            return

        selected_item = self.parent.tree_manager.tree.currentItem()
        data = selected_item.data(0, Qt.ItemDataRole.UserRole)

        title_idx = data["title"]
        angle = data.get("angle")
        title_info = self.parent.title_info.get((title_idx, angle), {})

        debug(debug_mapping["dumping_title_angle_info"].format(title_idx, angle))

        # Get paths
        suggested_name = self._get_suggested_filename(title_idx, title_info)
        output_path = self._get_save_path(suggested_name)

        if not output_path:
            debug(debug_mapping["save_dialog_cancelled"])
            return

        self._dump_title(title_idx, output_path, angle)

    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available and supports DVD video."""

        debug(debug_mapping["checking_ffmpeg"])

        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-h", "demuxer=dvdvideo"], capture_output=True, text=True, check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            error(error_mapping["ffmpeg_check_failed"].format(e, format_exc()))
            QMessageBox.critical(self.parent, "Error", error_mapping["ffmpeg_not_found"])

            return False

        if "dvdvideo" not in result.stdout:
            error(error_mapping["ffmpeg_no_dvd_support"])
            QMessageBox.critical(self.parent, "Error", error_mapping["ffmpeg_no_dvd_support_msg"])

            return False

        return True

    def _get_suggested_filename(self, title_idx: int, title_info: dict[str, Any] | int) -> str:
        """Get suggested filename for title dump."""

        # Normalize title_info and angle
        if isinstance(title_info, int):
            angle = title_info
            title_info = self.parent.title_info.get((title_idx, angle)) or self.parent.title_info.get(
                (title_idx, None), {}
            )
            if angle is not None and title_info:
                title_info = {**title_info, "angle": angle}
        else:
            angle = title_info.get("angle", 1)  # Default to angle 1 if not specified

        # Build filename components
        base_name = self.parent.iso_path.stem
        title_str = f"title_{title_idx:02d}"

        angle_str = ""
        chapter_str = ""

        # Add angle if title has multiple angles
        has_multiple_angles = title_info.get("angle_count", 1) > 1

        if has_multiple_angles:
            angle_str = f"_angle_{angle:02d}"

        # Add chapter range if not using full range
        if hasattr(self, "chapter_start") and hasattr(self, "chapter_end"):
            chapter_count = title_info.get("chapter_count", 1)

            if not (self.chapter_start == 1 and self.chapter_end == chapter_count):
                chapter_str = (
                    f"_ch{self.chapter_start:02d}"
                    if self.chapter_start == self.chapter_end
                    else f"_ch{self.chapter_start:02d}-{self.chapter_end:02d}"
                )

        suggested_name = f"{base_name}_{title_str}{angle_str}{chapter_str}.mkv"
        debug(debug_mapping["suggested_filename"].format(suggested_name))

        return suggested_name

    def _get_save_path(self, filename: str = "") -> str | None:
        """Get the save path for the video/audio output."""

        debug(debug_mapping["getting_save_path"])

        title_idx = self.parent.current_title._title + 1
        angle = getattr(self.parent.current_title, "angle", 1)
        title_key = (title_idx, angle)

        if title_key not in self.parent.title_info:
            title_key = (title_idx, None)

            if title_key not in self.parent.title_info:
                error(
                    error_mapping["title_info_lookup_failed"].format(
                        title_key, list(self.parent.title_info.keys()), title_idx, angle
                    )
                )
                QMessageBox.critical(self.parent, "Error", error_mapping["title_info_not_found"])
                return

        suggested_name = filename or self._get_suggested_filename(title_idx, angle)

        output_path, _ = QFileDialog.getSaveFileName(
            self.parent,
            "Save Title",
            (self.last_dump_path / suggested_name).to_str(),
            "Matroska Video (*.mkv);;All files (*.*)",
        )

        if output_path:
            self.last_dump_path = SPath(output_path).get_folder()
            debug(debug_mapping["selected_output_path"].format(output_path))

        return output_path

    def _build_ffmpeg_command(
        self, file_path: str, output_path: str | SPath, angle: int | None, title_idx: int | None = None
    ) -> list[str]:
        """Build the ffmpeg command for video and all audio extraction."""

        if title_idx is None:
            title_idx = self.parent.current_title._title + 1

        # Try both with and without angle for title info lookup
        title_key = (title_idx, angle)
        title_info = self.parent.title_info.get(title_key, {})

        if not title_info:
            title_key = (title_idx, None)
            title_info = self.parent.title_info.get(title_key, {})

        debug(debug_mapping["building_ffmpeg_command"].format(title_key))
        debug(debug_mapping["title_info"].format(title_info))

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-f",
            "dvdvideo",
            "-preindex",
            "True",
        ]

        # Add chapter trimming parameters if available and needed
        if "chapters" in title_info:
            chapter_count = title_info.get("chapter_count", 0)

            # Only add chapter_start if it's not the first chapter
            if self.chapter_start is not None and self.chapter_start > 1:
                debug(debug_mapping["adding_chapter_start"].format(self.chapter_start))
                cmd.extend(["-chapter_start", str(self.chapter_start)])

            # Only add chapter_end if it's not the last chapter
            if self.chapter_end is not None and self.chapter_end < chapter_count:
                debug(debug_mapping["adding_chapter_end"].format(self.chapter_end))
                cmd.extend(["-chapter_end", str(self.chapter_end)])
        else:
            debug(debug_mapping["no_chapters"])

        cmd.extend(["-title", str(title_idx)])

        if angle is not None:
            cmd.extend(["-angle", str(angle)])

        cmd.extend(["-i", f'"{file_path}"'])

        cmd.extend(["-map", "0:v:0", "-c:v", "copy"])

        if title_key not in self.parent.title_info:
            title_key = (title_idx, None)
            if title_key not in self.parent.title_info:
                error(error_mapping["title_info_not_found_key"].format(title_key))
                return cmd

        title_info = self.parent.title_info[title_key]
        audio_tracks = title_info["audio_tracks"]

        for idx, audio_info in enumerate(audio_tracks):
            cmd.extend(["-map", f"0:a:{idx}"])

            if "pcm" in audio_info.lower():
                warning(debug_mapping["pcm_audio_detected"].format(idx))
                cmd.extend([f"-c:a:{idx}", "flac", "-compression_level", "8"])
            else:
                debug(debug_mapping["audio_detected"].format(audio_info, idx))
                cmd.extend([f"-c:a:{idx}", "copy"])

            if lang_info := title_info.get("audio_langs", [])[idx] if "audio_langs" in title_info else None:
                debug(debug_mapping["setting_language_metadata"].format(idx, lang_info))
                cmd.extend(
                    [
                        f"-metadata:s:a:{idx}",
                        f"language={lang_info}",
                        f"-metadata:s:a:{idx}",
                        f"title=Audio Track {idx + 1} ({lang_info.upper()})",
                    ]
                )

        # Properly quote the output path
        if isinstance(output_path, SPath):
            output_path = output_path.to_str()

        quoted_output = f'"{output_path}"'
        cmd.append(quoted_output)
        return cmd

    def _run_ffmpeg_process(self, cmd: list[str]) -> None:
        """Run FFmpeg process and handle output."""

        # Get input and output paths from command
        input_idx = cmd.index("-i") + 1
        input_path = cmd[input_idx].strip('"')
        output_path = cmd[-1].strip('"')

        # Delete output file if it exists (ffmpeg will hang if it's there, even with -y)
        if SPath(output_path).exists():
            debug(debug_mapping["deleting_output"].format(output_path))
            SPath(output_path).unlink()

        # Properly escape paths for subprocess
        cmd[input_idx] = SPath(input_path).as_posix()
        cmd[-1] = SPath(output_path).as_posix()

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        error_output = ""

        while True:
            output = process.stderr.readline()
            if output == "" and process.poll() is not None:
                break

            if output:
                error_output += output

                if "time=" in output:
                    debug(output.strip())

        if process.returncode != 0:
            if "looks empty (may consist of padding cells)" in error_output:
                warning(error_mapping["empty_title"])
                return

            if "Unrecognized option" in error_output:
                error(error_mapping["ffmpeg_unrecognized_option_full"].format(error_output))

                QMessageBox.critical(
                    self.parent,
                    error_mapping["ffmpeg_unrecognized_option"],
                    " ".join([error_mapping["ffmpeg_unrecognized_option"], error_mapping["ffmpeg_build_without_gpl"]])
                    + f"\n\n{error_output}",
                )
                return

            error(error_mapping["ffmpeg_process_failed"].format(error_output))

            QMessageBox.critical(
                self.parent,
                "Failed to dump titles",
                error_mapping["dump_titles_failed"].format(error_output),
            )

    def _dump_title(self, title_idx: int, output_path: str, angle: int | None = None) -> None:
        """Dump a single title."""

        input_path = (
            self.parent.iso_path.get_folder() if self.parent.iso_path.suffix.lower() == ".ifo" else self.parent.iso_path
        )

        cmd = self._build_ffmpeg_command(input_path.to_str(), output_path, angle, title_idx)
        self._run_ffmpeg_process(cmd)

    @property
    def last_dump_path(self) -> SPath:
        """Get the last dump path."""

        if not hasattr(self, "_last_dump_path"):
            self._last_dump_path = SPath(".")
        elif not self._last_dump_path.exists():
            debug(debug_mapping["last_dump_path_not_found"].format(self._last_dump_path))
            self._last_dump_path = SPath(".")

        return self._last_dump_path

    @last_dump_path.setter
    def last_dump_path(self, path: SPath) -> None:
        """Set the last dump path."""

        if path != self._last_dump_path:
            self._last_dump_path = path
            debug(debug_mapping["last_dump_path"].format(self._last_dump_path))


error_mapping: dict[str, str] = {
    "ffmpeg_build_without_gpl": "Please ensure FFmpeg was built with GPL library support "
    "and is configured with --enable-libdvdnav and --enable-libdvdread "
    "and is in your PATH.",
    "ffmpeg_unrecognized_option": "Unrecognized option in FFmpeg command!",
    "ffmpeg_unrecognized_option_full": "Unrecognized option in FFmpeg command!\n\n{}",
    "ffmpeg_check_failed": "FFmpeg check failed: {}\n{}",
    "ffmpeg_not_found": "FFmpeg not found. Please install FFmpeg and make sure it's in your PATH.",
    "ffmpeg_no_dvd_support": "FFmpeg installation does not support DVD video demuxing!",
    "ffmpeg_no_dvd_support_msg": "FFmpeg installation does not support DVD video demuxing. "
    "Please ensure FFmpeg was built with GPL library support.",
    "title_info_lookup_failed": "Title info lookup failed:\n"
    "Title key: {}\n"
    "Title info keys: {}\n"
    "Title number: {}\n"
    "Title angle: {}",
    "title_info_not_found": "Title information not found!",
    "title_info_not_found_key": "Title info not found for {}",
    "empty_title": "Skipping empty title (padding cells)",
    "ffmpeg_process_failed": "FFmpeg process failed:\n{}",
    "dump_titles_failed": "Failed to dump titles:\n\n{}",
    "dump_title_failed": "Failed to dump title {}: {}",
    "dump_title_angle_failed": "Failed to dump title {} angle {}: {}",
    "unexpected_dump_error": "Unexpected error dumping title {}: {}\n{}",
}


debug_mapping: dict[str, str] = {
    "building_ffmpeg_command": "Building FFmpeg command for title_key={}",
    "title_info": "Title info: {}",
    "unique_titles_found": "Found {} unique titles",
    "processing_title_angles": "Processing title {} with {} angles",
    "dumping_title": "Dumping title {} to {}",
    "dumping_title_angle": "Dumping title {} angle {} to {}",
    "dumping_title_start": "Dumping title",
    "dumping_title_angle_info": "Dumping title {} (angle {})",
    "getting_save_path": "Getting save path for title dump",
    "suggested_filename": "Suggested filename: {}",
    "save_dialog_cancelled": "User cancelled save dialog",
    "checking_ffmpeg": "Checking FFmpeg installation and DVD video support",
    "selected_output_path": "Selected output path: {}",
    "adding_chapter_start": "Adding chapter start: {}",
    "adding_chapter_end": "Adding chapter end: {}",
    "no_chapters": "No chapters available in title info",
    "pcm_audio_detected": "PCM audio detected for track {}, re-encoding to FLAC",
    "audio_detected": "{} audio detected for track {}, copying stream",
    "setting_language_metadata": "Setting language metadata for audio track {}: {}",
    "deleting_output": "Deleting existing output file: {}",
    "last_dump_path_not_found": "Last dump path no longer exists: {}",
    "last_dump_path": "Dump path updated to: {}",
}
