from __future__ import annotations

import asyncio
import logging
from typing import NamedTuple

from jetpytools import CustomRuntimeError, SPath
from PyQt6.QtCore import QObject, pyqtSignal
from vskernels import Bilinear
from vspreview.core import PackingType, VideoOutput
from vspreview.main import MainWindow
from vstools import FieldBased, Matrix, core, get_video_format, get_w, vs

from .models import MatchInfo, SearchResult

plugin_name = "dev.lightarrowsexe.trace_moe"
logger = logging.getLogger(plugin_name)


class WorkerConfiguration(NamedTuple):
    uuid: str
    node: VideoOutput
    path: SPath
    main: MainWindow
    frame_num: int
    api_key: str
    anilist_id: int | None
    cut_black_borders: bool
    min_similarity: float


class Worker(QObject):
    finished = pyqtSignal(str, object)

    _is_finished = False

    def is_finished(self) -> bool:
        if self._is_finished:
            self.deleteLater()
        return self._is_finished

    def run(self, conf: WorkerConfiguration) -> None:
        path = conf.path / f"frame_{conf.frame_num}_{conf.uuid}"
        path.get_folder().mkdir(parents=True, exist_ok=True)

        src_filename = self._write_frame(conf, path)

        search_result = asyncio.run(self._get_response_from_api(conf, src_filename))

        filtered_matches = [match for match in search_result.matches if match.similarity >= conf.min_similarity]
        search_result.matches = filtered_matches

        src_filename.unlink(missing_ok=True)

        asyncio.run(self._download_match_images(search_result.matches))

        self.finished.emit(path.to_str(), search_result)

    def _write_frame(self, conf: WorkerConfiguration, path: SPath) -> SPath:
        path = path.with_suffix(".png")
        path.get_folder().mkdir(parents=True, exist_ok=True)

        if conf.frame_num >= conf.node.source.clip.num_frames:
            raise CustomRuntimeError(
                f"Frame number {conf.frame_num} is greater than the number of frames "
                f"in the clip ({conf.node.source.clip.num_frames})!",
                plugin_name,
            )

        clip = self._prepare_frame(
            conf.node.prepare_vs_output(
                conf.node.source.clip,
                not hasattr(core, "fpng"),
                PackingType.CURRENT.vs_format.replace(bits_per_sample=8, sample_type=vs.INTEGER),
            )
        )

        if hasattr(core, "fpng"):
            format_filename = path.with_stem(path.stem + "_%d").to_str()
            clip = core.fpng.Write(clip, filename=format_filename)
            clip.get_frame(conf.frame_num)

            if (generated_file := path.with_stem(path.stem + f"_{conf.frame_num}")).exists():
                generated_file.rename(path)
        else:
            conf.main.current_output.frame_to_qimage(clip.get_frame(0)).save(path.to_str(), "PNG", 100)

        if not path.exists():
            raise CustomRuntimeError(f"Failed to save source frame ({conf.frame_num})!", plugin_name)

        return path

    def _prepare_frame(self, clip: vs.VideoNode) -> vs.VideoNode:
        if (
            clip.height == 360
            and (vformat := get_video_format(clip)).color_family is vs.RGB
            and vformat.bits_per_sample == 8
        ):
            logger.debug(f"Frame {clip.height}x{clip.width} is already 360p and RGB24, returning as is")
            return clip

        clip = FieldBased.PROGRESSIVE.apply(clip)

        return Bilinear().scale(
            clip, get_w(360, clip), 360, format=vs.RGB24, matrix=Matrix.from_video(clip, strict=False)
        )

    async def _get_response_from_api(self, conf: WorkerConfiguration, image_path: SPath) -> SearchResult:
        return await asyncio.to_thread(
            SearchResult.from_image,
            image_path,
            conf.api_key,
            conf.anilist_id,
            conf.cut_black_borders,
        )

    async def _download_match_images(self, matches: list[MatchInfo]) -> list[SPath]:
        if not matches:
            err_msg = "No matches found in the response!"
            logger.warning(err_msg)
            raise CustomRuntimeError(err_msg, plugin_name)

        download_tasks = []
        for i, match in enumerate(matches):
            if not match.image_url:
                logger.warning(f"[{i}/{len(matches)}] No image URL found for match!")
                logger.debug(f"[{i}/{len(matches)}] API match object: {match}")
                continue

            download_tasks.append(asyncio.to_thread(match.download_image))

        if not download_tasks:
            logger.warning("No valid image URLs found in any matches!")
            return []

        return await asyncio.gather(*download_tasks)
