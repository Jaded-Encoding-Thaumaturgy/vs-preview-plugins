from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from time import time
from typing import Any

import requests
from jetpytools import CustomRuntimeError, FileWasNotFoundError, SPath, SPathLike

__all__ = ["AnilistInfo", "MatchInfo", "SearchResult"]

logger = logging.getLogger("dev.lightarrowsexe.trace_moe")
plugin_name = "dev.lightarrowsexe.trace_moe"

# idk the best way to deal with this and just get it from vspreview
plugin_dir = SPath.cwd() / ".vsjet" / "vspreview" / "trace-moe"


@dataclass
class BaseModel:
    """Base class for API response models."""

    @classmethod
    def _safe_get(cls, data: dict[str, Any], key: str, default: Any = None) -> Any:
        """Safely get value from dictionary with default."""

        return data.get(key, default)


@dataclass
class AnilistInfo(BaseModel):
    """Information about an anime."""

    anilist_id: int
    title_native: str
    title_romaji: str | None = None
    title_english: str | None = None
    media_info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | int) -> AnilistInfo:
        if isinstance(data, int):
            return cls(
                anilist_id=data,
                title_native="",
                title_romaji=None,
                title_english=None,
            )

        return cls(
            anilist_id=cls._safe_get(data, "id"),
            title_native=cls._safe_get(data, "title", {}).get("native", ""),
            title_romaji=cls._safe_get(data, "title", {}).get("romaji"),
            title_english=cls._safe_get(data, "title", {}).get("english"),
        )

    def fetch_media(self) -> None:
        """
        Build GraphQL query and variable dict for AniList API for this anime's anilist_id.

        See the AniList API documentation for more information: <https://docs.anilist.co/guide/graphql/queries/media#get-media-by-id>
        """

        if self._load_cache():
            return

        api_url = "https://graphql.anilist.co"
        variables = {"id": self.anilist_id}
        query = (
            "query ($id: Int) {"
            " Media(id: $id) {"
            " id"
            " title { romaji english native }"
            " coverImage { large }"
            " episodes format status"
            " startDate { year month day }"
            " endDate { year month day }"
            " genres"
            " }"
            "}"
        )

        response = requests.post(api_url, json={"query": query, "variables": variables}, timeout=15.0)

        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            logger.error(f"AniList API returned a non-successful status code for Anilist ID {self.anilist_id}: {e}")
            logger.debug(f"AniList API response: {response.text}")
            return

        data = dict(self._safe_get(json.loads(response.text), "data", {}))
        media = dict(self._safe_get(data, "Media", {}))

        media.pop("id", None)
        media.pop("title", None)

        if not self.title_native:
            titles = self._safe_get(data, "title", {})

            self.title_native = self._safe_get(titles, "native", "")
            self.title_romaji = self._safe_get(titles, "romaji", None)
            self.title_english = self._safe_get(titles, "english", None)

            logger.debug(
                f"Updated titles for Anilist ID {self.anilist_id}: "
                f"{self.title_native=}, {self.title_romaji=}, {self.title_english=}"
            )

        self.media_info.update(media)
        logger.debug(f"Updated media info for Anilist ID {self.anilist_id}: {media=}")

        self._save_cache()

    @property
    def _cache_dir(self) -> SPath:
        return plugin_dir / ".cache" / "anilist-queries"

    def _load_cache(self) -> bool:
        cache_file = self._cache_dir / f"{self.anilist_id}.json"

        if not cache_file.exists() or cache_file.stat().st_size == 0:
            return False

        mtime = cache_file.stat().st_mtime

        if (time() - mtime) > 3 * 86400:
            logger.debug(f"Cache file {cache_file} is older than 3 days. Deleting cache file.")
            cache_file.unlink(missing_ok=True)

            return False

        try:
            with open(cache_file.to_str(), "r") as f:
                data = json.load(f)

                self.title_native = self._safe_get(data, "title_native", "")
                self.title_romaji = self._safe_get(data, "title_romaji", None)
                self.title_english = self._safe_get(data, "title_english", None)
                self.media_info = self._safe_get(data, "media_info", {})

            logger.debug(f"Loaded cache for Anilist ID {self.anilist_id}")

            return True
        except (PermissionError, OSError, json.JSONDecodeError) as e:
            print(f"Failed to load cache: {e}")

            return False

    def _save_cache(self) -> None:
        cache_file = self._cache_dir / f"{self.anilist_id}.json"
        cache_file.get_folder().mkdir(parents=True, exist_ok=True)

        with open(cache_file.to_str(), "w") as f:
            json.dump(
                {
                    "title_native": self.title_native,
                    "title_romaji": self.title_romaji,
                    "title_english": self.title_english,
                    "media_info": self.media_info,
                },
                f,
                indent=4,
            )

        if not cache_file.exists():
            raise FileWasNotFoundError(f"Failed to save cache to {cache_file}!", plugin_name)


@dataclass
class MatchInfo(BaseModel):
    """
    Information about a match from trace.moe API.

    See the API documentation for more information: <https://soruly.github.io/trace.moe-api/#/docs?id=response-format>
    """

    anilist: AnilistInfo
    episode: int | None
    timestamp: float
    similarity: float
    video_url: str
    image_url: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatchInfo:
        anilist_data = data.get("anilist", {})

        if isinstance(anilist_data, int):
            anilist = AnilistInfo(
                anilist_id=anilist_data,
                title_native="",
                title_romaji=None,
                title_english=None,
            )
        else:
            anilist = AnilistInfo.from_dict(anilist_data)

        return cls(
            anilist=anilist,
            episode=cls._safe_get(data, "episode"),
            timestamp=cls._safe_get(data, "at", 0.0),
            similarity=cls._safe_get(data, "similarity", 0.0),
            video_url=cls._safe_get(data, "video", ""),
            image_url=cls._safe_get(data, "image", ""),
        )

    def format_timestamp(self) -> str:
        total_seconds = int(self.timestamp)

        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours == 0:
            return f"{minutes:02d}:{seconds:02d}"

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @property
    def similarity_as_percentage(self) -> str:
        return f"{self.similarity * 100:.2f}%"

    def download_image(self) -> SPath:
        """Download the image from the match."""

        url_key = self.image_url.rsplit("/", 1)[-1]

        if out_spath := self._check_image_cached(self.image_url):
            return out_spath

        out_spath = self._image_cache_dir / f"{url_key}.jpg"
        out_spath.get_folder().mkdir(parents=True, exist_ok=True)

        response = requests.get(self.image_url, timeout=15.0)
        response.raise_for_status()
        image_content = response.content

        with open(out_spath.to_str(), "wb") as f:
            f.write(image_content)

        if not out_spath.exists():
            raise FileWasNotFoundError(
                f"Failed to download image {SPath(url_key).with_suffix(out_spath.suffix)}!",
                plugin_name,
            )

        return out_spath

    @property
    def _image_cache_dir(self) -> SPath:
        return plugin_dir / ".cache" / "trace-moe-thumbnails"

    def _check_image_cached(self, url: str) -> SPath | None:
        url_key = url.rsplit("/", 1)[-1]
        cache_file = self._image_cache_dir / f"{url_key}.jpg"

        if cache_file.exists() and cache_file.stat().st_size > 0:
            logger.debug(f"Image {url_key} found in cache at {cache_file}")
            return cache_file

        return None


@dataclass
class SearchResult(BaseModel):
    """
    Complete search result from trace.moe API.

    See the API documentation for more information: <https://soruly.github.io/trace.moe-api/#/docs?id=response-format>
    """

    frame_count: int
    error: str | None
    matches: list[MatchInfo]

    @classmethod
    def from_image(
        cls,
        image_path: SPathLike,
        api_key: str = "",
        anilist_id: int | None = None,
        cut_black_borders: bool = False,
    ) -> SearchResult:
        """Create SearchResult from an image path."""

        spath = SPath(image_path)

        if not spath.exists():
            raise FileWasNotFoundError(f"Image file {spath} does not exist!", plugin_name)

        base_url = "https://api.trace.moe/search?anilistInfo"

        if api_key:
            base_url += f"&apikey={api_key}"

        if anilist_id is not None:
            base_url += f"&anilistID={anilist_id}"

        if cut_black_borders:
            base_url += "&cutBorders"

        try:
            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": "curl/8.16.0",
                    "Accept": "*/*",
                }
            )

            # Send image as multipart form data with parameters
            with open(image_path, "rb") as image_file:
                files = {"image": (spath.name, image_file, "image/png")}
                response = session.post(
                    base_url,
                    files=files,
                    timeout=15.0,
                )

            response.raise_for_status()

            return cls.from_response_json(response.content)
        except Exception as e:
            err_msg = f"Failed to get response from API: {e}"
            logger.error(err_msg)
            raise CustomRuntimeError(err_msg, plugin_name)

    @classmethod
    def from_response_json(cls, json_data: bytes) -> SearchResult:
        """Create SearchResult from API response bytes."""

        json_parsed = json.loads(json_data)
        json_parsed = dict(json_parsed)

        return cls(
            frame_count=cls._safe_get(json_parsed, "frameCount", 0),
            error=cls._safe_get(json_parsed, "error", None),
            matches=[MatchInfo.from_dict(result) for result in json_parsed.get("result", [])],
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchResult:
        """Create SearchResult from API response dictionary."""

        data = dict(data)

        return cls(
            frame_count=cls._safe_get(data, "frameCount", 0),
            error=cls._safe_get(data, "error", None),
            matches=[MatchInfo.from_dict(result) for result in data.get("result", [])],
        )
