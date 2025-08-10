from dataclasses import dataclass

__all__ = [
    "TitleInfo",
]


@dataclass
class TitleInfo:
    """Information about a title."""

    title_num: int
    """Index of the title."""

    angle: int | None
    """Angle number if multi-angle, None otherwise."""

    chapter_count: int
    """Number of chapters in the title."""

    chapters: list[int]
    """List of chapter start frames."""

    audio_tracks: list[str]
    """List of audio track descriptions."""

    angle_count: int
    """Total number of angles available."""

    width: int
    """Video width in pixels."""

    height: int
    """Video height in pixels."""

    fps: float
    """Video framerate."""

    duration: float
    """Duration in seconds."""
