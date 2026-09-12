import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from whisper_ui.domain import JobStage, ProgressCallback
from whisper_ui.errors import ValidationError
from whisper_ui.media import duration_limit_message, enforce_media

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}

_ACCESSIBLE_AVAILABILITY = {None, "public", "unlisted"}
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]+")
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
_VIDEO_PATH = re.compile(r"/(shorts|embed|live)/([A-Za-z0-9_-]{11})")


def _invalid_youtube_url() -> ValidationError:
    return ValidationError(
        "Enter a valid YouTube video URL.", code="invalid_youtube_url"
    )


def validate_youtube_url(url: str) -> str:
    try:
        parsed = urlsplit(url.strip())
        host = (parsed.hostname or "").lower().rstrip(".")
        username = parsed.username
        password = parsed.password
        _ = parsed.port
        query = parse_qs(parsed.query, keep_blank_values=True)
    except ValueError as error:
        raise _invalid_youtube_url() from error

    if parsed.scheme not in {"http", "https"} or host not in YOUTUBE_HOSTS:
        raise _invalid_youtube_url()
    if username or password:
        raise ValidationError(
            "Authenticated URLs are not supported.", code="invalid_youtube_url"
        )
    if "list" in query:
        raise _invalid_youtube_url()

    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.removeprefix("/")
        if not _VIDEO_ID.fullmatch(video_id):
            raise _invalid_youtube_url()
    elif parsed.path == "/watch":
        video_ids = query.get("v", [])
        if len(video_ids) != 1 or not _VIDEO_ID.fullmatch(video_ids[0]):
            raise _invalid_youtube_url()
    else:
        match = _VIDEO_PATH.fullmatch(parsed.path)
        if match is None or match.group(2) == "videoseries":
            raise _invalid_youtube_url()

    return parsed.geturl()


def _validate_video_info(info: Any, max_duration_seconds: int) -> dict[str, Any]:
    if not isinstance(info, dict) or "entries" in info:
        raise ValidationError(
            "Enter a valid YouTube video URL.", code="invalid_youtube_url"
        )
    if info.get("is_live") is True:
        raise ValidationError(
            "Live YouTube videos are not supported.",
            code="youtube_live_not_supported",
        )
    if info.get("availability") not in _ACCESSIBLE_AVAILABILITY:
        raise ValidationError(
            "Only public YouTube videos are supported.",
            code="youtube_video_unavailable",
        )

    age_limit = info.get("age_limit")
    if age_limit is not None:
        try:
            is_age_restricted = float(age_limit) > 0
        except (TypeError, ValueError) as error:
            raise ValidationError(
                "The YouTube video metadata is invalid.",
                code="youtube_download_failed",
            ) from error
        if is_age_restricted:
            raise ValidationError(
                "Age-restricted YouTube videos are not supported.",
                code="youtube_age_restricted",
            )

    duration = info.get("duration")
    if duration is not None:
        try:
            duration_seconds = float(duration)
        except (TypeError, ValueError) as error:
            raise ValidationError(
                "The YouTube video metadata is invalid.",
                code="youtube_download_failed",
            ) from error
        if duration_seconds > max_duration_seconds:
            raise ValidationError(
                duration_limit_message(max_duration_seconds),
                code="media_too_long",
            )

    return info


def _sanitize_title(value: Any) -> str:
    title = value if isinstance(value, str) else ""
    title = _CONTROL_CHARACTERS.sub(" ", title)
    title = " ".join(title.split())
    return title[:160] or "YouTube video"


def download_youtube(
    url: str,
    directory: Path,
    max_duration_seconds: int,
    progress: ProgressCallback,
) -> tuple[Path, str]:
    normalized_url = validate_youtube_url(url)
    options: dict[str, Any] = {
        "noplaylist": True,
        "playlist_items": "1",
        "format": "bestaudio/best",
        "outtmpl": str(directory / "youtube.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "js_runtimes": {"node": {}},
    }

    progress(JobStage.DOWNLOADING)
    try:
        with YoutubeDL(options) as preflight_downloader:
            preflight_info = preflight_downloader.extract_info(
                normalized_url, download=False
            )
        _validate_video_info(preflight_info, max_duration_seconds)

        with YoutubeDL(options) as downloader:
            downloaded_info = downloader.extract_info(normalized_url, download=True)
            info = _validate_video_info(downloaded_info, max_duration_seconds)
            downloaded_path = Path(downloader.prepare_filename(info))
    except DownloadError as error:
        raise ValidationError(
            "The public YouTube video could not be downloaded.",
            code="youtube_download_failed",
        ) from error

    if not downloaded_path.is_file():
        raise ValidationError(
            "The public YouTube video could not be downloaded.",
            code="youtube_download_failed",
        )

    enforce_media(downloaded_path, max_duration_seconds)
    return downloaded_path, _sanitize_title(info.get("title"))
