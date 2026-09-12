import json
import logging
import math
import subprocess
from pathlib import Path

from whisper_ui.domain import MediaInfo
from whisper_ui.errors import ValidationError

logger = logging.getLogger(__name__)


def duration_limit_message(max_duration_seconds: int) -> str:
    if max_duration_seconds % 3600 == 0:
        amount = max_duration_seconds // 3600
        unit = "hour"
    elif max_duration_seconds % 60 == 0:
        amount = max_duration_seconds // 60
        unit = "minute"
    else:
        amount = max_duration_seconds
        unit = "second"
    if amount != 1:
        unit += "s"
    return f"The media exceeds the configured duration limit of {amount} {unit}."


def probe_media(path: Path) -> MediaInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-show_streams",
        "-select_streams",
        "a",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command, shell=False, check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as error:
        logger.warning("ffprobe failed for %s: %s", path, error.stderr)
        raise ValidationError(
            "The media file could not be read.", code="media_probe_failed"
        ) from error
    except OSError as error:
        logger.warning("ffprobe could not run for %s: %s", path, error)
        raise ValidationError(
            "The media file could not be read.", code="media_probe_failed"
        ) from error

    try:
        payload = json.loads(result.stdout)
        info = MediaInfo(
            duration_seconds=float(payload["format"]["duration"]),
            has_audio=bool(payload["streams"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValidationError(
            "The media file could not be read.", code="media_probe_failed"
        ) from error

    if not info.has_audio:
        raise ValidationError(
            "The media file does not contain audio.", code="missing_audio"
        )
    if not math.isfinite(info.duration_seconds) or info.duration_seconds <= 0:
        raise ValidationError(
            "The media file has an invalid duration.",
            code="invalid_media_duration",
        )

    return info


def enforce_media(path: Path, max_duration_seconds: int) -> MediaInfo:
    info = probe_media(path)
    if info.duration_seconds > max_duration_seconds:
        raise ValidationError(
            duration_limit_message(max_duration_seconds),
            code="media_too_long",
        )
    return info


def normalize_audio(source: Path, destination: Path) -> Path:
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    try:
        subprocess.run(
            command, shell=False, check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as error:
        logger.warning("ffmpeg failed for %s: %s", source, error.stderr)
        raise ValidationError(
            "The audio could not be prepared.", code="media_processing_failed"
        ) from error
    except OSError as error:
        logger.warning("ffmpeg could not run for %s: %s", source, error)
        raise ValidationError(
            "The audio could not be prepared.", code="media_processing_failed"
        ) from error

    return destination
