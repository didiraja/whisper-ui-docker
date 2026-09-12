import math
from pathlib import Path
from typing import Any


def render_plain_text(segments: list[dict[str, Any]]) -> str:
    return " ".join(
        str(segment.get("text", "")).strip()
        for segment in segments
        if str(segment.get("text", "")).strip()
    )


def _is_timestamp(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _format_timestamp(seconds: float) -> str:
    total_milliseconds = round(seconds * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def render_srt(segments: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for segment in segments:
        start_value = segment.get("start")
        end_value = segment.get("end")
        text = str(segment.get("text", "")).strip()
        if not (_is_timestamp(start_value) and _is_timestamp(end_value) and text):
            continue

        start = max(0.0, float(start_value))
        end = max(start, float(end_value))
        blocks.append(
            "\n".join(
                (
                    str(len(blocks) + 1),
                    f"{_format_timestamp(start)} --> {_format_timestamp(end)}",
                    text,
                )
            )
        )

    return "\n\n".join(blocks)


def write_outputs(
    directory: Path, segments: list[dict[str, Any]]
) -> tuple[str, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    plain_text = render_plain_text(segments)
    text_path = directory / "transcript.txt"
    srt_path = directory / "transcript.srt"
    text_path.write_text(f"{plain_text}\n", encoding="utf-8")
    srt_path.write_text(f"{render_srt(segments)}\n", encoding="utf-8")
    return plain_text, text_path, srt_path
