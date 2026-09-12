from pathlib import Path

from fastapi import UploadFile

from whisper_ui.errors import ValidationError

ALLOWED_SUFFIXES = {".mp3", ".wav"}
CHUNK_SIZE = 1024 * 1024


def _megabyte_label(byte_count: int) -> str:
    return f"{byte_count / (1024 * 1024):g}"


async def save_upload(upload: UploadFile, destination: Path, max_bytes: int) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValidationError(
            "Only MP3 and WAV files are supported.", code="unsupported_file"
        )

    target = destination.with_suffix(suffix)
    written = 0
    try:
        with target.open("wb") as output:
            while chunk := await upload.read(CHUNK_SIZE):
                written += len(chunk)
                if written > max_bytes:
                    limit_mb = _megabyte_label(max_bytes)
                    raise ValidationError(
                        f"The upload exceeds the {limit_mb} MB limit.",
                        code="upload_too_large",
                    )
                output.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    return target
