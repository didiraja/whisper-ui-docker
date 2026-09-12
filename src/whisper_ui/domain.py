from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

ProgressCallback = Callable[["JobStage"], None]
SourceKind = Literal["upload", "youtube"]


class JobState(StrEnum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStage(StrEnum):
    VALIDATING = "validating"
    RECEIVING = "receiving"
    DOWNLOADING = "downloading"
    PREPARING = "preparing"
    LOADING_MODEL = "loading_model"
    TRANSCRIBING = "transcribing"
    ALIGNING = "aligning"
    FORMATTING = "formatting"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class JobRequest:
    source_kind: SourceKind
    model: str
    language: str | None
    source_path: Path | None = None
    source_url: str | None = None


@dataclass(slots=True)
class JobRecord:
    id: str
    directory: Path
    request: JobRequest
    display_name: str
    state: JobState = JobState.ACCEPTED
    stage: JobStage = JobStage.VALIDATING
    transcript: str | None = None
    text_path: Path | None = None
    srt_path: Path | None = None
    detected_language: str | None = None
    warning: str | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None

    def public_snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "state": self.state.value,
            "stage": self.stage.value,
            "transcript": self.transcript,
            "detected_language": self.detected_language,
            "warning": self.warning,
            "error": self.error,
            "downloads": {
                "txt": f"/api/jobs/{self.id}/transcript.txt" if self.text_path else None,
                "srt": f"/api/jobs/{self.id}/transcript.srt" if self.srt_path else None,
            },
        }


@dataclass(frozen=True, slots=True)
class MediaInfo:
    duration_seconds: float
    has_audio: bool


@dataclass(frozen=True, slots=True)
class TranscriptResult:
    segments: list[dict[str, Any]]
    language: str
    alignment_warning: str | None
