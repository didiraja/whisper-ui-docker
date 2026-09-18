from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

ProgressCallback = Callable[["JobStage"], None]
StageProgressCallback = Callable[["JobStage", float], None]
ActivityLogCallback = Callable[[str], None]
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
    pipeline_started_at: datetime | None = None
    pipeline_finished_at: datetime | None = None
    stage_started_at: datetime | None = None
    stage_progress_percent: float | None = None
    progress_started_at: datetime | None = None
    progress_started_percent: float | None = None
    progress_observations: int = 0
    logged_progress_percent: int | None = None
    alignment_fallback: bool = False
    log_lines: tuple[str, ...] = ()
    expires_at: datetime | None = None

    def public_snapshot(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        elapsed_seconds: int | None = None
        stage_elapsed: float | None = None
        stage_elapsed_seconds: int | None = None
        estimated_remaining_seconds: int | None = None
        if self.pipeline_started_at is not None:
            pipeline_end = self.pipeline_finished_at or now
            elapsed_seconds = max(
                0, int((pipeline_end - self.pipeline_started_at).total_seconds())
            )
        if self.stage_started_at is not None:
            stage_end = self.pipeline_finished_at or now
            stage_elapsed = max(
                0.0, (stage_end - self.stage_started_at).total_seconds()
            )
            stage_elapsed_seconds = int(stage_elapsed)
        if (
            self.state == JobState.RUNNING
            and self.progress_started_at is not None
            and self.progress_started_percent is not None
            and self.progress_observations >= 2
            and self.stage_progress_percent is not None
            and self.progress_started_percent < self.stage_progress_percent < 100
        ):
            measured_seconds = (now - self.progress_started_at).total_seconds()
            measured_progress = (
                self.stage_progress_percent - self.progress_started_percent
            )
            if measured_seconds >= 3 and measured_progress > 0:
                estimated_remaining_seconds = max(
                    0,
                    round(
                        measured_seconds
                        * (100 - self.stage_progress_percent)
                        / measured_progress
                    ),
                )
        return {
            "id": self.id,
            "display_name": self.display_name,
            "state": self.state.value,
            "stage": self.stage.value,
            "transcript": self.transcript,
            "detected_language": self.detected_language,
            "warning": self.warning,
            "error": self.error,
            "pipeline": {
                "elapsed_seconds": elapsed_seconds,
                "stage_elapsed_seconds": stage_elapsed_seconds,
                "estimated_remaining_seconds": estimated_remaining_seconds,
                "progress_percent": self.stage_progress_percent,
                "alignment_fallback": self.alignment_fallback,
                "log_lines": list(self.log_lines),
            },
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
