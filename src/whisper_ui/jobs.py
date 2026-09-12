import logging
import shutil
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, BinaryIO, Literal
from uuid import uuid4

from whisper_ui.config import Settings
from whisper_ui.domain import (
    JobRecord,
    JobRequest,
    JobStage,
    JobState,
    MediaInfo,
    ProgressCallback,
    TranscriptResult,
)
from whisper_ui.errors import AppError, BusyError, NotFoundError, ValidationError
from whisper_ui.media import enforce_media, normalize_audio
from whisper_ui.outputs import write_outputs
from whisper_ui.youtube import download_youtube

if TYPE_CHECKING:
    from whisper_ui.whisper_service import WhisperService

logger = logging.getLogger(__name__)
GENERIC_ERROR = "Transcription failed. Check the container logs for details."
_ACTIVE = {JobState.ACCEPTED, JobState.RUNNING}
RETIRED_CLEANUP_BATCH_SIZE = 4


class ResultLease:
    def __init__(
        self, manager: "JobManager", job_id: str, stream: BinaryIO
    ) -> None:
        self._manager = manager
        self._job_id = job_id
        self._stream = stream
        self._closed = False

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.close()
        finally:
            self._manager._release_result(self._job_id)


class JobManager:
    def __init__(
        self,
        settings: Settings,
        whisper: "WhisperService",
        *,
        media_validator: Callable[[Path, int], MediaInfo] = enforce_media,
        audio_normalizer: Callable[[Path, Path], Path] = normalize_audio,
        youtube_downloader: Callable[
            [str, Path, int, ProgressCallback], tuple[Path, str]
        ] = download_youtube,
    ) -> None:
        self.settings = settings
        self.whisper = whisper
        self._media_validator = media_validator
        self._audio_normalizer = audio_normalizer
        self._youtube_downloader = youtube_downloader
        self._jobs_dir = settings.jobs_dir.resolve()
        self._jobs_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="transcription"
        )
        self._current: JobRecord | None = None
        self._submitted: tuple[str, Future[None]] | None = None
        self._result_leases: dict[str, int] = {}
        self._retired_directories: dict[str, Path] = {}
        self._closed = False

    def reserve(self, request: JobRequest, display_name: str) -> JobRecord:
        with self._lock:
            if self._closed:
                raise AppError(
                    "The transcription service is shutting down.",
                    code="service_unavailable",
                    status_code=503,
                )
            if self._current is not None:
                if self._current.state in _ACTIVE:
                    raise BusyError()
                self._retire(self._current)
                self._current = None
            job_id = str(uuid4())
            directory = self._jobs_dir / job_id
            self._check_directory(directory)
            directory.mkdir(mode=0o700)
            self._current = JobRecord(
                id=job_id,
                directory=directory,
                request=request,
                display_name=display_name,
                stage=(
                    JobStage.RECEIVING
                    if request.source_kind == "upload"
                    else JobStage.VALIDATING
                ),
            )
            return replace(self._current)

    def set_request(self, job_id: str, request: JobRequest) -> None:
        with self._lock:
            record = self._lookup(job_id)
            if record.state != JobState.ACCEPTED:
                raise NotFoundError()
            record.request = request

    def start(self, job_id: str) -> None:
        with self._lock:
            record = self._lookup(job_id)
            if self._closed or record.state != JobState.ACCEPTED:
                raise NotFoundError()
            # Claim the record before submission so repeated starts cannot queue it twice.
            record.state = JobState.RUNNING
            try:
                future = self._executor.submit(self._run, job_id)
                self._submitted = (job_id, future)
            except Exception:
                logger.exception("Job %s failed", job_id)
                self._fail(record, GENERIC_ERROR)
                return
        # An already-finished future invokes callbacks immediately, outside our lock.
        future.add_done_callback(lambda completed: self._on_finished(job_id, completed))

    def _on_finished(self, job_id: str, future: Future[None]) -> None:
        with self._lock:
            if self._submitted != (job_id, future):
                return
            self._submitted = None
            if (
                future.cancelled()
                and self._current is not None
                and self._current.id == job_id
                and self._current.state == JobState.RUNNING
            ):
                self._fail(self._current, GENERIC_ERROR)

    def fail_reserved(self, job_id: str, message: str) -> None:
        """Finish an upload reservation with a caller-provided safe public message."""
        with self._lock:
            record = self._lookup(job_id)
            if record.state != JobState.ACCEPTED:
                raise NotFoundError()
            self._fail(record, message)

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            return replace(self._lookup(job_id))

    def current(self) -> JobRecord | None:
        with self._lock:
            if self._current is None or self._expired(self._current):
                return None
            return replace(self._current)

    def open_result(
        self, job_id: str, format_name: Literal["txt", "srt"]
    ) -> ResultLease:
        with self._lock:
            record = self._lookup(job_id)
            if record.state != JobState.COMPLETED or format_name not in {"txt", "srt"}:
                raise NotFoundError()
            path = record.text_path if format_name == "txt" else record.srt_path
            if path is None or not self._is_result_file(record.directory, path):
                raise NotFoundError()
            try:
                stream = path.open("rb")
            except OSError as error:
                raise NotFoundError() from error
            self._result_leases[job_id] = self._result_leases.get(job_id, 0) + 1
            return ResultLease(self, job_id, stream)

    def purge_expired(self) -> None:
        with self._lock:
            if self._current is not None and self._expired(self._current):
                self._retire(self._current)
                self._current = None
            self._retry_retired_directories()

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            submitted = self._submitted
        self._executor.shutdown(wait=False, cancel_futures=True)
        # Cancellation may precede start() attaching its callback.
        if submitted is not None and submitted[1].cancelled():
            self._on_finished(*submitted)

    def _lookup(self, job_id: str) -> JobRecord:
        record = self._current
        if record is None or record.id != job_id or self._expired(record):
            raise NotFoundError()
        return record

    @staticmethod
    def _expired(record: JobRecord) -> bool:
        return (
            record.state not in _ACTIVE
            and record.expires_at is not None
            and record.expires_at <= datetime.now(timezone.utc)
        )

    def _set_stage(self, job_id: str, stage: JobStage) -> None:
        with self._lock:
            if self._current is not None and self._current.id == job_id:
                if self._current.state == JobState.RUNNING:
                    self._current.stage = stage

    def _set_display_name(self, job_id: str, display_name: str) -> None:
        with self._lock:
            if self._current is not None and self._current.id == job_id:
                if self._current.state == JobState.RUNNING:
                    self._current.display_name = display_name

    def _run(self, job_id: str) -> None:
        try:
            record = self.get(job_id)
            request = record.request

            def progress(stage: JobStage) -> None:
                self._set_stage(job_id, stage)

            source = request.source_path
            if request.source_kind == "youtube":
                if not request.source_url or not request.source_url.strip():
                    raise ValidationError("Enter a valid YouTube video URL.")
                source, title = self._youtube_downloader(
                    request.source_url,
                    record.directory,
                    self.settings.max_media_duration_seconds,
                    progress,
                )
                self._set_display_name(job_id, title)
            elif request.source_kind == "upload":
                if source is None or not source.is_file():
                    raise ValidationError("The uploaded media file is missing.")
                progress(JobStage.VALIDATING)
                self._media_validator(source, self.settings.max_media_duration_seconds)
            else:
                raise ValidationError("Choose an upload or a YouTube video.")
            if source is None or not source.is_file():
                raise ValidationError("The media file is missing.")
            progress(JobStage.PREPARING)
            normalized = self._audio_normalizer(
                source, record.directory / "normalized.wav"
            )
            transcript = self.whisper.transcribe(
                normalized, request.model, request.language, progress
            )
            progress(JobStage.FORMATTING)
            plain, text_path, srt_path = write_outputs(
                record.directory, transcript.segments
            )
            self._complete(job_id, transcript, plain, text_path, srt_path)
        except AppError as error:
            self._fail_running(job_id, error.message)
        except Exception:
            logger.exception("Job %s failed", job_id)
            self._fail_running(job_id, GENERIC_ERROR)

    def _complete(
        self,
        job_id: str,
        transcript: TranscriptResult,
        plain: str,
        text_path: Path,
        srt_path: Path,
    ) -> None:
        with self._lock:
            record = self._lookup(job_id)
            if record.state != JobState.RUNNING:
                return
            for path in (text_path, srt_path):
                if not self._is_result_file(record.directory, path):
                    raise RuntimeError("Output file is missing or outside the job directory")
            # Cleanup and publication are atomic with respect to a new reservation.
            self._clean_payload(record.directory, {text_path, srt_path})
            record.state = JobState.COMPLETED
            record.stage = JobStage.COMPLETE
            record.transcript = plain
            record.text_path = text_path
            record.srt_path = srt_path
            record.detected_language = transcript.language
            record.warning = transcript.alignment_warning
            record.expires_at = self._expiration()

    def _fail_running(self, job_id: str, message: str) -> None:
        with self._lock:
            if self._current is not None and self._current.id == job_id:
                if self._current.state == JobState.RUNNING:
                    self._fail(self._current, message)

    def _fail(self, record: JobRecord, message: str) -> None:
        record.state = JobState.FAILED
        record.error = message
        record.transcript = None
        record.text_path = None
        record.srt_path = None
        record.expires_at = self._expiration()
        try:
            self._remove_directory(record.directory)
        except Exception:
            # Keep the safe terminal record even when filesystem cleanup fails.
            logger.exception("Could not remove payload for job %s", record.id)

    def _expiration(self) -> datetime:
        return datetime.now(timezone.utc) + timedelta(
            minutes=self.settings.result_retention_minutes
        )

    def _check_directory(self, directory: Path) -> None:
        # Job paths must be direct children of the original, resolved jobs root.
        if (
            self._jobs_dir.resolve() != self._jobs_dir
            or directory.parent != self._jobs_dir
            or directory == self._jobs_dir
            or directory.name in {"", ".", ".."}
        ):
            raise RuntimeError("Unsafe job directory")

    def _remove_directory(self, directory: Path) -> None:
        self._check_directory(directory)
        if directory.is_symlink():
            directory.unlink()
        elif directory.exists():
            shutil.rmtree(directory)

    def _retire(self, record: JobRecord) -> None:
        if self._result_leases.get(record.id, 0):
            self._retired_directories[record.id] = record.directory
            return
        self._remove_directory(record.directory)

    def _release_result(self, job_id: str) -> None:
        with self._lock:
            count = self._result_leases.get(job_id, 0)
            if count <= 1:
                self._result_leases.pop(job_id, None)
                directory = self._retired_directories.get(job_id)
                if directory is not None:
                    try:
                        self._remove_directory(directory)
                    except Exception:
                        logger.exception(
                            "Could not remove retired result directory for job %s",
                            job_id,
                        )
                    else:
                        self._retired_directories.pop(job_id, None)
            else:
                self._result_leases[job_id] = count - 1

    def _retry_retired_directories(self) -> None:
        batch_size = min(
            RETIRED_CLEANUP_BATCH_SIZE, len(self._retired_directories)
        )
        for _ in range(batch_size):
            job_id, directory = next(iter(self._retired_directories.items()))
            self._retired_directories.pop(job_id)
            if self._result_leases.get(job_id, 0):
                self._retired_directories[job_id] = directory
                continue
            try:
                self._remove_directory(directory)
            except Exception:
                self._retired_directories[job_id] = directory
                logger.exception(
                    "Could not remove retired result directory for job %s; "
                    "cleanup will retry",
                    job_id,
                )

    def _is_result_file(self, directory: Path, path: Path) -> bool:
        self._check_directory(directory)
        return (
            not directory.is_symlink()
            and path.parent == directory
            and not path.is_symlink()
            and path.is_file()
        )

    def _clean_payload(self, directory: Path, keep: set[Path]) -> None:
        self._check_directory(directory)
        if directory.is_symlink():
            raise RuntimeError("Unsafe job directory")
        for path in directory.iterdir():
            if path in keep:
                continue
            if path.is_symlink() or not path.is_dir():
                path.unlink()
            else:
                shutil.rmtree(path)
