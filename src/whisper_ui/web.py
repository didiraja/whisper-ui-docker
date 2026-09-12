import logging
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Literal, cast

from fastapi import APIRouter, Form, Request, UploadFile, status
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.types import Receive, Scope, Send
from whisperx.utils import LANGUAGES

from whisper_ui.config import MODELS, Settings
from whisper_ui.domain import JobRequest
from whisper_ui.errors import AppError, ValidationError
from whisper_ui.jobs import JobManager
from whisper_ui.uploads import save_upload
from whisper_ui.youtube import validate_youtube_url

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def _manager(request: Request) -> JobManager:
    return cast(JobManager, request.app.state.job_manager)


def _settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


class ResultStreamingResponse(StreamingResponse):
    def __init__(self, *args, lease, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lease = lease

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.lease.close()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> Response:
    settings = _settings(request)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "models": MODELS,
            "default_model": settings.default_model,
            "languages": sorted(LANGUAGES.items(), key=lambda item: item[1]),
            "default_language": settings.default_language,
            "max_upload_mb": settings.max_upload_mb,
            "max_duration_hours": settings.max_media_duration_seconds / 3600,
        },
    )


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    return {
        "status": "ok",
        "device": cast(str, request.app.state.whisper_service.device),
    }


@router.post("/api/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    request: Request,
    youtube_url: Annotated[str | None, Form()] = None,
    audio_file: UploadFile | None = None,
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
) -> dict[str, object]:
    normalized_url = _optional_text(youtube_url)
    normalized_model = _optional_text(model)
    normalized_language = _optional_text(language)
    upload = audio_file if audio_file and _optional_text(audio_file.filename) else None

    if (normalized_url is None) == (upload is None):
        raise ValidationError("Choose either a YouTube URL or one audio file.")
    if normalized_model not in MODELS:
        raise ValidationError("Choose a supported transcription model.", code="invalid_model")
    if normalized_language is not None and normalized_language not in LANGUAGES:
        raise ValidationError("Choose a supported language.", code="invalid_language")

    manager = _manager(request)
    if normalized_url is not None:
        source_url = validate_youtube_url(normalized_url)
        job_request = JobRequest(
            source_kind="youtube",
            model=normalized_model,
            language=normalized_language,
            source_url=source_url,
        )
        record = manager.reserve(job_request, "YouTube video")
    else:
        assert upload is not None
        display_name = Path(upload.filename or "").name
        job_request = JobRequest(
            source_kind="upload",
            model=normalized_model,
            language=normalized_language,
        )
        record = manager.reserve(job_request, display_name)
        try:
            source_path = await save_upload(
                upload,
                record.directory / "source",
                _settings(request).max_upload_bytes,
            )
        except AppError as error:
            manager.fail_reserved(record.id, error.message)
            raise
        except Exception as error:
            logger.exception("Upload persistence failed for job %s", record.id)
            public_error = AppError(
                "The upload could not be saved.",
                code="upload_failed",
                status_code=500,
            )
            manager.fail_reserved(record.id, public_error.message)
            raise public_error from error
        job_request = replace(job_request, source_path=source_path)
        manager.set_request(record.id, job_request)

    manager.start(record.id)
    return manager.get(record.id).public_snapshot()


@router.get("/api/jobs/current", response_model=None)
async def current_job(request: Request) -> Response | dict[str, object]:
    record = _manager(request).current()
    if record is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return record.public_snapshot()


@router.get("/api/jobs/{job_id}")
async def get_job(request: Request, job_id: str) -> dict[str, object]:
    return _manager(request).get(job_id).public_snapshot()


def _download_response(
    manager: JobManager, job_id: str, format_name: Literal["txt", "srt"]
) -> StreamingResponse:
    lease = manager.open_result(job_id, format_name)
    filename = f"transcript.{format_name}"

    async def content():
        try:
            while chunk := lease.read(64 * 1024):
                yield chunk
        finally:
            lease.close()

    return ResultStreamingResponse(
        content(),
        lease=lease,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/api/jobs/{job_id}/transcript.txt")
async def download_text(request: Request, job_id: str) -> StreamingResponse:
    return _download_response(_manager(request), job_id, "txt")


@router.get("/api/jobs/{job_id}/transcript.srt")
async def download_srt(request: Request, job_id: str) -> StreamingResponse:
    return _download_response(_manager(request), job_id, "srt")
