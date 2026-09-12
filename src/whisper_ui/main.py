import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from whisper_ui.config import get_settings
from whisper_ui.errors import AppError
from whisper_ui.jobs import JobManager
from whisper_ui.security import LocalRequestGuardMiddleware
from whisper_ui.web import router
from whisper_ui.whisper_service import WhisperService


async def cleanup_loop(manager: JobManager, interval: int) -> None:
    while True:
        await asyncio.sleep(interval)
        manager.purge_expired()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    whisper = WhisperService(settings)
    manager = JobManager(settings, whisper)
    app.state.settings = settings
    app.state.whisper_service = whisper
    app.state.job_manager = manager
    cleanup_task = asyncio.create_task(
        cleanup_loop(manager, settings.cleanup_interval_seconds)
    )
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            with suppress(asyncio.CancelledError):
                await cleanup_task
        finally:
            manager.shutdown()


app = FastAPI(title="Whisper UI", lifespan=lifespan)
app.add_middleware(LocalRequestGuardMiddleware, settings_provider=get_settings)
INVALID_REQUEST_MESSAGE = "Submit a valid transcription request."


def invalid_request_response(status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {"code": "invalid_input", "message": INVALID_REQUEST_MESSAGE}
        },
    )


@app.exception_handler(AppError)
async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    _request: Request, _exc: RequestValidationError
) -> JSONResponse:
    return invalid_request_response(status_code=422)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(
    request: Request, exc: StarletteHTTPException
) -> Response:
    if exc.status_code == 400:
        return invalid_request_response(status_code=400)
    return await http_exception_handler(request, exc)


static_directory = Path(__file__).parent / "static"
if static_directory.is_dir() and not static_directory.is_symlink():
    app.mount("/static", StaticFiles(directory=static_directory), name="static")
app.include_router(router)
