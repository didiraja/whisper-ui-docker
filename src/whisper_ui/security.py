from __future__ import annotations

from collections.abc import Callable
from typing import Any

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from whisper_ui.config import Settings

MULTIPART_OVERHEAD_BYTES = 1024 * 1024


class RequestBodyTooLarge(Exception):
    pass


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


class RequestSizeLimitMiddleware:
    """Cap the job request before form parsing."""

    def __init__(
        self, app: ASGIApp, settings_provider: Callable[[], Settings]
    ) -> None:
        self.app = app
        self.settings_provider = settings_provider

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        settings = self.settings_provider()

        is_job_submission = (
            scope.get("method") == "POST" and scope.get("path") == "/api/jobs"
        )
        if not is_job_submission:
            await self.app(scope, receive, send)
            return

        request_limit = settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES
        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                declared_length = -1
            if declared_length < 0:
                response = _error_response(
                    400, "invalid_input", "Submit a valid transcription request."
                )
                await response(scope, receive, send)
                return
            if declared_length > request_limit:
                await self._send_too_large(scope, receive, send, settings)
                return

        received = 0
        exceeded = False
        downstream_started = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > request_limit:
                    exceeded = True
                    raise RequestBodyTooLarge
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal downstream_started
            if exceeded:
                return
            if message["type"] == "http.response.start":
                downstream_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not exceeded:
                raise

        if exceeded and not downstream_started:
            await self._send_too_large(scope, receive, send, settings)

    @staticmethod
    async def _send_too_large(
        scope: Scope, receive: Receive, send: Send, settings: Settings
    ) -> None:
        response = _error_response(
            413,
            "upload_too_large",
            f"The upload exceeds the {settings.max_upload_mb:g} MB limit.",
        )
        await response(scope, receive, send)
