class AppError(Exception):
    def __init__(self, message: str, *, code: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class ConfigError(RuntimeError):
    """Raised when startup configuration cannot run on this installation."""


class ValidationError(AppError):
    def __init__(self, message: str, *, code: str = "invalid_input") -> None:
        super().__init__(message, code=code, status_code=422)


class BusyError(AppError):
    def __init__(self) -> None:
        super().__init__("A transcription is already running.", code="job_busy", status_code=409)


class NotFoundError(AppError):
    def __init__(self, message: str = "Job not found.") -> None:
        super().__init__(message, code="not_found", status_code=404)
