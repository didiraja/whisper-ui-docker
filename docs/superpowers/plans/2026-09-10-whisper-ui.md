# Whisper UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Dockerized local web application that transcribes one public YouTube video or MP3/WAV upload with WhisperX and returns copyable plain text plus TXT and SRT downloads.

**Architecture:** A single-worker FastAPI application serves a Jinja page and JSON API. An in-memory `JobManager` admits one job at a time and dispatches the blocking yt-dlp, FFmpeg, and WhisperX pipeline to a one-thread executor; only model caches survive container restarts.

**Tech Stack:** Python 3.12, FastAPI 0.141.1, Uvicorn 0.52.4, Jinja2 3.1.6, Pydantic Settings 2.15.0, WhisperX 3.8.6 with PyTorch 2.8/CUDA 12.8 dependencies, yt-dlp 2026.8.19, FFmpeg/ffprobe, Node.js for yt-dlp EJS, vanilla JavaScript, Docker Compose, and uv.

**Spec:** `docs/superpowers/specs/2026-09-10-whisper-ui-design.md`

## Global Constraints

- This is a local, single-user application with no authentication, database, persistent job history, or multi-user queue.
- Accept exactly one public single-video YouTube URL or one MP3/WAV upload per job.
- Default upload limit: 500 MB; default duration limit: 10,800 seconds; both are configurable.
- Model choices are exactly `tiny`, `base`, `small`, `medium`, and `large-v3`; default model is `small`.
- Language defaults to Portuguese (Brazil), represented by Whisper language code `pt`; Auto-detect remains available as an empty language value.
- Download a transcription or alignment model only when a job first needs it; persist all model caches under `/models`.
- Use CUDA with `float16` when visible to PyTorch, otherwise CPU with `int8`; allow explicit environment overrides.
- Return continuous UTF-8 plain text and SubRip subtitles. If phoneme alignment fails, use Whisper segment timestamps and expose a warning.
- Run exactly one Uvicorn worker and one transcription executor thread.
- Bind Docker's published port to `127.0.0.1` by default.
- Do not add automated tests, test dependencies, test directories, or test configuration. Use only the manual verification commands and checks in this plan.
- The workspace currently contains an empty read-only `.git` mount. Do not alter or unmount it. Commit checkpoints below are deferred until the workspace is provided with writable Git metadata.

---

## Planned File Structure

```text
.
├── .dockerignore                         # Docker build exclusions
├── .gitignore                            # Python, cache, and local-runtime exclusions
├── .python-version                       # Python 3.12 selection for uv
├── Dockerfile                            # One CPU/GPU-capable runtime image
├── README.md                             # Startup, limits, cache, and smoke-check instructions
├── compose.gpu.yaml                      # NVIDIA GPU exposure override
├── compose.yaml                          # Local CPU/default service and model volume
├── pyproject.toml                        # Package metadata and pinned direct dependencies
├── uv.lock                               # Fully resolved dependency lock
└── src/whisper_ui/
    ├── __init__.py                       # Package marker
    ├── config.py                         # Environment-backed settings and allow-lists
    ├── domain.py                         # Job enums, request, record, and public snapshot
    ├── errors.py                         # Safe application error hierarchy
    ├── jobs.py                           # Single-job admission and pipeline orchestration
    ├── main.py                           # FastAPI factory, lifespan, and periodic cleanup
    ├── media.py                          # ffprobe validation and FFmpeg normalization
    ├── outputs.py                        # Plain-text and SRT rendering
    ├── uploads.py                        # Size-limited UploadFile persistence
    ├── web.py                            # Page, health, job, status, and download routes
    ├── whisper_service.py                # Device choice, lazy model reuse, ASR, and alignment
    ├── youtube.py                        # YouTube URL validation and yt-dlp download
    ├── static/
    │   ├── app.js                        # Submission, polling, copy, and result behavior
    │   └── styles.css                    # Responsive single-page presentation
    └── templates/
        └── index.html                    # Form, progress, errors, and result markup
```

## Shared Interfaces

These signatures are fixed across tasks:

```python
# src/whisper_ui/domain.py
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
```

---

### Task 1: Reproducible Python and Container Foundation

**Files:**
- Create: `.python-version`
- Create: `.gitignore`
- Create: `.dockerignore`
- Create: `pyproject.toml`
- Create: `uv.lock`
- Create: `Dockerfile`
- Create: `compose.yaml`
- Create: `compose.gpu.yaml`
- Create: `src/whisper_ui/__init__.py`

**Interfaces:**
- Consumes: none.
- Produces: importable `whisper_ui` package; Python 3.12 uv environment; service listening on container port 8000; persistent `/models` volume; optional NVIDIA GPU exposure.

- [ ] **Step 1: Add package metadata and pinned direct dependencies**

Create `.python-version` containing `3.12`. Create `pyproject.toml` with this project definition:

```toml
[project]
name = "whisper-ui"
version = "0.1.0"
description = "Local WhisperX transcription web UI"
requires-python = ">=3.12,<3.13"
dependencies = [
  "fastapi==0.141.1",
  "jinja2==3.1.6",
  "pydantic-settings==2.15.0",
  "python-multipart==0.0.32",
  "uvicorn[standard]==0.52.4",
  "whisperx==3.8.6",
  "yt-dlp[default,curl-cffi]==2026.8.19",
]

[build-system]
requires = ["setuptools>=80"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.package-data]
whisper_ui = ["templates/*.html", "static/*.css", "static/*.js"]

[tool.setuptools.packages.find]
where = ["src"]

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch" }
torchaudio = { index = "pytorch" }
torchvision = { index = "pytorch" }
triton = { index = "pytorch", marker = "sys_platform == 'linux' and platform_machine == 'x86_64'" }
```

- [ ] **Step 2: Resolve and inspect the dependency lock**

Run:

```bash
uv lock
uv tree --depth 1
```

Expected: resolution succeeds on Python 3.12, `whisperx==3.8.6` selects `torch~=2.8.0`, and `uv.lock` is created. If uv reports a genuine version conflict, keep WhisperX at 3.8.6 and select the newest mutually compatible version of the conflicting direct web dependency; record the resulting exact pin in `pyproject.toml` before regenerating the lock.

- [ ] **Step 3: Add ignore rules**

Use these entries:

```gitignore
# .gitignore
__pycache__/
*.py[cod]
.venv/
.env
.mypy_cache/
.ruff_cache/
.DS_Store
```

```dockerignore
# .dockerignore
.git
.venv
__pycache__
*.pyc
.env
docs
```

- [ ] **Step 4: Build the runtime image definition**

Create `Dockerfile` from `python:3.12-slim-bookworm`, copy uv 0.11.18 from its official image, install `ffmpeg`, `nodejs`, and `ca-certificates`, install the frozen dependency set before copying source, then run one Uvicorn worker:

```dockerfile
FROM ghcr.io/astral-sh/uv:0.11.18 AS uv
FROM python:3.12-slim-bookworm

COPY --from=uv /uv /uvx /bin/
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch

EXPOSE 8000
CMD ["uvicorn", "whisper_ui.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

- [ ] **Step 5: Add CPU/default and GPU Compose definitions**

Create the default service with localhost-only publishing, cache volume, configuration defaults, and a health check:

```yaml
# compose.yaml
services:
  app:
    build: .
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      MAX_UPLOAD_MB: "500"
      MAX_MEDIA_DURATION_SECONDS: "10800"
      RESULT_RETENTION_MINUTES: "60"
      DEFAULT_MODEL: "small"
      DEFAULT_LANGUAGE: "pt"
      MODEL_CACHE_DIR: "/models"
      DEVICE: "auto"
    volumes:
      - whisper-model-cache:/models
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 20s

volumes:
  whisper-model-cache:
```

```yaml
# compose.gpu.yaml
services:
  app:
    gpus: all
```

- [ ] **Step 6: Perform static foundation checks**

Run:

```bash
uv run python -c "import fastapi, whisperx, yt_dlp; print('dependencies import')"
docker compose config --quiet
docker compose -f compose.yaml -f compose.gpu.yaml config --quiet
```

Expected: both Compose configurations parse and imports print `dependencies import`. These checks may download packages but must not download a Whisper model.

- [ ] **Step 7: Record the deferred commit checkpoint**

Once writable Git metadata exists, commit these files with:

```bash
git add .python-version .gitignore .dockerignore pyproject.toml uv.lock Dockerfile compose.yaml compose.gpu.yaml src/whisper_ui/__init__.py
git commit -m "build: add Whisper UI runtime foundation"
```

---

### Task 2: Configuration, Domain Types, and Safe Errors

**Files:**
- Create: `src/whisper_ui/config.py`
- Create: `src/whisper_ui/domain.py`
- Create: `src/whisper_ui/errors.py`

**Interfaces:**
- Consumes: `pydantic_settings.BaseSettings`.
- Produces: `Settings`, `settings`, `MODELS`, `JobState`, `JobStage`, `JobRequest`, `JobRecord`, `MediaInfo`, `TranscriptResult`, `AppError`, `BusyError`, `NotFoundError`, and `ValidationError`.

- [ ] **Step 1: Implement validated settings and fixed model choices**

Define these settings and derive byte/duration paths without reading environment variables elsewhere:

```python
# src/whisper_ui/config.py
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MODELS = ("tiny", "base", "small", "medium", "large-v3")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    max_upload_mb: int = Field(default=500, gt=0)
    max_media_duration_seconds: int = Field(default=10_800, gt=0)
    result_retention_minutes: int = Field(default=60, gt=0)
    default_model: str = "small"
    default_language: str = "pt"
    model_cache_dir: Path = Path("/models")
    jobs_dir: Path = Path("/tmp/whisper-ui")
    device: Literal["auto", "cpu", "cuda"] = "auto"
    cpu_compute_type: str = "int8"
    cuda_compute_type: str = "float16"
    cpu_batch_size: int = Field(default=4, gt=0)
    cuda_batch_size: int = Field(default=16, gt=0)
    cleanup_interval_seconds: int = Field(default=60, gt=0)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @model_validator(mode="after")
    def validate_defaults(self) -> "Settings":
        if self.default_model not in MODELS:
            raise ValueError(f"DEFAULT_MODEL must be one of {', '.join(MODELS)}")
        return self

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 2: Implement the shared domain types**

Create the enums and dataclasses shown in **Shared Interfaces**. Implement `JobRecord.public_snapshot()` so it returns only JSON-safe public fields:

```python
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
```

Use timezone-aware UTC datetimes (`datetime.now(timezone.utc)`) for `created_at` and expiration rather than the abbreviated constructor shown in the interface sketch.

- [ ] **Step 3: Implement safe application errors**

Create one error with explicit HTTP status and subclasses used by routes and services:

```python
# src/whisper_ui/errors.py
class AppError(Exception):
    def __init__(self, message: str, *, code: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code

class ValidationError(AppError):
    def __init__(self, message: str, *, code: str = "invalid_input") -> None:
        super().__init__(message, code=code, status_code=422)

class BusyError(AppError):
    def __init__(self) -> None:
        super().__init__("A transcription is already running.", code="job_busy", status_code=409)

class NotFoundError(AppError):
    def __init__(self, message: str = "Job not found.") -> None:
        super().__init__(message, code="not_found", status_code=404)
```

- [ ] **Step 4: Check imports and configuration validation manually**

Run:

```bash
uv run python -c "from whisper_ui.config import get_settings; s=get_settings(); assert s.max_upload_bytes == 524288000; assert s.default_language == 'pt'; print('config valid')"
uv run python -c "from whisper_ui.domain import JobState, JobStage; print(JobState.ACCEPTED, JobStage.TRANSCRIBING)"
```

Expected: `config valid` and both enum values print without import errors.

- [ ] **Step 5: Record the deferred commit checkpoint**

```bash
git add src/whisper_ui/config.py src/whisper_ui/domain.py src/whisper_ui/errors.py
git commit -m "feat: add configuration and job domain"
```

---

### Task 3: Upload and YouTube Media Ingestion

**Files:**
- Create: `src/whisper_ui/uploads.py`
- Create: `src/whisper_ui/youtube.py`
- Create: `src/whisper_ui/media.py`

**Interfaces:**
- Consumes: `Settings`, `MediaInfo`, `ValidationError`, `JobStage`, and `ProgressCallback`.
- Produces: `save_upload(upload: UploadFile, destination: Path, max_bytes: int) -> Path`; `validate_youtube_url(url: str) -> str`; `download_youtube(url: str, directory: Path, max_duration_seconds: int, progress: ProgressCallback) -> tuple[Path, str]`; `probe_media(path: Path) -> MediaInfo`; `enforce_media(path: Path, max_duration_seconds: int) -> MediaInfo`; `normalize_audio(source: Path, destination: Path) -> Path`.

- [ ] **Step 1: Stream uploads with an enforced byte ceiling**

In `uploads.py`, accept only `.mp3` and `.wav`, write 1 MiB chunks, delete the partial destination on failure, and reject content above `max_bytes`:

```python
ALLOWED_SUFFIXES = {".mp3", ".wav"}
CHUNK_SIZE = 1024 * 1024

async def save_upload(upload: UploadFile, destination: Path, max_bytes: int) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValidationError("Only MP3 and WAV files are supported.", code="unsupported_file")
    target = destination.with_suffix(suffix)
    written = 0
    try:
        with target.open("wb") as output:
            while chunk := await upload.read(CHUNK_SIZE):
                written += len(chunk)
                if written > max_bytes:
                    raise ValidationError("The upload exceeds the 500 MB limit.", code="upload_too_large")
                output.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return target
```

Pass the configured size into the error message so non-default limits report their actual value instead of hard-coding `500 MB` in the final implementation.

- [ ] **Step 2: Probe and normalize media without shell execution**

In `media.py`, call `subprocess.run(command, shell=False, check=True, capture_output=True, text=True)` with the exact argument arrays in this step. `probe_media()` must run:

```python
[
    "ffprobe", "-v", "error", "-show_entries", "format=duration",
    "-show_streams", "-select_streams", "a", "-of", "json", str(path),
]
```

Parse JSON into `MediaInfo(duration_seconds=float(payload["format"]["duration"]), has_audio=bool(payload["streams"]))`. Convert missing audio, unreadable JSON, a non-positive duration, and ffprobe failure to safe `ValidationError` messages. `enforce_media()` rejects duration above `max_duration_seconds`. `normalize_audio()` runs:

```python
[
    "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(source),
    "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination),
]
```

Map `CalledProcessError` to `ValidationError("The audio could not be prepared.", code="media_processing_failed")` and never return captured stderr to the browser.

- [ ] **Step 3: Restrict YouTube URLs**

Implement normalized parsing with this allow-list:

```python
YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
    "youtu.be", "www.youtu.be",
}

def validate_youtube_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme not in {"http", "https"} or host not in YOUTUBE_HOSTS:
        raise ValidationError("Enter a valid YouTube video URL.", code="invalid_youtube_url")
    if parsed.username or parsed.password:
        raise ValidationError("Authenticated URLs are not supported.", code="invalid_youtube_url")
    return parsed.geturl()
```

Reject channel, playlist-only, and non-video paths before yt-dlp where obvious; yt-dlp's extracted metadata remains the authority for ambiguous URLs.

- [ ] **Step 4: Download one public audio source through yt-dlp**

Use yt-dlp's Python API twice: metadata-only preflight, then download. Configure `noplaylist=True`, `playlist_items="1"`, `format="bestaudio/best"`, `outtmpl=str(directory / "youtube.%(ext)s")`, `quiet=True`, `no_warnings=True`, and `js_runtimes={"node": {}}`. Reject metadata containing `entries`, `is_live=True`, inaccessible availability, or duration over the configured maximum. After download, resolve the path with `YoutubeDL.prepare_filename(info)`, require it to exist, and call `enforce_media()` again.

Wrap `yt_dlp.utils.DownloadError` as:

```python
raise ValidationError(
    "The public YouTube video could not be downloaded.",
    code="youtube_download_failed",
) from error
```

Return `(downloaded_path, sanitized_title)` and limit the display title to 160 characters. Emit `JobStage.DOWNLOADING` through the callback before network work.

- [ ] **Step 5: Exercise local media helpers manually**

Create ephemeral files outside the repository and run the helpers:

```bash
ffmpeg -f lavfi -i sine=frequency=1000:duration=1 -y /tmp/whisper-ui-check.wav
uv run python -c "from pathlib import Path; from whisper_ui.media import enforce_media, normalize_audio; p=Path('/tmp/whisper-ui-check.wav'); print(enforce_media(p, 10800)); normalize_audio(p, Path('/tmp/whisper-ui-normalized.wav'))"
ffprobe -v error -show_entries stream=codec_name,sample_rate,channels -of json /tmp/whisper-ui-normalized.wav
```

Expected: the first command creates one second of audio, the helper reports an audio stream, and the normalized file reports PCM audio at 16 kHz with one channel.

- [ ] **Step 6: Record the deferred commit checkpoint**

```bash
git add src/whisper_ui/uploads.py src/whisper_ui/youtube.py src/whisper_ui/media.py
git commit -m "feat: add validated media ingestion"
```

---

### Task 4: WhisperX Transcription and Output Formatting

**Files:**
- Create: `src/whisper_ui/whisper_service.py`
- Create: `src/whisper_ui/outputs.py`

**Interfaces:**
- Consumes: `Settings`, `TranscriptResult`, `JobStage`, and `ProgressCallback`.
- Produces: `WhisperService.transcribe(audio_path: Path, model_name: str, language: str | None, progress: ProgressCallback) -> TranscriptResult`; `render_plain_text(segments: list[dict[str, Any]]) -> str`; `render_srt(segments: list[dict[str, Any]]) -> str`; `write_outputs(directory: Path, segments: list[dict[str, Any]]) -> tuple[str, Path, Path]`.

- [ ] **Step 1: Implement deterministic device selection**

In `WhisperService.__init__`, resolve settings as follows:

```python
if settings.device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("DEVICE=cuda was requested but CUDA is unavailable")
self.device = (
    settings.device
    if settings.device != "auto"
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
self.compute_type = (
    settings.cuda_compute_type if self.device == "cuda" else settings.cpu_compute_type
)
self.batch_size = (
    settings.cuda_batch_size if self.device == "cuda" else settings.cpu_batch_size
)
```

Create `/models/whisper`, `/models/alignment`, `/models/huggingface`, and `/models/torch` beneath `settings.model_cache_dir` during service initialization.

- [ ] **Step 2: Lazily load and reuse exactly one ASR model**

Track `_model_name: str | None` and `_model: Any | None`. `_ensure_model(model_name)` must validate against `MODELS`, return the loaded model when the name matches, and otherwise release the old model, run `gc.collect()`, call `torch.cuda.empty_cache()` when CUDA is available, and load:

```python
whisperx.load_model(
    model_name,
    self.device,
    compute_type=self.compute_type,
    download_root=str(self.settings.model_cache_dir / "whisper"),
)
```

No constructor or application-startup path may call `_ensure_model`.

- [ ] **Step 3: Transcribe and align with a timestamp fallback**

`transcribe()` emits `LOADING_MODEL`, loads audio with `whisperx.load_audio`, emits `TRANSCRIBING`, and calls:

```python
result = model.transcribe(
    audio,
    batch_size=self.batch_size,
    language=language,
)
```

Use `result["language"]` as the detected/effective language. Emit `ALIGNING`, then perform alignment with these exact calls:

```python
align_model, metadata = whisperx.load_align_model(
    language_code=effective_language,
    device=self.device,
    model_dir=str(alignment_cache),
)
aligned = whisperx.align(
    result["segments"],
    align_model,
    metadata,
    audio,
    self.device,
    return_char_alignments=False,
)
```

Always release the alignment model in `finally` and clear unused CUDA memory. Catch alignment exceptions only around the alignment phase, log the exception, retain the original ASR segments, and return this warning:

```text
Precise alignment was unavailable; SRT timing uses Whisper segment timestamps.
```

Do not convert ASR or model-download failures into alignment warnings; allow those failures to mark the job failed.

- [ ] **Step 4: Render continuous text and valid SubRip**

Normalize whitespace for plain text:

```python
def render_plain_text(segments: list[dict[str, Any]]) -> str:
    return " ".join(
        str(segment.get("text", "")).strip()
        for segment in segments
        if str(segment.get("text", "")).strip()
    )
```

Format timestamps with millisecond carry handling and render one block per segment containing numeric `start`, numeric `end`, and non-empty text:

```text
1
00:00:00,000 --> 00:00:02,450
First subtitle text.

2
00:00:02,450 --> 00:00:05,120
Second subtitle text.
```

Clamp negative starts to zero and ensure each end is at least its start. `write_outputs()` writes `transcript.txt` and `transcript.srt` as UTF-8 with a final newline and returns `(plain_text, text_path, srt_path)`.

- [ ] **Step 5: Check formatting and syntax without loading a model**

Run:

```bash
uv run python -m compileall -q src/whisper_ui
uv run python -c "from whisper_ui.outputs import render_plain_text, render_srt; s=[{'start':0.0,'end':1.25,'text':' Olá mundo '}]; assert render_plain_text(s)=='Olá mundo'; assert '00:00:01,250' in render_srt(s); print('outputs valid')"
```

Expected: `outputs valid`. This check must not create `/models` content or download weights.

- [ ] **Step 6: Record the deferred commit checkpoint**

```bash
git add src/whisper_ui/whisper_service.py src/whisper_ui/outputs.py
git commit -m "feat: add WhisperX transcription pipeline"
```

---

### Task 5: Single-Job Manager and Application Lifespan

**Files:**
- Create: `src/whisper_ui/jobs.py`
- Create: `src/whisper_ui/main.py`

**Interfaces:**
- Consumes: every service interface from Tasks 2-4.
- Produces: `JobManager.reserve(request: JobRequest, display_name: str) -> JobRecord`; `JobManager.set_request(job_id: str, request: JobRequest) -> None`; `JobManager.start(job_id: str) -> None`; `JobManager.fail_reserved(job_id: str, message: str) -> None`; `JobManager.get(job_id: str) -> JobRecord`; `JobManager.current() -> JobRecord | None`; `JobManager.result_path(job_id: str, format_name: Literal["txt", "srt"]) -> Path`; `JobManager.purge_expired() -> None`; `JobManager.shutdown() -> None`; FastAPI `app`.

- [ ] **Step 1: Implement atomic single-job admission**

`JobManager` owns one `threading.Lock`, one `ThreadPoolExecutor(max_workers=1, thread_name_prefix="transcription")`, one current `JobRecord`, and injected media/transcription services. `reserve()` must, while holding the lock:

1. Reject when the current record is `ACCEPTED` or `RUNNING`.
2. Remove an existing terminal job directory before replacing it.
3. Create a UUID job ID and `settings.jobs_dir / job_id` with mode `0o700`.
4. Store and return an `ACCEPTED` record.

Never hold the lock during file download, FFmpeg, or WhisperX work. `set_request()` replaces only the request on the matching `ACCEPTED` job and raises `NotFoundError` for another identifier or state.

- [ ] **Step 2: Dispatch the pipeline and publish stages safely**

`start(job_id)` submits `_run(job_id)` to the executor. `_set_stage(job_id, stage)` updates the matching current record under the lock. `_run` performs:

```python
source = request.source_path
if request.source_kind == "youtube":
    source, title = download_youtube(
        request.source_url or "", record.directory,
        self.settings.max_media_duration_seconds, progress,
    )
    self._set_display_name(job_id, title)
else:
    progress(JobStage.VALIDATING)
    enforce_media(source, self.settings.max_media_duration_seconds)

progress(JobStage.PREPARING)
normalized = normalize_audio(source, record.directory / "normalized.wav")
transcript = self.whisper.transcribe(
    normalized, request.model, request.language, progress,
)
progress(JobStage.FORMATTING)
plain, text_path, srt_path = write_outputs(record.directory, transcript.segments)
self._complete(job_id, transcript, plain, text_path, srt_path)
```

Validate non-optional path/URL fields before service calls. `_complete()` sets state `COMPLETED`, stage `COMPLETE`, result fields, and an expiration of current UTC plus the retention interval.

- [ ] **Step 3: Convert pipeline failures into safe terminal state**

Catch `AppError` and store its public message. For every other exception, call `logger.exception("Job %s failed", job_id)` and store `Transcription failed. Check the container logs for details.` Never put exception representations in `JobRecord.error`. Set `FAILED`, preserve the last meaningful stage, and set the same expiration deadline as successful jobs. Delete source and normalized media after terminal state while retaining only TXT/SRT results for a successful job.

- [ ] **Step 4: Implement lookup, downloads, cleanup, and shutdown**

`get()` and `result_path()` reject unknown, expired, unfinished, or missing-file jobs with `NotFoundError`. `purge_expired()` removes an expired terminal job directory with `shutil.rmtree` and clears the current pointer under the lock. `shutdown()` rejects new admission and calls `executor.shutdown(wait=False, cancel_futures=True)`.

- [ ] **Step 5: Wire app lifespan and periodic cleanup**

In `main.py`, use an `asynccontextmanager` lifespan that creates settings, the Whisper service, and the manager, saves them on `app.state`, and starts this cleanup loop:

```python
async def cleanup_loop(manager: JobManager, interval: int) -> None:
    while True:
        await asyncio.sleep(interval)
        manager.purge_expired()
```

On shutdown, cancel and await the cleanup task while suppressing `asyncio.CancelledError`, then call `manager.shutdown()`. Build `app = FastAPI(title="Whisper UI", lifespan=lifespan)`, mount `/static`, include the web router from Task 6, and register the `AppError` JSON handler.

- [ ] **Step 6: Check application construction**

Run:

```bash
JOBS_DIR=/tmp/whisper-ui-plan-check MODEL_CACHE_DIR=/tmp/whisper-ui-model-check uv run python -c "from whisper_ui.main import app; print(app.title)"
```

Expected: `Whisper UI`; no model files are downloaded.

- [ ] **Step 7: Record the deferred commit checkpoint**

```bash
git add src/whisper_ui/jobs.py src/whisper_ui/main.py
git commit -m "feat: orchestrate one transcription job"
```

---

### Task 6: Web API and Browser Interface

**Files:**
- Create: `src/whisper_ui/web.py`
- Create: `src/whisper_ui/templates/index.html`
- Create: `src/whisper_ui/static/app.js`
- Create: `src/whisper_ui/static/styles.css`

**Interfaces:**
- Consumes: `Settings`, `MODELS`, `JobRequest`, `JobManager`, `save_upload`, `validate_youtube_url`, and WhisperX `LANGUAGES`.
- Produces: fixed routes `GET /`, `GET /health`, `POST /api/jobs`, `GET /api/jobs/current`, `GET /api/jobs/{job_id}`, `GET /api/jobs/{job_id}/transcript.txt`, and `GET /api/jobs/{job_id}/transcript.srt`; DOM IDs consumed by `app.js`.

- [ ] **Step 1: Implement page, health, and safe error responses**

Create an `APIRouter`, `Jinja2Templates`, and helpers that read `request.app.state.job_manager` and settings. Render `/` with:

```python
{
    "request": request,
    "models": MODELS,
    "default_model": settings.default_model,
    "languages": sorted(LANGUAGES.items(), key=lambda item: item[1]),
    "default_language": settings.default_language,
    "max_upload_mb": settings.max_upload_mb,
    "max_duration_hours": settings.max_media_duration_seconds / 3600,
}
```

Return health JSON containing `status: "ok"` and the resolved transcription device. Register `AppError` handling in `main.py` with the stable body `{"error": {"code": exc.code, "message": exc.message}}`.

- [ ] **Step 2: Implement multipart job submission**

Accept fields `youtube_url`, `audio_file`, `model`, and `language`. Normalize blank strings to `None`; require exactly one source; validate model against `MODELS`; validate non-empty language against `LANGUAGES`. Use `None` for Auto-detect. For YouTube, validate the URL before reservation. For uploads, derive a safe display name with `Path(upload.filename).name`, reserve a record, and save to `record.directory / "source"` using `save_upload()`. The browser shows the receiving stage immediately while the multipart POST is pending.

If upload persistence fails after reservation, call `manager.fail_reserved(record.id, public_message)` and re-raise the safe error. Replace the frozen request's `source_path` using `dataclasses.replace`, update it through `manager.set_request(job_id, request)`, then call `manager.start(job_id)`. Return the snapshot with HTTP 202.

- [ ] **Step 3: Implement status and result downloads**

`GET /api/jobs/current` returns HTTP 204 when no job exists, otherwise the public snapshot. `GET /api/jobs/{job_id}` returns the snapshot. Both download routes use `FileResponse` with media type `text/plain; charset=utf-8`, filename `transcript.txt` or `transcript.srt`, and a path returned only by `manager.result_path()`.

- [ ] **Step 4: Build accessible HTML with a stable DOM contract**

The template must include these IDs used by JavaScript:

```text
source-youtube, source-upload, youtube-panel, upload-panel,
youtube-url, audio-file, model, language, transcribe,
form-error, job-panel, job-stage, job-warning, result-panel,
transcript, copy-transcript, download-txt, download-srt
```

Use radio-backed source tabs, one `<form id="transcription-form">`, a model select, and a language select whose first option is Auto-detect and whose `pt` option is selected by default. Display `MP3 or WAV, up to 500 MB and 3 hours` from template values. Keep the result panel hidden until completion and use `aria-live="polite"` for stage/error text.

Use this document structure, expanding only labels and non-functional presentation markup:

```html
<main class="shell">
  <header><h1>Whisper UI</h1><p>Local transcription with WhisperX</p></header>
  <form id="transcription-form" class="card">
    <fieldset class="source-tabs">
      <legend>Source</legend>
      <label><input id="source-youtube" type="radio" name="source_kind" value="youtube" checked> YouTube</label>
      <label><input id="source-upload" type="radio" name="source_kind" value="upload"> Audio file</label>
    </fieldset>
    <div id="youtube-panel"><label for="youtube-url">YouTube URL</label><input id="youtube-url" name="youtube_url" type="url"></div>
    <div id="upload-panel" hidden><label for="audio-file">MP3 or WAV</label><input id="audio-file" name="audio_file" type="file" accept=".mp3,.wav,audio/mpeg,audio/wav" disabled></div>
    <label for="model">Model</label>
    <select id="model" name="model">{% for item in models %}<option value="{{ item }}" {% if item == default_model %}selected{% endif %}>{{ item }}</option>{% endfor %}</select>
    <label for="language">Language</label>
    <select id="language" name="language">
      <option value="">Auto-detect</option>
      {% for code, name in languages %}<option value="{{ code }}" {% if code == default_language %}selected{% endif %}>{{ "Portuguese (Brazil)" if code == "pt" else name|title }}</option>{% endfor %}
    </select>
    <button id="transcribe" type="submit">Transcribe</button>
  </form>
  <p id="form-error" class="alert" role="alert" hidden></p>
  <section id="job-panel" class="card" aria-live="polite" hidden><p id="job-stage"></p><p id="job-warning"></p></section>
  <section id="result-panel" class="card" hidden>
    <label for="transcript">Transcript</label><textarea id="transcript" readonly></textarea>
    <div class="actions"><button id="copy-transcript" type="button">Copy</button><a id="download-txt" role="button">Download TXT</a><a id="download-srt" role="button">Download SRT</a></div>
  </section>
</main>
```

- [ ] **Step 5: Implement submission, polling, refresh recovery, and clipboard behavior**

In `app.js`, define one `pollTimer` and these functions with matching responsibilities:

```javascript
const form = document.querySelector("#transcription-form");
const sourceYoutube = document.querySelector("#source-youtube");
const sourceUpload = document.querySelector("#source-upload");
const youtubePanel = document.querySelector("#youtube-panel");
const uploadPanel = document.querySelector("#upload-panel");
const youtubeUrl = document.querySelector("#youtube-url");
const audioFile = document.querySelector("#audio-file");
const formError = document.querySelector("#form-error");
const jobPanel = document.querySelector("#job-panel");
const jobStage = document.querySelector("#job-stage");
const jobWarning = document.querySelector("#job-warning");
const resultPanel = document.querySelector("#result-panel");
const transcript = document.querySelector("#transcript");
const copyButton = document.querySelector("#copy-transcript");
const downloadTxt = document.querySelector("#download-txt");
const downloadSrt = document.querySelector("#download-srt");
let pollTimer = null;

const stageLabels = {
  validating: "Validating input…",
  receiving: "Receiving upload…",
  downloading: "Downloading YouTube audio…",
  preparing: "Preparing audio…",
  loading_model: "Downloading or loading model…",
  transcribing: "Transcribing…",
  aligning: "Aligning subtitles…",
  formatting: "Preparing downloads…",
  complete: "Transcription complete",
};

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  if (response.status === 204) return null;
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error?.message || "Request failed.");
  return body;
}

function showError(message) {
  formError.textContent = message || "Something went wrong.";
  formError.hidden = false;
}

function clearError() {
  formError.textContent = "";
  formError.hidden = true;
}

function showPollingError(error) {
  showError(error.message || "Status could not be refreshed.");
  setBusy(false);
}

async function submitJob(event) {
  event.preventDefault();
  clearError();
  resultPanel.hidden = true;
  jobPanel.hidden = false;
  const payload = new FormData(form);
  setBusy(true);
  jobStage.textContent = sourceUpload.checked ? "Receiving upload…" : "Submitting URL…";
  try {
    const job = await requestJson("/api/jobs", { method: "POST", body: payload });
    renderJob(job);
    await pollJob(job.id);
  } catch (error) {
    showError(error.message);
    setBusy(false);
  }
}

async function pollJob(jobId) {
  const job = await requestJson(`/api/jobs/${jobId}`);
  renderJob(job);
  if (job.state === "accepted" || job.state === "running") {
    pollTimer = window.setTimeout(() => pollJob(jobId).catch(showPollingError), 1500);
  }
}

function renderJob(job) {
  jobPanel.hidden = false;
  jobStage.textContent = stageLabels[job.stage] || job.stage;
  jobWarning.textContent = job.warning || "";
  if (job.state === "completed") {
    transcript.value = job.transcript;
    downloadTxt.href = job.downloads.txt;
    downloadSrt.href = job.downloads.srt;
    resultPanel.hidden = false;
    setBusy(false);
  } else if (job.state === "failed") {
    showError(job.error);
    setBusy(false);
  }
}

function setBusy(isBusy) {
  for (const control of form.elements) control.disabled = isBusy;
  if (!isBusy) updateSourcePanels();
}

function updateSourcePanels() {
  const uploadSelected = sourceUpload.checked;
  uploadPanel.hidden = !uploadSelected;
  youtubePanel.hidden = uploadSelected;
  audioFile.disabled = !uploadSelected;
  youtubeUrl.disabled = uploadSelected;
  if (uploadSelected) youtubeUrl.value = "";
  else audioFile.value = "";
}

async function recoverCurrentJob() {
  const job = await requestJson("/api/jobs/current");
  if (!job) return;
  renderJob(job);
  if (job.state === "accepted" || job.state === "running") await pollJob(job.id);
}

async function copyTranscript() {
  try {
    await navigator.clipboard.writeText(transcript.value);
  } catch {
    transcript.select();
    document.execCommand("copy");
  }
}

sourceYoutube.addEventListener("change", updateSourcePanels);
sourceUpload.addEventListener("change", updateSourcePanels);
form.addEventListener("submit", submitJob);
copyButton.addEventListener("click", copyTranscript);
window.addEventListener("DOMContentLoaded", () => {
  updateSourcePanels();
  recoverCurrentJob().catch(showPollingError);
});
```

Parse non-2xx JSON errors and show `error.message`; use a generic network message only when no safe response exists. Stop polling on `completed` or `failed`. On completion, set download link `href` values from `job.downloads`. Source-tab changes must clear and disable the inactive source field so the submitted form always contains one source.

- [ ] **Step 6: Add responsive styling**

Create a centered, single-column layout with a maximum content width near 800 px, a distinct input card, obvious source tabs, visible focus states, a progress/status panel, an error alert, and a large read-only transcript area. Use CSS custom properties for colors and spacing, preserve readable contrast, stack controls on narrow screens, and keep action buttons at least 44 px tall. Do not add a CSS framework or external font request.

Start from these layout and accessibility rules, then style the same semantic classes without changing the DOM contract:

```css
:root { color-scheme: light; --space: 1rem; --radius: .75rem; --accent: #4f46e5; }
* { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, sans-serif; line-height: 1.5; background: #f5f7fb; color: #172033; }
.shell { width: min(100% - 2rem, 50rem); margin: 3rem auto; }
.card { margin-block: var(--space); padding: 1.25rem; border: 1px solid #d9deea; border-radius: var(--radius); background: #fff; }
label, legend { font-weight: 650; }
input, select, textarea, button, a[role="button"] { font: inherit; }
input, select, textarea { width: 100%; margin-block: .35rem 1rem; padding: .75rem; }
button, a[role="button"] { min-height: 44px; display: inline-flex; align-items: center; justify-content: center; padding: .65rem 1rem; border-radius: .5rem; }
button:focus-visible, a:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible { outline: 3px solid #a5b4fc; outline-offset: 2px; }
textarea { min-height: 18rem; resize: vertical; }
.actions { display: flex; flex-wrap: wrap; gap: .75rem; }
.alert { padding: .75rem 1rem; border-left: .3rem solid #b42318; background: #fee4e2; color: #7a271a; }
[hidden] { display: none !important; }
@media (max-width: 36rem) { .shell { margin-block: 1rem; } .actions > * { width: 100%; } }
```

- [ ] **Step 7: Check Python and JavaScript syntax and start the lightweight web surface**

Run:

```bash
uv run python -m compileall -q src/whisper_ui
node --check src/whisper_ui/static/app.js
JOBS_DIR=/tmp/whisper-ui-web-check MODEL_CACHE_DIR=/tmp/whisper-ui-model-check uv run uvicorn whisper_ui.main:app --host 127.0.0.1 --port 8000
```

While Uvicorn is running, check in another terminal:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ | grep -F "Portuguese (Brazil)"
```

Expected: health returns JSON with `status: ok`; the page contains the default-language label; no model download begins.

- [ ] **Step 8: Record the deferred commit checkpoint**

```bash
git add src/whisper_ui/web.py src/whisper_ui/templates/index.html src/whisper_ui/static/app.js src/whisper_ui/static/styles.css
git commit -m "feat: add transcription web interface"
```

---

### Task 7: Documentation and End-to-End Manual Verification

**Files:**
- Create: `README.md`
- Modify only if verification exposes a defect: files created in Tasks 1-6.

**Interfaces:**
- Consumes: complete application and Compose definitions.
- Produces: reproducible CPU/GPU startup instructions, configuration reference, cache/cleanup explanation, limitations, troubleshooting, and verified local workflow.

- [ ] **Step 1: Document the normal CPU startup path**

README sections must include Requirements, Quick Start, Using the App, Configuration, NVIDIA GPU, Model Cache, Limits and Privacy, Troubleshooting, and Manual Verification. The primary commands are:

```bash
docker compose build
docker compose up
```

Tell the user to open `http://127.0.0.1:8000`, that Portuguese (Brazil) and `small` are defaults, that first use of each model/alignment language downloads weights, and that CPU transcription can be slow.

- [ ] **Step 2: Document NVIDIA GPU startup and fallback semantics**

Require a compatible NVIDIA driver plus NVIDIA Container Toolkit, then provide:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build
docker compose -f compose.yaml -f compose.gpu.yaml logs app
```

Explain that the app chooses CUDA only when PyTorch can see it inside the container; `DEVICE=cuda` makes missing CUDA a startup error, while the default `DEVICE=auto` falls back to CPU.

- [ ] **Step 3: Document exact configuration and lifecycle behavior**

List every environment variable from `Settings`, its default, units, and effect. State that only MP3/WAV uploads and public single-video YouTube URLs are accepted, results expire after 60 minutes by default, page refresh recovers the in-memory current job, container restart loses job state/results, and the named `whisper-model-cache` volume survives container replacement. Include the explicit cache deletion command as an opt-in destructive operation:

```bash
docker compose down
docker volume rm whisper-ui_whisper-model-cache
```

Warn that this deletion forces future model downloads.

- [ ] **Step 4: Build and start the final CPU container**

Run:

```bash
docker compose config --quiet
docker compose build
docker compose up -d
docker compose ps
docker compose logs --tail=100 app
curl --fail http://127.0.0.1:8000/health
```

Expected: the service becomes healthy, reports CPU unless a GPU was explicitly exposed, and logs show no model download before a job is submitted.

- [ ] **Step 5: Manually verify upload and result behavior**

Using a user-provided short speech MP3 and WAV through the browser, verify each of these observable outcomes:

1. Portuguese (Brazil) and `small` are initially selected; Auto-detect can be selected.
2. Submitting disables the form and shows stage changes.
3. Completion shows continuous text; Copy places the same text on the clipboard.
4. TXT is UTF-8 plain text and SRT opens with sequential blocks and timestamps.
5. Refresh during and after a job restores its status/result.
6. A second submission while processing returns the busy message.
7. A renamed non-audio file and unsupported extension are rejected.

If no speech fixtures are available, report upload transcription as unverified rather than substituting generated tones, which cannot validate recognition output.

- [ ] **Step 6: Manually verify model caching and YouTube flow**

Submit a short public single-video YouTube URL with `tiny` to reduce download and CPU time. After completion, inspect the named volume and submit a second job with `tiny`:

```bash
docker compose exec app python -c "from pathlib import Path; print(sum(p.stat().st_size for p in Path('/models').rglob('*') if p.is_file()))"
docker compose logs --since=10m app
```

Expected: `/models` contains weight files; the second `tiny` job reuses them rather than fetching the ASR model again; TXT and SRT actions work for the YouTube result. Record YouTube as unverified if outbound network access is unavailable.

- [ ] **Step 7: Manually verify configured rejection paths**

Use the browser or `curl` to confirm that a non-YouTube remote URL, a playlist URL, simultaneous submission, an unsupported file, and an upload over the configured limit return safe messages with no traceback or internal path. Temporarily set `MAX_MEDIA_DURATION_SECONDS=1`, restart, submit a longer valid audio file, and confirm duration rejection; restore the value to `10800` afterward.

- [ ] **Step 8: Verify GPU mode when compatible hardware is available**

Run:

```bash
docker compose down
docker compose -f compose.yaml -f compose.gpu.yaml up -d
docker compose -f compose.yaml -f compose.gpu.yaml exec app python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
curl --fail http://127.0.0.1:8000/health
```

Expected on a configured NVIDIA host: `True` and the GPU name, with health reporting CUDA. If no compatible host is available, state explicitly in the handoff that GPU execution was not verified.

- [ ] **Step 9: Stop the app without removing the model volume**

Run:

```bash
docker compose down
docker volume ls | grep -F whisper-model-cache
```

Expected: containers and network stop; the model cache volume remains.

- [ ] **Step 10: Record the deferred final commit checkpoint**

```bash
git add README.md
git commit -m "docs: add local usage and verification guide"
```

Do not run any commit command until writable Git metadata is available.
