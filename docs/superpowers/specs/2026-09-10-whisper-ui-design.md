# Whisper UI Design

## Purpose

Build a simple, local, single-user web application that transcribes either one public YouTube video or one uploaded MP3/WAV file with WhisperX. The user can choose a common Whisper model and a language, monitor the current job, copy the finished plain-text transcript, and download TXT and SRT outputs.

The application is intentionally local and ephemeral. It has no accounts, database, multi-user scheduling, or transcription history. Model files persist in a Docker volume; source media and generated results expire after a cleanup period.

## Scope

### Included

- One public YouTube video URL per job.
- One MP3 or WAV upload per job, with a default 500 MB maximum.
- A default maximum media duration of three hours.
- Common model choices: `tiny`, `base`, `small`, `medium`, and `large-v3`.
- Portuguese (Brazil) as the default language, mapped to Whisper's `pt` language code.
- An auto-detect language option and the language choices supported by the installed WhisperX version.
- Automatic CUDA use when an NVIDIA GPU is exposed to the container, with CPU fallback.
- Persistent on-disk model caching; a model downloads only when first selected.
- Plain-text display, browser clipboard copying, TXT download, and SRT download.
- Stage-based progress and actionable error messages.
- Docker Compose entry points for CPU and NVIDIA GPU hosts.

### Excluded

- Private, age-restricted, or login-required YouTube content.
- Playlists, channels, and non-YouTube remote URLs.
- Upload formats other than MP3 and WAV.
- Speaker diarization.
- Persistent transcription history, accounts, authentication, multiple concurrent jobs, or distributed queues.
- Automated tests and test tooling. Verification is manual only, as requested.

## Architecture

Use one FastAPI application served by a single Uvicorn worker. FastAPI serves a Jinja-rendered page and static CSS/JavaScript as well as the JSON job API and result downloads. Small vanilla JavaScript submits jobs, polls status, updates the interface, copies text, and initiates downloads. There is no Node-based frontend build.

An in-memory job manager accepts at most one active job. It executes the blocking media and WhisperX pipeline through one dedicated executor so that the HTTP event loop stays responsive. A second submission while a job is active receives a conflict response. Because state is in memory, a container restart cancels the current job and removes access to its result.

The implementation is divided into these units:

- **Web layer:** page, health, job, status, and download routes.
- **Job manager:** the current job state, single-job admission, executor dispatch, progress stages, terminal result, and cleanup scheduling.
- **Input service:** exactly-one-source validation, upload streaming, YouTube URL allow-listing, and media-limit enforcement.
- **Media service:** yt-dlp integration plus FFmpeg/ffprobe inspection, audio extraction, and normalization.
- **Transcription service:** device selection, lazy WhisperX model loading, transcription, alignment, and GPU-memory cleanup.
- **Output service:** continuous UTF-8 text generation and SubRip formatting.
- **Cleanup service:** expiration and deletion of job media and outputs without deleting cached model weights.

No Redis, Celery, database, or separate worker service is included.

## User Interface

The page presents one input card with mutually exclusive source tabs:

1. **YouTube URL** accepts a single public `youtube.com` or `youtu.be` video URL.
2. **Audio file** supports drag-and-drop and file selection for one `.mp3` or `.wav` file.

The card also contains:

- A model dropdown containing `tiny`, `base`, `small`, `medium`, and `large-v3`, defaulting to `small`.
- A language dropdown defaulting to Portuguese (Brazil), with Auto-detect and the installed WhisperX language choices available.
- A Transcribe button.

After submission, source and option controls are disabled. The UI polls the job status endpoint and shows meaningful stages rather than fabricated percentages: validating, receiving or downloading, preparing audio, downloading or loading the model, transcribing, aligning subtitles, formatting outputs, and complete.

On completion, the page shows the continuous transcript in a readable text area with Copy, Download TXT, and Download SRT actions. If precise phoneme alignment was unavailable and segment timestamps were used, a warning appears beside the SRT action. A refresh reconnects to the current in-memory job by querying the current-job endpoint.

## API and Job State

The web API uses these routes:

- `GET /` renders the application.
- `GET /health` reports process health.
- `POST /api/jobs` accepts multipart form data containing exactly one source plus the selected model and language, returns HTTP 202 and a generated job identifier, or HTTP 409 when another job is active.
- `GET /api/jobs/current` lets a refreshed browser discover the current job while the process remains alive.
- `GET /api/jobs/{job_id}` returns state, stage, safe user-facing error details, warnings, detected language, and completion metadata.
- `GET /api/jobs/{job_id}/transcript.txt` and `GET /api/jobs/{job_id}/transcript.srt` return completed files with attachment headers.

Job states are `accepted`, `running`, `completed`, and `failed`. A running job also records its current stage. Terminal jobs remain available until the configured retention deadline. Submitting after a terminal job is allowed; the previous job is cleaned up immediately or by the scheduled cleanup pass.

## Processing Flow

1. Validate the model and language against server-owned allow-lists and require exactly one source.
2. Create a UUID-named job directory. Never incorporate an uploaded filename or remote title into a filesystem path.
3. For an upload, stream the body to disk while enforcing the byte limit. Require an `.mp3` or `.wav` suffix, then use ffprobe to validate the actual media stream and duration.
4. For YouTube, require an HTTP(S) URL whose normalized hostname is an allowed YouTube hostname. Use yt-dlp's Python API with playlist processing disabled. Check metadata duration before download when available, download only the best suitable audio, then probe the resulting file and enforce the duration again.
5. Normalize the media through FFmpeg into a WhisperX-compatible audio file.
6. Select `cuda` when PyTorch reports an available CUDA device; otherwise select `cpu`. Use a GPU-appropriate compute type on CUDA and an int8 CPU configuration by default, with environment-variable overrides.
7. Load the selected WhisperX model with the persistent cache directory as its download root. No startup model prefetching occurs.
8. Transcribe using Portuguese when the default remains selected, the selected language otherwise, or no fixed language when Auto-detect is selected.
9. Load the language-specific alignment model and align the transcript for accurate subtitle timing. Alignment models use the same persistent model-cache volume.
10. If alignment is unavailable or fails after transcription, retain the Whisper segment timestamps, record a warning, and continue producing SRT rather than failing the whole job.
11. Join normalized segment text into the plain transcript and write UTF-8 TXT and SRT files.
12. Mark the job completed, expose results until expiration, and remove source media and results after the retention deadline.

Only the current selected transcription model is held in memory. When a later job selects another model, release the prior model and clear unused CUDA memory before loading the new one. Cached weights remain on disk.

## Limits and Configuration

Configuration is supplied through environment variables with these defaults:

- `MAX_UPLOAD_MB=500`
- `MAX_MEDIA_DURATION_SECONDS=10800`
- `RESULT_RETENTION_MINUTES=60`
- `DEFAULT_MODEL=small`
- `DEFAULT_LANGUAGE=pt`
- `MODEL_CACHE_DIR=/models`

Document optional compute type, batch size, worker-thread, and device overrides for constrained or unusual hardware. Values are validated at startup, and invalid configuration prevents startup with a clear log message.

The 500 MB upload limit is enforced while streaming rather than relying only on `Content-Length`. The three-hour limit is based on probed media duration. YouTube downloads with missing preliminary duration metadata may begin, but the downloaded media must pass the final probe before transcription.

## Containers and Hardware

Provide a pinned project definition and dependency lockfile. The Docker image contains Python, WhisperX, PyTorch, yt-dlp, FFmpeg/ffprobe, and the JavaScript runtime and yt-dlp support components required by the pinned yt-dlp release. Use stable releases pinned to a mutually compatible Python, PyTorch, WhisperX, and CUDA combination, and verify that combination with a Docker build and smoke run before treating it as the project baseline.

`compose.yaml` binds the application port to `127.0.0.1` and starts the CPU-capable configuration. A small GPU Compose override requests an NVIDIA GPU. The same application code then detects CUDA at runtime. The documentation makes clear that NVIDIA Container Toolkit must be installed and the GPU override must be used for Docker to expose the GPU; otherwise the application falls back to CPU.

A named Docker volume mounted at `/models` persists transcription and alignment model weights. Job files stay on the container's ephemeral filesystem and are not mounted to the host.

Run one Uvicorn worker. Multiple web workers are unsupported because they would create independent in-memory job managers and could run simultaneous transcriptions.

## Error Handling and Safety

- Reject malformed inputs, unsupported model/language values, multiple sources, unsupported file extensions or contents, playlists, disallowed URL hosts, oversized uploads, and overlong media with specific client errors.
- Restrict remote input to normalized YouTube hostnames instead of passing arbitrary URLs to yt-dlp.
- Invoke external programs with argument arrays and without a shell.
- Keep internal paths, exception traces, and command output out of browser-visible errors. Log technical context to standard output for `docker compose logs`.
- Convert yt-dlp, FFmpeg, model-download, disk-space, CUDA-memory, alignment, and transcription failures into concise user-facing messages.
- Treat alignment failure as a completed job with an SRT timing warning when usable Whisper segment timestamps exist. Treat transcription failure as a failed job.
- Clean partial job files after failure. Do not remove valid shared cache entries.
- Bind only to localhost by default. Authentication is unnecessary within the defined local-only deployment scope.

## Manual Verification

No automated tests or test framework will be added. Before completion, manually verify:

1. Build the image and start the CPU Compose configuration.
2. Confirm the health endpoint and initial page load.
3. Transcribe one small MP3 and one small WAV and confirm displayed text, clipboard copying, valid TXT download, and valid SRT numbering/timestamps.
4. Confirm Portuguese (Brazil) is selected initially and Auto-detect remains selectable.
5. Confirm an initial small-model job downloads weights and a second job reuses the cache.
6. Confirm a public single-video YouTube URL completes through the same output flow.
7. Confirm unsupported files, a disallowed URL, a second simultaneous job, and an over-limit input produce actionable errors.
8. On an NVIDIA-equipped host with NVIDIA Container Toolkit, start with the GPU override and confirm the application reports CUDA selection. If such a host is unavailable, document that GPU execution was not manually verified.

## Acceptance Criteria

The design is satisfied when a local user can start the application with Docker Compose, submit either allowed source, choose a common model and language, observe honest processing stages, copy a completed continuous transcript, download valid TXT and SRT outputs, and reuse model weights across container restarts. The application must default to Portuguese (Brazil), enforce the configured 500 MB and three-hour limits, fall back to CPU when CUDA is not visible, reject concurrent work, and surface failures without crashing the web process.
