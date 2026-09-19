# Whisper UI

Whisper UI is a single-user web interface for transcribing one audio file or one public YouTube video at a time with WhisperX. It shows real processing stages, keeps the current job recoverable across page refreshes, and produces continuous plain text plus TXT and SRT downloads.

## Requirements

- Docker Engine with Docker Compose v2 (`docker compose`). Docker Desktop is suitable on macOS and Windows/WSL 2.
- Internet access for the first use of each transcription model and each alignment language. YouTube transcription also needs outbound access to YouTube.
- Enough disk space for the container image, downloaded model weights, and temporary media. Larger Whisper models require substantially more memory, storage, and processing time.
- No host Python, FFmpeg, Node.js, or frontend toolchain is required; they are included in the image. Node.js 22.22.0 is pinned because yt-dlp requires Node.js 22 or newer for its enabled JavaScript runtime.

The Compose configuration runs transcription on the CPU and publishes the app on port `8000` on every host interface.

## Quick Start

From this directory, build and start the application:

```bash
docker compose build
docker compose up
```

Open <http://127.0.0.1:8000> locally, or `http://your-domain-or-vps-ip:8000` from another computer. The VPS firewall and hosting provider must allow inbound TCP port 8000. Portuguese (Brazil) and the `base` model are selected by default. The first job with each transcription model downloads that model's weights, and the first alignment for each language downloads separate alignment weights. CPU transcription can be slow, especially with larger models or long media.

Keep the terminal open while using the app. Press Ctrl+C to stop the foreground process, then remove the stopped container and network without removing the model cache:

```bash
docker compose down
```

## Using the App

1. Choose **YouTube** and enter one public single-video YouTube URL, or choose **Audio file** and select or drag-and-drop one MP3 or WAV file. The native file chooser remains keyboard accessible.
2. Choose a model. `tiny` is fastest and least accurate; larger models generally need more memory and time. `base` is the default.
3. Keep **Portuguese (Brazil)**, select another supported language, or choose **Auto-detect**.
4. Select **Transcribe**. The form is disabled while the one allowed active job progresses through validation/download and audio preparation. From model loading onward, a four-step tracker shows the current stage, elapsed time, a measured current-stage estimate when enough progress is available, and the latest five activity lines.
5. When complete, copy the continuous transcript or download UTF-8 plain-text TXT and timestamped SubRip SRT output. Select **Reset session** when you want to clear the completed job and return to a fresh form.

Only one job can be active. A second submission while a job is accepted or running returns `A transcription is already running.` A page refresh recovers the current job's status or result while the same application process is alive. Temporary polling failures keep the form locked and retry with capped exponential backoff because the server may still own the active job. Resetting a completed session removes its result from the current session; starting another job also replaces the prior terminal job once any download already in progress closes.

## Configuration

Configuration uses environment variables. `Settings` also reads a `.env` file when the app is run directly, but `compose.yaml` supplies several literal container values. For Compose deployments, put overrides in a `compose.override.yaml` file so the settings are explicit and reproducible:

```yaml
services:
  app:
    environment:
      MAX_UPLOAD_MB: "250"
      DEFAULT_MODEL: "tiny"
      COMPUTE_TYPE: "int8"
```

Compose merges that file automatically for ordinary `docker compose` commands. Recreate the service after changing configuration:

```bash
docker compose up -d --build --force-recreate
```

Every application setting is listed below. Positive numeric settings reject zero and negative values. `DEFAULT_MODEL` is restricted to the listed values. Startup rejects an unknown default language and a compute type unsupported by the installed CPU CTranslate2 build; this does not download a model.

| Variable | Default | Units / values | Effect |
| --- | --- | --- | --- |
| `MAX_UPLOAD_MB` | `500` | MiB (`value × 1024 × 1024` bytes; shown as MB in the UI) | Maximum uploaded MP3/WAV size, enforced while streaming the request to disk. |
| `MAX_MEDIA_DURATION_SECONDS` | `10800` | seconds (3 hours) | Maximum probed media duration. Applied to uploads and both YouTube metadata, when present, and the downloaded media. |
| `RESULT_RETENTION_MINUTES` | `60` | minutes | Time a completed or failed current job remains available before it expires. A newer job removes a prior terminal job immediately. |
| `DEFAULT_MODEL` | `base` | `tiny`, `base`, `small`, `medium`, or `large-v3` | Model initially selected in the page. |
| `DEFAULT_LANGUAGE` | `pt` | WhisperX language code | Language initially selected in the page; `pt` is displayed as Portuguese (Brazil). Users may select Auto-detect per job. |
| `MODEL_CACHE_DIR` | `/models` | container directory | Root the app passes directly to WhisperX for ASR (`whisper/`) and alignment (`alignment/`) caches. The container's auxiliary Hugging Face/Torch cache variables and Compose volume target are configured separately. |
| `JOBS_DIR` | `/tmp/whisper-ui` | container directory | Ephemeral working directory for the one current job's upload/download and results. It is not host-mounted by the supplied Compose file. |
| `COMPUTE_TYPE` | `int8` | CTranslate2 compute-type string | Numeric compute type used for CPU transcription. |
| `BATCH_SIZE` | `4` | segments per transcription batch | WhisperX transcription batch size. Reduce it if memory is constrained. |
| `CLEANUP_INTERVAL_SECONDS` | `60` | seconds | Interval between checks that remove an expired terminal job and its result directory. |

The image intentionally runs one Uvicorn worker. Multiple workers are unsupported because each worker would have independent in-memory job state and could start a separate transcription.

## Model Cache

Compose mounts the named `whisper-model-cache` volume at `/models`. Under the default Compose project name, Docker normally shows it as `whisper-ui_whisper-model-cache`. The volume survives container replacement and ordinary `docker compose down`, so later jobs can reuse downloaded transcription and alignment weights. The currently selected transcription model is also reused in memory until another model is selected or the process restarts.

Keep `MODEL_CACHE_DIR=/models` for the supplied container layout. Changing only that setting moves the app's ASR/alignment paths but does not move the volume or the auxiliary caches: `compose.yaml` still mounts the named volume at `/models`, and the Dockerfile still sets `HF_HOME=/models/huggingface` and `TORCH_HOME=/models/torch`. If you intentionally choose another container cache root, update the volume target and set matching `HF_HOME` and `TORCH_HOME` values in the same Compose configuration; otherwise caches can be split between persistent and container-ephemeral locations.

Inspect cached file size and recent model-related logs with:

```bash
docker compose exec app python -c "from pathlib import Path; print(sum(p.stat().st_size for p in Path('/models').rglob('*') if p.is_file()))"
docker compose logs --since=10m app
docker volume ls | grep -F whisper-model-cache
```

Cache removal is optional and destructive. The following commands permanently delete the default project volume and force future jobs to download model weights again. Do not run them if you want to keep the cache; a custom Compose project name changes the actual volume name.

```bash
docker compose down
docker volume rm whisper-ui_whisper-model-cache
```

## Limits and Privacy

- Uploads are limited to filenames ending in `.mp3` or `.wav` (case-insensitive). Renaming another format is not enough: FFprobe must find readable media with an audio stream and a valid duration.
- Remote input is restricted to one HTTP(S) YouTube video URL. Playlists, channels, searches, arbitrary remote hosts, live, private, login-required, and age-restricted videos are rejected.
- The defaults allow up to 500 MiB per upload and 10,800 seconds (3 hours) per media item. Before multipart parsing, the job endpoint caps the request stream at the file limit plus 1 MiB of bounded form overhead; file persistence separately enforces the exact configured byte limit. YouTube duration is checked before download when metadata is available and always checked on the downloaded file.
- The app accepts requests addressed to any hostname or IP address and does not apply origin filtering.
- Uvicorn listens on `0.0.0.0:8000` inside the container, and the supplied Compose file publishes port 8000 on every host interface. The app has no accounts or authentication, so anyone who can reach that port can use it and consume VPS resources.
- Uploads and downloaded source media are held inside the container's job directory. On successful completion, source and intermediate media are removed and only TXT/SRT results remain. Failed-job payloads are removed. Terminal job data expires after 60 minutes by default and is removed on a later cleanup pass.
- Job state is only in application memory. Refreshing the page can recover it while the process lives, but restarting or replacing the container loses the current job state and access to its results. Container replacement also removes its ephemeral job files. The separate model-cache volume survives.
- There is no database, account, transcription history, telemetry integration, or remote transcription API. YouTube jobs contact YouTube through yt-dlp, and model/alignment downloads contact their upstream model hosts.
- Alignment failure does not discard an otherwise usable transcript. The app completes with Whisper segment timing and displays a warning that precise alignment was unavailable.

## Troubleshooting

**`docker compose` is missing in WSL 2**

Enable Docker Desktop's integration for the relevant WSL distribution, make sure Docker Desktop is running, open a new WSL shell, and verify `docker compose version` before retrying.

**The page does not open or the health check fails**

```bash
docker compose ps
docker compose logs --tail=100 app
curl --fail http://127.0.0.1:8000/health
```

Check for startup configuration errors and confirm nothing else is using local port 8000. A healthy response is JSON containing `"status":"ok"` and the resolved `"device"`.

**Startup rejects a language or compute type**

Use a language code shown in the WhisperX-backed list. For a compute-type error, choose one of the supported values printed in the startup message for the CPU device, update `COMPUTE_TYPE`, and recreate the service.

**A job appears stuck at model loading**

First use can download large transcription or alignment weights. Follow logs with `docker compose logs -f app`, check outbound network access and free disk space, and inspect the cache size. Do not interrupt a download by deleting the shared cache.

**Transcription is very slow or runs out of memory**

CPU processing can be much slower than media duration. Choose a smaller model and reduce `BATCH_SIZE` if memory is constrained.

**A YouTube URL is rejected or cannot download**

Use a public, non-live, single-video `youtube.com` or `youtu.be` URL without a playlist parameter. Private, age-restricted, login-required, playlist, channel, and non-YouTube URLs are unsupported. Then check container logs and outbound connectivity.

**The job completes with an alignment warning**

The transcript is still usable. SRT timing falls back to the transcription model's segment timestamps because precise language alignment could not be loaded or run. Check logs and model-download connectivity for details.

**A result disappeared**

Completed and failed jobs expire after `RESULT_RETENTION_MINUTES`, a newer job replaces a terminal job immediately, and a process/container restart loses in-memory access to prior results. Download needed TXT/SRT files before those events.

## Manual Verification

Use short, user-provided speech MP3 and WAV files. Do not substitute generated tones: tones cannot verify speech recognition. The first real transcription can download model weights and can take substantial time on CPU.

### CPU startup

```bash
docker compose config --quiet
docker compose build
docker compose up -d
docker compose ps
docker compose logs --tail=100 app
curl --fail http://127.0.0.1:8000/health
```

Confirm the service becomes healthy, health reports `cpu`, and startup logs contain no model download before any job is submitted.

### Upload and browser behavior

With each user-provided speech file, verify:

1. Portuguese (Brazil) and `base` are initially selected and Auto-detect can be selected.
2. Submission disables the form and displays changing processing stages.
3. Completion displays continuous text and **Copy** places the same text on the clipboard.
4. The TXT file is UTF-8 plain text; the SRT opens with sequential blocks and `HH:MM:SS,mmm --> HH:MM:SS,mmm` timestamps.
5. Refreshing during a job and after completion restores its status or result; selecting **Reset session** clears the result and leaves no current job after refresh.
6. A second submission from another browser tab while the first is processing returns the busy message.
7. An unsupported extension and a non-audio file renamed with `.mp3` or `.wav` are both rejected safely.

Also verify safe rejection of a non-YouTube remote URL, a playlist URL, and an upload over `MAX_UPLOAD_MB`. Browser-visible errors must not include a traceback, command output, or an internal filesystem path.

To check the duration limit, temporarily set `MAX_MEDIA_DURATION_SECONDS: "1"` in `compose.override.yaml`, recreate the service, and submit a valid user-provided speech file longer than one second. Confirm the configured-duration message, restore the value to `"10800"` (or remove the override), and recreate the service again.

### YouTube and cache reuse

Submit a short public single-video URL with `tiny`. After it completes, inspect `/models`, submit another `tiny` job, and inspect recent logs:

```bash
docker compose exec app python -c "from pathlib import Path; print(sum(p.stat().st_size for p in Path('/models').rglob('*') if p.is_file()))"
docker compose logs --since=10m app
```

Confirm the cache contains weight files, the second job reuses the ASR model instead of downloading it again, and TXT/SRT actions work for the YouTube result.

### Stop without deleting the cache

```bash
docker compose down
docker volume ls | grep -F whisper-model-cache
```

Confirm the containers and network stop while the model-cache volume remains.
