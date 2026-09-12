FROM ghcr.io/astral-sh/uv:0.11.18 AS uv
FROM node:22.22.0-bookworm-slim AS node
FROM python:3.12-slim-bookworm

COPY --from=uv /uv /uvx /bin/
COPY --from=node /usr/local/bin/node /usr/local/bin/node
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg libatomic1 \
    && rm -rf /var/lib/apt/lists/* \
    && node --version \
    && node -e "const major = Number(process.versions.node.split('.')[0]); if (major < 22) process.exit(1)"

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
