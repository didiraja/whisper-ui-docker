import gc
import logging
from pathlib import Path
from typing import Any

import torch
import whisperx
import ctranslate2
from whisperx.utils import LANGUAGES

from whisper_ui.config import MODELS, Settings
from whisper_ui.domain import JobStage, ProgressCallback, TranscriptResult
from whisper_ui.errors import ConfigError

logger = logging.getLogger(__name__)

ALIGNMENT_WARNING = (
    "Precise alignment was unavailable; SRT timing uses Whisper segment timestamps."
)


class WhisperService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if settings.default_language not in LANGUAGES:
            raise ConfigError(
                "DEFAULT_LANGUAGE must be a language code supported by WhisperX."
            )
        if settings.device == "cuda" and not torch.cuda.is_available():
            raise ConfigError("DEVICE=cuda was requested but CUDA is unavailable.")
        self.device = (
            settings.device
            if settings.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.compute_type = (
            settings.cuda_compute_type
            if self.device == "cuda"
            else settings.cpu_compute_type
        )
        try:
            supported_compute_types = ctranslate2.get_supported_compute_types(
                self.device
            )
        except Exception as error:
            raise ConfigError(
                f"Could not query CTranslate2 compute types for {self.device}."
            ) from error
        if self.compute_type not in supported_compute_types:
            setting_name = (
                "CUDA_COMPUTE_TYPE"
                if self.device == "cuda"
                else "CPU_COMPUTE_TYPE"
            )
            supported = ", ".join(sorted(supported_compute_types))
            raise ConfigError(
                f"{setting_name}={self.compute_type!r} is unsupported for "
                f"{self.device}; choose one of: {supported}."
            )
        self.batch_size = (
            settings.cuda_batch_size
            if self.device == "cuda"
            else settings.cpu_batch_size
        )

        for cache_name in ("whisper", "alignment", "huggingface", "torch"):
            (settings.model_cache_dir / cache_name).mkdir(parents=True, exist_ok=True)

        self._model_name: str | None = None
        self._model: Any | None = None

    def _ensure_model(self, model_name: str) -> Any:
        if model_name not in MODELS:
            raise ValueError(f"model_name must be one of {', '.join(MODELS)}")
        if self._model_name == model_name and self._model is not None:
            return self._model

        old_model = self._model
        self._model = None
        self._model_name = None
        if old_model is not None:
            del old_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = whisperx.load_model(
            model_name,
            self.device,
            compute_type=self.compute_type,
            download_root=str(self.settings.model_cache_dir / "whisper"),
        )
        self._model = model
        self._model_name = model_name
        return model

    def transcribe(
        self,
        audio_path: Path,
        model_name: str,
        language: str | None,
        progress: ProgressCallback,
    ) -> TranscriptResult:
        progress(JobStage.LOADING_MODEL)
        model = self._ensure_model(model_name)
        audio = whisperx.load_audio(str(audio_path))

        progress(JobStage.TRANSCRIBING)
        if language is None:
            # WhisperX 3.8.6 detects only when its pipeline tokenizer is None.
            # Clear any tokenizer left by an earlier call (including a failed call)
            # so every auto-detect job examines its own audio.
            model.tokenizer = None
        result = model.transcribe(
            audio,
            batch_size=self.batch_size,
            language=language,
        )
        effective_language = result["language"]
        segments = result["segments"]

        progress(JobStage.ALIGNING)
        align_model: Any | None = None
        alignment_warning: str | None = None
        try:
            alignment_cache = self.settings.model_cache_dir / "alignment"
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
            segments = aligned["segments"]
        except Exception:
            logger.exception("WhisperX alignment failed; using ASR segment timestamps")
            alignment_warning = ALIGNMENT_WARNING
        finally:
            if align_model is not None:
                del align_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return TranscriptResult(
            segments=segments,
            language=effective_language,
            alignment_warning=alignment_warning,
        )
