"""Lyrics / text analysis features: transcription, embedding, and similarity."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

WHISPER_MODEL_SIZE = "large-v2"
SBERT_MODEL_NAME = "all-mpnet-base-v2"
SBERT_DIM = 768


def _resolve_device(device: str) -> torch.device:
    """Resolve device string to a :class:`torch.device`."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class LyricsFeatureExtractor(ABC):
    """Abstract base for lyrics / text feature extractors with lazy model loading."""

    def __init__(self, device: str = "auto") -> None:
        self.device: torch.device = _resolve_device(device)
        self._model: object | None = None

    @property
    def model(self) -> object:
        """Lazily load the underlying model on first access."""
        if self._model is None:
            self.load_model()
        return self._model  # type: ignore[return-value]

    @abstractmethod
    def load_model(self) -> None:
        """Load the backing model into ``self._model``."""

    @abstractmethod
    def extract(self, *args: object, **kwargs: object) -> object:
        """Run feature extraction (signature varies by subclass)."""


class WhisperTranscriber(LyricsFeatureExtractor):
    """Speech-to-text transcription via faster-whisper."""

    def load_model(self) -> None:
        device_str: str = "cuda" if self.device.type == "cuda" else "cpu"
        compute_type: str = "float16" if device_str == "cuda" else "float32"
        logger.info(
            "Loading Whisper model '%s' on %s (%s)",
            WHISPER_MODEL_SIZE,
            device_str,
            compute_type,
        )
        self._model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=device_str,
            compute_type=compute_type,
        )

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file and return the full text.

        Uses VAD filtering to skip silent regions.  Returns an empty string
        when no speech is detected.
        """
        whisper_model: WhisperModel = self.model  # type: ignore[assignment]
        try:
            segments, info = whisper_model.transcribe(audio_path, vad_filter=True)
            text: str = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        except Exception:
            logger.exception("Whisper transcription failed for '%s'", audio_path)
            return ""

        if not text:
            logger.debug(
                "No speech detected in '%s' (lang prob %.2f)",
                audio_path,
                info.language_probability,
            )
            return ""

        logger.debug(
            "Transcribed '%s': %d chars, language=%s",
            audio_path,
            len(text),
            info.language,
        )
        return text

    def extract(self, audio_path: str) -> str:  # type: ignore[override]
        """Alias for :meth:`transcribe`."""
        return self.transcribe(audio_path)


class SBERTEmbedder(LyricsFeatureExtractor):
    """Sentence-BERT text embedding via sentence-transformers."""

    def load_model(self) -> None:
        logger.info("Loading SBERT model '%s'", SBERT_MODEL_NAME)
        self._model = SentenceTransformer(SBERT_MODEL_NAME, device=str(self.device))

    def extract(self, text: str) -> np.ndarray:  # type: ignore[override]
        """Return a 768-d L2-normalised embedding for *text*.

        Returns a zero vector when *text* is empty or whitespace-only.
        """
        if not text or not text.strip():
            return np.zeros(SBERT_DIM, dtype=np.float32)

        sbert_model: SentenceTransformer = self.model  # type: ignore[assignment]
        embedding: np.ndarray = sbert_model.encode(text, normalize_embeddings=True)
        return embedding


def compute_lyrical_similarity(
    track_a_path: str,
    track_b_path: str,
    device: str = "auto",
) -> float | None:
    """Compute cosine similarity between the lyrics of two audio tracks.

    Returns ``None`` when **both** transcriptions are empty (no speech in
    either track).
    """
    transcriber = WhisperTranscriber(device=device)
    embedder = SBERTEmbedder(device=device)

    text_a: str = transcriber.transcribe(track_a_path)
    text_b: str = transcriber.transcribe(track_b_path)

    if not text_a and not text_b:
        logger.info("Both tracks have empty transcriptions; returning None")
        return None

    emb_a: np.ndarray = embedder.extract(text_a)
    emb_b: np.ndarray = embedder.extract(text_b)

    dot: float = float(np.dot(emb_a, emb_b))
    norm_a: float = float(np.linalg.norm(emb_a))
    norm_b: float = float(np.linalg.norm(emb_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    similarity: float = dot / (norm_a * norm_b)
    logger.info(
        "Lyrical similarity: %.4f  (text_a=%d chars, text_b=%d chars)",
        similarity,
        len(text_a),
        len(text_b),
    )
    return similarity
