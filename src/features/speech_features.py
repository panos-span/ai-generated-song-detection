"""Speech and vocal feature extraction for AI-generated music detection.

Extracts speech-level embeddings using Wav2Vec2 and Whisper encoder models,
provides voice activity detection via Silero VAD, and computes vocal similarity
between track pairs using cosine similarity of speech embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF
from numpy.typing import NDArray
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Processor,
    WhisperFeatureExtractor,
    WhisperModel,
)

from src.features.lyrics_features import LyricsFeatureExtractor

logger = logging.getLogger(__name__)

WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-large-960h"
WAV2VEC2_DIM = 1024
WAV2VEC2_SR = 16000

WHISPER_MODEL_NAME = "openai/whisper-large-v2"
WHISPER_DIM = 1280
WHISPER_SR = 16000

MIN_AUDIO_SAMPLES = 400  # Minimum samples for meaningful processing


def _resolve_device(device: str) -> torch.device:
    """Resolve device string to a torch device. ``"auto"`` picks CUDA if available."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_audio_mono_16k(audio_path: str) -> tuple[torch.Tensor, int]:
    """Load an audio file, convert to mono, and resample to 16 kHz.

    Returns:
        Tuple of (waveform tensor of shape (samples,), sample rate 16000).
        Returns an empty tensor if loading fails.
    """
    path = Path(audio_path)
    if not path.exists():
        logger.warning("Audio file does not exist: %s", audio_path)
        return torch.zeros(0), WAV2VEC2_SR

    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception:
        logger.exception("Failed to load audio: %s", audio_path)
        return torch.zeros(0), WAV2VEC2_SR

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16 kHz
    if sr != WAV2VEC2_SR:
        waveform = AF.resample(waveform, sr, WAV2VEC2_SR)

    return waveform.squeeze(0), WAV2VEC2_SR


class Wav2Vec2Extractor(LyricsFeatureExtractor):
    """Extract speech embeddings using Wav2Vec2-Large-960h (Apache-2.0).

    Produces a 1024-dimensional mean-pooled hidden-state vector from the
    final encoder layer. Model loading is lazy -- the checkpoint is fetched
    only on the first call to :meth:`extract`.
    """

    def __init__(self, device: str = "auto") -> None:
        self.device = _resolve_device(device)
        self._processor: Wav2Vec2Processor | None = None
        self._model: Wav2Vec2Model | None = None

    def load_model(self) -> None:
        """Load Wav2Vec2 processor and model (lazy, idempotent)."""
        if self._model is not None:
            return
        logger.info("Loading Wav2Vec2 model: %s", WAV2VEC2_MODEL_NAME)
        self._processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL_NAME)
        self._model = Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL_NAME).to(self.device).eval()
        logger.info("Wav2Vec2 model loaded on %s", self.device)

    def extract(self, audio_path: str) -> NDArray[np.floating]:
        """Extract a 1024-d mean-pooled Wav2Vec2 embedding from *audio_path*.

        Returns ``np.zeros(1024)`` for empty, missing, or very short audio.
        """
        self.load_model()
        assert self._processor is not None
        assert self._model is not None

        waveform, sr = _load_audio_mono_16k(audio_path)

        if waveform.numel() < MIN_AUDIO_SAMPLES:
            logger.warning(
                "Audio too short (%d samples) for Wav2Vec2: %s",
                waveform.numel(),
                audio_path,
            )
            return np.zeros(WAV2VEC2_DIM, dtype=np.float32)

        inputs = self._processor(
            waveform.numpy(),
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Mean-pool over time dimension -> (1024,)
        hidden_states = outputs.last_hidden_state  # (1, T, 1024)
        embedding = hidden_states.mean(dim=1).squeeze(0)  # (1024,)
        return embedding.cpu().numpy().astype(np.float32)


class WhisperEncoderExtractor(LyricsFeatureExtractor):
    """Extract encoder hidden states from Whisper-Large-v2 (MIT).

    Uses only the encoder -- no decoder / ASR pass is performed.  This is
    useful when Whisper is already loaded for transcription elsewhere in
    the pipeline, avoiding a second large model download.

    Produces a 1280-dimensional mean-pooled encoder vector.
    """

    def __init__(self, device: str = "auto") -> None:
        self.device = _resolve_device(device)
        self._feature_extractor: WhisperFeatureExtractor | None = None
        self._model: WhisperModel | None = None

    def load_model(self) -> None:
        """Load Whisper feature extractor and model (lazy, idempotent)."""
        if self._model is not None:
            return
        logger.info("Loading Whisper model: %s", WHISPER_MODEL_NAME)
        self._feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL_NAME)
        self._model = WhisperModel.from_pretrained(WHISPER_MODEL_NAME).to(self.device).eval()
        logger.info("Whisper encoder loaded on %s", self.device)

    def extract(self, audio_path: str) -> NDArray[np.floating]:
        """Extract a 1280-d mean-pooled Whisper encoder embedding.

        Returns ``np.zeros(1280)`` for empty, missing, or very short audio.
        """
        self.load_model()
        assert self._feature_extractor is not None
        assert self._model is not None

        waveform, sr = _load_audio_mono_16k(audio_path)

        if waveform.numel() < MIN_AUDIO_SAMPLES:
            logger.warning(
                "Audio too short (%d samples) for Whisper: %s",
                waveform.numel(),
                audio_path,
            )
            return np.zeros(WHISPER_DIM, dtype=np.float32)

        inputs = self._feature_extractor(
            waveform.numpy(),
            sampling_rate=sr,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device)  # (1, 80, 3000)

        with torch.no_grad():
            # Use only the encoder -- skip decoder entirely
            encoder_outputs = self._model.encoder(input_features)

        # Mean-pool over time dimension -> (1280,)
        hidden_states = encoder_outputs.last_hidden_state  # (1, T, 1280)
        embedding = hidden_states.mean(dim=1).squeeze(0)  # (1280,)
        return embedding.cpu().numpy().astype(np.float32)


class SileroVAD:
    """Voice Activity Detection wrapper around Silero VAD.

    Model is loaded lazily from ``torch.hub`` on the first call to
    :meth:`has_speech`.
    """

    def __init__(self, device: str = "auto") -> None:
        self.device = _resolve_device(device)
        self._model: torch.nn.Module | None = None
        self._utils: tuple | None = None

    def _load_model(self) -> None:
        """Load Silero VAD model from torch.hub (lazy, idempotent)."""
        if self._model is not None:
            return
        logger.info("Loading Silero VAD from torch.hub")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._model = model.to(self.device)
        self._utils = utils
        logger.info("Silero VAD loaded on %s", self.device)

    def has_speech(self, audio_path: str, threshold: float = 0.5) -> bool:
        """Return True if speech is detected above *threshold* in *audio_path*.

        Falls back to False on any loading / processing error.
        """
        self._load_model()
        assert self._model is not None
        assert self._utils is not None

        try:
            (
                get_speech_timestamps,
                _save_audio,
                _read_audio,
                _VADIterator,
                _collect_chunks,
            ) = self._utils

            wav = _read_audio(audio_path, sampling_rate=WAV2VEC2_SR)
            if wav.numel() < MIN_AUDIO_SAMPLES:
                logger.warning("Audio too short for VAD: %s", audio_path)
                return False

            wav = wav.to(self.device)
            speech_timestamps = get_speech_timestamps(
                wav, self._model, threshold=threshold, sampling_rate=WAV2VEC2_SR
            )
            return len(speech_timestamps) > 0
        except Exception:
            logger.exception("Silero VAD failed for: %s", audio_path)
            return False


def compute_vocal_similarity(
    track_a_path: str,
    track_b_path: str,
    device: str = "auto",
) -> float | None:
    """Compute cosine similarity of Wav2Vec2 speech embeddings between two tracks.

    Workflow:
      1. Check for speech in both tracks via Silero VAD.
      2. If either track has no speech, return None.
      3. Extract Wav2Vec2 embeddings from both tracks.
      4. Return cosine similarity in [-1, 1].

    Args:
        track_a_path: Path to the first audio file.
        track_b_path: Path to the second audio file.
        device: Device string (``"auto"``, ``"cpu"``, or ``"cuda"``).

    Returns:
        Cosine similarity as a float, or None if no speech is detected
        in at least one track.
    """
    vad = SileroVAD(device=device)

    has_speech_a = vad.has_speech(track_a_path)
    has_speech_b = vad.has_speech(track_b_path)

    if not has_speech_a or not has_speech_b:
        logger.info(
            "No speech detected -- skipping vocal similarity (A=%s, B=%s)",
            has_speech_a,
            has_speech_b,
        )
        return None

    extractor = Wav2Vec2Extractor(device=device)
    emb_a = extractor.extract(track_a_path)
    emb_b = extractor.extract(track_b_path)

    # Guard against zero vectors (e.g. from fallback on short audio)
    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        logger.warning("Zero-norm embedding encountered -- returning None")
        return None

    cosine_sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
    return cosine_sim
