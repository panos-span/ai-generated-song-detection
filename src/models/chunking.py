import numpy as np


def chunk_audio(
    audio: np.ndarray,
    sr: int,
    window_sec: float = 10.0,
    stride_sec: float = 5.0,
) -> list[np.ndarray]:
    """Split audio into overlapping chunks. Returns list of audio segments."""
    window_samples = int(sr * window_sec)
    stride_samples = int(sr * stride_sec)

    if len(audio) <= window_samples:
        padded = np.zeros(window_samples, dtype=audio.dtype)
        padded[: len(audio)] = audio
        return [padded]

    chunks: list[np.ndarray] = []
    start = 0
    while start < len(audio):
        end = start + window_samples
        chunk = audio[start:end]
        if len(chunk) < window_samples:
            padded = np.zeros(window_samples, dtype=audio.dtype)
            padded[: len(chunk)] = chunk
            chunk = padded
        chunks.append(chunk)
        start += stride_samples
        if end >= len(audio):
            break

    return chunks


def chunk_features(features: np.ndarray, num_chunks: int) -> np.ndarray:
    """Reshape a flat feature vector repeated for each chunk position. For pre-computed features."""
    if features.ndim == 1:
        return np.tile(features, (num_chunks, 1))
    return np.tile(features, (num_chunks, 1))
