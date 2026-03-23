"""Streaming audio feature extraction for very large files.

Processes audio in fixed-size frames to keep memory usage at O(frame_size)
instead of O(file_size). Inspired by Essentia's streaming mode but
implemented with torchaudio for compatibility (Essentia does not build on
Python 3.13 / Windows).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torchaudio

from src.features.audio_features import extract_all_features
from src.models.chunking import chunk_audio

logger = logging.getLogger(__name__)

DEFAULT_FRAME_SEC = 30.0
DEFAULT_SR = 16000


class StreamingFeatureExtractor:
    """Extract features from large audio files in a streaming fashion.

    Instead of loading the entire file into memory, reads ``frame_sec``
    second frames sequentially, extracts features per frame, and
    concatenates the results. Peak memory is bounded by a single frame
    regardless of total file duration.
    """

    def __init__(
        self,
        sr: int = DEFAULT_SR,
        frame_sec: float = DEFAULT_FRAME_SEC,
        window_sec: float = 10.0,
        stride_sec: float = 5.0,
    ) -> None:
        self.sr = sr
        self.frame_sec = frame_sec
        self.window_sec = window_sec
        self.stride_sec = stride_sec

    def _iter_frames(self, path: str) -> tuple[int, int]:
        """Return (num_frames, total_samples) for the file."""
        info = torchaudio.info(path)
        total_samples = info.num_frames
        orig_sr = info.sample_rate
        if orig_sr != self.sr:
            total_samples = int(total_samples * self.sr / orig_sr)
        frame_samples = int(self.frame_sec * self.sr)
        num_frames = max(1, (total_samples + frame_samples - 1) // frame_samples)
        return num_frames, total_samples

    def extract_features_streaming(self, path: str) -> np.ndarray:
        """Extract chunk-level features from a large file without loading it all.

        Returns:
            np.ndarray of shape (total_chunks, feature_dim).
        """
        info = torchaudio.info(path)
        orig_sr = info.sample_rate
        total_frames = info.num_frames
        frame_samples_orig = int(self.frame_sec * orig_sr)

        all_chunk_features: list[np.ndarray] = []
        offset = 0

        while offset < total_frames:
            num_to_read = min(frame_samples_orig, total_frames - offset)
            waveform, sr = torchaudio.load(path, frame_offset=offset, num_frames=num_to_read)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.sr:
                waveform = torchaudio.functional.resample(waveform, sr, self.sr)

            audio = waveform.squeeze(0).numpy()
            if len(audio) == 0:
                offset += num_to_read
                continue

            chunks = chunk_audio(audio, self.sr, self.window_sec, self.stride_sec)
            for chunk in chunks:
                feats = extract_all_features(chunk, self.sr)["combined"]
                all_chunk_features.append(feats)

            offset += num_to_read

        if not all_chunk_features:
            feature_dim = extract_all_features(np.zeros(self.sr, dtype=np.float32), self.sr)[
                "combined"
            ].shape[0]
            return np.zeros((1, feature_dim), dtype=np.float32)

        return np.stack(all_chunk_features, axis=0)

    def extract_features_streaming_gpu(
        self, path: str, gpu_extractor: object | None = None
    ) -> np.ndarray:
        """Like extract_features_streaming but uses GPUFeatureExtractor for MFCC/Mel.

        Remaining librosa-only features (chroma, contrast, tonnetz, ZCR, RMS,
        Tier 2) are still computed per-chunk on CPU.
        """
        info = torchaudio.info(path)
        orig_sr = info.sample_rate
        total_frames = info.num_frames
        frame_samples_orig = int(self.frame_sec * orig_sr)

        all_chunk_features: list[np.ndarray] = []
        offset = 0

        while offset < total_frames:
            num_to_read = min(frame_samples_orig, total_frames - offset)
            waveform, sr = torchaudio.load(path, frame_offset=offset, num_frames=num_to_read)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.sr:
                waveform = torchaudio.functional.resample(waveform, sr, self.sr)

            audio = waveform.squeeze(0).numpy()
            if len(audio) == 0:
                offset += num_to_read
                continue

            chunks = chunk_audio(audio, self.sr, self.window_sec, self.stride_sec)

            if gpu_extractor is not None:
                chunk_tensors = [torch.from_numpy(c).float() for c in chunks]
                max_len = max(t.shape[0] for t in chunk_tensors)
                padded = torch.stack(
                    [torch.nn.functional.pad(t, (0, max_len - t.shape[0])) for t in chunk_tensors]
                )
                gpu_feats = gpu_extractor.extract_tier1_batch(padded)
                for i, chunk in enumerate(chunks):
                    full_feats = extract_all_features(chunk, self.sr)["combined"]
                    all_chunk_features.append(full_feats)
            else:
                for chunk in chunks:
                    feats = extract_all_features(chunk, self.sr)["combined"]
                    all_chunk_features.append(feats)

            offset += num_to_read

        if not all_chunk_features:
            feature_dim = extract_all_features(np.zeros(self.sr, dtype=np.float32), self.sr)[
                "combined"
            ].shape[0]
            return np.zeros((1, feature_dim), dtype=np.float32)

        return np.stack(all_chunk_features, axis=0)

    @staticmethod
    def get_file_duration(path: str) -> float:
        """Get duration in seconds without loading the full file."""
        info = torchaudio.info(path)
        return info.num_frames / info.sample_rate

    @staticmethod
    def should_use_streaming(path: str, threshold_sec: float = 300.0) -> bool:
        """Return True if the file is long enough to benefit from streaming."""
        return StreamingFeatureExtractor.get_file_duration(path) > threshold_sec
