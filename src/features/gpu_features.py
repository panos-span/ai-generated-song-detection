"""GPU-accelerated spectral feature extraction using torchaudio.transforms and nnAudio.

Provides batch-capable alternatives to the librosa-based Tier 1 features.
When a CUDA device is available features are computed on GPU for 5-50x speedup.
Falls back to CPU transparently.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio.transforms as T
from nnAudio.features.mel import MelSpectrogram as NNMel
from nnAudio.features.stft import STFT as NNSTFT


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class GPUFeatureExtractor:
    """Batch-capable spectral feature extraction on GPU via torchaudio + nnAudio."""

    def __init__(
        self,
        sr: int = 16000,
        n_mfcc: int = 20,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        device: str = "auto",
    ) -> None:
        self.sr = sr
        self.device = _resolve_device(device)

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        ).to(self.device)

        self.mfcc_transform = T.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={"n_mels": n_mels, "n_fft": n_fft, "hop_length": hop_length},
        ).to(self.device)

        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80).to(self.device)

        self.nn_stft = NNSTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            sr=sr,
            output_format="Magnitude",
        ).to(self.device)

        self.nn_mel = NNMel(
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        ).to(self.device)

    def _mean_std(self, x: torch.Tensor) -> torch.Tensor:
        """Mean and std along time axis. x shape (B, F, T) returns (B, F*2)."""
        means = x.mean(dim=-1)
        stds = x.std(dim=-1)
        return torch.cat([means, stds], dim=-1)

    @torch.no_grad()
    def extract_mfcc_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """MFCC + delta + delta2 mean/std. Returns (batch, 120)."""
        waveforms = waveforms.to(self.device)
        mfcc = self.mfcc_transform(waveforms)
        delta1 = T.ComputeDeltas()(mfcc)
        delta2 = T.ComputeDeltas()(delta1)
        return torch.cat(
            [self._mean_std(mfcc), self._mean_std(delta1), self._mean_std(delta2)], dim=-1
        )

    @torch.no_grad()
    def extract_mel_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Mel spectrogram mean/std via torchaudio. Returns (batch, 256)."""
        waveforms = waveforms.to(self.device)
        mel = self.mel_transform(waveforms)
        mel_db = self.amplitude_to_db(mel)
        return self._mean_std(mel_db)

    @torch.no_grad()
    def extract_mel_batch_nnaudio(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Mel spectrogram via nnAudio (10-100x faster, differentiable). Returns (batch, 256)."""
        waveforms = waveforms.to(self.device)
        mel = self.nn_mel(waveforms)
        mel_db = self.amplitude_to_db(mel)
        return self._mean_std(mel_db)

    @torch.no_grad()
    def extract_stft_batch_nnaudio(self, waveforms: torch.Tensor) -> torch.Tensor:
        """STFT magnitude statistics via nnAudio. Returns (batch, n_fft//2+1)."""
        waveforms = waveforms.to(self.device)
        stft_mag = self.nn_stft(waveforms)
        return stft_mag.mean(dim=-1)

    @torch.no_grad()
    def extract_tier1_batch(self, waveforms: torch.Tensor) -> np.ndarray:
        """GPU-accelerated MFCC(120) + Mel(256) = 376-dim features.

        Covers the two heaviest components of the 430-dim Tier 1 vector.
        The remaining features (chroma, contrast, tonnetz, ZCR, RMS) are
        still computed via librosa as they lack GPU-batch equivalents.
        """
        mfcc_feats = self.extract_mfcc_batch(waveforms)
        mel_feats = self.extract_mel_batch_nnaudio(waveforms)
        combined = torch.cat([mfcc_feats, mel_feats], dim=-1)
        return combined.cpu().numpy()
