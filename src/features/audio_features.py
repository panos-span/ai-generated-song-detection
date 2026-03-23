"""Audio feature extraction for AI-music detection.

Tier 1 (430-dim): MFCC × 3 (with delta/delta2), Mel spectrogram, Chroma,
    Spectral Contrast, Tonnetz, ZCR, RMS.
Tier 2 (22-dim): Phase features, HNR, Spectral Flatness, Rolloff ratio,
    SSM Novelty, Fourier artifact fingerprint.

Performance notes:
    MFCC and Mel use torchaudio (C++ FFT backend, 2–4× faster than librosa on CPU).
    STFT is computed once per chunk and shared between phase/artifact functions.
    Chroma is computed once per chunk and shared between Tier 1 and SSM novelty.

    If rebuilding the feature cache after updating this file, delete the existing
    cache first to avoid mixing librosa-based and torchaudio-based feature vectors::

        uv run python -c "import shutil; shutil.rmtree('data/feature_cache', True)"
"""

from __future__ import annotations

import librosa
import numpy as np
import parselmouth
import soundfile as sf
import torch
import torchaudio.functional as _TA_F
import torchaudio.transforms as _TA_T
from scipy.signal import fftconvolve

# ---------------------------------------------------------------------------
# Module-level transform cache — avoids re-initialising on every chunk call
# ---------------------------------------------------------------------------
_TRANSFORM_CACHE: dict = {}


def _mfcc_transform(sr: int) -> "_TA_T.MFCC":
    key = ("mfcc", sr)
    if key not in _TRANSFORM_CACHE:
        _TRANSFORM_CACHE[key] = _TA_T.MFCC(
            sample_rate=sr,
            n_mfcc=20,
            melkwargs={
                "n_fft": 2048,
                "hop_length": 512,
                "n_mels": 128,
                "window_fn": torch.hann_window,
            },
        )
    return _TRANSFORM_CACHE[key]


def _mel_db_transform(sr: int) -> torch.nn.Sequential:
    key = ("mel_db", sr)
    if key not in _TRANSFORM_CACHE:
        _TRANSFORM_CACHE[key] = torch.nn.Sequential(
            _TA_T.MelSpectrogram(
                sample_rate=sr,
                n_fft=2048,
                hop_length=512,
                n_mels=128,
                window_fn=torch.hann_window,
            ),
            _TA_T.AmplitudeToDB(top_db=80),
        )
    return _TRANSFORM_CACHE[key]


def load_audio(path: str, sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate.

    Uses soundfile for robust WAV loading (avoids torchaudio backend issues).
    Falls back to librosa for formats soundfile cannot read.
    """
    try:
        audio, orig_sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        audio, orig_sr = librosa.load(path, sr=None, mono=True)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix to mono
    if orig_sr != sr:
        # torchaudio.functional.resample uses a C++ backend and is 2-5x faster
        # than librosa.resample (scipy) on CPU.
        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        audio = _TA_F.resample(waveform, orig_sr, sr).squeeze(0).numpy()
    audio = audio.astype(np.float32)
    if len(audio) == 0:
        audio = np.zeros(sr, dtype=np.float32)
    return audio, sr


def _safe_mean_std(feature: np.ndarray) -> np.ndarray:
    if feature.ndim == 1:
        feature = feature.reshape(1, -1)
    if feature.shape[1] == 0:
        return np.zeros(feature.shape[0] * 2, dtype=np.float64)
    means = np.mean(feature, axis=1)
    stds = np.std(feature, axis=1)
    return np.concatenate([means, stds])


def _pad_if_short(audio: np.ndarray, sr: int, min_duration: float = 0.5) -> np.ndarray:
    min_samples = int(sr * min_duration)
    if len(audio) < min_samples:
        audio = np.pad(audio, (0, min_samples - len(audio)), mode="constant")
    return audio


def extract_tier1_features(
    audio: np.ndarray,
    sr: int,
    *,
    _chroma: np.ndarray | None = None,
) -> np.ndarray:
    """Extract 430-dim Tier 1 features.

    Args:
        audio: Waveform array.
        sr: Sample rate.
        _chroma: Pre-computed chroma matrix (12, n_frames). When provided by
            ``extract_all_features``, avoids a redundant librosa.stft() call.
    """
    audio = _pad_if_short(audio, sr)
    waveform = torch.from_numpy(audio).float().unsqueeze(0)  # (1, n_samples)

    # MFCC via torchaudio (C++ FFT, 2–3× faster than librosa/scipy on CPU)
    mfccs = _mfcc_transform(sr)(waveform).squeeze(0).numpy()  # (20, n_frames)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    mfcc_feats = np.concatenate(
        [
            _safe_mean_std(mfccs),
            _safe_mean_std(mfcc_delta),
            _safe_mean_std(mfcc_delta2),
        ]
    )

    # Mel spectrogram via torchaudio (2–4× faster than librosa on CPU)
    mel_db = _mel_db_transform(sr)(waveform).squeeze(0).numpy()  # (128, n_frames)
    mel_feats = _safe_mean_std(mel_db)

    # Chroma: reuse pre-computed if provided (avoids second redundant STFT)
    if _chroma is None:
        _chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12)
    chroma_feats = _safe_mean_std(_chroma)

    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
    contrast_feats = _safe_mean_std(contrast)

    # Pass pre-computed chroma to tonnetz — skips HPSS (librosa.effects.harmonic),
    # saving ~250 ms per 10-second chunk (the single biggest remaining bottleneck).
    # librosa.feature.tonnetz(chroma=…) computes the same 6 tonal-centroid features.
    tonnetz = librosa.feature.tonnetz(chroma=_chroma)
    tonnetz_feats = _safe_mean_std(tonnetz)

    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_feats = _safe_mean_std(zcr)

    rms = librosa.feature.rms(y=audio)
    rms_feats = _safe_mean_std(rms)

    return np.concatenate(
        [
            mfcc_feats,
            mel_feats,
            chroma_feats,
            contrast_feats,
            tonnetz_feats,
            zcr_feats,
            rms_feats,
        ]
    )


def _phase_continuity_index(
    audio: np.ndarray, sr: int, *, _stft: np.ndarray | None = None
) -> float:
    if _stft is None:
        _stft = librosa.stft(audio)
    phase = np.angle(_stft)
    if phase.shape[1] < 3:
        return 0.0
    inst_freq = np.diff(phase, axis=1)
    inst_freq = np.mod(inst_freq + np.pi, 2 * np.pi) - np.pi
    expected = inst_freq[:, :-1]
    actual = inst_freq[:, 1:]
    deviation = np.abs(actual - expected)
    deviation = np.mod(deviation + np.pi, 2 * np.pi) - np.pi
    return float(np.mean(np.abs(deviation)))


def _harmonic_to_noise_ratio(audio: np.ndarray, sr: int) -> float:
    """Compute mean HNR using Praat via parselmouth."""
    snd = parselmouth.Sound(audio, sampling_frequency=sr)
    hnr = snd.to_harmonicity()
    values = hnr.values[hnr.values != -200]
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def _spectral_flatness_features(audio: np.ndarray, sr: int) -> np.ndarray:
    sf = librosa.feature.spectral_flatness(y=audio)
    return _safe_mean_std(sf)


def _high_freq_rolloff_ratio(audio: np.ndarray, sr: int) -> float:
    rolloff_85 = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)
    rolloff_95 = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.95)
    mean_85 = np.mean(rolloff_85)
    mean_95 = np.mean(rolloff_95)
    if mean_95 == 0:
        return 0.0
    return float(mean_85 / mean_95)


def _temporal_ssm_novelty(
    audio: np.ndarray, sr: int, *, _chroma: np.ndarray | None = None
) -> float:
    """Compute mean novelty from recurrence-based SSM using vectorized checkerboard kernel."""
    if _chroma is None:
        _chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    if _chroma.shape[1] < 4:
        return 0.0
    chroma_norm = librosa.util.normalize(_chroma, axis=0)
    ssm = np.dot(chroma_norm.T, chroma_norm)
    n = ssm.shape[0]
    kernel_size = min(8, n // 2)
    if kernel_size < 2:
        return 0.0
    # Build checkerboard kernel
    kern = np.ones((2 * kernel_size, 2 * kernel_size))
    kern[:kernel_size, kernel_size:] = -1
    kern[kernel_size:, :kernel_size] = -1
    nov = fftconvolve(ssm, kern, mode="same")
    novelty_curve = np.diag(nov)
    novelty_curve = np.maximum(novelty_curve, 0)
    if len(novelty_curve) == 0:
        return 0.0
    return float(np.mean(novelty_curve))


def _expanded_phase_features(
    audio: np.ndarray, sr: int, *, _stft: np.ndarray | None = None
) -> np.ndarray:
    """Compute expanded phase spectrogram summary (7 features)."""
    if _stft is None:
        _stft = librosa.stft(audio)
    phase = np.angle(_stft)

    if phase.shape[1] < 3:
        return np.zeros(7, dtype=np.float64)

    inst_freq = np.diff(phase, axis=1)
    inst_freq = np.mod(inst_freq + np.pi, 2 * np.pi) - np.pi
    expected = inst_freq[:, :-1]
    actual = inst_freq[:, 1:]
    deviation = np.abs(actual - expected)
    deviation = np.mod(deviation + np.pi, 2 * np.pi) - np.pi
    pci = float(np.mean(np.abs(deviation)))

    phase_dev_per_band = np.std(phase, axis=1)
    phase_dev_mean = float(np.mean(phase_dev_per_band))
    phase_dev_std = float(np.std(phase_dev_per_band))

    if_stability = np.std(inst_freq, axis=1)
    if_stab_mean = float(np.mean(if_stability))
    if_stab_std = float(np.std(if_stability))

    group_delay = -np.diff(phase, axis=0)
    group_delay = np.mod(group_delay + np.pi, 2 * np.pi) - np.pi
    gd_dev_per_time = np.std(group_delay, axis=0)
    gd_mean = float(np.mean(gd_dev_per_time))
    gd_std = float(np.std(gd_dev_per_time))

    return np.array(
        [pci, phase_dev_mean, phase_dev_std, if_stab_mean, if_stab_std, gd_mean, gd_std]
    )


def extract_fourier_artifacts(
    audio: np.ndarray, sr: int, *, _stft: np.ndarray | None = None
) -> np.ndarray:
    """Detect deconvolution-induced spectral peaks from AI music generators.

    AI synthesis modules (Suno/Udio/MusicGen) use transposed convolutions
    that produce mathematically predictable peaks at frequencies tied to
    common upsampling ratios (2x, 4x, 8x). Based on Afchar et al.
    "A Fourier Explanation of AI-music Artifacts" (ISMIR 2025).

    Returns a 10-dim feature vector::

        [peak_to_bg_2x, peak_to_bg_4x, peak_to_bg_8x,
         peak_count_2x, peak_count_4x, peak_count_8x,
         overall_peak_regularity, max_peak_to_bg,
         spectral_periodicity_score, artifact_energy_ratio]

    Args:
        audio: Waveform array.
        sr: Sample rate.
        _stft: Pre-computed complex STFT (n_bins, n_frames). When provided by
            ``extract_tier2_features``, avoids a redundant librosa.stft() call.
    """
    if _stft is None:
        _stft = librosa.stft(audio)
    magnitude = np.abs(_stft)

    if magnitude.shape[1] < 2:
        return np.zeros(10, dtype=np.float64)

    # FFT along frequency axis per time frame to find periodic spectral patterns
    freq_fft = np.abs(np.fft.rfft(magnitude, axis=0))
    avg_freq_fft = np.mean(freq_fft, axis=1)

    n_bins = len(avg_freq_fft)
    if n_bins < 10:
        return np.zeros(10, dtype=np.float64)

    background = np.median(avg_freq_fft[1:])
    if background < 1e-10:
        background = 1e-10

    features = []

    # Check for peaks at upsampling ratios (2x, 4x, 8x)
    for ratio in [2, 4, 8]:
        target_bin = n_bins // ratio
        search_range = max(1, n_bins // (ratio * 10))
        lo = max(1, target_bin - search_range)
        hi = min(n_bins - 1, target_bin + search_range)

        region = avg_freq_fft[lo : hi + 1]
        peak_val = float(np.max(region))
        peak_to_bg = peak_val / background

        # Count peaks above threshold in region
        threshold = background * 3.0
        peak_count = int(np.sum(region > threshold))

        features.append(float(np.clip(peak_to_bg / 20.0, 0.0, 1.0)))
        features.append(float(np.clip(peak_count / 10.0, 0.0, 1.0)))

    # Overall peak regularity: std of inter-peak distances
    threshold = background * 2.5
    peak_positions = np.where(avg_freq_fft[1:] > threshold)[0]
    if len(peak_positions) > 2:
        inter_peak = np.diff(peak_positions).astype(np.float64)
        regularity = 1.0 / (1.0 + float(np.std(inter_peak) / (np.mean(inter_peak) + 1e-8)))
    else:
        regularity = 0.0
    features.append(regularity)

    # Max peak-to-background across all bins
    max_peak_to_bg = float(np.max(avg_freq_fft[1:])) / background
    features.append(float(np.clip(max_peak_to_bg / 30.0, 0.0, 1.0)))

    # Spectral periodicity: autocorrelation of the frequency-domain FFT
    norm_fft = avg_freq_fft[1:] - np.mean(avg_freq_fft[1:])
    if np.std(norm_fft) > 1e-10:
        autocorr = np.correlate(norm_fft, norm_fft, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        autocorr = autocorr / (autocorr[0] + 1e-10)
        # Find first significant peak after lag 0
        if len(autocorr) > 2:
            peaks_ac = np.where((autocorr[1:-1] > autocorr[:-2]) & (autocorr[1:-1] > autocorr[2:]))[
                0
            ]
            periodicity = float(np.max(autocorr[peaks_ac + 1])) if len(peaks_ac) > 0 else 0.0
        else:
            periodicity = 0.0
    else:
        periodicity = 0.0
    features.append(float(np.clip(periodicity, 0.0, 1.0)))

    # Artifact energy ratio: energy at artifact bins / total energy
    artifact_bins = []
    for ratio in [2, 4, 8]:
        target_bin = n_bins // ratio
        artifact_bins.extend(range(max(1, target_bin - 1), min(n_bins, target_bin + 2)))
    artifact_energy = float(np.sum(avg_freq_fft[artifact_bins] ** 2))
    total_energy = float(np.sum(avg_freq_fft[1:] ** 2)) + 1e-10
    features.append(float(np.clip(artifact_energy / total_energy, 0.0, 1.0)))

    return np.array(features, dtype=np.float64)


def extract_tier2_features(
    audio: np.ndarray,
    sr: int,
    *,
    skip_hnr: bool = False,
    _stft: np.ndarray | None = None,
    _chroma: np.ndarray | None = None,
) -> np.ndarray:
    """Extract 22-dim Tier 2 features.

    Args:
        audio: Waveform array.
        sr: Sample rate.
        skip_hnr: Skip Praat HNR computation (slow; use for fast inference).
        _stft: Pre-computed complex STFT. Shared from ``extract_all_features``
            to avoid recomputing ``librosa.stft()`` for both phase and artifacts.
        _chroma: Pre-computed chroma. Shared from ``extract_all_features``
            to avoid recomputing ``librosa.feature.chroma_stft()`` for SSM novelty.
    """
    audio = _pad_if_short(audio, sr)

    # Compute STFT once; reused by _expanded_phase_features AND extract_fourier_artifacts.
    if _stft is None:
        _stft = librosa.stft(audio)

    phase_feats = _expanded_phase_features(audio, sr, _stft=_stft)
    hnr = 0.0 if skip_hnr else _harmonic_to_noise_ratio(audio, sr)
    sf_feats = _spectral_flatness_features(audio, sr)
    rolloff_ratio = _high_freq_rolloff_ratio(audio, sr)
    ssm_novelty = _temporal_ssm_novelty(audio, sr, _chroma=_chroma)
    fourier_feats = extract_fourier_artifacts(audio, sr, _stft=_stft)

    return np.concatenate(
        [
            phase_feats,
            np.array([hnr]),
            sf_feats,
            np.array([rolloff_ratio]),
            np.array([ssm_novelty]),
            fourier_feats,
        ]
    )


def extract_all_features(
    audio: np.ndarray, sr: int, *, skip_hnr: bool = False
) -> dict[str, np.ndarray]:
    """Extract all features, sharing STFT and chroma across Tier 1 + Tier 2."""
    audio = _pad_if_short(audio, sr)

    # Compute shared spectra once to avoid redundant librosa STFT calls:
    #   _stft   → _expanded_phase_features + extract_fourier_artifacts
    #   _chroma → extract_tier1_features (chroma bins) + _temporal_ssm_novelty
    _stft = librosa.stft(audio)
    _chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12)

    tier1 = extract_tier1_features(audio, sr, _chroma=_chroma)
    tier2 = extract_tier2_features(audio, sr, skip_hnr=skip_hnr, _stft=_stft, _chroma=_chroma)
    return {
        "tier1": tier1,
        "tier2": tier2,
        "combined": np.concatenate([tier1, tier2]),
    }


def extract_features_batch(chunks: list[np.ndarray], sr: int, *, skip_hnr: bool = False) -> np.ndarray:
    """Extract features for multiple audio chunks.

    Returns array of shape (num_chunks, feature_dim).
    """
    return np.stack([extract_all_features(c, sr, skip_hnr=skip_hnr)["combined"] for c in chunks])


def extract_features_batch_gpu(
    chunks: list[np.ndarray],
    sr: int,
    gpu_extractor: object | None = None,
    *,
    skip_hnr: bool = False,
) -> np.ndarray:
    """Extract features using GPU-accelerated MFCC/Mel + CPU librosa for the rest.

    If ``gpu_extractor`` is provided, uses it for the MFCC (120-dim) and Mel
    (256-dim) components, falling back to librosa only for chroma, contrast,
    tonnetz, ZCR, RMS, and Tier 2 features. Otherwise identical to
    ``extract_features_batch``.
    """
    if gpu_extractor is None:
        return extract_features_batch(chunks, sr, skip_hnr=skip_hnr)

    import torch

    chunk_tensors = [torch.from_numpy(c).float() for c in chunks]
    max_len = max(t.shape[0] for t in chunk_tensors)
    padded = torch.stack(
        [torch.nn.functional.pad(t, (0, max_len - t.shape[0])) for t in chunk_tensors]
    )

    gpu_feats = gpu_extractor.extract_tier1_batch(padded)

    all_feats = []
    for i, chunk in enumerate(chunks):
        chunk = _pad_if_short(chunk, sr)

        # Compute shared per-chunk STFT and chroma once for both tier1 remainder and tier2
        _stft = librosa.stft(chunk)
        _chroma = librosa.feature.chroma_stft(y=chunk, sr=sr, n_chroma=12)

        chroma_feats = _safe_mean_std(_chroma)

        contrast = librosa.feature.spectral_contrast(y=chunk, sr=sr, n_bands=6)
        contrast_feats = _safe_mean_std(contrast)

        tonnetz = librosa.feature.tonnetz(chroma=_chroma)
        tonnetz_feats = _safe_mean_std(tonnetz)

        zcr = librosa.feature.zero_crossing_rate(chunk)
        zcr_feats = _safe_mean_std(zcr)

        rms = librosa.feature.rms(y=chunk)
        rms_feats = _safe_mean_std(rms)

        tier2 = extract_tier2_features(chunk, sr, skip_hnr=skip_hnr, _stft=_stft, _chroma=_chroma)

        librosa_only = np.concatenate(
            [chroma_feats, contrast_feats, tonnetz_feats, zcr_feats, rms_feats]
        )
        combined = np.concatenate([gpu_feats[i], librosa_only, tier2])
        all_feats.append(combined)

    return np.stack(all_feats)
