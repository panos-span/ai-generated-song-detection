"""Stochastic audio augmentation pipeline for training robustness."""

from __future__ import annotations

import random

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from scipy.signal import fftconvolve as _fftconvolve


class AudioAugmentor:
    """Apply random augmentations to raw audio waveforms.

    Each augmentation is applied independently with probability ``p``.
    When ``enabled=False``, pass through unchanged (for validation/inference).
    """

    def __init__(self, sr: int = 16000, p: float = 0.3, enabled: bool = True) -> None:
        self.sr = sr
        self.p = p
        self.enabled = enabled

    def _should_apply(self) -> bool:
        return random.random() < self.p

    def _pitch_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pitch shift by +/-2 semitones via interpolation.

        Uses ``torch.nn.functional.interpolate`` (linear) instead of
        ``torchaudio.functional.pitch_shift`` (phase vocoder) or
        ``torchaudio.functional.resample`` (polyphase filter).  Both
        alternatives are 500–2 000× slower on CPU for arbitrary rate
        ratios.  Linear interpolation is ~2 ms for 30 s of audio and
        perfectly adequate for data-augmentation quality.
        """
        semitones = random.uniform(-2.0, 2.0)
        ratio = 2.0 ** (semitones / 12.0)
        orig_len = waveform.shape[-1]
        new_len = max(1, int(round(orig_len / ratio)))
        # interpolate expects (N, C, L)
        shifted = torch.nn.functional.interpolate(
            waveform.unsqueeze(0), size=new_len, mode="linear", align_corners=False
        ).squeeze(0)
        if shifted.shape[-1] > orig_len:
            shifted = shifted[..., :orig_len]
        elif shifted.shape[-1] < orig_len:
            shifted = torch.nn.functional.pad(shifted, (0, orig_len - shifted.shape[-1]))
        return shifted

    def _time_stretch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Time stretch by 80-120% via linear interpolation."""
        rate = random.uniform(0.8, 1.2)
        orig_len = waveform.shape[-1]
        new_len = max(1, int(round(orig_len * rate)))
        stretched = torch.nn.functional.interpolate(
            waveform.unsqueeze(0), size=new_len, mode="linear", align_corners=False
        ).squeeze(0)
        if stretched.shape[-1] > orig_len:
            stretched = stretched[..., :orig_len]
        elif stretched.shape[-1] < orig_len:
            stretched = torch.nn.functional.pad(
                stretched,
                (0, orig_len - stretched.shape[-1]),
            )
        return stretched

    def _random_gain(self, waveform: torch.Tensor) -> torch.Tensor:
        """Random gain +/-6 dB."""
        gain_db = random.uniform(-6.0, 6.0)
        gain_linear = 10 ** (gain_db / 20.0)
        return waveform * gain_linear

    def _additive_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add white noise at SNR 20-40 dB."""
        snr_db = random.uniform(20.0, 40.0)
        signal_power = waveform.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise = torch.randn_like(waveform) * noise_power.sqrt()
        return waveform + noise

    def _codec_reencode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Re-encode through lossy codec via FFmpeg (torchaudio 2.x).

        Falls back to no-op if FFmpeg is not available.
        """
        try:
            codec = random.choice(["libmp3lame", "libvorbis"])
            fmt = "ogg" if "vorbis" in codec else "mp3"
            effector = torchaudio.io.AudioEffector(format=fmt, encoder=codec)
            result = effector.apply(waveform, self.sr)
            if result is not None and result.shape == waveform.shape:
                return result
        except Exception:  # noqa: BLE001
            pass  # FFmpeg not available or codec error
        return waveform

    def _eq_filter(self, waveform: torch.Tensor) -> torch.Tensor:
        """Random bandpass EQ."""
        center_freq = random.uniform(200.0, 4000.0)
        Q = random.uniform(0.5, 2.0)
        gain_db = random.uniform(-6.0, 6.0)
        return F.equalizer_biquad(waveform, self.sr, center_freq, gain_db, Q)

    def _reverb(self, waveform: torch.Tensor) -> torch.Tensor:
        """Simple synthetic reverb via FFT convolution with exponential decay IR."""
        ir_length = int(0.3 * self.sr)
        decay = torch.exp(-torch.linspace(0, 5, ir_length))
        ir = (torch.randn(ir_length) * decay).numpy()
        ir = ir / (max(abs(ir.max()), abs(ir.min())) + 1e-8)
        # FFT-based convolution is O(n log n) vs O(n*k) for regular conv1d,
        # giving ~100x speedup for long signals (30s audio + 4800-sample IR).
        wav_np = waveform.squeeze(0).numpy()
        convolved = _fftconvolve(wav_np, ir, mode="full")[: wav_np.shape[-1]]
        result = torch.from_numpy(convolved).float().unsqueeze(0)
        wet = random.uniform(0.1, 0.4)
        return (1 - wet) * waveform + wet * result

    def _short_crop(self, waveform: torch.Tensor) -> torch.Tensor:
        """Random crop to 5-30 seconds (from A7 broadcast findings)."""
        min_samples = int(5.0 * self.sr)
        max_samples = int(30.0 * self.sr)
        total = waveform.shape[-1]
        if total <= min_samples:
            return waveform
        crop_len = random.randint(min_samples, min(max_samples, total))
        start = random.randint(0, total - crop_len)
        cropped = waveform[..., start : start + crop_len]
        # Pad back to original length for consistent tensor shapes
        if cropped.shape[-1] < total:
            cropped = torch.nn.functional.pad(cropped, (0, total - cropped.shape[-1]))
        return cropped

    def _background_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add colored environmental noise at SNR 15-35 dB."""
        snr_db = random.uniform(15.0, 35.0)
        signal_power = waveform.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        # Pink noise approximation via filtered white noise
        noise = torch.randn_like(waveform)
        # Simple 1/f roll-off via cumulative sum
        noise = torch.cumsum(noise, dim=-1)
        noise = noise - noise.mean(dim=-1, keepdim=True)
        noise_rms = noise.pow(2).mean().sqrt() + 1e-8
        noise = noise * (noise_power.sqrt() / noise_rms)
        return waveform + noise

    def _bandreject_eq(self, waveform: torch.Tensor) -> torch.Tensor:
        """Band-reject EQ (Deezer AdversarialAugmenter pattern)."""
        center_freq = random.uniform(500.0, 4000.0)
        Q = random.uniform(1.0, 5.0)
        gain_db = random.uniform(-12.0, -3.0)
        return F.equalizer_biquad(waveform, self.sr, center_freq, gain_db, Q)

    def _bass_treble_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """Random bass or treble boost/cut."""
        if random.random() < 0.5:
            freq = random.uniform(80.0, 300.0)
        else:
            freq = random.uniform(4000.0, 8000.0)
        gain_db = random.uniform(-6.0, 6.0)
        Q = random.uniform(0.5, 1.5)
        return F.equalizer_biquad(waveform, self.sr, freq, gain_db, Q)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to waveform.

        Args:
            waveform: Tensor of shape ``(1, num_samples)`` or ``(num_samples,)``

        Returns:
            Augmented waveform, same shape as input.
        """
        if not self.enabled:
            return waveform

        squeeze = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True

        if self._should_apply():
            waveform = self._pitch_shift(waveform)
        if self._should_apply():
            waveform = self._time_stretch(waveform)
        if self._should_apply():
            waveform = self._random_gain(waveform)
        if self._should_apply():
            waveform = self._additive_noise(waveform)
        if self._should_apply():
            waveform = self._codec_reencode(waveform)
        if self._should_apply():
            waveform = self._eq_filter(waveform)
        if self._should_apply():
            waveform = self._reverb(waveform)
        if self._should_apply():
            waveform = self._short_crop(waveform)
        if self._should_apply():
            waveform = self._background_noise(waveform)
        if self._should_apply():
            waveform = self._bandreject_eq(waveform)
        if self._should_apply():
            waveform = self._bass_treble_shift(waveform)

        if squeeze:
            waveform = waveform.squeeze(0)
        return waveform
