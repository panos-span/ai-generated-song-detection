"""Deep learned embedding extraction using MERT and CLAP models."""

from __future__ import annotations

import numpy as np
import torch
import torchaudio.functional as AF
from transformers import AutoFeatureExtractor, AutoModel, ClapModel, ClapProcessor

MERT_SR = 24000
CLAP_SR = 48000
MAX_DURATION_S = 30


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    waveform = torch.from_numpy(audio).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    resampled = AF.resample(waveform, orig_sr, target_sr)
    return resampled.squeeze(0).numpy()


def _truncate(audio: np.ndarray, sr: int, max_seconds: float = MAX_DURATION_S) -> np.ndarray:
    max_samples = int(sr * max_seconds)
    if len(audio) > max_samples:
        return audio[:max_samples]
    return audio


class EmbeddingExtractor:
    """Extracts MERT and CLAP embeddings from raw audio waveforms."""

    def __init__(
        self,
        device: str = "auto",
        use_mert: bool = True,
        use_clap: bool = True,
    ) -> None:
        self.device = _resolve_device(device)
        self.use_mert = use_mert
        self.use_clap = use_clap

        self._mert_model = None
        self._mert_extractor = None
        self._clap_model = None
        self._clap_processor = None

    def _load_mert(self) -> None:
        if self._mert_model is not None:
            return
        last_exc: Exception | None = None
        for model_name in ("m-a-p/MERT-v1-330M", "m-a-p/MERT-v1-95M"):
            try:
                self._mert_model = (
                    AutoModel.from_pretrained(model_name, trust_remote_code=True)
                    .to(self.device)
                    .eval()
                )
                self._mert_extractor = AutoFeatureExtractor.from_pretrained(
                    model_name, trust_remote_code=True
                )
                return
            except Exception as exc:
                last_exc = exc
                continue
        raise RuntimeError(f"Failed to load any MERT model variant: {last_exc}") from last_exc

    def _load_clap(self) -> None:
        if self._clap_model is not None:
            return
        try:
            self._clap_model = (
                ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
                .to(self.device)
                .eval()
            )
            self._clap_processor = ClapProcessor.from_pretrained(
                "laion/larger_clap_music_and_speech"
            )
        except Exception as exc:
            raise RuntimeError("Failed to load CLAP model") from exc

    def extract_mert_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract MERT CLS-token embedding from audio.

        Returns a 1-D numpy array (768-dim for 330M, varies for other sizes).
        """
        self._load_mert()
        audio = _resample(audio, sr, MERT_SR)
        audio = _truncate(audio, MERT_SR)

        inputs = self._mert_extractor(audio, sampling_rate=MERT_SR, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._mert_model(**inputs, output_hidden_states=False)

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.squeeze(0).cpu().numpy()

    def extract_clap_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract CLAP audio embedding.

        Returns a 1-D numpy array (512-dim).
        """
        self._load_clap()
        audio = _resample(audio, sr, CLAP_SR)
        audio = _truncate(audio, CLAP_SR)

        inputs = self._clap_processor(audios=audio, sampling_rate=CLAP_SR, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._clap_model.get_audio_features(**inputs)

        return outputs.squeeze(0).cpu().numpy()

    def extract_all_embeddings(self, audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        """Extract all enabled embeddings.

        Returns a dict with 'mert' and/or 'clap' keys mapping to 1-D arrays.
        """
        results: dict[str, np.ndarray] = {}
        if self.use_mert:
            results["mert"] = self.extract_mert_embedding(audio, sr)
        if self.use_clap:
            results["clap"] = self.extract_clap_embedding(audio, sr)
        return results
