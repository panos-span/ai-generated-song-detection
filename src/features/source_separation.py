"""Optional Demucs-based vocal isolation for AI music detection.

Deezer repo marks Demucs as ultimately not used because it is not robust
to adversarial attacks.  This module is opt-in via the ``--use-demucs`` flag.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torchaudio

logger = logging.getLogger(__name__)


def separate_vocals(
    audio_path: str,
    output_dir: str,
    device: str = "auto",
) -> str:
    """Isolate vocals from an audio track using Demucs HTDemucs_ft.

    Deezer repo marks Demucs as ultimately not used because it is not
    robust to adversarial attacks.  This is opt-in via ``--use-demucs``
    flag.

    Parameters
    ----------
    audio_path:
        Path to the input audio file (any format supported by torchaudio).
    output_dir:
        Directory where the isolated vocal WAV will be saved.
    device:
        ``"auto"`` (default) picks CUDA when available, otherwise CPU.
        Pass ``"cpu"`` or ``"cuda"`` to override.

    Returns
    -------
    str
        Absolute path to the saved 16 kHz mono vocal WAV file.

    Raises
    ------
    RuntimeError
        If ``demucs`` is not installed.
    FileNotFoundError
        If *audio_path* does not exist.
    """
    # ---- optional dependency guard ---------------------------------- #
    try:
        from demucs.apply import apply_model  # type: ignore[import-untyped]
        from demucs.pretrained import get_model  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("demucs not installed. Run: uv pip install demucs")

    # ---- resolve paths ---------------------------------------------- #
    src = Path(audio_path)
    if not src.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_path = out / f"{src.stem}_vocals.wav"

    # ---- select device ---------------------------------------------- #
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    logger.info("Demucs device: %s", dev)

    # ---- load audio ------------------------------------------------- #
    logger.info("Loading audio: %s", audio_path)
    waveform, sr = torchaudio.load(str(src))

    # Ensure float32 for model input
    waveform = waveform.float()

    # ---- load pre-trained model ------------------------------------- #
    logger.info("Loading HTDemucs_ft model ...")
    model = get_model("htdemucs_ft")
    model.to(dev)
    model.eval()

    # ---- run separation --------------------------------------------- #
    logger.info("Running source separation ...")
    # apply_model expects (batch, channels, samples); add batch dim
    waveform_gpu = waveform.to(dev)
    sources = apply_model(model, waveform_gpu[None], device=dev)[0]
    # sources shape: (num_sources, channels, samples)

    # ---- extract vocal stem ----------------------------------------- #
    # Demucs source order: drums, bass, other, vocals
    source_names: list[str] = model.sources
    try:
        vocal_idx = source_names.index("vocals")
    except ValueError:
        logger.warning(
            "Model sources %s do not contain 'vocals'; falling back to last stem.",
            source_names,
        )
        vocal_idx = len(source_names) - 1

    vocals = sources[vocal_idx].cpu()  # (channels, samples)

    # ---- convert to 16 kHz mono ------------------------------------- #
    if vocals.shape[0] > 1:
        vocals = vocals.mean(dim=0, keepdim=True)

    if sr != 16000:
        vocals = torchaudio.functional.resample(vocals, sr, 16000)

    # ---- save ------------------------------------------------------- #
    torchaudio.save(str(output_path), vocals, 16000)
    logger.info("Saved isolated vocals -> %s", output_path)

    return str(output_path.resolve())
