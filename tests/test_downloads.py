"""Unit tests for data download scripts (SONICS & FakeMusicCaps)."""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests
import soundfile as sf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_songs_csv(path: Path, rows: list[dict]) -> None:
    """Write a minimal fake_songs.csv."""
    fieldnames = [
        "id", "filename", "title", "duration", "algorithm", "style",
        "source", "lyrics_features", "topic", "genre", "mood", "label",
        "target", "split",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row = {k: "" for k in fieldnames}
            row.update(r)
            writer.writerow(row)


def _make_real_songs_csv(path: Path, rows: list[dict]) -> None:
    """Write a minimal real_songs.csv."""
    fieldnames = [
        "id", "filename", "title", "artist", "year", "lyrics",
        "lyrics_features", "duration", "youtube_id", "label",
        "artist_overlap", "target", "skip_time", "no_vocal", "split",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row = {k: "" for k in fieldnames}
            row.update(r)
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Tests: metadata schema
# ---------------------------------------------------------------------------

class TestSonicsMetadataSchema:
    """VAL-SONICS-002: metadata.csv has exact columns filename,label,generator,split."""

    def test_sonics_metadata_schema(self, tmp_path: Path) -> None:
        """The metadata CSV produced by the script has exactly the right columns."""
        from data.download_sonics import build_metadata_row_fake, build_metadata_row_real

        fake_row = build_metadata_row_fake(
            filename="fake_001.wav",
            source="suno",
            label="full fake",
            split="train",
        )
        real_row = build_metadata_row_real(
            filename="real_001.wav",
            split="train",
        )

        expected_cols = {"filename", "label", "generator", "split"}
        assert set(fake_row.keys()) == expected_cols
        assert set(real_row.keys()) == expected_cols

    def test_sonics_metadata_no_extra_columns(self, tmp_path: Path) -> None:
        """No unexpected columns sneak into the metadata rows."""
        from data.download_sonics import build_metadata_row_fake

        row = build_metadata_row_fake(
            filename="f.wav", source="udio", label="half fake", split="test",
        )
        assert len(row) == 4, f"Expected 4 columns, got {len(row)}: {list(row.keys())}"


# ---------------------------------------------------------------------------
# Tests: label mapping
# ---------------------------------------------------------------------------

class TestSonicsLabelMapping:
    """VAL-SONICS-003: fake variants -> 'ai', real -> 'real'."""

    @pytest.mark.parametrize("raw_label", ["full fake", "half fake", "mostly fake"])
    def test_sonics_label_mapping_fake(self, raw_label: str) -> None:
        from data.download_sonics import build_metadata_row_fake

        row = build_metadata_row_fake(
            filename="x.wav", source="suno", label=raw_label, split="train",
        )
        assert row["label"] == "ai", f"Expected 'ai' for '{raw_label}', got '{row['label']}'"

    def test_sonics_generator_mapping_suno(self) -> None:
        from data.download_sonics import build_metadata_row_fake

        row = build_metadata_row_fake(
            filename="x.wav", source="suno", label="full fake", split="train",
        )
        assert row["generator"] == "suno"

    def test_sonics_generator_mapping_udio(self) -> None:
        from data.download_sonics import build_metadata_row_fake

        row = build_metadata_row_fake(
            filename="x.wav", source="udio", label="half fake", split="train",
        )
        assert row["generator"] == "udio"

    def test_sonics_label_mapping_real(self) -> None:
        from data.download_sonics import build_metadata_row_real

        row = build_metadata_row_real(filename="r.wav", split="train")
        assert row["label"] == "real"
        assert row["generator"] == "real"


# ---------------------------------------------------------------------------
# Tests: idempotency
# ---------------------------------------------------------------------------

class TestSonicsIdempotency:
    """VAL-SONICS-005: already-processed filenames are skipped."""

    def test_sonics_idempotency(self, tmp_path: Path) -> None:
        from data.download_sonics import get_existing_filenames

        csv_path = tmp_path / "metadata.csv"
        df = pd.DataFrame(
            [
                {"filename": "sonics_00000.wav", "label": "ai", "generator": "suno", "split": "train"},
                {"filename": "sonics_00001.wav", "label": "real", "generator": "real", "split": "train"},
            ]
        )
        df.to_csv(csv_path, index=False)

        existing = get_existing_filenames(csv_path)
        assert "sonics_00000.wav" in existing
        assert "sonics_00001.wav" in existing
        assert len(existing) == 2

    def test_sonics_idempotency_no_file(self, tmp_path: Path) -> None:
        from data.download_sonics import get_existing_filenames

        csv_path = tmp_path / "metadata.csv"
        existing = get_existing_filenames(csv_path)
        assert len(existing) == 0


# ---------------------------------------------------------------------------
# Tests: credential / CLI error handling
# ---------------------------------------------------------------------------

class TestSonicsCredentialErrors:
    """VAL-SONICS-006: clear error when kaggle credentials missing."""

    def test_sonics_missing_credentials(self) -> None:
        from data.download_sonics import download_kaggle_dataset

        with patch("data.download_sonics.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError(
                "[Errno 2] No such file or directory: 'kaggle'"
            )
            with pytest.raises(SystemExit) as exc_info:
                download_kaggle_dataset("awsaf49/sonics-dataset", Path("/tmp/staging"))

            # Should exit with non-zero code
            assert exc_info.value.code != 0

    def test_sonics_kaggle_auth_error(self) -> None:
        from data.download_sonics import download_kaggle_dataset

        with patch("data.download_sonics.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1,
                stdout="",
                stderr="401 - Unauthorized",
            )
            with pytest.raises(SystemExit) as exc_info:
                download_kaggle_dataset("awsaf49/sonics-dataset", Path("/tmp/staging"))

            assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# Tests: YouTube download failure resilience
# ---------------------------------------------------------------------------

class TestSonicsYouTubeResilience:
    """VAL-SONICS-007: individual YT failures logged, not fatal."""

    def test_sonics_youtube_failure_logged(self) -> None:
        from data.download_sonics import download_real_song

        with patch("data.download_sonics.yt_dlp.YoutubeDL") as mock_ydl_cls:
            mock_ydl = MagicMock()
            mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
            mock_ydl.__exit__ = MagicMock(return_value=False)
            mock_ydl.download.side_effect = Exception("Video unavailable")
            mock_ydl_cls.return_value = mock_ydl

            result = download_real_song("bad_id", Path("/tmp/out.wav"))
            assert result is False, "Expected False for failed download"

    def test_sonics_youtube_success(self, tmp_path: Path) -> None:
        from data.download_sonics import download_real_song

        out_file = tmp_path / "test.wav"

        with patch("data.download_sonics.yt_dlp.YoutubeDL") as mock_ydl_cls:
            mock_ydl = MagicMock()
            mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
            mock_ydl.__exit__ = MagicMock(return_value=False)
            mock_ydl.download.return_value = 0
            mock_ydl_cls.return_value = mock_ydl

            # Create file to simulate yt-dlp output
            out_file.write_bytes(b"fake wav data")

            result = download_real_song("good_id", out_file)
            assert result is True


# ===========================================================================
# FakeMusicCaps tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Tests: FakeMusicCaps metadata schema
# ---------------------------------------------------------------------------

class TestFakeMusicCapsMetadataSchema:
    """VAL-FMC-002: metadata.csv has exact columns filename,model,caption."""

    def test_fakemusiccaps_metadata_schema(self) -> None:
        from data.download_fakemusiccaps import build_metadata_row

        row = build_metadata_row(
            filename="MusicGen_medium_abc123.wav",
            model="MusicGen_medium",
            caption="A lively piano piece",
        )
        expected_cols = {"filename", "model", "caption"}
        assert set(row.keys()) == expected_cols

    def test_fakemusiccaps_metadata_no_extra_columns(self) -> None:
        from data.download_fakemusiccaps import build_metadata_row

        row = build_metadata_row(
            filename="musicldm_xyz.wav", model="musicldm", caption="test",
        )
        assert len(row) == 3, f"Expected 3 columns, got {len(row)}: {list(row.keys())}"


# ---------------------------------------------------------------------------
# Tests: FakeMusicCaps model name validation
# ---------------------------------------------------------------------------

class TestFakeMusicCapsModelValidation:
    """VAL-FMC-003 / VAL-FMC-007: only known TTM models, no MusicCaps."""

    @pytest.mark.parametrize(
        "model_name",
        ["MusicGen_medium", "musicldm", "audioldm2", "stable_audio_open", "mustango"],
    )
    def test_fakemusiccaps_valid_model_names(self, model_name: str) -> None:
        from data.download_fakemusiccaps import is_valid_model

        assert is_valid_model(model_name), f"{model_name} should be valid"

    def test_fakemusiccaps_musiccaps_excluded(self) -> None:
        """MusicCaps folder must NOT be treated as a valid model."""
        from data.download_fakemusiccaps import is_valid_model

        assert not is_valid_model("MusicCaps"), "MusicCaps should be excluded"

    def test_fakemusiccaps_unknown_model_excluded(self) -> None:
        from data.download_fakemusiccaps import is_valid_model

        assert not is_valid_model("some_random_model")

    def test_fakemusiccaps_discover_excludes_musiccaps(self, tmp_path: Path) -> None:
        """discover_model_folders must skip MusicCaps/ directory."""
        from data.download_fakemusiccaps import discover_model_folders

        # Create model folders + MusicCaps folder
        (tmp_path / "FakeMusicCaps" / "MusicCaps").mkdir(parents=True)
        (tmp_path / "FakeMusicCaps" / "MusicGen_medium").mkdir(parents=True)
        (tmp_path / "FakeMusicCaps" / "musicldm").mkdir(parents=True)

        results = discover_model_folders(tmp_path)
        model_names = [name for name, _ in results]

        assert "MusicCaps" not in model_names
        assert "MusicGen_medium" in model_names
        assert "musicldm" in model_names


# ---------------------------------------------------------------------------
# Tests: FakeMusicCaps caption matching
# ---------------------------------------------------------------------------

class TestFakeMusicCapsCaptionMatching:
    """VAL-FMC-004: captions populated via MusicCaps CSV matching."""

    def test_fakemusiccaps_caption_lookup_basic(self) -> None:
        from data.download_fakemusiccaps import build_caption_lookup

        df = pd.DataFrame({
            "ytid": ["abc123", "def456", "ghi789"],
            "caption": [
                "A piano melody",
                "Drum solo with cymbals",
                "Orchestral piece",
            ],
        })
        lookup = build_caption_lookup(df)

        assert lookup["abc123"] == "A piano melody"
        assert lookup["def456"] == "Drum solo with cymbals"
        assert lookup["ghi789"] == "Orchestral piece"

    def test_fakemusiccaps_caption_lookup_missing_column(self) -> None:
        """Gracefully handle DataFrame without expected columns."""
        from data.download_fakemusiccaps import build_caption_lookup

        df = pd.DataFrame({"some_col": [1, 2, 3]})
        lookup = build_caption_lookup(df)
        assert lookup == {}

    def test_fakemusiccaps_caption_lookup_nan(self) -> None:
        """NaN captions should become empty strings."""
        from data.download_fakemusiccaps import build_caption_lookup

        df = pd.DataFrame({
            "ytid": ["abc123", "def456"],
            "caption": ["A real caption", None],
        })
        lookup = build_caption_lookup(df)
        assert lookup["abc123"] == "A real caption"
        assert lookup["def456"] == ""

    def test_fakemusiccaps_metadata_row_has_caption(self) -> None:
        from data.download_fakemusiccaps import build_metadata_row

        row = build_metadata_row(
            filename="audioldm2_abc123.wav",
            model="audioldm2",
            caption="A gentle flute melody",
        )
        assert row["caption"] == "A gentle flute melody"

    def test_fakemusiccaps_metadata_row_empty_caption(self) -> None:
        from data.download_fakemusiccaps import build_metadata_row

        row = build_metadata_row(
            filename="mustango_xyz.wav",
            model="mustango",
            caption="",
        )
        assert row["caption"] == ""


# ---------------------------------------------------------------------------
# Tests: FakeMusicCaps idempotency
# ---------------------------------------------------------------------------

class TestFakeMusicCapsIdempotency:
    """VAL-FMC-006: already-processed filenames are skipped."""

    def test_fakemusiccaps_idempotency(self, tmp_path: Path) -> None:
        from data.download_fakemusiccaps import get_existing_filenames

        csv_path = tmp_path / "metadata.csv"
        df = pd.DataFrame([
            {"filename": "MusicGen_medium_abc.wav", "model": "MusicGen_medium", "caption": "x"},
            {"filename": "musicldm_def.wav", "model": "musicldm", "caption": "y"},
        ])
        df.to_csv(csv_path, index=False)

        existing = get_existing_filenames(csv_path)
        assert "MusicGen_medium_abc.wav" in existing
        assert "musicldm_def.wav" in existing
        assert len(existing) == 2

    def test_fakemusiccaps_idempotency_no_file(self, tmp_path: Path) -> None:
        from data.download_fakemusiccaps import get_existing_filenames

        csv_path = tmp_path / "nonexistent.csv"
        existing = get_existing_filenames(csv_path)
        assert len(existing) == 0


# ---------------------------------------------------------------------------
# Tests: FakeMusicCaps Zenodo API
# ---------------------------------------------------------------------------

class TestFakeMusicCapsZenodoAPI:
    """VAL-FMC-001: Zenodo API integration."""

    def test_fakemusiccaps_zenodo_api_success(self) -> None:
        from data.download_fakemusiccaps import get_zenodo_download_url

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "files": [
                {
                    "key": "FakeMusicCaps.zip",
                    "size": 12900000000,
                    "links": {
                        "self": "https://zenodo.org/api/records/15063698/files/FakeMusicCaps.zip/content",
                    },
                }
            ],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("data.download_fakemusiccaps.requests.get", return_value=mock_response):
            url, size = get_zenodo_download_url("15063698")

        assert "FakeMusicCaps.zip" in url
        assert size == 12900000000

    def test_fakemusiccaps_zenodo_api_failure(self) -> None:
        from data.download_fakemusiccaps import get_zenodo_download_url

        with patch("data.download_fakemusiccaps.requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException("Network error")
            with pytest.raises(SystemExit) as exc_info:
                get_zenodo_download_url("15063698")
            assert exc_info.value.code != 0

    def test_fakemusiccaps_zenodo_no_files(self) -> None:
        from data.download_fakemusiccaps import get_zenodo_download_url

        mock_response = MagicMock()
        mock_response.json.return_value = {"files": []}
        mock_response.raise_for_status = MagicMock()

        with patch("data.download_fakemusiccaps.requests.get", return_value=mock_response):
            with pytest.raises(SystemExit) as exc_info:
                get_zenodo_download_url("15063698")
            assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# Tests: FakeMusicCaps zip download (resume support)
# ---------------------------------------------------------------------------

class TestFakeMusicCapsDownloadZip:
    """Download resume / skip logic."""

    def test_fakemusiccaps_download_skip_when_complete(self, tmp_path: Path) -> None:
        """Skip download if file exists with correct size."""
        from data.download_fakemusiccaps import download_zip

        dest = tmp_path / "FakeMusicCaps.zip"
        dest.write_bytes(b"x" * 100)

        # Should not make any HTTP requests
        with patch("data.download_fakemusiccaps.requests.get") as mock_get:
            download_zip("https://example.com/f.zip", dest, expected_size=100)
            mock_get.assert_not_called()

    def test_fakemusiccaps_download_redownload_wrong_size(self, tmp_path: Path) -> None:
        """Re-download if file exists but size is wrong."""
        from data.download_fakemusiccaps import download_zip

        dest = tmp_path / "FakeMusicCaps.zip"
        dest.write_bytes(b"x" * 50)  # smaller than expected

        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": "100"}
        mock_resp.iter_content.return_value = [b"x" * 100]
        mock_resp.raise_for_status = MagicMock()

        with patch("data.download_fakemusiccaps.requests.get", return_value=mock_resp):
            download_zip("https://example.com/f.zip", dest, expected_size=100)

        assert dest.exists()


# ---------------------------------------------------------------------------
# Tests: FakeMusicCaps audio re-save
# ---------------------------------------------------------------------------

class TestFakeMusicCapsResaveAudio:
    """VAL-FMC-005: audio re-saved as PCM_16."""

    def test_fakemusiccaps_resave_creates_pcm16(self, tmp_path: Path) -> None:
        import numpy as np
        from data.download_fakemusiccaps import resave_wav_pcm16

        # Create a float32 WAV source file
        src = tmp_path / "source.wav"
        dst = tmp_path / "dest.wav"
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        sf.write(str(src), audio, 16000, subtype="FLOAT")

        ok = resave_wav_pcm16(src, dst)
        assert ok is True
        assert dst.exists()

        info = sf.info(str(dst))
        assert info.subtype == "PCM_16"
        assert info.samplerate == 16000

    def test_fakemusiccaps_resave_stereo_to_mono(self, tmp_path: Path) -> None:
        import numpy as np
        from data.download_fakemusiccaps import resave_wav_pcm16

        src = tmp_path / "stereo.wav"
        dst = tmp_path / "mono.wav"
        audio = np.random.randn(16000, 2).astype(np.float32) * 0.5
        sf.write(str(src), audio, 16000, subtype="FLOAT")

        ok = resave_wav_pcm16(src, dst)
        assert ok is True

        info = sf.info(str(dst))
        assert info.channels == 1
        assert info.subtype == "PCM_16"

    def test_fakemusiccaps_resave_handles_error(self, tmp_path: Path) -> None:
        from data.download_fakemusiccaps import resave_wav_pcm16

        src = tmp_path / "nonexistent.wav"
        dst = tmp_path / "out.wav"
        ok = resave_wav_pcm16(src, dst)
        assert ok is False