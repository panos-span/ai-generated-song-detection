"""
Validate downloaded datasets for file integrity, sample rate, and format.

Usage:
    # Check all datasets (report only)
    uv run python data/validate_datasets.py --data_dir data

    # Check a single dataset
    uv run python data/validate_datasets.py --data_dir data --dataset fakemusiccaps

    # Delete invalid files + rewrite CSVs, then re-run Phase 1 download commands
    uv run python data/validate_datasets.py --data_dir data --fix

Exit code: 0 if all files valid, 1 if any invalid files found.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf


# ---------------------------------------------------------------------------
# Per-file validation
# ---------------------------------------------------------------------------

def validate_wav(
    path: Path,
    min_duration: float = 0.1,
    expected_sr: int = 16_000,
) -> tuple[bool, str]:
    """Header-only WAV validation via soundfile.info() (no full audio decode).

    Returns (is_valid, reason).  reason is empty string when valid.
    """
    if not path.exists():
        return False, "missing"
    if path.stat().st_size == 0:
        return False, "empty file (0 bytes)"
    try:
        info = sf.info(str(path))
    except Exception as exc:  # noqa: BLE001
        return False, f"corrupt (soundfile error: {exc})"
    if info.samplerate != expected_sr:
        return False, f"wrong sample rate ({info.samplerate} Hz, expected {expected_sr})"
    if info.channels != 1:
        return False, f"not mono ({info.channels} channels)"
    if info.duration <= min_duration:
        return False, f"too short ({info.duration:.3f}s, min {min_duration}s)"
    return True, ""


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------

_DatasetResult = dict[str, tuple[bool, str]]  # path_str -> (valid, reason)


def scan_dataset(
    name: str,
    audio_dir: Path,
    csv_path: Path,
    file_columns: list[str],
    min_duration: float,
    expected_sr: int,
) -> tuple[_DatasetResult, list[Path]]:
    """Validate all audio files referenced in csv_path.

    Returns:
        results: {absolute_path_str: (valid, reason)} for all CSV-referenced files
        orphaned: paths on disk that are not referenced in the CSV
    """
    results: _DatasetResult = {}

    if not csv_path.exists():
        print(f"  [{name}] metadata CSV not found: {csv_path}")
        return results, []

    df = pd.read_csv(csv_path)

    referenced: set[str] = set()
    for _, row in df.iterrows():
        for col in file_columns:
            if col not in df.columns:
                continue
            value = str(row[col]).strip()
            if not value or value == "nan":
                # MIPPIA rows where a track download previously failed — skip
                continue
            abs_path = audio_dir / value
            path_str = str(abs_path)
            referenced.add(path_str)
            if path_str not in results:
                valid, reason = validate_wav(abs_path, min_duration, expected_sr)
                results[path_str] = (valid, reason)

    # Orphaned: on disk but not in CSV — report only, never delete
    orphaned: list[Path] = []
    if audio_dir.exists():
        for p in sorted(audio_dir.iterdir()):
            if p.is_file() and str(p) not in referenced:
                orphaned.append(p)

    return results, orphaned


# ---------------------------------------------------------------------------
# Fix: delete invalid files + rewrite CSV
# ---------------------------------------------------------------------------

def fix_dataset(
    invalid_paths: list[str],
    csv_path: Path,
    file_columns: list[str],
    audio_dir: Path,
) -> tuple[int, int]:
    """Delete invalid files and remove their rows from the metadata CSV.

    Returns (files_deleted, csv_rows_removed).
    """
    if not invalid_paths:
        return 0, 0

    # Build set of bare filenames that were deleted (for CSV matching)
    deleted_names: set[str] = set()
    files_deleted = 0
    for path_str in invalid_paths:
        p = Path(path_str)
        if p.exists():
            p.unlink()
            files_deleted += 1
        deleted_names.add(p.name)

    rows_removed = 0
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        original_len = len(df)
        mask = pd.Series([True] * len(df), index=df.index)
        for col in file_columns:
            if col not in df.columns:
                continue
            # Keep rows only if their file column value is not in deleted_names
            col_names = df[col].astype(str).str.strip().apply(
                lambda v: Path(v).name if v and v != "nan" else ""
            )
            mask &= ~col_names.isin(deleted_names)
        df = df[mask]
        rows_removed = original_len - len(df)
        df.to_csv(csv_path, index=False)

    return files_deleted, rows_removed


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _categorise(results: _DatasetResult) -> dict[str, list[str]]:
    cats: dict[str, list[str]] = {
        "ok": [],
        "missing": [],
        "corrupt": [],
        "wrong_format": [],
    }
    for path_str, (valid, reason) in results.items():
        if valid:
            cats["ok"].append(path_str)
        elif reason == "missing":
            cats["missing"].append(path_str)
        elif reason.startswith("corrupt"):
            cats["corrupt"].append(path_str)
        else:
            cats["wrong_format"].append(path_str)
    return cats


def _print_summary(
    name: str,
    cats: dict[str, list[str]],
    orphaned: list[Path],
) -> None:
    total = sum(len(v) for v in cats.values())
    n_ok = len(cats["ok"])
    n_miss = len(cats["missing"])
    n_corrupt = len(cats["corrupt"])
    n_fmt = len(cats["wrong_format"])
    n_orphan = len(orphaned)

    status = "OK" if total == n_ok else "ISSUES FOUND"
    print(f"\n{'='*60}")
    print(f"  Dataset: {name}  [{status}]")
    print(f"  Total: {total}  Valid: {n_ok}  Missing: {n_miss}  "
          f"Corrupt: {n_corrupt}  Wrong format: {n_fmt}  Orphaned: {n_orphan}")
    print(f"{'='*60}")

    for label, paths in [
        ("MISSING", cats["missing"]),
        ("CORRUPT", cats["corrupt"]),
        ("WRONG FORMAT", cats["wrong_format"]),
    ]:
        for p in paths:
            print(f"  [{label}] {p}")

    if orphaned:
        print(f"  [{len(orphaned)} ORPHANED files — on disk but not in CSV, not deleted]")
        for p in orphaned[:5]:
            print(f"    {p}")
        if len(orphaned) > 5:
            print(f"    ... and {len(orphaned) - 5} more")


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

def _build_registry(data_dir: Path) -> dict[str, dict]:
    """Return per-dataset config: audio_dir, csv_path, file_columns."""
    return {
        "fakemusiccaps": {
            "audio_dir": data_dir / "fakemusiccaps" / "audio",
            "csv_path":  data_dir / "fakemusiccaps" / "metadata.csv",
            "file_columns": ["filename"],
        },
        "mippia": {
            "audio_dir": data_dir / "mippia" / "audio",
            "csv_path":  data_dir / "mippia" / "metadata.csv",
            "file_columns": ["track_a", "track_b"],
        },
        "sonics": {
            "audio_dir": data_dir / "sonics" / "audio",
            "csv_path":  data_dir / "sonics" / "metadata.csv",
            "file_columns": ["filename"],
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate downloaded audio datasets for integrity, sample rate, and format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", default="data", help="Root data directory")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", "fakemusiccaps", "mippia", "sonics"],
        help="Which dataset to validate",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Delete invalid files and remove their rows from metadata CSVs",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=0.1,
        help="Minimum valid audio duration in seconds",
    )
    parser.add_argument(
        "--expected_sr",
        type=int,
        default=16_000,
        help="Expected audio sample rate in Hz",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    registry = _build_registry(data_dir)

    datasets_to_check = (
        list(registry.keys()) if args.dataset == "all" else [args.dataset]
    )

    any_invalid = False

    for name in datasets_to_check:
        cfg = registry[name]
        print(f"\nScanning {name} ...")
        results, orphaned = scan_dataset(
            name=name,
            audio_dir=cfg["audio_dir"],
            csv_path=cfg["csv_path"],
            file_columns=cfg["file_columns"],
            min_duration=args.min_duration,
            expected_sr=args.expected_sr,
        )

        cats = _categorise(results)
        _print_summary(name, cats, orphaned)

        invalid_paths = cats["missing"] + cats["corrupt"] + cats["wrong_format"]
        if invalid_paths:
            any_invalid = True

        if args.fix and invalid_paths:
            files_del, rows_rm = fix_dataset(
                invalid_paths=invalid_paths,
                csv_path=cfg["csv_path"],
                file_columns=cfg["file_columns"],
                audio_dir=cfg["audio_dir"],
            )
            print(
                f"\n  [FIX] Deleted {files_del} invalid file(s), "
                f"removed {rows_rm} CSV row(s) from {cfg['csv_path']}"
            )

    print()
    if any_invalid:
        if args.fix:
            print(
                "Invalid files have been removed. Re-run the Phase 1 download commands\n"
                "to refetch missing files (the scripts are idempotent and will skip\n"
                "files that already exist on disk):\n\n"
                "  uv run python data/download_fakemusiccaps.py --num_samples 1000 "
                "--output_dir data/fakemusiccaps\n"
                "  uv run python data/download_mippia.py --output_dir data/mippia\n"
                "  uv run python data/download_sonics.py --num_samples 500 "
                "--output_dir data/sonics"
            )
        else:
            print(
                "Invalid files found. Re-run with --fix to delete them and rewrite CSVs,\n"
                "then re-run the Phase 1 download commands to refetch the missing files."
            )
        return 1
    else:
        print("All files valid.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
