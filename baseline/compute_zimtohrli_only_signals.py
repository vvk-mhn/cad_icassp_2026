# recipes/cad_icassp_2026/baseline/compute_zimtohrli.py
"""Compute Zimtohrli distance scores between a reference and a processed signal (no separation)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import zimtohrli
from clarity.utils.file_io import read_jsonl, write_jsonl
from clarity.utils.flac_encoder import read_flac_signal
from clarity.utils.signal_processing import resample
from recipes.cad_icassp_2026.baseline.shared_predict_utils import input_align

logger = logging.getLogger(__name__)


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x[:, None]
    return x


def _compute_single_z(
    z: zimtohrli.Pyohrli,
    reference: np.ndarray,
    processed: np.ndarray,
    fsamp: int,
    zimtohrli_fsamp: int = 48000,
) -> float:
    """Compute Zimtohrli distance for one channel."""
    # Resample to target rate (Zimtohrli expects 48kHz)
    ref48 = resample(reference, fsamp, zimtohrli_fsamp)
    proc48 = resample(processed, fsamp, zimtohrli_fsamp)

    # Align signals
    ref48, proc48 = input_align(ref48, proc48, fsamp=int(zimtohrli_fsamp))

    # Cast to float32 for bindings
    ref48 = ref48.astype(np.float32)
    proc48 = proc48.astype(np.float32)

    return float(z.distance(ref48, proc48))


def zimtohrli_pair_distance(
    reference_signal: np.ndarray,
    processed_signal: np.ndarray,
    fsamp: int,
    zimtohrli_fsamp: int = 48000,
    channel_reduce: str = "min",
) -> float:
    """Compute Zimtohrli distance between two multichannel signals."""
    reference_signal = _ensure_2d(reference_signal)
    processed_signal = _ensure_2d(processed_signal)

    # Normalize both by reference peak to keep scale consistent
    ref_norm = np.max(np.abs(reference_signal)) or 1.0
    reference_signal = reference_signal / ref_norm
    processed_signal = processed_signal / ref_norm

    z = zimtohrli.Pyohrli()
    C = min(reference_signal.shape[1], processed_signal.shape[1])

    dists = []
    for ch in range(C):
        d = _compute_single_z(
            z,
            reference_signal[:, ch],
            processed_signal[:, ch],
            fsamp=fsamp,
            zimtohrli_fsamp=zimtohrli_fsamp,
        )
        dists.append(float(d))

    if channel_reduce == "mean":
        return float(np.mean(dists))
    return float(np.min(dists))


def _resolve_subdirs_from_cfg(cfg: DictConfig) -> Tuple[str, str]:
    """Decide subdirectories used for reference and processed signals."""
    # Highest priority: explicit cfg.paths.*
    ref_subdir = getattr(cfg.paths, "reference_subdir", None) if hasattr(cfg, "paths") else None
    proc_subdir = getattr(cfg.paths, "processed_subdir", None) if hasattr(cfg, "paths") else None

    if ref_subdir and proc_subdir:
        return ref_subdir, proc_subdir

    # Convenience: baseline.reference can hint the reference subdir name
    # Example: baseline.reference: unprocessed -> reference_subdir="unprocessed"
    ref_hint = getattr(cfg.baseline, "reference", None)
    if ref_subdir is None:
        if ref_hint:
            ref_subdir = str(ref_hint)
        else:
            ref_subdir = "reference_signals"  # sensible default
    if proc_subdir is None:
        proc_subdir = "signals"  # default processed subdir

    return ref_subdir, proc_subdir


def _resolve_paths(
    dataroot: Path,
    split: str,
    record: Dict[str, Any],
    reference_subdir: str,
    processed_subdir: str,
) -> Tuple[Path, Path, str]:
    """
    Determine reference and processed paths from either metadata fields or subdir pattern.

    Priority:
      1) record['reference'] and record['processed'] if present (absolute or relative to dataroot)
      2) dataroot / split / <subdir> / f"{record['signal']}.flac"
    """
    signal_name = record.get("signal")
    ref_field = record.get("reference")
    proc_field = record.get("processed")

    if ref_field and proc_field:
        ref_path = Path(ref_field)
        proc_path = Path(proc_field)
        if not ref_path.is_absolute():
            ref_path = dataroot / ref_path
        if not proc_path.is_absolute():
            proc_path = dataroot / proc_path
        if not signal_name:
            signal_name = ref_path.stem
    else:
        if signal_name is None:
            raise ValueError("Record missing 'signal' and no explicit 'reference'/'processed' fields.")
        ref_path = dataroot / split / reference_subdir / f"{signal_name}_unproc.flac"
        proc_path = dataroot / split / processed_subdir / f"{signal_name}.flac"

    return ref_path, proc_path, signal_name


# pylint: disable=no-value-for-parameter
@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_compute_zimtohrli(cfg: DictConfig) -> None:
    """
    Computes Zimtohrli distances between reference and processed waveforms.
    No source separation or vocals-only logic.
    """
    assert cfg.baseline.name == "zimtohrli", "Use baseline=zimtohrli for this script."

    logger.info(f"Running {cfg.baseline.system} baseline on {cfg.split} set (no separation).")
    logger.debug("Config:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    dataroot = Path(cfg.data.cadenza_data_root) / cfg.data.dataset
    dataset_filename = dataroot / "metadata" / f"{cfg.split}_metadata.json"
    if not dataset_filename.exists():
        raise FileNotFoundError(f"Metadata not found: {dataset_filename}")

    with dataset_filename.open("r", encoding="utf-8") as fp:
        records = json.load(fp)
    total_records = len(records)

    # Batch helper in filename to allow multi-batch runs
    batch_str = (
        f".{cfg.baseline.batch}_{cfg.baseline.n_batches}"
        if cfg.baseline.n_batches > 1
        else ""
    )
    results_file = Path(f"{cfg.data.dataset}.{cfg.split}.{cfg.baseline.system}{batch_str}.jsonl")

    # Load existing results (so we can resume)
    results = read_jsonl(str(results_file)) if results_file.exists() else []
    results_index = {result["signal"]: result for result in results}

    # Only compute missing ones
    records = [r for r in records if (r.get("signal") and r["signal"] not in results_index)]
    # Stride for batch selection
    records = records[cfg.baseline.batch - 1 :: cfg.baseline.n_batches]

    logger.info(f"Computing scores for {len(records)} out of {total_records} signals")
    reference_subdir, processed_subdir = _resolve_subdirs_from_cfg(cfg)
    logger.info(f"Reference subdir: {reference_subdir} | Processed subdir: {processed_subdir}")

    target_sr = int(cfg.data.sample_rate)
    zim_fs = int(getattr(cfg.baseline, "zimtohrli_sample_rate", 48000))

    for record in tqdm(records):
        ref_path, proc_path, signal_name = _resolve_paths(
            dataroot,
            cfg.split,
            record,
            reference_subdir=reference_subdir,
            processed_subdir=processed_subdir,
        )

        if not ref_path.exists():
            logger.warning(f"Missing reference file: {ref_path}; skipping {signal_name}")
            continue
        if not proc_path.exists():
            logger.warning(f"Missing processed file: {proc_path}; skipping {signal_name}")
            continue

        # Load both signals
        reference, ref_sr = read_flac_signal(ref_path)
        processed, proc_sr = read_flac_signal(proc_path)

        # Resample to a common working rate before Zimtohrli-internal resample
        if ref_sr != target_sr:
            reference = resample(reference, ref_sr, target_sr)
        if proc_sr != target_sr:
            processed = resample(processed, proc_sr, target_sr)

        # Compute distance (lower is better)
        zim_distance = zimtohrli_pair_distance(
            reference_signal=reference,
            processed_signal=processed,
            fsamp=target_sr,
            zimtohrli_fsamp=zim_fs,
            channel_reduce="min",
        )

        # Write one JSONL record per signal
        result = {"signal": signal_name, f"{cfg.baseline.system}": float(zim_distance)}
        write_jsonl(str(results_file), [result])

    logger.info(f"Done. Results appended to {results_file}")


if __name__ == "__main__":
    run_compute_zimtohrli()
