"""
Compute multiple WhisperX correctness scores by sampling with temperature.

This script runs WhisperX transcription N times for each signal 
with a non-zero temperature to generate different hypotheses. 
It then computes statistics (mean, median, std, min, max) over the 
N correctness scores.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
# from typing import Module
from torch.nn import Module

import torchaudio as ta

import hydra
import numpy as np
import torch
import whisperx  # Use whisperx
from omegaconf import DictConfig
from tqdm import tqdm

from clarity.utils.file_io import read_jsonl, write_jsonl
from recipes.cad_icassp_2026.baseline.shared_predict_utils import (
    load_mixture,
)
from recipes.cad_icassp_2026.baseline.transcription_scorer import SentenceScorer

logger = logging.getLogger(__name__)


def compute_asr_stats_for_signal(
    cfg: DictConfig, record: dict, signal: np.ndarray, asr_model: Module
) -> dict:
    """
    Compute correctness score statistics for a given signal over N samples.

    Args:
        cfg (DictConfig): configuration object.
        record (dict): the metadata dict for the signal.
        signal (np.ndarray): the signal to compute the score for.
        asr_model (Module): the ASR model to use for transcription.

    Returns:
        dict: A dictionary of statistics (mean, median, std, min, max).
    """
    reference = record["prompt"]
    n_samples = cfg.baseline.n_samples
    temperature = cfg.baseline.temperature
    batch_size = cfg.baseline.whisperx_batch_size

    # We collect N scores for left and N scores for right
    scores_left = []
    scores_right = []

    scorer = SentenceScorer(cfg.baseline.contractions_file)

    logger.info(
        f"Computing {n_samples} transcriptions for signal {record['signal']}..."
    )
    for _ in range(n_samples):
        score_left = compute_correctness(
            signal[:, 0],
            reference,
            asr_model,
            scorer,
            temperature,
            batch_size,
        )
        score_right = compute_correctness(
            signal[:, 1],
            reference,
            asr_model,
            scorer,
            temperature,
            batch_size,
        )
        scores_left.append(score_left)
        scores_right.append(score_right)

    # From the N runs, find the best score for each run
    # (max of left/right)
    scores_best = np.max([scores_left, scores_right], axis=0)

    # Now compute statistics over these N best scores
    stats = {
        "wx_correct_mean": np.mean(scores_best),
        "wx_correct_median": np.median(scores_best),
        "wx_correct_std": np.std(scores_best),
        "wx_correct_min": np.min(scores_best),
        "wx_correct_max": np.max(scores_best),
    }
    return stats


# def compute_correctness(
#     signal: np.ndarray,
#     reference: str,
#     asr_model: Module,
#     scorer: SentenceScorer,
#     temperature: float,
#     batch_size: int,
# ) -> float:
#     """
#     Compute the correctness score for a given signal.

#     Args:
#         signal (np.ndarray): the signal to compute the score for
#         reference (str): the reference transcription
#         asr_model (Module): the ASR model to use for transcription
#         scorer (SentenceScorer): The sentence scorer instance.
#         temperature (float): Temperature for sampling.
#         batch_size (int): Batch size for WhisperX.

#     Returns:
#         float: correctness score.
#     """
    
#     # Run WhisperX ASR
#     # WhisperX's transcribe method takes the numpy array directly
#     result = asr_model.transcribe(
#         signal,
#         batch_size=batch_size,
#         temperature=temperature,
#         # We don't need alignment, just the transcript
#         align_model=None, 
#     )

#     # Join text from all segments
#     hypothesis = " ".join([seg["text"] for seg in result["segments"]]).strip()

#     # Score the transcription
#     results = scorer.score([reference], [hypothesis])
#     total_words = results.substitutions + results.deletions + results.hits

#     if total_words == 0:
#         return 0.0  # Avoid division by zero if reference is empty

#     return results.hits / total_words


def compute_correctness(
    signal: np.ndarray,
    reference: str,
    asr_model: Module,
    scorer: SentenceScorer,
    temperature: float,  # Note: temperature is already set in the model
    batch_size: int,
) -> float:
    # Resample to 16kHz as before
    orig_sr = 48000
    target_sr = 16000
    if orig_sr != target_sr:
        wav = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        resampler = ta.transforms.Resample(orig_sr, target_sr)
        wav_16k = resampler(wav).squeeze(0).numpy().astype(np.float32)
    else:
        wav_16k = signal.astype(np.float32)

    # Call transcribe with only supported parameters
    result = asr_model.transcribe(
        audio=wav_16k,
        batch_size=batch_size,
        language="en",
        task="transcribe",
        # Remove vad and decode_options - they're not supported here
    )

    # Join text from segments
    hypothesis = " ".join(seg.get("text", "") for seg in result.get("segments", []))
    hypothesis = hypothesis.strip()

    # Score
    results = scorer.score([reference], [hypothesis])
    total_words = results.substitutions + results.deletions + results.hits
    return (results.hits / total_words) if total_words > 0 else 0.0


def run_asr_from_mixture(
    dataroot: Path, records: list, results_file: Path, cfg: DictConfig
) -> None:
    """
    Load the mixture signal for a given record.

    Args:
        dataroot (Path): the root path to the dataset.
        records (list): list of records to process.
        results_file (Path): path to the results file.
        cfg (DictConfig): configuration object.
    """
    # Prepare dnn models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Use WhisperX load_model
    # Note: whisper_version is now the model name, e.g., "base.en" or "large-v2"
    asr_options = {
        "beam_size": 1,  # CRITICAL: Must be 1 for sampling
        "best_of": 1,    # CRITICAL: Must be 1 for sampling
        "temperatures": [cfg.baseline.temperature],  # Single temperature
        "patience": 1,
        "length_penalty": 1.0,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": True,
    }

    asr_model = whisperx.load_model(
        cfg.baseline.whisper_version,
        device,
        compute_type=cfg.baseline.compute_type,
        asr_options=asr_options,
        language="en",
        task="transcribe"
    )
    
    
    # We don't load the alignment model here to save VRAM, 
    # as it's not needed for transcription.

    # Iterate through the signals that need scoring
    for record in tqdm(records):
        signal_name = record["signal"]

        # Load mixture
        signal_to_whisper, _ = load_mixture(dataroot, record, cfg)

        # Compute ASR stats
        stats = compute_asr_stats_for_signal(
            cfg, record, signal_to_whisper, asr_model
        )

        # Results are appended to the results file to allow interruption
        result = {"signal": signal_name, **stats}
        write_jsonl(str(results_file), [result])


# pylint: disable = no-value-for-parameter
@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_compute_whisperx_multi(cfg: DictConfig) -> None:
    """
    Run WhisperX N times to compute correctness score statistics.
    """
    assert cfg.baseline.name == "whisperx_multi"

    logger.info(
        f"Running {cfg.baseline.system} baseline on {cfg.split} set..."
    )

    # Load the set of signal for which we need to compute scores
    dataroot = Path(cfg.data.cadenza_data_root) / cfg.data.dataset

    dataset_filename = dataroot / "metadata" / f"{cfg.split}_metadata.json"

    with dataset_filename.open("r", encoding="utf-8") as fp:
        records = json.load(fp)

    total_records = len(records)
    # Load existing results file if present
    batch_str = (
        f".{cfg.baseline.batch}_{cfg.baseline.n_batches}"
        if cfg.baseline.n_batches > 1
        else ""
    )

    results_file = Path(
        f"{cfg.data.dataset}.{cfg.split}.{cfg.baseline.system}{batch_str}.jsonl"
    )
    results = read_jsonl(str(results_file)) if results_file.exists() else []
    results_index = {result["signal"]: result for result in results}

    # Find signals for which we don't have scores
    records = [
        record
        for record in records
        if record["signal"] not in results_index.keys()
    ]
    records = records[cfg.baseline.batch - 1 :: cfg.baseline.n_batches]

    # Iterate over the signals that need scoring
    logger.info(
        f"Computing scores for {len(records)} out of {total_records} signals"
    )

    run_asr_from_mixture(dataroot, records, results_file, cfg)


if __name__ == "__main__":
    run_compute_whisperx_multi()