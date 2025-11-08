"""
Compute combined Whisper correctness scores:
1.  Multi-sample statistics (mean, median, std) using OpenAI's original
    Whisper library with temperature-based sampling.
2.  A single, deterministic correctness score using WhisperX (faster-whisper).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import torchaudio as ta
import whisper  # Original OpenAI Whisper
import whisperx  # WhisperX
from omegaconf import DictConfig
from torch.nn import Module
from tqdm import tqdm

from clarity.utils.file_io import read_jsonl, write_jsonl, write_signal
from recipes.cad_icassp_2026.baseline.shared_predict_utils import (
    load_mixture,
)
from recipes.cad_icassp_2026.baseline.transcription_scorer import SentenceScorer

logger = logging.getLogger(__name__)


# --- OpenAI Whisper (Sampling) Functions ---

def compute_openai_stats(
    cfg: DictConfig,
    record: dict,
    signal: np.ndarray,
    sample_rate: int,
    asr_model: Module,
    scorer: SentenceScorer,
) -> dict:
    """
    Compute correctness score statistics over N samples using OpenAI Whisper.
    """
    reference = record["prompt"]
    n_samples = cfg.baseline.n_samples
    temperature = cfg.baseline.temperature

    scores_left = []
    scores_right = []

    # Use a persistent temp file for speed
    path_temp_left = Path("temp_left.flac")
    path_temp_right = Path("temp_right.flac")
    write_signal(
        filename=path_temp_left,
        signal=signal[:, 0],
        sample_rate=sample_rate,
        floating_point=False,
    )
    write_signal(
        filename=path_temp_right,
        signal=signal[:, 1],
        sample_rate=sample_rate,
        floating_point=False,
    )

    for _ in range(n_samples):
        score_left = compute_correctness_openai(
            str(path_temp_left), reference, asr_model, scorer, temperature
        )
        score_right = compute_correctness_openai(
            str(path_temp_right), reference, asr_model, scorer, temperature
        )
        scores_left.append(score_left)
        scores_right.append(score_right)

    # Clean up temp files
    path_temp_left.unlink()
    path_temp_right.unlink()

    # From the N runs, find the best score for each run
    scores_best = np.max([scores_left, scores_right], axis=0)

    # Compute statistics over these N best scores
    stats = {
        "oa_correct_mean": float(np.mean(scores_best)),
        "oa_correct_median": float(np.median(scores_best)),
        "oa_correct_std": float(np.std(scores_best)),
        "oa_correct_min": float(np.min(scores_best)),
        "oa_correct_max": float(np.max(scores_best)),
    }
    return stats


def compute_correctness_openai(
    audio_path: str,
    reference: str,
    asr_model: Module,
    scorer: SentenceScorer,
    temperature: float,
) -> float:
    """
    Compute correctness for a single run using OpenAI Whisper.
    """
    # Run Whisper ASR with temperature sampling
    result = asr_model.transcribe(
        audio_path,
        fp16=torch.cuda.is_available(),
        language="en",
        temperature=temperature,
        beam_size=1,  # CRITICAL: Use sampling, not beam search
        best_of=1,    # CRITICAL: Single sample for randomness
    )
    hypothesis = result["text"].strip()

    # Score the transcription
    results = scorer.score([reference], [hypothesis])
    total_words = results.substitutions + results.deletions + results.hits

    return (results.hits / total_words) if total_words > 0 else 0.0


# --- WhisperX (Deterministic) Functions ---

def compute_whisperx_single(
    cfg: DictConfig,
    record: dict,
    signal: np.ndarray,
    sample_rate: int,
    asr_model: Module,
    scorer: SentenceScorer,
) -> float:
    """
    Compute a single deterministic correctness score using WhisperX.
    """
    reference = record["prompt"]
    batch_size = cfg.baseline.whisperx_batch_size

    score_left = compute_correctness_whisperx(
        signal[:, 0], reference, asr_model, scorer, batch_size, sample_rate
    )
    score_right = compute_correctness_whisperx(
        signal[:, 1], reference, asr_model, scorer, batch_size, sample_rate
    )

    return max(score_left, score_right)


def compute_correctness_whisperx(
    signal: np.ndarray,
    reference: str,
    asr_model: Module,
    scorer: SentenceScorer,
    batch_size: int,
    sample_rate: int = 48000,
) -> float:
    """
    Compute correctness for a single run using WhisperX.
    
    CRITICAL FIXES:
    1. Resample to 16kHz (WhisperX expects 16kHz)
    2. Convert to float32 (pyannote VAD expects float32, not float64)
    """
    # CRITICAL FIX: WhisperX expects 16kHz float32 audio
    target_sr = 16000
    
    # Resample if needed
    if sample_rate != target_sr:
        wav = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        resampler = ta.transforms.Resample(sample_rate, target_sr)
        wav_16k = resampler(wav).squeeze(0).numpy()
    else:
        wav_16k = signal
    
    # CRITICAL FIX: Ensure float32 (not float64) to avoid dtype mismatch with VAD model
    # This fixes: "torch.cuda.DoubleTensor does not equal torch.cuda.FloatTensor"
    wav_16k = wav_16k.astype(np.float32)
    
    # Run WhisperX ASR
    result = asr_model.transcribe(
        audio=wav_16k,
        batch_size=batch_size,
        language="en",
        task="transcribe",
    )

    # Join text from all segments
    hypothesis = " ".join([seg["text"] for seg in result["segments"]]).strip()

    # Score the transcription
    results = scorer.score([reference], [hypothesis])
    total_words = results.substitutions + results.deletions + results.hits

    return (results.hits / total_words) if total_words > 0 else 0.0


# --- Main Runner ---

def run_asr_from_mixture(
    dataroot: Path, records: list, results_file: Path, cfg: DictConfig
) -> None:
    """
    Load mixture signals and run both ASR models.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 1. Load OpenAI Whisper model (for sampling)
    logger.info(
        f"Loading OpenAI Whisper model: {cfg.baseline.whisper_version}"
    )
    openai_model = whisper.load_model(
        cfg.baseline.whisper_version, device=device
    )

    # 2. Load WhisperX model (for deterministic score)
    logger.info(
        f"Loading WhisperX model: {cfg.baseline.whisper_version} "
        f"({cfg.baseline.compute_type})"
    )
    whisperx_model = whisperx.load_model(
        cfg.baseline.whisper_version,
        device,
        compute_type=cfg.baseline.compute_type,
        language="en",  # Set language to avoid detection overhead
        task="transcribe",
    )

    # 3. Initialize Scorer
    scorer = SentenceScorer(cfg.baseline.contractions_file)

    # Iterate through the signals that need scoring
    for record in tqdm(records):
        signal_name = record["signal"]

        # Load mixture
        signal_to_process, sample_rate = load_mixture(dataroot, record, cfg)

        # Compute OpenAI (sampling) stats
        openai_stats = compute_openai_stats(
            cfg, record, signal_to_process, sample_rate, openai_model, scorer
        )

        # Compute WhisperX (deterministic) score
        whisperx_score = compute_whisperx_single(
            cfg, record, signal_to_process, sample_rate, whisperx_model, scorer
        )

        # Combine results and append to file
        result = {
            "signal": signal_name,
            **openai_stats,
            "whisperx_correct": float(whisperx_score),
        }
        write_jsonl(str(results_file), [result])


# pylint: disable = no-value-for-parameter
@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_compute_whisper_combined(cfg: DictConfig) -> None:
    """
    Run the combined Whisper/WhisperX pipeline.
    """
    assert cfg.baseline.name == "whisperx_multi"
    logger.info(
        f"Running {cfg.baseline.system} baseline on {cfg.split} set..."
    )

    dataroot = Path(cfg.data.cadenza_data_root) / cfg.data.dataset
    dataset_filename = dataroot / "metadata" / f"{cfg.split}_metadata.json"
    with dataset_filename.open("r", encoding="utf-8") as fp:
        records = json.load(fp)

    total_records = len(records)
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

    records = [
        record
        for record in records
        if record["signal"] not in results_index.keys()
    ]
    records = records[cfg.baseline.batch - 1 :: cfg.baseline.n_batches]

    logger.info(
        f"Computing scores for {len(records)} out of {total_records} signals"
    )
    run_asr_from_mixture(dataroot, records, results_file, cfg)


if __name__ == "__main__":
    run_compute_whisper_combined()