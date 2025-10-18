"""Compute the Songprep correctness scores."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.nn import Module
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoProcessor # Use appropriate classes
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from recipes.cad_icassp_2026.baseline.transcription_scorer import SentenceScorer
from clarity.utils.signal_processing import resample # Ensure resample is imported

from clarity.utils.file_io import read_jsonl, write_jsonl, write_signal
from recipes.cad_icassp_2026.baseline.shared_predict_utils import (
    load_mixture,
)
from recipes.cad_icassp_2026.baseline.transcription_scorer import SentenceScorer

logger = logging.getLogger(__name__)


def compute_asr_for_signal(
    cfg: DictConfig, record: dict, signal: np.ndarray, asr_model: Module
) -> float:
    """Compute the correctness score for a given signal.

    Args:

        cfg (DictConfig): configuration object.
        record (dict): the metadata dict for the signal.
        signal (np.ndarray): the signal to compute the score for.
        asr_model (Module): the ASR model to use for transcription.

    Returns:

        float: correctness score
    """
    reference = record["prompt"]

    score_left = compute_correctness(
        signal[:, 0],
        cfg.data.sample_rate,
        reference,
        asr_model,
        cfg.baseline.contractions_file,
    )
    score_right = compute_correctness(
        signal[:, 1],
        cfg.data.sample_rate,
        reference,
        asr_model,
        cfg.baseline.contractions_file,
    )

    return np.max([score_left, score_right])


def compute_correctness(
    signal: np.ndarray,
    sample_rate: int,
    reference: str,
    asr_model: Module,
    contraction_file: str,
) -> float:
    """Compute the correctness score for a given signal.

    Args:
        signal (np.ndarray): the signal to compute the score for
        sample_rate (int): the sample rate of the signal
        reference (str): the reference transcription
        asr_model (Module): the ASR model to use for transcription
        contraction_file (str): path to the contraction file for the scorer

    Returns:
        float: correctness score.
    """
    scorer = SentenceScorer(contraction_file)

    # create a temporary file to store the signal as flac
    # for Whisper to open it
    path_temp = Path("temp.flac")
    write_signal(
        filename=path_temp, signal=signal, sample_rate=sample_rate, floating_point=False
    )

    # Run Whisper ASR
    hypothesis = asr_model.transcribe(
        str(path_temp), fp16=False, language="en", temperature=0.0
    )["text"]

    # Score the transcription
    results = scorer.score([reference], [hypothesis])
    total_words = results.substitutions + results.deletions + results.hits

    # Delete temporal file
    Path(path_temp).unlink()

    return results.hits / total_words


def run_asr_from_mixture(
    dataroot: Path, records: list, results_file: Path, cfg: DictConfig
) -> None:
    """Load the mixture signal for a given record.

    Args:

        dataroot (Path): the root path to the dataset.
        records (list): list of records to process.
        results_file (Path): path to the results file.
        cfg (DictConfig): configuration object.
    """
    # Prepare dnn models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    asr_model = whisper.load_model(cfg.baseline.whisper_version, device=device)

    # Iterate through the signals that need scoring
    for record in tqdm(records):
        signal_name = record["signal"]

        # Load mixture
        signal_to_whisper, _ = load_mixture(dataroot, record, cfg)

        # Compute ASR
        correct = compute_asr_for_signal(cfg, record, signal_to_whisper, asr_model)

        # Results are appended to the results file to allow interruption
        result = {"signal": signal_name, f"{cfg.baseline.system}": correct}
        write_jsonl(str(results_file), [result])


# pylint: disable = no-value-for-parameter
@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_compute_songprep(cfg: DictConfig) -> None:
    """Run the songprep to compute correctness hits/total words."""
    assert cfg.baseline.name == "songprep"

    logger.info(f"Running {cfg.baseline.system} baseline on {cfg.split} set...")

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
        record for record in records if record["signal"] not in results_index.keys()
    ]
    records = records[cfg.baseline.batch - 1 :: cfg.baseline.n_batches]

    # Iterate over the signals that need scoring
    logger.info(f"Computing scores for {len(records)} out of {total_records} signals")

    run_songprep_from_mixture(dataroot, records, results_file, cfg)

def compute_songprep_score_for_signal(
    signal: np.ndarray,
    sample_rate: int,
    reference_prompt: str,
    songprep_model,
    songprep_processor,
    scorer: SentenceScorer,
    device: str,
) -> float:
    """
    Compute correctness score for a signal using SongPrep.
    This function replaces compute_asr_for_signal and compute_correctness.
    """
    
    # SongPrep might expect a specific sample rate, e.g., 16000 Hz
    # Let's assume 16000 for this example
    TARGET_SR = 16000
    if sample_rate != TARGET_SR:
        signal = resample(signal, sample_rate, TARGET_SR)

    # Convert to mono for transcription if needed
    if signal.ndim > 1 and signal.shape[1] > 1:
        signal = signal.mean(axis=1)


    # inputs = songprep_processor(signal, sampling_rate=TARGET_SR, return_tensors="pt").to(device)

    # predicted_ids = songprep_model.generate(**inputs)
    # hypothesis = songprep_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    if isinstance(signal, np.ndarray):
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        signal = signal.astype("float32")

    # Prepare inputs for Whisper
    inputs = songprep_processor(
        audio=signal,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
    )

    # Move tensors to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # For Whisper, the key is "input_features"
    predicted_ids = songprep_model.generate(inputs["input_features"])

    # Decode using the tokenizer inside the processor
    hypothesis = songprep_processor.tokenizer.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0]
    
    # The output might be structured, e.g., "[verse][...
    # We need to parse out just the lyric text. A simple regex can work for this.
    import re
    lyrics_only = re.sub(r'\[.*?\]', '', hypothesis).strip()

    # Score the transcription against the ground-truth prompt
    results = scorer.score([reference_prompt], [lyrics_only])
    total_words = results.substitutions + results.deletions + results.hits
    
    if total_words == 0:
        return 0.0 # Avoid division by zero if reference is empty or no words are matched

    return results.hits / total_words


def run_songprep_from_mixture(
    dataroot: Path, records: list, results_file: Path, cfg: DictConfig
) -> None:
    """
    Main processing loop for SongPrep.
    This function replaces run_asr_from_mixture.
    """
    # Prepare DNN models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load the SongPrep model and processor
    # Replace "your-songprep-model-name" with the actual model from Hugging Face
    # model_name = "tencent/SongPrep-7B"
    # processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

    model_name = getattr(cfg.baseline, "tencent/SongPrep-7B", "openai/whisper-small.en")

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    # Initialize the scorer for comparing transcriptions
    scorer = SentenceScorer(cfg.baseline.contractions_file)

    # Iterate through the signals that need scoring
    for record in tqdm(records):
        signal_name = record["signal"]
        
        # Load the audio to be transcribed
        # We use the "processed" signal (with hearing loss) as input
        signal, sr = load_mixture(dataroot, record, cfg)

        # Compute the correctness score
        correctness_score = compute_songprep_score_for_signal(
            signal, sr, record["prompt"], model, processor, scorer, device
        )

        # Results are appended to the results file to allow interruption
        result = {"signal": signal_name, f"{cfg.baseline.system}": correctness_score}
        write_jsonl(str(results_file), [result])


if __name__ == "__main__":
    run_compute_songprep()


