"""
Computes all supplementary features for the Cadenza baseline in a single pass.
This script is self-contained and performs:
1. Vocal separation (Demucs) to compute vocal SNR.
2. ASR (Canary) to generate hypotheses.
3. Scoring (BERTScore, Word Correctness).
4. Linguistic feature extraction (NLTK) for both reference and hypothesis.
5. Ratio calculation for all linguistic features.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

import hydra
import librosa  # Using librosa for energy calculation
import nemo.collections.asr.models as asr_models
import nltk
import numpy as np
import torch
from omegaconf import DictConfig
from torchmetrics.functional.text.bert import bert_score
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from tqdm import tqdm

# Import utilities from the baseline
# These are assumed to be in the same recipe folder
try:
    from clarity.utils.file_io import read_jsonl, write_jsonl, write_signal
    from clarity.utils.signal_processing import resample
    from recipes.cad_icassp_2026.baseline.shared_predict_utils import load_mixture
    from recipes.cad_icassp_2026.baseline.transcription_scorer import SentenceScorer
except ImportError as e:
    print(
        f"Error: Could not import baseline utilities: {e}. "
        "Make sure this script is run from the correct directory "
        "and your environment is set up."
    )
    exit(1)

# Set up logging
logger = logging.getLogger(__name__)


def download_nltk_resources():
    """Download required NLTK data."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        logger.info("Downloading NLTK resource: punkt")
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        logger.info("Downloading NLTK resource: averaged_perceptron_tagger")
        nltk.download("averaged_perceptron_tagger", quiet=True)


def calculate_linguistic_features(text: str) -> dict:
    """
    Calculate a set of linguistic features for a given text.
    Returns a dictionary of features.
    """
    if not text or not isinstance(text, str) or text.strip() == "":
        return {
            "n_chars": 0,
            "avg_word_len": 0.0,
            "syllables_per_word": 0.0,
            "pos_nouns": 0.0,
            "pos_verbs": 0.0,
            "pos_adj": 0.0,
            "pos_adv": 0.0,
            "content_ratio": 0.0,
        }

    try:
        words = nltk.word_tokenize(text.lower())
        if not words:
            return calculate_linguistic_features("")  # Return all zeros

        pos_tags = nltk.pos_tag(words)
        n_words = len(words)
        n_chars = len("".join(words))

        # Basic counts
        word_lengths = [len(w) for w in words]
        avg_word_len = np.mean(word_lengths) if word_lengths else 0.0

        # Syllables (simple estimation)
        # For a more accurate count, `syllables` library could be used:
        # `import syllables; total_syllables = sum(syllables.estimate(w) for w in words)`
        # Using a simple vowel-based heuristic for now
        def count_syllables_heuristic(word):
            count = 0
            vowels = "aeiouy"
            word = word.lower()
            if word and word[0] in vowels:
                count += 1
            for index in range(1, len(word)):
                if word[index] in vowels and word[index - 1] not in vowels:
                    count += 1
            if word.endswith("e"):
                count -= 1
            return max(1, count)

        total_syllables = sum(count_syllables_heuristic(w) for w in words)
        syllables_per_word = total_syllables / n_words if n_words > 0 else 0.0

        # POS Ratios
        n_nouns = sum(1 for _, tag in pos_tags if tag.startswith("NN"))
        n_verbs = sum(1 for _, tag in pos_tags if tag.startswith("VB"))
        n_adj = sum(1 for _, tag in pos_tags if tag.startswith("JJ"))
        n_adv = sum(1 for _, tag in pos_tags if tag.startswith("RB"))
        
        # Content words = N, V, Adj, Adv
        n_content = n_nouns + n_verbs + n_adj + n_adv

        return {
            "n_chars": n_chars,
            "avg_word_len": avg_word_len,
            "syllables_per_word": syllables_per_word,
            "pos_nouns": n_nouns / n_words,
            "pos_verbs": n_verbs / n_words,
            "pos_adj": n_adj / n_words,
            "pos_adv": n_adv / n_words,
            "content_ratio": n_content / n_words,
        }
    except Exception as e:
        logger.warning(f"Linguistic feature calculation failed for text '{text}': {e}")
        return calculate_linguistic_features("")  # Return all zeros


def calculate_vocal_snr(vocals_signal: np.ndarray, other_signal: np.ndarray) -> float:
    """
    Calculate the energy ratio (SNR) between vocals and other stems in dB.
    """
    try:
        # Calculate RMS energy. Add epsilon to avoid log(0)
        epsilon = np.finfo(np.float32).eps
        vocals_energy = np.sum(np.square(vocals_signal)) + epsilon
        other_energy = np.sum(np.square(other_signal)) + epsilon

        snr = 10 * np.log10(vocals_energy / other_energy)
        return float(snr)
    except Exception as e:
        logger.warning(f"Could not compute vocal SNR: {e}")
        return None  # Use None to indicate failure


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """
    Calculate a safe ratio, handling None or zero in the denominator.
    """
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


# pylint: disable = no-value-for-parameter
@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_compute_all_features(cfg: DictConfig) -> None:
    """Run the main feature computation pipeline."""
    logger.info(f"Running all-in-one feature computation on {cfg.split} set...")

    # --- 1. Load Metadata and Set Up Paths ---
    dataroot = Path(cfg.data.cadenza_data_root) / cfg.data.dataset
    dataset_filename = dataroot / "metadata" / f"{cfg.split}_metadata.json"

    with dataset_filename.open("r", encoding="utf-8") as fp:
        records = json.load(fp)

    total_records = len(records)
    
    # Define and load existing results to allow resuming
    results_file = Path(
        f"{cfg.data.dataset}.{cfg.split}.{cfg.baseline.system}.jsonl"
    )
    results = read_jsonl(str(results_file)) if results_file.exists() else []
    results_index = {result["signal"]: result for result in results}

    # Filter records to process only those not in the results file
    records_to_process = [
        record for record in records if record["signal"] not in results_index.keys()
    ]
    logger.info(
        f"Found {len(results)} existing results in {results_file}."
    )
    logger.info(
        f"Processing {len(records_to_process)} / {total_records} signals."
    )
    
    if not records_to_process:
        logger.info("All signals already processed. Exiting.")
        return

    # --- 2. Load All Models ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load Canary
    logger.info("Loading Canary ASR model (nvidia/canary-1b-v2)...")
    canary_model = asr_models.EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b-v2")
    canary_model.to(device)

    # Load Demucs
    logger.info("Loading Demucs (HDEMUCS_HIGH_MUSDB_PLUS) model...")
    separation_model = HDEMUCS_HIGH_MUSDB_PLUS.get_model()
    separation_model.to(device)

    # Load SentenceScorer (for canary_correct)
    logger.info("Initializing SentenceScorer...")
    scorer = SentenceScorer(cfg.baseline.contractions_file)

    # Warm up BERTScore model
    logger.info("Initializing BERTScore model (roberta-large)...")
    bert_score(
        ["test"], ["test"], model_name_or_path="roberta-large", device=device
    )
    
    # Download NLTK models
    download_nltk_resources()

    # --- 3. Main Processing Loop ---
    for record in tqdm(records_to_process, desc="Computing features"):
        signal_name = record["signal"]
        reference_text = record["prompt"]
        
        # This will be the dict written to the jsonl file
        result = {"signal": signal_name, "reference": reference_text}
        
        # Create a temp file path for ASR
        temp_wav_path = Path(f"temp_{signal_name}.wav")

        try:
            # Load mixture audio
            mixture_signal, sr = load_mixture(dataroot, record, cfg)
            
            # Ensure correct sample rate
            if sr != cfg.data.sample_rate:
                mixture_signal = resample(
                    mixture_signal, sr, cfg.data.sample_rate
                )
            
            # --- 3a. Demucs Vocal Separation & SNR ---
            # Prepare tensor for Demucs. We need (B, C, T).
            # load_mixture gives (T, C) [e.g., (148201, 2)] or (T,) [e.g., (148201,)]
            mixture_tensor = torch.tensor(mixture_signal, dtype=torch.float32).to(device)

            if mixture_tensor.ndim == 1:
                # It's MONO (T,). We need to make it (1, 2, T).
                # 1. Add channel dim: (1, T)
                mixture_tensor = mixture_tensor.unsqueeze(0)
                # 2. Duplicate mono to stereo (Demucs expects stereo): (2, T)
                mixture_tensor = mixture_tensor.repeat(2, 1)
                # 3. Add batch dim: (1, 2, T)
                mixture_tensor = mixture_tensor.unsqueeze(0)
            
            elif mixture_tensor.ndim == 2:
                # It's STEREO (T, C). We need to make it (1, C, T).
                # 1. Transpose: (C, T)
                mixture_tensor = mixture_tensor.T
                # 2. Add batch dim: (1, C, T)
                mixture_tensor = mixture_tensor.unsqueeze(0)
            
            else:
                raise ValueError(f"Unexpected audio shape from load_mixture: {mixture_tensor.shape}")

            # Run separation
            with torch.no_grad():
                # mixture_tensor is now guaranteed to be (1, 2, T)
                stems = separation_model(mixture_tensor)
            
            # Stems: 0=drums, 1=bass, 2=other, 3=vocals
            vocals_tensor = stems[:, 3, :, :]  # Shape: (1, 2, T)
            other_tensor = stems[:, :3, :, :].sum(dim=1) 

            # Convert to numpy [C, T] then transpose to [T, C]
            vocals_np = vocals_tensor.squeeze(0).detach().cpu().numpy().T  # (T, C)
            other_np = other_tensor.squeeze(0).detach().cpu().numpy().T  # (T, C)
            logger.debug(f"Stems shape: {stems.shape}")  # Should be (1, 4, 2, T)
            result["vocal_snr_ratio"] = calculate_vocal_snr(vocals_np, other_np)

            # --- 3b. Canary ASR ---
            # Canary ASR expects a mono signal.     
            # We must convert our stereo `mixture_signal` (shape T, 2) to mono (shape T,).
            
            asr_signal = mixture_signal  # Start with the original signal
            
            if asr_signal.ndim == 2 and asr_signal.shape[1] == 2:
                # It's stereo (T, 2), so average the channels to get (T,)
                asr_signal = np.mean(asr_signal, axis=1)
            elif asr_signal.ndim == 2 and asr_signal.shape[1] == 1:
                # It's mono but with an extra dim (T, 1)
                asr_signal = asr_signal.squeeze(axis=1)

            # Now, asr_signal is guaranteed to be a 1D numpy array (mono)
            write_signal(
                temp_wav_path, asr_signal, cfg.data.sample_rate, floating_point=True
            )
            
            # This will now load the mono WAV and work as expected
            transcription = canary_model.transcribe([str(temp_wav_path)])
            
            # FIX: Access the .text attribute from the Hypothesis object
            if transcription and transcription[0]:
                hypothesis_text = transcription[0].text  # This is the fix
            else:
                hypothesis_text = ""
            
            # Ensure it's a string (for safety, in case .text is None)
            hypothesis_text = str(hypothesis_text) if hypothesis_text is not None else ""
            
            # Now save the *clean text* to the log, not the object
            result["hypothesis"] = hypothesis_text

            # --- 3c. Scoring (BERT & Correctness) ---
            
            # NEW FIX 2: Only run bert_score if hypothesis is not empty
            # --- 3c. Scoring (BERT & Correctness) ---

            # NEW FIX 2: Only run bert_score if hypothesis is not empty
            if hypothesis_text and reference_text:
                bert_scores = bert_score(
                    [hypothesis_text], 
                    [reference_text], 
                    model_name_or_path="roberta-large", 
                    device=device
                )
                # bert_score returns a dict with keys: 'precision', 'recall', 'f1'
                result["bertscore_f1"] = bert_scores["f1"].item()
            else:
                # ASR produced no text, BERTScore is undefined.
                result["bertscore_f1"] = 0.0

            # The SentenceScorer is usually robust to empty strings,
            # so this part can stay the same.
            scores = scorer.score([reference_text], [hypothesis_text])
            total_words = scores.substitutions + scores.deletions + scores.hits
            result["canary_correct"] = (scores.hits / total_words) if total_words > 0 else 0.0

            # --- 3d. Linguistic Features ---
            ref_features = calculate_linguistic_features(reference_text)
            hyp_features = calculate_linguistic_features(hypothesis_text)
            
            # Add to result with prefixes
            result.update({f"ref_{k}": v for k, v in ref_features.items()})
            result.update({f"hyp_{k}": v for k, v in hyp_features.items()})

            # --- 3e. Linguistic Ratios ---
            for key in ref_features:
                result[f"ratio_{key}"] = safe_ratio(
                    hyp_features.get(key), ref_features.get(key)
                )

            # Append this result to the file
            write_jsonl(str(results_file), [result])

        except Exception as e:
            logger.error(
                f"--- FAILED to process {signal_name} ---"
            )
            logger.error(f"Reference: {reference_text}")
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())
            
            # Write a failure record so we don't retry it
            result_on_fail = {"signal": signal_name, "error": str(e)}
            write_jsonl(str(results_file), [result_on_fail])

        finally:
            # Clean up temp file
            if temp_wav_path.exists():
                temp_wav_path.unlink()
                
    logger.info("Feature computation complete.")


if __name__ == "__main__":
    run_compute_all_features()