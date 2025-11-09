"""
Compute multiple feature sets for the Cadenza CLIP challenge:
1. Legacy Text: Whisper Correctness, Semantic Sim, NLI
2. Acoustic/Spectral: MFCCs, Spectral Contrast, etc.
3. Hearing-Loss Metrics: STOI, PESQ
4. Phoneme-Level: Phonological distance (panphon)
5. ASR Embeddings: Whisper encoder and wav2vec2 hidden states
6. Optional: LLM-based ASR correction
"""



from __future__ import annotations
from huggingface_hub import login
import scipy
from scipy import signal as scipy_signal
import os
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if token:
    login(token=token)
import json
import logging
from pathlib import Path
import nltk
nltk.download('averaged_perceptron_tagger_eng')

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable to avoid fork warnings
os.environ['HF_HOME'] = '/scratch/work/peterr2/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/work/peterr2/.cache/huggingface'
import hydra
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from clarity.utils.audiogram import (
    Audiogram,
    AUDIOGRAM_REF_CLARITY,
    AUDIOGRAM_MILD, 
    AUDIOGRAM_MODERATE,
    AUDIOGRAM_MODERATE_SEVERE,
)
import whisper
from clarity.evaluator.haaqi import haaqi_v1
from clarity.evaluator.haspi import haspi_v2
from g2p_en import G2p
from omegaconf import DictConfig
from panphon.distance import Distance
from pystoi import stoi
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from torch.nn import Module
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoModelForCausalLM, # For LLM
    AutoTokenizer, # For LLM
    BitsAndBytesConfig, # For 4-bit
    Wav2Vec2ForCTC,
    Wav2Vec2Model,
    Wav2Vec2Processor,
    WhisperModel,
    WhisperProcessor,
    Wav2Vec2FeatureExtractor
)

STANDARD_AUDIOGRAMS = {
    "normal": AUDIOGRAM_REF_CLARITY,
    "mild": AUDIOGRAM_MILD,
    "moderate": AUDIOGRAM_MODERATE,
    "moderate-severe": AUDIOGRAM_MODERATE_SEVERE,
    "severe": AUDIOGRAM_MODERATE_SEVERE,  # Use moderate-severe as proxy for severe
}

# --- NEW: Handle optional imports ---
try:
    import pysepm
    HAVE_PYSEPM = True
except ImportError:
    HAVE_PYSEPM = False
    logging.warning("pysepm not found. Skipping fwSNRseg feature.")

try:
    import pesq
    from pesq import PesqError
    HAVE_PESQ = True
except ImportError:
    HAVE_PESQ = False
    logging.warning("pesq not found. Skipping PESQ feature.")
# ---

# Demucs (source separation)
try:
    import torch.hub
    HAVE_DEMUCS = True
except Exception:
    HAVE_DEMUCS = False
    logging.warning("Demucs not available (torch.hub load failed). Skipping vocal-based metrics.")

def separate_vocals_demucs(
    signal_stereo: np.ndarray,
    sr: int,
    model: torch.nn.Module,
    device: torch.device,
    target_sr: int,
) -> np.ndarray:
    """
    Separate vocals using HTDemucs and return a mono vocals track resampled to target_sr.
    signal_stereo: shape (N, 2), float32
    sr: current sample rate (can be 16k)
    model: Demucs model, expects 44.1kHz
    device: torch device
    target_sr: the desired output sample rate (e.g., 16000)
    """
    if signal_stereo.ndim == 1:
        signal_stereo = np.stack([signal_stereo, signal_stereo], axis=1)
    assert signal_stereo.shape[1] == 2, "Demucs requires stereo input."

    # Resample to 44100 for Demucs
    demucs_sr = 44100
    if sr != demucs_sr:
        x_lr = librosa.resample(signal_stereo.T, orig_sr=sr, target_sr=demucs_sr, axis=1)  # (2, T)
    else:
        x_lr = signal_stereo.T  # (2, T)

    # To tensor [1, C, T]
    x = torch.from_numpy(x_lr).float().unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        # Forward returns [B, S, C, T]
        sources = model(x)
    # Find vocals index
    if hasattr(model, "sources") and "vocals" in model.sources:
        v_idx = model.sources.index("vocals")
    else:
        # Fallback: assume last source is vocals (common for Demucs variants)
        v_idx = -1
    vocals = sources[0, v_idx]  # [C, T]
    vocals_np = vocals.detach().cpu().numpy().T  # (T, 2)

    # Resample back to target_sr and mono-ize
    if demucs_sr != target_sr:
        vocals_np = librosa.resample(vocals_np.T, orig_sr=demucs_sr, target_sr=target_sr, axis=1).T  # (T, 2)
    vocals_mono = vocals_np.mean(axis=1).astype(np.float32)
    return vocals_mono


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """
    Resample audio to target sample rate using GPU acceleration if available.
    
    Args:
        audio: Input audio array (mono or stereo)
        orig_sr: Original sample rate
        target_sr: Target sample rate (default 16000)
    
    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio
    
    # Try GPU resampling first
    try:
        import torchaudio
        import torchaudio.functional as F
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dim
        
        # Move to device
        audio_tensor = audio_tensor.to(device)
        
        # Resample
        resampled = F.resample(audio_tensor, orig_sr, target_sr)
        
        # Convert back
        resampled = resampled.cpu().numpy()
        if audio.ndim == 1:
            resampled = resampled.squeeze(0)
            
        logger.debug(f"GPU resampling: {orig_sr}Hz -> {target_sr}Hz")
        return resampled
        
    except (ImportError, Exception) as e:
        # Fallback to librosa CPU resampling
        logger.debug(f"GPU resampling unavailable ({e}), using librosa CPU fallback")
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


from clarity.utils.file_io import read_jsonl, write_jsonl, write_signal
from recipes.cad_icassp_2026.baseline.shared_predict_utils import (
    load_mixture,
)
from recipes.cad_icassp_2026.baseline.transcription_scorer import SentenceScorer

logger = logging.getLogger(__name__)


# --- Helper Function for Reference Audio (Unchanged) ---
def load_reference_audio(dataroot: Path, record: dict, cfg: DictConfig) -> np.ndarray:
    """
    FIXME: Implement this function to load the CLEAN reference audio.
    This is just a placeholder. You must adapt it to your data structure.
    """
    # ... (same as before) ...
    # Example: Assumes reference path is in metadata
    if "reference_signal_path" not in record:
        logger.error("Metadata record missing 'reference_signal_path'. Cannot compute HASPI/HAAQI.")
        raise FileNotFoundError("Reference signal path not in metadata.")
        
    ref_path = dataroot / record["reference_signal_path"]
    signal, sr = sf.read(ref_path, dtype="float32")
    
    if sr != cfg.data.sample_rate:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=cfg.data.sample_rate)
    
    # Ensure mono for some metrics if needed, or handle stereo
    if signal.ndim > 1:
        signal = signal.mean(axis=1) # Or process stereo
        
    return signal


def correct_with_llm(
    hypothesis: str, 
    confidence: float, 
    cfg: DictConfig, 
    models: dict
) -> str:
    """Corrects a hypothesis using the loaded LLM if confidence is low."""
    
    llm_model = models["llm_model"]
    llm_tokenizer = models["llm_tokenizer"]
    
    logger.warning(
        f"Signal confidence {confidence:.3f} is below threshold "
        f"{cfg.baseline.llm_correction.confidence_threshold}. "
        f"Attempting LLM correction."
    )
    
    terminators = [
        llm_tokenizer.eos_token_id,
        llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    messages = [
        {"role": "system", "content": "You are an expert at correcting Automatic Speech Recognition (ASR) errors. Only output the corrected text, with no preamble or quotation marks."},
        {"role": "user", "content": f"Correct the following ASR transcription:\n\n{hypothesis}"}
    ]
    
    prompt = llm_tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    
    try:
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=100,
                eos_token_id=terminators,
                temperature=0.3,
                do_sample=True,
            )
        
        response = outputs[0][inputs["input_ids"].shape[-1]:]
        corrected_text = llm_tokenizer.decode(response, skip_special_tokens=True).strip()
        
        # Clean up common LLM artifacts
        if corrected_text.startswith('"') and corrected_text.endswith('"'):
            corrected_text = corrected_text[1:-1]
        
        if corrected_text: # Don't replace with an empty string
            logger.info(f"LLM Correction: '{hypothesis}' -> '{corrected_text}'")
            return corrected_text
        else:
            logger.warning("LLM generated empty correction. Reverting to original.")
            return hypothesis
            
    except Exception as e:
        logger.error(f"LLM correction failed: {e}. Reverting to original hypothesis.")
        return hypothesis


# --- Main Feature Computation Function ---
def compute_all_features_for_signal(
    cfg: DictConfig,
    record: dict,
    processed_signal: np.ndarray,
    reference_signal: np.ndarray,
    models: dict,
    sample_rate: int,
) -> dict:
    """
    Computes all feature groups for a given stereo signal.
    All resampling is done ONCE at the beginning for efficiency.
    """
    features = {}
    reference_prompt = record["prompt"]
    
    # Get model references
    scorer = models["scorer"]
    asr_model = models["whisper_asr"]
    semantic_model = models["semantic_model"]
    nli_model = models["nli_model"]
    g2p = models["g2p"]
    phoneme_model = models["wav2vec2_phoneme_model"]
    phoneme_processor = models["wav2vec2_phoneme_processor"]
    panphon_dist = models["panphon_dist"]
    whisper_processor = models["whisper_processor"]
    w2v_robust_model = models["wav2vec2_robust_model"]
    w2v_robust_processor = models["wav2vec2_robust_processor"]
    device = models["whisper_embed_model"].device

    # ========================================================================
    # STEP 1: RESAMPLE EVERYTHING TO 16kHz ONCE (if needed)
    # ========================================================================
    TARGET_SR = 16000
    
    if sample_rate != TARGET_SR:
        logger.debug(f"Resampling from {sample_rate}Hz to {TARGET_SR}Hz (GPU-accelerated)")
        
        # Resample stereo processed signal
        processed_signal = np.column_stack([
            resample_audio(processed_signal[:, 0], sample_rate, TARGET_SR),
            resample_audio(processed_signal[:, 1], sample_rate, TARGET_SR)
        ])
        
        # Resample reference signal (handle mono/stereo)
        if reference_signal.ndim == 1:
            reference_signal = resample_audio(reference_signal, sample_rate, TARGET_SR)
        else:
            reference_signal = np.column_stack([
                resample_audio(reference_signal[:, 0], sample_rate, TARGET_SR),
                resample_audio(reference_signal[:, 1], sample_rate, TARGET_SR)
            ])
        
        # Update sample rate - we're now working at 16kHz
        sample_rate = TARGET_SR
        logger.debug(f"All audio now at {sample_rate}Hz")
    
    # ========================================================================
    # STEP 2: ASR TRANSCRIPTION (Whisper)
    # ========================================================================
    path_temp = Path("temp.flac")
    
    # Left Channel
    write_signal(
        filename=path_temp,
        signal=processed_signal[:, 0],
        sample_rate=sample_rate,
        floating_point=False,
    )
    
    result_left = asr_model.transcribe(
        str(path_temp), fp16=False, language="en", temperature=0.0
    )
    hyp_left = result_left["text"]
    segments_left = result_left.get("segments", [])
    avg_logprob_left = (
        np.mean([s["avg_logprob"] for s in segments_left if "avg_logprob" in s])
        if segments_left else 0.0
    )

    # Right Channel
    write_signal(
        filename=path_temp,
        signal=processed_signal[:, 1],
        sample_rate=sample_rate,
        floating_point=False,
    )
    
    result_right = asr_model.transcribe(
        str(path_temp), fp16=False, language="en", temperature=0.0
    )
    hyp_right = result_right["text"]
    segments_right = result_right.get("segments", [])
    avg_logprob_right = (
        np.mean([s["avg_logprob"] for s in segments_right if "avg_logprob" in s])
        if segments_right else 0.0
    )

    path_temp.unlink()

    # ========================================================================
    # STEP 3: OPTIONAL LLM CORRECTION
    # ========================================================================
    if cfg.baseline.llm_correction.enabled:
        if avg_logprob_left < cfg.baseline.llm_correction.confidence_threshold:
            hyp_left = correct_with_llm(hyp_left, avg_logprob_left, cfg, models)

        if avg_logprob_right < cfg.baseline.llm_correction.confidence_threshold:
            hyp_right = correct_with_llm(hyp_right, avg_logprob_right, cfg, models)

    # ========================================================================
    # STEP 4: TEXT SCORING (Correctness)
    # ========================================================================
    results_left = scorer.score([reference_prompt], [hyp_left])
    total_left = results_left.substitutions + results_left.deletions + results_left.hits
    score_left = results_left.hits / total_left if total_left > 0 else 0.0

    results_right = scorer.score([reference_prompt], [hyp_right])
    total_right = results_right.substitutions + results_right.deletions + results_right.hits
    score_right = results_right.hits / total_right if total_right > 0 else 0.0

    # ========================================================================
    # STEP 5: SEMANTIC SIMILARITY & NLI (Batched)
    # ========================================================================
    ref_emb, hyp_left_emb, hyp_right_emb = semantic_model.encode(
        [reference_prompt, hyp_left, hyp_right], convert_to_tensor=True
    )
    sem_sim_left = util.cos_sim(ref_emb, hyp_left_emb).item()
    sem_sim_right = util.cos_sim(ref_emb, hyp_right_emb).item()

    nli_scores = nli_model.predict(
        [(reference_prompt, hyp_left), (reference_prompt, hyp_right)],
        show_progress_bar=False,
    )
    nli_entail_left = float(nli_scores[0][1])
    nli_entail_right = float(nli_scores[1][1])

    # ========================================================================
    # STEP 6: SELECT BEST CHANNEL
    # ========================================================================
    if score_left >= score_right:
        features["whisper_correct"] = score_left
        features["semantic_sim"] = sem_sim_left
        features["nli_entail"] = nli_entail_left
        hypothesis = hyp_left
        mono_signal = processed_signal[:, 0]
    else:
        features["whisper_correct"] = score_right
        features["semantic_sim"] = sem_sim_right
        features["nli_entail"] = nli_entail_right
        hypothesis = hyp_right
        mono_signal = processed_signal[:, 1]

    features["hypothesis"] = hypothesis

    # ========================================================================
    # STEP 7: PREPARE MONO REFERENCE (align lengths)
    # ========================================================================
    if reference_signal.ndim > 1:
        reference_signal_mono = reference_signal.mean(axis=1)
    else:
        reference_signal_mono = reference_signal

    min_len = min(len(mono_signal), len(reference_signal_mono))
    mono_signal = mono_signal[:min_len]
    reference_signal_mono = reference_signal_mono[:min_len]

    # ========================================================================
    # STEP 8: ACOUSTIC & SPECTRAL FEATURES (librosa)
    # ========================================================================
    mfcc = librosa.feature.mfcc(y=mono_signal, sr=sample_rate, n_mfcc=40)
    features["mfcc_mean"] = list(mfcc.mean(axis=1))
    features["mfcc_std"] = list(mfcc.std(axis=1))

    spec_cent = librosa.feature.spectral_centroid(y=mono_signal, sr=sample_rate)
    features["spectral_centroid_mean"] = float(spec_cent.mean())
    features["spectral_centroid_std"] = float(spec_cent.std())

    spec_contrast = librosa.feature.spectral_contrast(y=mono_signal, sr=sample_rate)
    features["spectral_contrast_mean"] = list(spec_contrast.mean(axis=1))

    # ========================================================================
    # STEP 9: PYSEPM METRICS
    # ========================================================================
    # if HAVE_PYSEPM:
    #     try:
    #         features["fwSNRseg"] = float(
    #             pysepm.fwSNRseg(reference_signal_mono, mono_signal, sample_rate)
    #         )
    #     except Exception as e:
    #         logger.warning(f"pysepm.fwSNRseg failed for {record['signal']}: {e}")
    #         features["fwSNRseg"] = 0.0
    # else:
    #     # features["fwSNRseg"] = 0.0


    # ========================================================================
    # STEP 11: PREPARE STEREO SIGNALS FOR HEARING LOSS METRICS
    # ========================================================================
    # Ensure reference_signal is stereo (N, 2)
    if reference_signal.ndim == 1:
        reference_signal = np.stack([reference_signal, reference_signal], axis=1)
    elif reference_signal.shape[1] == 1:
        reference_signal = np.concatenate([reference_signal, reference_signal], axis=1)

    # Ensure processed_signal is stereo (N, 2)
    if processed_signal.ndim == 1:
        processed_signal = np.stack([processed_signal, processed_signal], axis=1)
    elif processed_signal.shape[1] == 1:
        processed_signal = np.concatenate([processed_signal, processed_signal], axis=1)

    # Align lengths by padding
    n_proc = len(processed_signal)
    n_ref = len(reference_signal)

    if n_proc > n_ref:
        padding = n_proc - n_ref
        reference_signal_stereo = np.pad(
            reference_signal, ((0, padding), (0, 0)), "constant"
        )
        processed_signal_stereo = processed_signal
    elif n_ref > n_proc:
        padding = n_ref - n_proc
        processed_signal_stereo = np.pad(
            processed_signal, ((0, padding), (0, 0)), "constant"
        )
        reference_signal_stereo = reference_signal
    else:
        processed_signal_stereo = processed_signal
        reference_signal_stereo = reference_signal

    # ========================================================================
    # STEP 11.1: OPTIONAL VOCALS-ONLY STEMS WITH DEMUCS (for STOI/PESQ)
    # ========================================================================
    vocals_ref_mono = None
    vocals_proc_mono = None
    if models.get("demucs", None) is not None:
        try:
            demucs_model = models["demucs"]
            # Separate vocals for both reference and processed stereo
            vocals_ref_mono = separate_vocals_demucs(
                reference_signal_stereo, sample_rate, demucs_model, demucs_model.device, sample_rate
            )
            vocals_proc_mono = separate_vocals_demucs(
                processed_signal_stereo, sample_rate, demucs_model, demucs_model.device, sample_rate
            )
            # Trim to same length as earlier min_len, for safety
            vmin = min(len(vocals_ref_mono), len(vocals_proc_mono))
            vocals_ref_mono = vocals_ref_mono[:vmin]
            vocals_proc_mono = vocals_proc_mono[:vmin]
        except Exception as e:
            logger.warning(f"Demucs vocal separation failed for {record['signal']}: {e}")
            vocals_ref_mono, vocals_proc_mono = None, None

    # ========================================================================
    # STEP 10: PESQ (No resampling needed - already at 16kHz!)
    # ========================================================================
    if HAVE_PESQ:
        try:
            features["pesq_wb"] = float(
                pesq.pesq(sample_rate, reference_signal_mono, mono_signal, "wb")
            )
        except (PesqError, Exception) as e:
            logger.warning(f"PESQ failed for {record['signal']}: {e}")
            features["pesq_wb"] = 0.0
    else:
        features["pesq_wb"] = 0.0

    # NEW: PESQ on vocals-only stems if available (also wideband at 16k)
    if HAVE_PESQ and (vocals_ref_mono is not None and vocals_proc_mono is not None):
        try:
            features["pesq_wb_vocals"] = float(
                pesq.pesq(sample_rate, vocals_ref_mono, vocals_proc_mono, "wb")
            )
        except (PesqError, Exception) as e:
            logger.warning(f"PESQ (vocals) failed for {record['signal']}: {e}")
            features["pesq_wb_vocals"] = 0.0
    else:
        features["pesq_wb_vocals"] = 0.0
    # ========================================================================
    # STEP 12: HEARING LOSS METRICS (STOI, HASPI, HAAQI)
    # ========================================================================
    audiogram_left = record["audiogram_left"]
    audiogram_right = record["audiogram_right"]

    # STOI
    if vocals_ref_mono is not None and vocals_proc_mono is not None:
        try:
            features["stoi_vocals"] = float(
                stoi(vocals_ref_mono, vocals_proc_mono, sample_rate, extended=True)
            )
        except Exception as e:
            logger.warning(f"STOI (vocals) failed for {record['signal']}: {e}")
            features["stoi_vocals"] = 0.0
    else:
        features["stoi_vocals"] = 0.0


    # HASPI (Better-Ear) - Commented out in original, keeping commented
    # try:
    #     score_left_haspi, _ = haspi_v2(
    #         reference_signal_stereo[:, 0],
    #         sample_rate,
    #         processed_signal_stereo[:, 0],
    #         sample_rate,
    #         audiogram_left,
    #     )
    # except Exception as e:
    #     logger.warning(f"HASPI v2 (left) failed for {record['signal']}: {e}")
    #     score_left_haspi = 0.0
    #
    # try:
    #     score_right_haspi, _ = haspi_v2(
    #         reference_signal_stereo[:, 1],
    #         sample_rate,
    #         processed_signal_stereo[:, 1],
    #         sample_rate,
    #         audiogram_right,
    #     )
    # except Exception as e:
    #     logger.warning(f"HASPI v2 (right) failed for {record['signal']}: {e}")
    #     score_right_haspi = 0.0
    #
    # features["haspi"] = float(max(score_left_haspi, score_right_haspi))

    # HAAQI (Better-Ear) - Commented out in original, keeping commented
    # EQUALISATION_MODE = 1
    # try:
    #     score_left_haaqi, _, _, _ = haaqi_v1(
    #         reference_signal_stereo[:, 0],
    #         sample_rate,
    #         processed_signal_stereo[:, 0],
    #         sample_rate,
    #         audiogram_left,
    #         EQUALISATION_MODE,
    #     )
    # except Exception as e:
    #     logger.warning(f"HAAQI v1 (left) failed for {record['signal']}: {e}")
    #     score_left_haaqi = 0.0
    #
    # try:
    #     score_right_haaqi, _, _, _ = haaqi_v1(
    #         reference_signal_stereo[:, 1],
    #         sample_rate,
    #         processed_signal_stereo[:, 1],
    #         sample_rate,
    #         audiogram_right,
    #         EQUALISATION_MODE,
    #     )
    # except Exception as e:
    #     logger.warning(f"HAAQI v1 (right) failed for {record['signal']}: {e}")
    #     score_right_haaqi = 0.0
    #
    # features["haaqi"] = float(max(score_left_haaqi, score_right_haaqi))

    # ========================================================================
    # STEP 13: PHONEME-LEVEL FEATURES
    # ========================================================================
    reference_phonemes = " ".join(g2p(reference_prompt))

    inputs = phoneme_processor(
        mono_signal, sampling_rate=sample_rate, return_tensors="pt"
    ).to(phoneme_model.device)

    with torch.no_grad():
        logits = phoneme_model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    hypothesis_phonemes = phoneme_processor.batch_decode(predicted_ids)[0]

    features["phoneme_feat_dist"] = float(
        panphon_dist.feature_edit_distance(reference_phonemes, hypothesis_phonemes)
    )
    features["phoneme_weighted_dist"] = float(
        panphon_dist.weighted_feature_edit_distance(
            reference_phonemes, hypothesis_phonemes
        )
    )

    # ========================================================================
    # STEP 14: ASR EMBEDDINGS (Whisper & Wav2Vec2)
    # ========================================================================
    # Whisper embeddings
    inputs_whisper = whisper_processor(
        mono_signal, sampling_rate=sample_rate, return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        whisper_out = models["whisper_embed_model"].encoder(
            inputs_whisper.input_features
        )
        features["whisper_embed_mean"] = list(
            whisper_out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        )

    # Wav2Vec2 embeddings
    inputs_w2v = w2v_robust_processor(
        mono_signal, sampling_rate=sample_rate, return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        w2v_out = w2v_robust_model(
            inputs_w2v.input_values, output_hidden_states=True
        )
        last_4_layers = torch.stack([h for h in w2v_out.hidden_states[-4:]]).mean(dim=0)
        features["w2v_embed_mean"] = list(
            last_4_layers.mean(dim=1).squeeze().cpu().numpy()
        )

    return features
    

def flatten_features(features_dict: dict) -> dict:
    """
    Flattens a dictionary where some values might be lists or 1D arrays.
    
    Example: {"a": 1, "b": [10, 20]} -> {"a": 1, "b_0": 10, "b_1": 20}
    """
    flat_dict = {}
    for key, value in features_dict.items():
        if isinstance(value, (list, np.ndarray)):
            # Ensure it's a 1D iterable
            if isinstance(value, np.ndarray):
                value = value.flatten()
            
            for i, item in enumerate(value):
                # Ensure item is a JSON-serializable type (e.g., float)
                flat_dict[f"{key}_{i}"] = float(item)
        else:
            # Keep scalar values (floats, ints, strings) as-is
            flat_dict[key] = value
    return flat_dict


def run_feature_extraction(
    dataroot: Path, records: list, results_file: Path, cfg: DictConfig
) -> None:
    """
    Loads all models and iterates through records to compute all features.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    models = {}

    # --- 1. Load Legacy Models ---
    logger.info(f"Loading Whisper ASR: {cfg.baseline.whisper_version}")
    models["whisper_asr"] = whisper.load_model(
        cfg.baseline.whisper_version, device=device
    )
    models["scorer"] = SentenceScorer(cfg.baseline.contractions_file)

    logger.info(f"Loading Semantic model: {cfg.baseline.semantic_model}")
    models["semantic_model"] = SentenceTransformer(
        cfg.baseline.semantic_model, device=device
    )

    logger.info(f"Loading NLI model: {cfg.baseline.nli_model}")
    models["nli_model"] = CrossEncoder(
        cfg.baseline.nli_model, device=device, max_length=256
    )

    # --- 2. Load New Phoneme & ASR Models ---
    logger.info(
        f"Loading Whisper Embedding model: {cfg.baseline.whisper_version}"
    )
    models["whisper_embed_model"] = WhisperModel.from_pretrained(
        f"openai/whisper-{cfg.baseline.whisper_version}"
    ).to(device)
    models["whisper_processor"] = WhisperProcessor.from_pretrained(
        f"openai/whisper-{cfg.baseline.whisper_version}"
    )

    logger.info(
        f"Loading wav2vec2-phoneme: {cfg.baseline.wav2vec2_phoneme_model}"
    )
    models["wav2vec2_phoneme_model"] = Wav2Vec2ForCTC.from_pretrained(
        cfg.baseline.wav2vec2_phoneme_model
    ).to(device)
    models["wav2vec2_phoneme_processor"] = Wav2Vec2Processor.from_pretrained(
        cfg.baseline.wav2vec2_phoneme_model
    )

    logger.info(
        f"Loading wav2vec2-robust: {cfg.baseline.wav2vec2_robust_model}"
    )
    models["wav2vec2_robust_model"] = Wav2Vec2Model.from_pretrained(
        cfg.baseline.wav2vec2_robust_model
    ).to(device)
    # --- FIX from last time ---
    models["wav2vec2_robust_processor"] = (
        Wav2Vec2FeatureExtractor.from_pretrained(
            cfg.baseline.wav2vec2_robust_model
        )
    )

    # --- 3. Load Phoneme Helpers ---
    models["g2p"] = G2p()
    models["panphon_dist"] = Distance()

        # --- 5. Optional: Load Demucs for vocal-based metrics ---
    if HAVE_DEMUCS:
        try:
            logger.info("Loading Demucs (htdemucs) for vocal-based metrics...")
            models["demucs"] = torch.hub.load("facebookresearch/demucs", "htdemucs").to(device).eval()
            logger.info("Demucs loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load Demucs: {e}. Vocal-based metrics will be skipped.")
            models["demucs"] = None
    else:
        models["demucs"] = None


    # --- 4. NEW: Load Optional LLM ---
    if cfg.baseline.llm_correction.enabled:
        logger.info(
            f"Loading LLM corrector: {cfg.baseline.llm_correction.model_id}"
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        models["llm_model"] = AutoModelForCausalLM.from_pretrained(
            cfg.baseline.llm_correction.model_id,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            token=token
        )
        models["llm_tokenizer"] = AutoTokenizer.from_pretrained(
            cfg.baseline.llm_correction.model_id
        )
        logger.info("LLM loaded successfully.")
    else:
        logger.info("LLM correction is disabled.")

    logger.info("All models loaded successfully.")

    # --- NEW: Define Target Sample Rate ---
    # TARGET_SR = 16000  # All models expect 16kHz
    # logger.info(
    #     f"Ensuring all audio is resampled to {TARGET_SR}Hz for processing."
    # )

    # --- Iterate through signals ---
    # for record in tqdm(records):
    #     signal_name = record["signal"]

    #     try:
    #         # 1. Load processed (mixture) audio
    #         # load_mixture returns audio at cfg.data.sample_rate
    #         signal_to_process, original_sr = load_mixture(
    #             dataroot, record, cfg
    #         )
            
    #         # # 2. Resample processed signal to TARGET_SR
    #         # if original_sr != TARGET_SR:
    #         #     signal_to_process = librosa.resample(
    #         #         signal_to_process,
    #         #         orig_sr=original_sr,
    #         #         target_sr=TARGET_SR,
    #         #         axis=0,  # resample along time axis for stereo
    #         #     )

    #         # 3. Load clean (reference) audio (with your _unproc fix)
    #         reference_path = (
    #             dataroot
    #             / cfg.split
    #             / "unprocessed"
    #             / f"{record['signal']}_unproc.flac"
    #         )

    #         if not reference_path.exists():
    #             raise FileNotFoundError(
    #                 f"Missing reference file: {reference_path}"
    #             )

    #         reference_signal, ref_sr = sf.read(
    #             reference_path, dtype="float32"
    #         )

    #         # # 4. Resample reference to TARGET_SR
    #         # if ref_sr != TARGET_SR:
    #         #     reference_signal = librosa.resample(
    #         #         reference_signal, orig_sr=ref_sr, target_sr=TARGET_SR
    #         #     )

    #         # 5. Compute all scores, passing TARGET_SR
    #         current_processing_sr = original_sr # The rate we'll use for features
    #         if ref_sr != current_processing_sr:
    #              logger.warning(f"Reference SR ({ref_sr}) differs from Processed SR ({current_processing_sr}) for {signal_name}. Resampling reference FOR STOI ONLY.")
    #              reference_signal = librosa.resample(
    #                  reference_signal, orig_sr=ref_sr, target_sr=current_processing_sr
    #              )
    #              ref_sr = current_processing_sr # Update the reference sample rate variable


    #         # 5. Compute all scores, passing the ORIGINAL sample rate
    #         all_features_dict = compute_all_features_for_signal(
    #             cfg, record, signal_to_process, reference_signal, models,
    #             current_processing_sr, # Pass the actual sample rate being used
    #         )
    for record in tqdm(records):
        signal_name = record["signal"]
        try:
            # 1. Load processed (mixture) audio
            signal_to_process, original_sr = load_mixture(dataroot, record, cfg)
            logger.debug(f"Loaded processed signal for {signal_name} with SR={original_sr}") # Optional logging

            # 2. Resample processed signal to TARGET_SR -- REMOVED
            # if original_sr != TARGET_SR:
            #     signal_to_process = librosa.resample(...)

            # 3. Load clean (reference) audio
            reference_path = (
             dataroot
             / cfg.split
             / "unprocessed"
             / f"{record['signal']}_unproc.flac"
            )
            if not reference_path.exists(): raise FileNotFoundError(f"Missing ref: {reference_path}")
            reference_signal, ref_sr = sf.read(reference_path, dtype="float32")
            logger.debug(f"Loaded reference signal for {signal_name} with SR={ref_sr}") # Optional logging


            # 4. Resample reference to TARGET_SR -- REMOVED
            # if ref_sr != TARGET_SR:
            #     reference_signal = librosa.resample(...)

            # **** NEW: Resample reference ONLY IF rates mismatch for STOI ****
            current_processing_sr = original_sr # The rate we'll use for features
            if ref_sr != current_processing_sr:
                 logger.warning(f"Reference SR ({ref_sr}) differs from Processed SR ({current_processing_sr}) for {signal_name}. Resampling reference FOR STOI ONLY.")
                 reference_signal = librosa.resample(
                     reference_signal, orig_sr=ref_sr, target_sr=current_processing_sr
                 )
                 ref_sr = current_processing_sr # Update the reference sample rate variable


            # 5. Compute all scores, passing the ORIGINAL sample rate
            all_features_dict = compute_all_features_for_signal(
                cfg, record, signal_to_process, reference_signal, models,
                current_processing_sr, # Pass the actual sample rate being used
            )

            # 6. Flatten dict for JSONL
            flat_features = flatten_features(all_features_dict)

            # 7. Results are appended to the results file
            result = {"signal": signal_name, **flat_features}
            write_jsonl(str(results_file), [result])

        except Exception as e:
            logger.error(
                f"Failed to process signal {signal_name}: {e}", exc_info=True
            )
            # Optionally write a failed record
            write_jsonl(
                str(results_file), [{"signal": signal_name, "error": str(e)}]
            )


# --- Main execution (Unchanged) ---
# pylint: disable = no-value-for-parameter
@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_compute_all_features(cfg: DictConfig) -> None:
    """Run the comprehensive feature computation."""
    assert cfg.baseline.name == "multimodal_features"
    logger.info(f"Running {cfg.baseline.system} feature extraction on {cfg.split} set...")

    dataroot = Path(cfg.data.cadenza_data_root) / cfg.data.dataset
    dataset_filename = dataroot / "metadata" / f"{cfg.split}_metadata.json"

    with dataset_filename.open("r", encoding="utf-8") as fp:
        records = json.load(fp)

    # --- NEW: Inject standard audiograms based on 'hearing_loss' key ---
    logger.info("Injecting standard audiograms into records...")
    missing_audiograms = 0
    
    for record in records:
        hl_profile = record.get("hearing_loss")
        
        if hl_profile:
            # Map "Mild" to "mild", "Moderate" to "moderate", etc.
            hl_key = hl_profile.lower().replace(" ", "-")  # Handle "Moderate Severe" -> "moderate-severe"
            
            if hl_key in STANDARD_AUDIOGRAMS:
                audiogram = STANDARD_AUDIOGRAMS[hl_key]
                logger.debug(f"Using '{hl_key}' audiogram for signal {record['signal']}")
            else:
                logger.warning(
                    f"Unknown hearing_loss profile '{hl_profile}' for signal {record['signal']}. "
                    f"Using 'normal' audiogram."
                )
                audiogram = STANDARD_AUDIOGRAMS["normal"]
                missing_audiograms += 1
        else:
            logger.warning(
                f"Record {record['signal']} missing 'hearing_loss' key. Using 'normal' audiogram."
            )
            audiogram = STANDARD_AUDIOGRAMS["normal"]
            missing_audiograms += 1
        
        # Assign the Audiogram objects directly
        # The HASPI/HAAQI functions expect Audiogram objects
        record["audiogram_left"] = audiogram
        record["audiogram_right"] = audiogram

    if missing_audiograms > 0:
        logger.warning(
            f"Could not find hearing_loss profile for {missing_audiograms} records. "
            f"Used 'normal' as default."
        )
    logger.info("Audiogram injection complete.")
    # --- END OF NEW BLOCK ---

    # --- Batching & Resuming Logic (same as before) ---
    total_records = len(records)
    batch_str = (
        f".{cfg.baseline.batch}_{cfg.baseline.n_batches}"
        if cfg.baseline.n_batches > 1
        else ""
    )

    results_file = Path(
        f"{cfg.data.dataset}.{cfg.split}.{cfg.baseline.system}{batch_str}_updated.jsonl"
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

    if len(records) > 0:
        run_feature_extraction(dataroot, records, results_file, cfg)
    else:
        logger.info("All signals already processed.")


if __name__ == "__main__":
    run_compute_all_features()