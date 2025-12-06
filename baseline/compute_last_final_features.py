"""
compute_last_final_features.py
"""
import json
import logging
import os
from pathlib import Path
import warnings

import hydra
import numpy as np
import torch
import torchaudio
import whisper
import librosa
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import Wav2Vec2Model, WavLMModel, Wav2Vec2FeatureExtractor
from clarity.utils.file_io import read_jsonl, write_jsonl
from recipes.cad_icassp_2026.baseline.shared_predict_utils import load_mixture

# Filter warnings for cleaner logs
warnings.simplefilter("ignore")
logger = logging.getLogger(__name__)

# --- Helper: Levenshtein Distance for "WER-like" consensus ---
def simple_levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x): matrix[x, 0] = x
    for y in range(size_y): matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x,y] = matrix[x-1, y-1]
            else:
                matrix[x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return matrix[size_x-1, size_y-1]

# --- Feature Function: Acoustic ---
def get_acoustic_features(signal, sr):
    """Computes Pitch and Band Energy features."""
    # 1. Pitch (F0)
    try:
        f0, _, _ = librosa.pyin(signal, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        f0 = f0[~np.isnan(f0)]
        if len(f0) > 0:
            pitch_median = float(np.median(f0))
            pitch_p95 = float(np.percentile(f0, 95))
            pitch_var = float(np.var(f0))
        else:
            pitch_median, pitch_p95, pitch_var = 0.0, 0.0, 0.0
    except:
        pitch_median, pitch_p95, pitch_var = 0.0, 0.0, 0.0

    # 2. Band Energy (1k-4k Hz) using Spectrogram
    S = np.abs(librosa.stft(signal))
    freqs = librosa.fft_frequencies(sr=sr)
    
    # Index range for 1kHz to 4kHz
    idx_1k = np.argmax(freqs >= 1000)
    idx_4k = np.argmax(freqs >= 4000)
    
    total_energy = np.sum(S)
    band_energy = np.sum(S[idx_1k:idx_4k, :])
    
    vocal_band_ratio = (band_energy / (total_energy + 1e-6))

    return {
        "pitch_median": pitch_median,
        "pitch_p95": pitch_p95,
        "pitch_var": pitch_var,
        "vocal_band_ratio_1k_4k": float(vocal_band_ratio)
    }

# --- Feature Function: Whisper Stats ---
def get_whisper_features(model, audio, n_samples=5, temp=0.7):
    # Pad/Trim
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # 1. Extract Embedding (Deterministic)
    with torch.no_grad():
        encoder_output = model.encoder(mel.unsqueeze(0)) 
        whisper_embed = encoder_output.squeeze(0).cpu().half() # (1500, 512)

    # 2. Greedy Decode (Deterministic Baseline)
    options_greedy = whisper.DecodingOptions(language="en", without_timestamps=True, temperature=0.0)
    res_greedy = model.decode(mel, options_greedy)
    text_greedy = res_greedy.text
    
    # Greedy Stats
    seg_avg_logprobs = [s.avg_logprob for s in res_greedy.segments] if hasattr(res_greedy, 'segments') else [res_greedy.avg_logprob]
    if not seg_avg_logprobs: seg_avg_logprobs = [-10.0]
    
    # 3. Stochastic Sampling Loop (Stability)
    options_sample = whisper.DecodingOptions(language="en", without_timestamps=True, temperature=temp)
    
    lev_distances = []
    sampled_texts = []
    
    for _ in range(n_samples):
        res_sample = model.decode(mel, options_sample)
        sampled_texts.append(res_sample.text)
        # Calculate distance from greedy text (proxy for WER)
        dist = simple_levenshtein(text_greedy, res_sample.text)
        lev_distances.append(dist)
    
    lev_distances = np.array(lev_distances)
    
    # Derived Statistics
    stats = {
        "whisper_avg_logprob": float(np.mean(seg_avg_logprobs)),
        "whisper_prob_drop_max": float(np.max(np.diff(seg_avg_logprobs)) if len(seg_avg_logprobs) > 1 else 0.0),
        "whisper_no_speech_prob": float(res_greedy.no_speech_prob) if hasattr(res_greedy, 'no_speech_prob') else 0.0,
        # Stability Stats
        "whisper_stability_avg_dist": float(np.mean(lev_distances)),
        "whisper_stability_std_dist": float(np.std(lev_distances)),
        "whisper_stability_max_dist": float(np.max(lev_distances)) if len(lev_distances) > 0 else 0.0,
    }

    return stats, whisper_embed

# --- Feature Function: HF Models ---
def extract_hf_embeddings(model, processor, signal, sample_rate, device):
    target_sr = 16000
    # Resample
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        signal = resampler(torch.tensor(signal, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    
    # Input
    inputs = processor(signal, sampling_rate=target_sr, return_tensors="pt").input_values.to(device)
    
    with torch.zno_grad():
        outputs = model(inputs)
        # (Time, Dim)
        embeddings = outputs.last_hidden_state.squeeze(0).cpu().half()
        print("emb_shape", embeddings.shape())
        
    return embeddings

@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_compute_features(cfg: DictConfig) -> None:
    dataroot = Path(cfg.data.cadenza_data_root) / cfg.data.dataset
    dataset_filename = dataroot / "metadata" / f"{cfg.split}_metadata.json"
    with dataset_filename.open("r", encoding="utf-8") as fp:
        records = json.load(fp)
        
    tensor_dir = dataroot / cfg.baseline.embedding_dir_name
    tensor_dir.mkdir(parents=True, exist_ok=True)
    results_file = Path(f"{cfg.data.dataset}.{cfg.split}.non_intrusive_extended.jsonl")
    
    # Resume logic
    existing_ids = set()
    if results_file.exists():
        existing_data = read_jsonl(str(results_file))
        existing_ids = {x['signal'] for x in existing_data}
    records = [r for r in records if r['signal'] not in existing_ids]
    
    # Load Models
    device = torch.device(cfg.baseline.device)
    logger.info(f"Loading models on {device}...")
    
    whisper_model = whisper.load_model(cfg.baseline.whisper_version, device=device)
    
    # FIX: Use FeatureExtractor, not Processor
    w2v_proc = Wav2Vec2FeatureExtractor.from_pretrained(cfg.baseline.wav2vec2_model)

    w2v_model = Wav2Vec2Model.from_pretrained(cfg.baseline.wav2vec2_model).to(device)

    
    wavlm_proc = Wav2Vec2FeatureExtractor.from_pretrained(cfg.baseline.wavlm_model)

    wavlm_model = WavLMModel.from_pretrained(cfg.baseline.wavlm_model).to(device)


    logger.info(f"Processing {len(records)} records...")
    
    for record in tqdm(records):
        signal_name = record["signal"]
        
        # Load and Mix to Mono
        signal_stereo, sr = load_mixture(dataroot, record, cfg)
        signal_mono = np.mean(signal_stereo, axis=1).astype(np.float32)

        
        # 1. Whisper (Stats + Embed)
        w_stats, w_embed = get_whisper_features(
            whisper_model, signal_mono, 
            n_samples=cfg.baseline.n_samples, 
            temp=cfg.baseline.temperature
        )
        
        # 2. Acoustic Features
        ac_stats = get_acoustic_features(signal_mono, sr)
        
        # 3. HF Embeddings
        w2v_embed = extract_hf_embeddings(w2v_model, w2v_proc, signal_mono, sr, device)
        wavlm_embed = extract_hf_embeddings(wavlm_model, wavlm_proc, signal_mono, sr, device)
        
        # Save Tensor
        torch.save({
            "whisper": w_embed,
            "wav2vec": w2v_embed,
            "wavlm": wavlm_embed
        }, tensor_dir / f"{signal_name}.pt")
        
        # Save Scalar
        full_stats = {"signal": signal_name, **w_stats, **ac_stats}
        write_jsonl(str(results_file), [full_stats])

if __name__ == "__main__":
    run_compute_features()