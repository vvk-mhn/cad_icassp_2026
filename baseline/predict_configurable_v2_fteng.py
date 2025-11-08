from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union
import glob

import hydra
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
import xgboost as xgb
from omegaconf import DictConfig
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from clarity.utils.file_io import read_jsonl

log = logging.getLogger(__name__)


def load_selected_features_safe():
    """
    SAFE: Load pre-selected features WITHOUT calling hydra.utils.get_original_cwd()
    This is called INSIDE run_experiment(), AFTER Hydra initializes
    """
    search_paths = [
        Path("exp") / "selected_features.txt",
        Path("selected_features.txt"),
        Path(".") / "selected_features.txt",
        Path("..") / "selected_features.txt",
        Path("../..") / "selected_features.txt",
    ]
    
    # Try glob pattern
    try:
        search_paths.extend([Path(p) for p in glob.glob("**/selected_features.txt", recursive=True)])
    except:
        pass
    
    search_paths = [p for p in search_paths if p is not None]
    
    for feature_file in search_paths:
        if feature_file.exists():
            log.info(f"✓ Found selected_features.txt at: {feature_file}")
            with open(feature_file, 'r') as f:
                features = [line.strip() for line in f if line.strip()]
            log.info(f"✓ Loaded {len(features)} pre-selected features")
            return features
    
    log.warning("selected_features.txt not found in any location. Using fallback feature set")
    return None


# ============================================================================
# FEATURE GROUPS DEFINITION (NO LOADING HERE!)
# ============================================================================

MFCC_MEAN_FEATURES = [f"mfcc_mean_{i}" for i in range(40)]
MFCC_STD_FEATURES = [f"mfcc_std_{i}" for i in range(40)]
SPECTRAL_CONTRAST_FEATURES = [f"spectral_contrast_mean_{i}" for i in range(7)]
ACOUSTIC_SCALAR_FEATURES = [
    "spectral_centroid_mean", "spectral_centroid_std",
    "stoi_vocals", "pesq_wb_vocals", "zimtohrli"
]

PHONEME_FEATURES = ["phoneme_feat_dist", "phoneme_weighted_dist"]
WHISPER_EMBED_FEATURES = [f"whisper_embed_mean_{i}" for i in range(512)]
W2V_EMBED_FEATURES = [f"w2v_embed_mean_{i}" for i in range(1024)]
EMBED_FEATURES = WHISPER_EMBED_FEATURES + W2V_EMBED_FEATURES
ACOUSTIC_FEATURES = (
    MFCC_MEAN_FEATURES + MFCC_STD_FEATURES + SPECTRAL_CONTRAST_FEATURES
)
TEXT_SCALAR_FEATURES = [
    "whisper_correct",
    "semantic_sim",
    "songprep_wer",
    "whisper",
    "lyricwhiz",
]
OPENAI_SAMPLED_FEATURES = [
    "oa_correct_mean", "oa_correct_median", "oa_correct_std",
    "oa_correct_min", "oa_correct_max",
]
WHISPERX_DETERMINISTIC_FEATURE = ["whisperx_correct"]

REMAINING_FEATURES = [
    "vocal_snr_ratio",
    "bertscore_f1",
    "canary_correct",
    "ref_n_chars", "ref_avg_word_len", "ref_syllables_per_word",
    "ref_pos_nouns", "ref_pos_verbs", "ref_pos_adj", "ref_pos_adv",
    "ref_content_ratio",
    "hyp_n_chars", "hyp_avg_word_len", "hyp_syllables_per_word",
    "hyp_pos_nouns", "hyp_pos_verbs", "hyp_pos_adj", "hyp_pos_adv",
    "hyp_content_ratio",
    "ratio_n_chars", "ratio_avg_word_len", "ratio_syllables_per_word",
    "ratio_pos_nouns", "ratio_pos_verbs", "ratio_pos_adj", "ratio_pos_adv",
    "ratio_content_ratio",
]

GATED_FEATURES = [
    "vocal_gate",
    "stoi_vocals_gated",
    "pesq_wb_vocals_gated"
]

ADVANCED_FEATURES = [
    'stoi_pesq_ratio',
    'whisper_wer_ratio',
    'whisper_correct_sq',
    'semantic_sim_sq',
    'vocal_quality_score',
    'text_complexity',
    'text_match_quality',
    'acoust_text_align'
]

ALL_FEATURES = (
    MFCC_MEAN_FEATURES
    + MFCC_STD_FEATURES
    + SPECTRAL_CONTRAST_FEATURES
    + ACOUSTIC_SCALAR_FEATURES
    + PHONEME_FEATURES
    + WHISPER_EMBED_FEATURES
    + W2V_EMBED_FEATURES
    + TEXT_SCALAR_FEATURES
    + OPENAI_SAMPLED_FEATURES
    + WHISPERX_DETERMINISTIC_FEATURE
    + REMAINING_FEATURES
    + GATED_FEATURES
    + ADVANCED_FEATURES
)
ALL_FEATURES = sorted(list(set(ALL_FEATURES)))
SCALAR_FEATURES = sorted(
    list(
        set(ALL_FEATURES) - set(EMBED_FEATURES) - set(ACOUSTIC_FEATURES)
    )
)
SCALAR_FEATURES = [f for f in SCALAR_FEATURES if f not in EMBED_FEATURES and f not in ACOUSTIC_FEATURES]
EMBED_FEATURES = [f for f in EMBED_FEATURES if f in ALL_FEATURES]
ACOUSTIC_FEATURES = [f for f in ACOUSTIC_FEATURES if f in ALL_FEATURES]

# WILL BE SET INSIDE run_experiment(), NOT HERE!
SELECTED_FEATURES = None


# ============================================================================
# UTILITY FUNCTIONS (same as original)
# ============================================================================

def rmse_score(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(np.mean((x - y) ** 2))

def ncc_score(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0 or np.isnan(x).any() or np.isnan(y).any():
        log.warning("NCC calculation failed (zero variance or NaN). Returning 0.0")
        return 0.0
    return pearsonr(x, y)[0]


def add_advanced_features(df):
    """Add interaction and polynomial features"""
    df_out = df.copy()
    
    try:
        if 'stoi_vocals_gated' in df_out.columns and 'pesq_wb_vocals_gated' in df_out.columns:
            df_out['stoi_pesq_ratio'] = df_out['stoi_vocals_gated'] / (df_out['pesq_wb_vocals_gated'] + 1e-6)
        if 'whisper_correct' in df_out.columns and 'songprep_wer' in df_out.columns:
            df_out['whisper_wer_ratio'] = df_out['whisper_correct'] / (df_out['songprep_wer'] + 1e-6)
        if 'whisper_correct' in df_out.columns:
            df_out['whisper_correct_sq'] = df_out['whisper_correct'] ** 2
        if 'semantic_sim' in df_out.columns:
            df_out['semantic_sim_sq'] = df_out['semantic_sim'] ** 2
        if all(c in df_out.columns for c in ['vocal_gate', 'stoi_vocals', 'pesq_wb_vocals']):
            df_out['vocal_quality_score'] = df_out['vocal_gate'] * df_out['stoi_vocals'] * df_out['pesq_wb_vocals']
        if 'ref_syllables_per_word' in df_out.columns and 'ref_avg_word_len' in df_out.columns:
            df_out['text_complexity'] = df_out['ref_syllables_per_word'] * df_out['ref_avg_word_len']
        if all(c in df_out.columns for c in ['whisper_correct', 'semantic_sim', 'bertscore_f1']):
            df_out['text_match_quality'] = df_out['whisper_correct'] * df_out['semantic_sim'] * df_out['bertscore_f1']
        if 'zimtohrli' in df_out.columns and 'whisper_correct' in df_out.columns:
            df_out['acoust_text_align'] = df_out['zimtohrli'] * df_out['whisper_correct']
    except Exception as e:
        log.warning(f"Could not create all advanced features: {e}")
    
    return df_out


def load_all_features_and_metadata(cfg: DictConfig, split: str, is_prediction_set: bool = False) -> tuple:
    """Load features - use selected_features if provided"""
    global SELECTED_FEATURES
    
    dataroot = Path(cfg.data.cadenza_data_root) / cfg.data.dataset
    log.info(f"Using dataroot: {dataroot}")
    metadata_file = dataroot / "metadata" / f"{split}_metadata.json"
    if not metadata_file.exists(): 
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    with metadata_file.open("r", encoding="utf-8") as fp: 
        metadata = json.load(fp)
    metadata_df_full = pd.DataFrame(metadata)
    log.info(f"Loaded metadata for '{split}' ({len(metadata_df_full)} rows)")

    columns_to_load = ["signal"]
    targets_col = None
    if not is_prediction_set:
        if "correctness" not in metadata_df_full.columns: 
            raise KeyError(f"'correctness' missing in {metadata_file}")
        columns_to_load.append("correctness")
        metadata_df = metadata_df_full[columns_to_load].copy()
        if metadata_df["correctness"].max() > 1.01:
            metadata_df["correctness"] = metadata_df["correctness"] / 100.0
        metadata_df["correctness"] = np.clip(metadata_df["correctness"], 0.0, 1.0)
        targets_col = metadata_df["correctness"]
    else:
        if "signal" not in metadata_df_full.columns: 
            raise KeyError(f"'signal' missing in {metadata_file}")
        metadata_df = metadata_df_full[["signal"]].copy()

    # Load all feature files - TRY LOCAL PATHS FIRST, then Hydra
    def load_feature_file(filename_pattern, cols_to_keep):
        filename = filename_pattern.format(split=split, system=cfg.baseline.system)
        file_path = Path(filename)
        if not file_path.exists():
            try:
                file_path = Path(hydra.utils.get_original_cwd()) / filename
            except:
                pass
            if not file_path.exists(): 
                raise FileNotFoundError(f"Feature file not found: {filename}")
        log.info(f"Loading features from {file_path}")
        df = pd.DataFrame(read_jsonl(str(file_path)))
        return df[["signal"] + [c for c in cols_to_keep if c in df.columns]]

    # Use selected features if available, otherwise all features
    features_to_use = SELECTED_FEATURES if SELECTED_FEATURES is not None else ALL_FEATURES
    
    main_features_filename = f"{cfg.data.dataset}.{{split}}.{cfg.baseline.system}_updated.jsonl"
    main_features_df = load_feature_file(main_features_filename, features_to_use)

    zimtohrli_filename = f"{cfg.data.dataset}.{{split}}.zimtohrli.jsonl"
    zimtohrli_df = load_feature_file(zimtohrli_filename, ["zimtohrli"])

    lyricwhiz_filename = f"{cfg.data.dataset}.{{split}}.lyricwhiz.jsonl"
    lyricwhiz_df = load_feature_file(lyricwhiz_filename, ["lyricwhiz"])

    whisperx_filename = f"{cfg.data.dataset}.{{split}}.whisperx_multi.jsonl"
    whisperx_df = load_feature_file(whisperx_filename, features_to_use)

    songprep_wer_filename = f"{cfg.data.dataset}.{{split}}.songprep_wer.jsonl"
    songprep_alt_df = load_feature_file(songprep_wer_filename, ["songprep_wer"])

    whisper_filename = f"{cfg.data.dataset}.{{split}}.whisper.jsonl"
    whisper_df = load_feature_file(whisper_filename, ["whisper"])

    remaining_filename = f"{cfg.data.dataset}.{{split}}.features_remaining.jsonl"
    remaining_df = load_feature_file(remaining_filename, REMAINING_FEATURES)

    # Merge all dataframes
    merged_df = pd.merge(main_features_df, metadata_df, on="signal", how="inner")
    merged_df = pd.merge(merged_df, whisper_df, on="signal", how="inner")
    merged_df = pd.merge(merged_df, songprep_alt_df, on="signal", how="inner")
    merged_df = pd.merge(merged_df, zimtohrli_df, on="signal", how="inner")
    merged_df = pd.merge(merged_df, lyricwhiz_df, on="signal", how="inner")
    merged_df = pd.merge(merged_df, whisperx_df, on="signal", how="inner")
    merged_df = pd.merge(merged_df, remaining_df, on="signal", how="inner")
    log.info(f"Final merged data has {len(merged_df)} samples.")
    if merged_df.empty: 
        raise ValueError("Merging resulted in empty DataFrame.")

    # Gating
    snr_col = 'vocal_snr_ratio' 
    if snr_col in merged_df.columns:
        alpha = float(getattr(cfg.baseline, 'vocal_gate_alpha', 1.0))
        snr0 = float(getattr(cfg.baseline, 'vocal_gate_snr0', 0.0))
        merged_df['vocal_gate'] = 1.0 / (1.0 + np.exp(-alpha * (merged_df[snr_col] - snr0)))
    else:
        log.warning(f"{snr_col} not found; setting vocal_gate=1 for all rows")
        merged_df['vocal_gate'] = 1.0

    for col in ['stoi_vocals', 'pesq_wb_vocals']:
        if col in merged_df.columns:
            merged_df[f'{col}_gated'] = merged_df[col] * merged_df['vocal_gate']
        else:
            log.warning(f"{col} not found in merged_df; skipping gating.")
            merged_df[f'{col}_gated'] = np.nan

    # Apply Advanced Features
    merged_df = add_advanced_features(merged_df)

    # Use selected features if available
    all_needed_columns = SELECTED_FEATURES if SELECTED_FEATURES is not None else ALL_FEATURES
    missing_cols = [col for col in all_needed_columns if col not in merged_df.columns]
    if missing_cols:
        log.warning(f"Missing {len(missing_cols)} expected columns. Filling with NaN/mean.")
        for col in missing_cols:
            merged_df[col] = np.nan
            
    # Replace inf and fill NaNs
    merged_df[all_needed_columns] = merged_df[all_needed_columns].replace([np.inf, -np.inf], np.nan)
    if merged_df[all_needed_columns].isnull().any().any():
        log.warning("NaNs detected in features! Filling with column means.")
        merged_df[all_needed_columns] = merged_df[all_needed_columns].fillna(merged_df[all_needed_columns].mean())
        if merged_df[all_needed_columns].isnull().any().any():
             log.error("NaNs still present after mean imputation. Check data.")
             merged_df[all_needed_columns] = merged_df[all_needed_columns].fillna(0.0)

    final_targets = targets_col.loc[merged_df.index].values if targets_col is not None else None
    return merged_df, final_targets

# ============================================================================
# MAIN RUN FUNCTION
# ============================================================================

@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_experiment(cfg: DictConfig):
    """
    FIXED: Load selected_features INSIDE Hydra context
    """
    global SELECTED_FEATURES
    
    # NOW it's safe to call load_selected_features
    SELECTED_FEATURES = load_selected_features_safe()
    if SELECTED_FEATURES is None:
        SELECTED_FEATURES = ALL_FEATURES
        log.info(f"Using ALL_FEATURES ({len(ALL_FEATURES)} features)")
    else:
        log.info(f"Using SELECTED_FEATURES ({len(SELECTED_FEATURES)} features)")
    
    log.info(f"Starting Prediction with {len(SELECTED_FEATURES)} features")

    predict_split = cfg.get("split", "valid")
    train_split = cfg.get("train_split", "train")
    log.info(f"Training on: '{train_split}', Predicting on: '{predict_split}'")

    try:
        train_val_df_full, y_train_val_full = load_all_features_and_metadata(cfg, train_split, is_prediction_set=False)
        predict_df_full, _ = load_all_features_and_metadata(cfg, predict_split, is_prediction_set=True)
    except Exception as e: 
        log.error(f"Failed to load data: {e}", exc_info=True)
        return

    predict_signals = predict_df_full['signal'].copy()
    train_val_signals = train_val_df_full['signal'].copy()

    # *** USE SELECTED FEATURES ***
    features_to_use = [f for f in SELECTED_FEATURES if f in train_val_df_full.columns]
    
    log.info(f"Using {len(features_to_use)} features (pre-selected)")
    X_train_val_full = train_val_df_full[features_to_use].values
    X_predict_full = predict_df_full[features_to_use].values
    log.info(f"Training data shape: {X_train_val_full.shape}")
    log.info(f"Prediction data shape: {X_predict_full.shape}")

if __name__ == "__main__":
    run_experiment()
