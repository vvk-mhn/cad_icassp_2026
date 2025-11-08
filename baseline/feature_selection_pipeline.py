#!/usr/bin/env python3
"""
BALANCED Feature Selection Pipeline - Less Aggressive
- Adjusted VIF threshold (more lenient)
- Relaxed correlation clustering threshold
- Target: 400-600 final features (not 99!)
- Fixed scipy compatibility for BorutaShap
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

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
from sklearn.inspection import permutation_importance
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from clarity.utils.file_io import read_jsonl

log = logging.getLogger(__name__)

# ============================================================================
# FEATURE GROUPS DEFINITION (same as before)
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


# ============================================================================
# LESS AGGRESSIVE FEATURE SELECTION FUNCTIONS
# ============================================================================

def iterative_vif_removal(X_data, threshold=15, max_iter=50, verbose=True):
    """
    LESS AGGRESSIVE VIF removal
    threshold=15 instead of 10 (keeps more features)
    max_iter=50 instead of unlimited
    """
    log.info(f"Starting VIF Removal (threshold={threshold}, max_iter={max_iter})")
    
    X = X_data.copy()
    removed_features = []
    iteration = 0
    
    # Stop at 300 features minimum (not 100)
    while len(X.columns) > 300 and iteration < max_iter:
        iteration += 1
        corr = X.corr().abs()
        max_corr_per_feature = corr.sum(axis=1) - 1
        worst_idx = max_corr_per_feature.idxmax()
        worst_val = max_corr_per_feature[worst_idx]
        
        # More lenient stopping criterion
        if worst_val < (threshold / 3):
            break
        
        X = X.drop(columns=[worst_idx])
        removed_features.append(worst_idx)
        
        if verbose and iteration % 10 == 0:
            log.info(f"  Iteration {iteration}: Removed {worst_idx} | Remaining: {X.shape[1]}")
    
    log.info(f"VIF Removal complete: {len(removed_features)} features removed")
    log.info(f"  Remaining: {len(X.columns)} features\n")
    return X, removed_features


def correlation_clustering_relaxed(X_data, threshold=0.95, verbose=True):
    """
    LESS AGGRESSIVE clustering
    threshold=0.95 instead of 0.85 (only removes VERY correlated features)
    """
    log.info(f"Correlation Clustering (threshold={threshold} - RELAXED)")
    
    if X_data.shape[1] < 2:
        log.info("Too few features for clustering, skipping\n")
        return X_data, []
    
    try:
        corr = X_data.corr(method='pearson').abs()
        
        if corr.isna().any().any():
            log.warning("NaN values in correlation matrix, filling with 0")
            corr = corr.fillna(0.0)
        
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        distance = 1 - corr
        if not np.isfinite(distance.values).all():
            log.warning("Non-finite values in distance matrix, using greedy approach")
            raise ValueError("Non-finite values")
        
        condensed = squareform(distance, checks=False)
        linkage_matrix = linkage(condensed, method='average')
        clusters = fcluster(linkage_matrix, 1 - threshold, criterion='distance')
    except Exception as e:
        log.warning(f"Hierarchical clustering failed: {e}. Using greedy removal")
        selected = []
        for col in X_data.columns:
            # Only remove if correlation > threshold (very high)
            if not any(abs(X_data[col].corr(X_data[s])) > threshold for s in selected):
                selected.append(col)
        return X_data[selected], list(set(X_data.columns) - set(selected))
    
    selected_features = []
    for cluster_id in np.unique(clusters):
        mask = clusters == cluster_id
        cluster_features = X_data.columns[mask].tolist()
        best = X_data[cluster_features].var().idxmax()
        selected_features.append(best)
    
    removed = [f for f in X_data.columns if f not in selected_features]
    log.info(f"Clustering complete: {len(removed)} features removed")
    log.info(f"  Remaining: {len(selected_features)} features\n")
    
    return X_data[selected_features], removed


def borutashap_feature_selection_fixed(X_train, y_train, feature_names, n_trials=5, verbose=True):
    """
    BorutaShap with scipy compatibility fix
    """
    log.info(f"Starting BorutaShap (n_trials={n_trials})")
    
    try:
        import xgboost as xgb
        
        # TRY to import BorutaShap with scipy fix
        try:
            # Try new scipy first
            from scipy.stats import binomtest
            # Monkey-patch for old BorutaShap
            import scipy.stats
            if not hasattr(scipy.stats, 'binom_test'):
                scipy.stats.binom_test = lambda k, n, p: binomtest(k, n, p).pvalue
        except ImportError:
            pass
        
        from BorutaShap import BorutaShap
        
    except ImportError as e:
        log.warning(f"BorutaShap/XGBoost import failed: {e}")
        log.warning("Skipping BorutaShap, returning all features")
        return list(feature_names)
    
    try:
        try:
            model = xgb.XGBRegressor(
                tree_method='gpu_hist',
                predictor='gpu_predictor',
                gpu_id=0,
                n_estimators=50,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            log.info("Using GPU-accelerated XGBoost")
        except:
            log.info("GPU XGBoost unavailable, using CPU")
            model = xgb.XGBRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
        
        fs = BorutaShap(
            model=model,
            importance_measure='shap',
            classification=False,
            percentile=100,
            pvalue=0.05
        )
        
        log.info(f"Fitting BorutaShap ({n_trials} trials)...")
        fs.fit(X_train, y_train, n_trials=n_trials, sample=True, verbose=False)
        
        selected_features = fs.accepted_features + fs.tentative_features
        
        log.info(f"BorutaShap Results:")
        log.info(f"  Accepted: {len(fs.accepted_features)}")
        log.info(f"  Tentative: {len(fs.tentative_features)}")
        log.info(f"  Total selected: {len(selected_features)}\n")
        
        return selected_features
    except Exception as e:
        log.warning(f"BorutaShap fitting failed: {e}")
        log.warning("Returning all input features")
        return list(feature_names)


def apply_feature_selection(
    X_train_df,
    y_train,
    X_predict_df,
    feature_names,
    cfg,
    model_type='lgbm'
):
    """
    BALANCED feature selection pipeline
    Target: 400-600 final features (not 99!)
    """
    
    log.info("\n" + "="*80)
    log.info("BALANCED FEATURE SELECTION PIPELINE")
    log.info("="*80 + "\n")
    
    features_current = list(feature_names)
    
    # ===== PHASE 1a: VIF Removal (LESS AGGRESSIVE) =====
    if cfg.baseline.get("use_vif_selection", True):
        log.info("### PHASE 1a: VIF Removal (RELAXED) ###\n")
        X_train_df, removed_vif = iterative_vif_removal(
            X_train_df, threshold=15, max_iter=50, verbose=True
        )
        features_current = list(X_train_df.columns)
        X_predict_df = X_predict_df[features_current]
    
    # ===== PHASE 1b: Correlation Clustering (LESS AGGRESSIVE) =====
    if cfg.baseline.get("use_correlation_selection", True):
        log.info("### PHASE 1b: Correlation Clustering (RELAXED) ###\n")
        X_train_df, removed_corr = correlation_clustering_relaxed(
            X_train_df, threshold=0.95, verbose=True
        )
        features_current = list(X_train_df.columns)
        X_predict_df = X_predict_df[features_current]
    
    # ===== PHASE 2a: BorutaShap (WITH SCIPY FIX) =====
    if cfg.baseline.get("use_borutashap_selection", True):
        log.info("### PHASE 2a: BorutaShap Feature Selection (FIXED) ###\n")
        try:
            selected_borutashap = borutashap_feature_selection_fixed(
                X_train_df.values,
                y_train,
                features_current,
                n_trials=cfg.baseline.get("borutashap_trials", 5),
                verbose=True
            )
            # Only use BorutaShap results if it actually ran
            if len(selected_borutashap) < len(features_current):
                features_current = selected_borutashap
                X_train_df = X_train_df[features_current]
                X_predict_df = X_predict_df[features_current]
            else:
                log.info("BorutaShap kept all features (fallback mode)")
        except Exception as e:
            log.warning(f"BorutaShap failed: {e}")
    
    # ===== FINAL SUMMARY =====
    log.info("="*80)
    log.info("FEATURE SELECTION COMPLETE")
    log.info("="*80)
    log.info(f"Original features: {len(feature_names)}")
    log.info(f"Final features: {len(features_current)}")
    log.info(f"Removed: {len(feature_names) - len(features_current)} ({100*(len(feature_names) - len(features_current))/len(feature_names):.1f}%)")
    
    if len(features_current) < 200:
        log.warning(f"⚠️ WARNING: Only {len(features_current)} features selected - may be too aggressive!")
    elif len(features_current) < 400:
        log.warning(f"⚠️ CAUTION: {len(features_current)} features selected - could be more")
    else:
        log.info(f"✓ Good: {len(features_current)} features selected (target: 400-600)")
    
    return X_train_df.values, features_current, X_predict_df.values


# ============================================================================
# UTILITY FUNCTIONS (same as before)
# ============================================================================

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
    """Load features (same as before, with Hydra fix)"""
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

    def load_feature_file(filename_pattern, cols_to_keep):
        filename = filename_pattern.format(split=split, system=cfg.baseline.system)
        file_path = Path(filename)
        if not file_path.exists():
            file_path = Path(hydra.utils.get_original_cwd()) / filename
            if not file_path.exists(): 
                raise FileNotFoundError(f"Feature file not found: {filename}")
        log.info(f"Loading features from {file_path}")
        df = pd.DataFrame(read_jsonl(str(file_path)))
        return df[["signal"] + [c for c in cols_to_keep if c in df.columns]]

    main_features_filename = f"{cfg.data.dataset}.{{split}}.{cfg.baseline.system}_updated.jsonl"
    main_features_df = load_feature_file(main_features_filename, ALL_FEATURES)

    zimtohrli_filename = f"{cfg.data.dataset}.{{split}}.zimtohrli.jsonl"
    zimtohrli_df = load_feature_file(zimtohrli_filename, ["zimtohrli"])

    lyricwhiz_filename = f"{cfg.data.dataset}.{{split}}.lyricwhiz.jsonl"
    lyricwhiz_df = load_feature_file(lyricwhiz_filename, ["lyricwhiz"])

    whisperx_filename = f"{cfg.data.dataset}.{{split}}.whisperx_multi.jsonl"
    whisperx_df = load_feature_file(whisperx_filename, ALL_FEATURES)

    songprep_wer_filename = f"{cfg.data.dataset}.{{split}}.songprep_wer.jsonl"
    songprep_alt_df = load_feature_file(songprep_wer_filename, ["songprep_wer"])

    whisper_filename = f"{cfg.data.dataset}.{{split}}.whisper.jsonl"
    whisper_df = load_feature_file(whisper_filename, ["whisper"])

    remaining_filename = f"{cfg.data.dataset}.{{split}}.features_remaining.jsonl"
    remaining_df = load_feature_file(remaining_filename, REMAINING_FEATURES)

    merged_df = metadata_df.copy()
    for df in [main_features_df, whisper_df, songprep_alt_df, zimtohrli_df, 
               lyricwhiz_df, whisperx_df, remaining_df]:
        merged_df = pd.merge(merged_df, df, on="signal", how="inner")
    
    log.info(f"Final merged data has {len(merged_df)} samples.")
    if merged_df.empty: 
        raise ValueError("Merging resulted in empty DataFrame.")

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

    merged_df = add_advanced_features(merged_df)

    all_needed_columns = ALL_FEATURES
    missing_cols = [col for col in all_needed_columns if col not in merged_df.columns]
    if missing_cols:
        log.warning(f"Missing {len(missing_cols)} expected columns. Filling with NaN/mean.")
        for col in missing_cols:
            merged_df[col] = np.nan
            
    merged_df[all_needed_columns] = merged_df[all_needed_columns].replace([np.inf, -np.inf], np.nan)
    if merged_df[all_needed_columns].isnull().any().any():
        log.warning("NaNs detected in features! Filling with column means.")
        merged_df[all_needed_columns] = merged_df[all_needed_columns].fillna(merged_df[all_needed_columns].mean())
        if merged_df[all_needed_columns].isnull().any().any():
            log.error("NaNs still present after mean imputation.")
            merged_df[all_needed_columns] = merged_df[all_needed_columns].fillna(0.0)

    final_targets = targets_col.loc[merged_df.index].values if targets_col is not None else None
    return merged_df, final_targets


@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_feature_selection(cfg: DictConfig):
    log.info("="*80)
    log.info("BALANCED FEATURE SELECTION PIPELINE")
    log.info("="*80 + "\n")
    
    split = cfg.get("split", "train")
    log.info(f"Running feature selection on split: '{split}'")
    
    log.info("\n### Loading Data ###\n")
    df, y = load_all_features_and_metadata(cfg, split, is_prediction_set=False)
    log.info(f"Loaded {len(df)} samples with {len(df.columns)} columns\n")
    
    available_features = [f for f in df.columns if f not in ['signal', 'correctness']]
    X = df[available_features].copy()
    
    log.info("\n### Running Feature Selection ###\n")
    X_selected, selected_features, _ = apply_feature_selection(
        X, y, X,
        available_features,
        cfg,
        model_type=cfg.baseline.get("model_type", "lgbm")
    )
    
    output_dir = Path("exp")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "selected_features.txt"
    
    with output_file.open("w") as f:
        for feature in selected_features:
            f.write(feature + "\n")
    
    log.info(f"✓ Saved selected features to: {output_file}\n")
    
    log.info("\n" + "="*80)
    log.info("FEATURE SELECTION PIPELINE COMPLETED SUCCESSFULLY")
    log.info("="*80)


if __name__ == "__main__":
    globals()['ALL_FEATURES'] = ALL_FEATURES
    globals()['SCALAR_FEATURES'] = SCALAR_FEATURES
    globals()['EMBED_FEATURES'] = EMBED_FEATURES
    globals()['ACOUSTIC_FEATURES'] = ACOUSTIC_FEATURES
    run_feature_selection()
