# predict_configurable_v2.py

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
import xgboost as xgb  # --- NEW (Suggestion #3) ---
from omegaconf import DictConfig
from scipy.stats import pearsonr
# --- NEW (Suggestion #2) ---
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from clarity.utils.file_io import read_jsonl

log = logging.getLogger(__name__)

# --- Feature Groups (Define ALL features) ---
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
    "vocal_snr_ratio", # non intrusive
    "bertscore_f1", # intrusve
    "canary_correct", # intrusve
    "ref_n_chars", "ref_avg_word_len", "ref_syllables_per_word", # intrusve
    "ref_pos_nouns", "ref_pos_verbs", "ref_pos_adj", "ref_pos_adv", # intrusve
    "ref_content_ratio", # intrusve
    "hyp_n_chars", "hyp_avg_word_len", "hyp_syllables_per_word", # non intrusve
    "hyp_pos_nouns", "hyp_pos_verbs", "hyp_pos_adj", "hyp_pos_adv", # non intrusive
    "hyp_content_ratio", # non instrusive
    "ratio_n_chars", "ratio_avg_word_len", "ratio_syllables_per_word", # intrusive
    "ratio_pos_nouns", "ratio_pos_verbs", "ratio_pos_adj", "ratio_pos_adv", # intrusive
    "ratio_content_ratio", # intrusive
]

GATED_FEATURES = [
    "vocal_gate", # non intrusive
    "stoi_vocals_gated", # non instrusive
    "pesq_wb_vocals_gated" # non instrusive
]

# --- NEW (Suggestion #4) ---
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

# --- NEW: Master list of ALL features ---
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
    + ADVANCED_FEATURES  # --- NEW (Suggestion #4) ---
)
# Remove potential duplicates
ALL_FEATURES = sorted(list(set(ALL_FEATURES)))
SCALAR_FEATURES = sorted(
    list(
        set(ALL_FEATURES) - set(EMBED_FEATURES) - set(ACOUSTIC_FEATURES)
    )
)
# Ensure embed/acoustic features are not in scalar
SCALAR_FEATURES = [f for f in SCALAR_FEATURES if f not in EMBED_FEATURES and f not in ACOUSTIC_FEATURES]
EMBED_FEATURES = [f for f in EMBED_FEATURES if f in ALL_FEATURES]
ACOUSTIC_FEATURES = [f for f in ACOUSTIC_FEATURES if f in ALL_FEATURES]


def rmse_score(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(np.mean((x - y) ** 2))

def ncc_score(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0 or np.isnan(x).any() or np.isnan(y).any():
        log.warning("NCC calculation failed (zero variance or NaN). Returning 0.0")
        return 0.0
    return pearsonr(x, y)[0]


# --- NEW (Suggestion #4) ---
def add_advanced_features(df):
    """Add interaction and polynomial features"""
    df_out = df.copy()
    
    # 1. Ratio features
    df_out['stoi_pesq_ratio'] = df_out['stoi_vocals_gated'] / (df_out['pesq_wb_vocals_gated'] + 1e-6)
    df_out['whisper_wer_ratio'] = df_out['whisper_correct'] / (df_out['songprep_wer'] + 1e-6)
    
    # 2. Polynomial combinations (assuming these are top features)
    df_out['whisper_correct_sq'] = df_out['whisper_correct'] ** 2
    df_out['semantic_sim_sq'] = df_out['semantic_sim'] ** 2
    
    # 3. Gate-weighted combinations
    df_out['vocal_quality_score'] = (
        df_out['vocal_gate'] * df_out['stoi_vocals'] * df_out['pesq_wb_vocals']
    )
    
    # 4. Text complexity interaction
    df_out['text_complexity'] = (
        df_out['ref_syllables_per_word'] * df_out['ref_avg_word_len']
    )
    df_out['text_match_quality'] = (
        df_out['whisper_correct'] * df_out['semantic_sim'] * df_out['bertscore_f1']
    )
    
    # 5. Acoustic-Text interaction
    df_out['acoust_text_align'] = (
        df_out['zimtohrli'] * df_out['whisper_correct']
    )
    
    return df_out


def load_all_features_and_metadata(cfg: DictConfig, split: str, is_prediction_set: bool = False) -> tuple:
    dataroot = Path(cfg.data.cadenza_data_root) / cfg.data.dataset
    log.info(f"Using dataroot: {dataroot}")
    metadata_file = dataroot / "metadata" / f"{split}_metadata.json"
    if not metadata_file.exists(): raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    with metadata_file.open("r", encoding="utf-8") as fp: metadata = json.load(fp)
    metadata_df_full = pd.DataFrame(metadata)
    log.info(f"Loaded metadata for '{split}' ({len(metadata_df_full)} rows)")

    columns_to_load = ["signal"]; targets_col = None
    if not is_prediction_set:
        if "correctness" not in metadata_df_full.columns: raise KeyError(f"'correctness' missing in {metadata_file}")
        columns_to_load.append("correctness")
        metadata_df = metadata_df_full[columns_to_load].copy()
        if metadata_df["correctness"].max() > 1.01:
            metadata_df["correctness"] = metadata_df["correctness"] / 100.0
        metadata_df["correctness"] = np.clip(metadata_df["correctness"], 0.0, 1.0)
        targets_col = metadata_df["correctness"]
    else:
        if "signal" not in metadata_df_full.columns: raise KeyError(f"'signal' missing in {metadata_file}")
        metadata_df = metadata_df_full[["signal"]].copy()

    # --- Load all existing feature files ---
    # (Loading code omitted for brevity... assuming it's the same as provided)
    main_features_filename = f"{cfg.data.dataset}.{split}.{cfg.baseline.system}_updated.jsonl"
    main_features_file = Path(main_features_filename)
    if not main_features_file.exists():
        hydra_cwd = Path(hydra.utils.get_original_cwd())
        main_features_file = hydra_cwd / main_features_filename
        if not main_features_file.exists(): raise FileNotFoundError(f"Main features file not found: {main_features_filename}")
    log.info(f"Loading main features from {main_features_file}")
    main_features_df = pd.DataFrame(read_jsonl(str(main_features_file)))

    zimtohrli_filename = f"{cfg.data.dataset}.{split}.zimtohrli.jsonl"
    zimtohrli_file = Path(zimtohrli_filename)
    if not zimtohrli_file.exists():
        hydra_cwd = Path(hydra.utils.get_original_cwd())
        zimtohrli_file = hydra_cwd / zimtohrli_filename
        if not zimtohrli_file.exists(): raise FileNotFoundError(f"zimtohrli features file not found: {zimtohrli_filename}")
    log.info(f"Loading zimtohrli features from {zimtohrli_file}")
    zimtohrli_df = pd.DataFrame(read_jsonl(str(zimtohrli_file)))[["signal", "zimtohrli"]]

    lyricwhiz_filename = f"{cfg.data.dataset}.{split}.lyricwhiz.jsonl"
    lyricwhiz_file = Path(lyricwhiz_filename)
    if not lyricwhiz_file.exists():
        hydra_cwd = Path(hydra.utils.get_original_cwd())
        lyricwhiz_file = hydra_cwd / lyricwhiz_filename
        if not lyricwhiz_file.exists(): raise FileNotFoundError(f"lyricwhiz features file not found: {lyricwhiz_filename}")
    log.info(f"Loading lyricwhiz features from {lyricwhiz_file}")
    lyricwhiz_df = pd.DataFrame(read_jsonl(str(lyricwhiz_file)))[["signal", "lyricwhiz"]]

    whisperx_filename = f"{cfg.data.dataset}.{split}.whisperx_multi.jsonl"
    whisperx_file = Path(whisperx_filename)
    if not whisperx_file.exists():
        hydra_cwd = Path(hydra.utils.get_original_cwd())
        whisperx_file = hydra_cwd / whisperx_filename
        if not whisperx_file.exists(): raise FileNotFoundError(f"WhisperX multi features file not found: {whisperx_filename}")
    log.info(f"Loading WhisperX-Multi features from {whisperx_file}")
    whisperx_df = pd.DataFrame(read_jsonl(str(whisperx_file)))
    
    songprep_wer_filename = f"{cfg.data.dataset}.{split}.songprep_wer.jsonl"
    songprep_alt_file = Path(songprep_wer_filename)
    if not songprep_alt_file.exists():
        hydra_cwd = Path(hydra.utils.get_original_cwd())
        songprep_alt_file = hydra_cwd / songprep_wer_filename
        if not songprep_alt_file.exists(): raise FileNotFoundError(f"songprep features file not found: {songprep_wer_filename}")
    log.info(f"Loading whisper features from {songprep_alt_file}")
    songprep_alt_df = pd.DataFrame(read_jsonl(str(songprep_alt_file)))[["signal", "songprep_wer"]]

    whisper_filename = f"{cfg.data.dataset}.{split}.whisper.jsonl"
    whisper_file = Path(whisper_filename)
    if not whisper_file.exists():
        hydra_cwd = Path(hydra.utils.get_original_cwd())
        whisper_file = hydra_cwd / whisper_file
        if not whisper_file.exists(): raise FileNotFoundError(f"whisper features file not found: {whisper_filename}")
    log.info(f"Loading whisper features from {whisper_file}")
    whisper_df = pd.DataFrame(read_jsonl(str(whisper_file)))[["signal", "whisper"]]

    remaining_filename = f"{cfg.data.dataset}.{split}.features_remaining.jsonl"
    remaining_file = Path(remaining_filename)
    if not remaining_file.exists():
        hydra_cwd = Path(hydra.utils.get_original_cwd())
        remaining_file = hydra_cwd / remaining_filename
        if not remaining_file.exists():
            raise FileNotFoundError(f"Remaining features file not found: {remaining_filename}")
    log.info(f"Loading remaining features from {remaining_file}")
    
    cols_to_load = ["signal"] + [col for col in REMAINING_FEATURES if col != "vocal_gate"]
    remaining_df = pd.DataFrame(read_jsonl(str(remaining_file)))
    cols_to_load = [col for col in cols_to_load if col in remaining_df.columns]
    remaining_df = remaining_df[cols_to_load]

    # --- Merge all dataframes ---
    merged_df = pd.merge(main_features_df, metadata_df, on="signal", how="inner")
    merged_df = pd.merge(merged_df, whisper_df, on="signal", how="inner")
    merged_df = pd.merge(merged_df, songprep_alt_df, on="signal", how="inner")
    merged_df = pd.merge(merged_df, zimtohrli_df, on="signal", how="inner")
    merged_df = pd.merge(merged_df, lyricwhiz_df, on="signal", how="inner")
    merged_df = pd.merge(merged_df, whisperx_df, on="signal", how="inner")
    merged_df = pd.merge(merged_df, remaining_df, on="signal", how="inner")
    log.info(f"Final merged data has {len(merged_df)} samples.")
    if merged_df.empty: raise ValueError("Merging resulted in empty DataFrame.")

    # --- Gating ---
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

    # --- NEW: Apply Advanced Features (Suggestion #4) ---
    merged_df = add_advanced_features(merged_df)

    # --- Use the complete ALL_FEATURES list ---
    all_needed_columns = ALL_FEATURES
    missing_cols = [col for col in all_needed_columns if col not in merged_df.columns]
    if missing_cols:
        log.warning(f"Missing expected columns after merge: {missing_cols}. They will be filled with NaN/mean.")
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


class TabularDataset(Dataset):
    """Dataset for multimodal tabular data."""
    def __init__(self, X_dict: dict, y: np.ndarray | None):
        self.X_embed = torch.tensor(X_dict['embeds'], dtype=torch.float32)
        self.X_acoustic = torch.tensor(X_dict['acoustic'], dtype=torch.float32)
        self.X_scalar = torch.tensor(X_dict['scalar'], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X_embed)

    def __getitem__(self, idx):
        if self.y is not None:
            return (
                self.X_embed[idx],
                self.X_acoustic[idx],
                self.X_scalar[idx],
                self.y[idx]
            )
        return self.X_embed[idx], self.X_acoustic[idx], self.X_scalar[idx]


# --- NEW (Suggestion #7): Improved Modality Tower MLP Model ---
class ImprovedModalityTower(nn.Module):
    def __init__(self, embed_dim, acoustic_dim, scalar_dim, hidden_dim=128):
        super().__init__()
        # Tower 1: Embeddings (with LayerNorm instead of BatchNorm)
        self.embed_tower = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),  # <-- Changed from BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),  # <-- Changed from BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.embed_attention = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
        
        # Tower 2: Acoustic Features (with LayerNorm)
        self.acoustic_tower = nn.Sequential(
            nn.Linear(acoustic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # <-- Changed from BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),  # <-- Changed from BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Tower 3: Scalar Features (with LayerNorm)
        self.scalar_tower = nn.Sequential(
            nn.Linear(scalar_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),  # <-- Changed from BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),  # <-- Changed from BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Combined Head (with LayerNorm)
        combined_dim = hidden_dim + (hidden_dim // 2) + (hidden_dim // 4)
        self.head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),  # <-- Changed from BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x_embed, x_acoustic, x_scalar):
        # Process embeddings with attention
        emb = self.embed_tower(x_embed)
        emb_attended, _ = self.embed_attention(
            emb.unsqueeze(1), emb.unsqueeze(1), emb.unsqueeze(1)
        )
        emb = emb_attended.squeeze(1)
        
        out_acoustic = self.acoustic_tower(x_acoustic)
        out_scalar = self.scalar_tower(x_scalar)
        
        combined = torch.cat([emb, out_acoustic, out_scalar], dim=1)
        output = self.head(combined)
        return output.squeeze(1)


# --- LGBM Model Training Function (Phase 1) ---
def train_lgbm_phase1(X_train_sub, y_train_sub, X_val_sub, y_val_sub, lgbm_params: dict) -> tuple[lgb.LGBMRegressor, int]:
    """ Trains LGBM with early stopping to find the best number of iterations. """
    log.info("Starting Phase 1 Training (LGBM - find best iteration)...")
    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(
        X_train_sub, y_train_sub,
        eval_set=[(X_val_sub, y_val_sub)],
        eval_metric='l2',
        callbacks=[lgb.early_stopping(100, verbose=False)] # Quieter logging
    )
    best_iteration = model.best_iteration_
    if best_iteration is None or best_iteration <= 0:
        log.warning(f"LGBM early stopping failed. Defaulting to {lgbm_params.get('n_estimators', 100)}.")
        best_iteration = lgbm_params.get('n_estimators', 100)
    log.info(f"Phase 1 (LGBM) finished. Best iteration: {best_iteration}")
    return model, best_iteration

def train_lgbm_phase2(X_train_full, y_train_full, lgbm_params: dict, best_iteration: int) -> lgb.LGBMRegressor:
    log.info(f"Starting Phase 2 Training (LGBM - final model): {best_iteration} fixed iterations...")
    final_params = lgbm_params.copy()
    final_params['n_estimators'] = best_iteration
    final_params.pop('metric', None)
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X_train_full, y_train_full)
    log.info("Phase 2 Training (LGBM) finished.")
    return model


# --- CatBoost Model Training Function (Phase 1) ---
def train_catboost_phase1(X_train_sub, y_train_sub, X_val_sub, y_val_sub, catboost_params: dict) -> tuple[CatBoostRegressor, int]:
    log.info("Starting Phase 1 Training (CatBoost - find best iteration)...")
    model = CatBoostRegressor(**catboost_params)
    model.fit(
        X_train_sub, y_train_sub,
        eval_set=[(X_val_sub, y_val_sub)],
        early_stopping_rounds=100,
        verbose=False # Quieter logging
    )
    best_iteration = model.get_best_iteration()
    if best_iteration is None or best_iteration <= 0:
        log.warning(f"CatBoost early stopping failed. Defaulting to {catboost_params.get('iterations', 1000)}.")
        best_iteration = catboost_params.get('iterations', 1000)
    log.info(f"Phase 1 (CatBoost) finished. Best iteration: {best_iteration}")
    return model, best_iteration

def train_catboost_phase2(X_train_full, y_train_full, catboost_params: dict, best_iteration: int) -> CatBoostRegressor:
    log.info(f"Starting Phase 2 Training (CatBoost - final model): {best_iteration} fixed iterations...")
    final_params = catboost_params.copy()
    final_params['iterations'] = best_iteration
    final_params.pop('eval_metric', None)
    final_params['verbose'] = False
    model = CatBoostRegressor(**final_params)
    model.fit(X_train_full, y_train_full, verbose=False)
    log.info("Phase 2 Training (CatBoost) finished.")
    return model


# --- NEW (Suggestion #3): XGBoost Model Training Functions ---
def train_xgboost_phase1(X_train_sub, y_train_sub, X_val_sub, y_val_sub, xgboost_params: dict) -> tuple[xgb.Booster, int]:
    """ Trains XGBoost with early stopping. """
    log.info("Starting Phase 1 Training (XGBoost - find best iteration)...")
    dtrain = xgb.DMatrix(X_train_sub, label=y_train_sub)
    dval = xgb.DMatrix(X_val_sub, label=y_val_sub)
    
    model = xgb.train(
        xgboost_params, dtrain,
        num_boost_round=4000, # High number for early stopping
        evals=[(dval, 'valid')],
        early_stopping_rounds=100,
        verbose_eval=False # Quieter logging
    )
    
    best_iteration = model.best_iteration
    log.info(f"Phase 1 (XGBoost) finished. Best iteration: {best_iteration}")
    return model, best_iteration

def train_xgboost_phase2(X_train_full, y_train_full, xgboost_params: dict, best_iteration: int) -> xgb.Booster:
    """ Trains the final XGBoost model. """
    log.info(f"Starting Phase 2 Training (XGBoost - final model): {best_iteration} fixed iterations...")
    dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full)
    
    model = xgb.train(
        xgboost_params, dtrain_full,
        num_boost_round=best_iteration
    )
    log.info("Phase 2 Training (XGBoost) finished.")
    return model

# --- NN Training Functions (Phase 1) ---
def train_nn_phase1(
    X_train_dict, y_train, X_val_dict, y_val, nn_params, device
) -> tuple[ImprovedModalityTower, int]:
    log.info("Starting Phase 1 Training (NN Tower - find best epoch)...")
    
    embed_dim = X_train_dict['embeds'].shape[1]
    acoustic_dim = X_train_dict['acoustic'].shape[1]
    scalar_dim = X_train_dict['scalar'].shape[1]

    # Use the new model
    model = ImprovedModalityTower(embed_dim, acoustic_dim, scalar_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=nn_params['lr'])
    loss_fn = nn.MSELoss()
    
    train_dataset = TabularDataset(X_train_dict, y_train)
    val_dataset = TabularDataset(X_val_dict, y_val)
    train_loader = DataLoader(train_dataset, batch_size=nn_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=nn_params['batch_size'] * 2)

    best_val_loss = float('inf')
    best_epoch = 0
    patience = 20
    epochs_no_improve = 0

    for epoch in range(nn_params['epochs']):
        model.train()
        for x_e, x_a, x_s, y_batch in train_loader:
            x_e, x_a, x_s, y_batch = x_e.to(device), x_a.to(device), x_s.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_e, x_a, x_s)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_e, x_a, x_s, y_batch in val_loader:
                x_e, x_a, x_s, y_batch = x_e.to(device), x_a.to(device), x_s.to(device), y_batch.to(device)
                preds = model(x_e, x_a, x_s)
                loss = loss_fn(preds, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), "nn_tower_best_model.pth")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            log.info(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch} (Val Loss: {best_val_loss:.6f})")
            break
            
    model.load_state_dict(torch.load("nn_tower_best_model.pth"))
    log.info(f"Phase 1 (NN) finished. Best epoch: {best_epoch}")
    return model, best_epoch


# --- NN Final Model Training Function (Phase 2) ---
def train_nn_phase2(
    X_full_dict, y_full, nn_params, device, best_epoch: int
) -> ImprovedModalityTower:
    log.info(f"Starting Phase 2 Training (NN Tower - final model): {best_epoch} fixed epochs...")

    embed_dim = X_full_dict['embeds'].shape[1]
    acoustic_dim = X_full_dict['acoustic'].shape[1]
    scalar_dim = X_full_dict['scalar'].shape[1]
    
    # Use the new model
    model = ImprovedModalityTower(embed_dim, acoustic_dim, scalar_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=nn_params['lr'])
    loss_fn = nn.MSELoss()
    
    train_dataset = TabularDataset(X_full_dict, y_full)
    train_loader = DataLoader(train_dataset, batch_size=nn_params['batch_size'], shuffle=True)

    for epoch in range(best_epoch):
        model.train()
        for x_e, x_a, x_s, y_batch in train_loader:
            x_e, x_a, x_s, y_batch = x_e.to(device), x_a.to(device), x_s.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_e, x_a, x_s)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0 or (epoch + 1) == best_epoch:
            log.info(f"Phase 2, Epoch {epoch+1}/{best_epoch}")
            
    log.info("Phase 2 Training (NN Tower) finished.")
    return model


# --- NN Prediction Function ---
def predict_nn(model: ImprovedModalityTower, X_predict_dict, device, nn_params) -> np.ndarray:
    log.info("Starting NN Prediction...")
    model.eval()
    pred_dataset = TabularDataset(X_predict_dict, y=None)
    pred_loader = DataLoader(pred_dataset, batch_size=nn_params['batch_size'] * 2)
    
    all_preds = []
    with torch.no_grad():
        for x_e, x_a, x_s in pred_loader:
            x_e, x_a, x_s = x_e.to(device), x_a.to(device), x_s.to(device)
            preds = model(x_e, x_a, x_s)
            all_preds.append(preds.cpu().numpy())
            
    return np.concatenate(all_preds)


# --- SHAP Analysis Function ---
def run_shap_analysis(model, X_val_sub, feature_names: list, model_type: str):
    log.info("\n" + "="*50)
    log.info(f"--- STARTING SHAP Analysis on Validation Set ({model_type}) ---")
    log.info("="*50)
    try:
        if model_type == 'xgboost':
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(xgb.DMatrix(X_val_sub))
        else: # lgbm, catboost
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_val_sub)

        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        shap_importance_df = pd.DataFrame(
            list(zip(feature_names, mean_abs_shap)),
            columns=['feature', 'mean_abs_shap']
        ).sort_values(by='mean_abs_shap', ascending=False)
        
        log.info(f"Top 20 Features (Mean Absolute SHAP):\n{shap_importance_df.head(20).to_string()}")

        gated_features_of_interest = [
            'stoi_vocals', 'pesq_wb_vocals',
            'stoi_vocals_gated', 'pesq_wb_vocals_gated',
            'vocal_gate', 'vocal_snr_ratio'
        ]
        gated_features_shap = shap_importance_df[
            shap_importance_df['feature'].isin(gated_features_of_interest)
        ]
        log.info(f"\nSHAP Analysis for Gating-Related Features:\n{gated_features_shap.to_string()}")
        log.info("="*50 + "\n")

    except Exception as e:
        log.error(f"SHAP analysis failed: {e}", exc_info=True)


# --- MODIFIED: Main Experiment Function ---
@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_experiment(cfg: DictConfig):
    log.info(f"Starting CV-Stacking Prediction (Two-Phase) - Config:\n{cfg}")

    predict_split = cfg.get("split", "valid")
    train_split = cfg.get("train_split", "train")
    log.info(f"Training on: '{train_split}', Predicting on: '{predict_split}'")

    try:
        train_val_df_full, y_train_val_full = load_all_features_and_metadata(cfg, train_split, is_prediction_set=False)
        predict_df_full, _ = load_all_features_and_metadata(cfg, predict_split, is_prediction_set=True)
    except Exception as e: log.error(f"Failed to load data: {e}", exc_info=True); return

    predict_signals = predict_df_full['signal'].copy()
    train_val_signals = train_val_df_full['signal'].copy() # For OOF saving

    features_to_use = list(cfg.baseline.get("features_to_use", ALL_FEATURES))
    features_to_use = [f for f in features_to_use if f in train_val_df_full.columns]
    
    log.info(f"Extracting {len(features_to_use)} features...")
    X_train_val_full = train_val_df_full[features_to_use].values
    X_predict_full = predict_df_full[features_to_use].values
    log.info(f"Training data shape: {X_train_val_full.shape}")
    log.info(f"Prediction data shape: {X_predict_full.shape}")

    model_type = cfg.baseline.get("model_type", "lgbm")
    
    # --- Define Model Parameters ---
    lgbm_params = {
        'objective': 'regression_l2', 'metric': ['l2', 'pearson'], 'n_estimators': 2000,
        'learning_rate': 0.02, 'n_jobs': -1, 'random_state': 42, 'boosting_type': 'gbdt',
        'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    }
    
    catboost_params = {
        'iterations': 4000, 'learning_rate': 0.03, 'depth': 8, 'loss_function': 'RMSE',
        'random_seed': 42, 'thread_count': -1, 'verbose': False, 'early_stopping_rounds': 100,
    }

    # --- NEW (Suggestion #3) ---
    xgboost_params = {
        'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'learning_rate': 0.02,
        'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 3,
        'alpha': 0.1, 'lambda': 0.1, 'seed': 42, 'nthread': -1,
    }

    nn_params = {'epochs': 200, 'batch_size': 128, 'lr': 1e-4}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # =========================================================================
    # --- PHASE 1: CV-Stacking to get OOF Preds & Best Iteration ---
    # =========================================================================
    n_folds = cfg.baseline.get("n_folds", 5) # Default to 5 folds
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    oof_preds_all = np.zeros(len(y_train_val_full))
    all_best_iterations = []
    
    log.info("\n" + "="*50)
    log.info(f"--- STARTING PHASE 1: {n_folds}-Fold CV Stacking ({model_type.upper()}) ---")
    log.info("="*50)

    # --- Prep for NN Tower (Scaling) ---
    X_train_val_full_nn = None
    X_predict_full_nn = None
    if model_type == 'nn_tower':
        log.info("Preparing data for NN Tower (scaling)...")
        scaler_embed = StandardScaler().fit(train_val_df_full[EMBED_FEATURES].values)
        scaler_acoustic = StandardScaler().fit(train_val_df_full[ACOUSTIC_FEATURES].values)
        scaler_scalar = StandardScaler().fit(train_val_df_full[SCALAR_FEATURES].values)
        
        X_train_val_full_nn = {
            'embeds': scaler_embed.transform(train_val_df_full[EMBED_FEATURES].values),
            'acoustic': scaler_acoustic.transform(train_val_df_full[ACOUSTIC_FEATURES].values),
            'scalar': scaler_scalar.transform(train_val_df_full[SCALAR_FEATURES].values),
        }
        X_predict_full_nn = {
            'embeds': scaler_embed.transform(predict_df_full[EMBED_FEATURES].values),
            'acoustic': scaler_acoustic.transform(predict_df_full[ACOUSTIC_FEATURES].values),
            'scalar': scaler_scalar.transform(predict_df_full[SCALAR_FEATURES].values),
        }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val_full, y_train_val_full)):
        log.info(f"\n--- Fold {fold+1}/{n_folds} ---")
        
        if model_type == 'nn_tower':
            X_train_sub_dict = {k: v[train_idx] for k, v in X_train_val_full_nn.items()}
            X_val_sub_dict = {k: v[val_idx] for k, v in X_train_val_full_nn.items()}
            y_train_sub, y_val_sub = y_train_val_full[train_idx], y_train_val_full[val_idx]

            model_fold, best_iter_fold = train_nn_phase1(
                X_train_sub_dict, y_train_sub, X_val_sub_dict, y_val_sub, nn_params, device
            )
            val_preds = predict_nn(model_fold, X_val_sub_dict, device, nn_params)
            
        else: # Tree models
            X_train_sub, X_val_sub = X_train_val_full[train_idx], X_train_val_full[val_idx]
            y_train_sub, y_val_sub = y_train_val_full[train_idx], y_train_val_full[val_idx]

            if model_type == 'lgbm':
                model_fold, best_iter_fold = train_lgbm_phase1(
                    X_train_sub, y_train_sub, X_val_sub, y_val_sub, lgbm_params
                )
                val_preds = model_fold.predict(X_val_sub)
            elif model_type == 'catboost':
                model_fold, best_iter_fold = train_catboost_phase1(
                    X_train_sub, y_train_sub, X_val_sub, y_val_sub, catboost_params
                )
                val_preds = model_fold.predict(X_val_sub)
            elif model_type == 'xgboost':
                model_fold, best_iter_fold = train_xgboost_phase1(
                    X_train_sub, y_train_sub, X_val_sub, y_val_sub, xgboost_params
                )
                val_preds = model_fold.predict(xgb.DMatrix(X_val_sub))
            else:
                log.error(f"Unknown model_type: {model_type}"); return

            # --- Run SHAP on first fold only ---
            if fold == 0 and cfg.baseline.get("run_shap", False):
                run_shap_analysis(model_fold, X_val_sub, features_to_use, model_type)

        oof_preds_all[val_idx] = val_preds
        all_best_iterations.append(best_iter_fold)
        del model_fold # Free up memory
        
    # --- CV Finished ---
    avg_best_iteration = int(np.mean(all_best_iterations))
    log.info(f"\n--- CV Stacking Finished ---")
    log.info(f"All 'best_iterations': {all_best_iterations}")
    log.info(f"Average best iteration: {avg_best_iteration}")

    # =========================================================================
    # --- Internal Validation (on FULL OOF set) ---
    # =========================================================================
    oof_preds_scaled = np.clip(oof_preds_all, 0.0, 1.0) * 100.0
    y_train_val_scaled = y_train_val_full * 100.0
    
    internal_rmse = rmse_score(oof_preds_scaled, y_train_val_scaled)
    internal_ncc = ncc_score(oof_preds_scaled, y_train_val_scaled)
    log.info("--- Internal Validation Scores (on FULL OOF set) ---")
    log.info(f"--- RMSE: {internal_rmse:.6f}")
    log.info(f"--- NCC:  {internal_ncc:.6f}")
    log.info("---------------------------------------------------------")
    
    oof_df = pd.DataFrame({
        'signal': train_val_signals, 
        'oof_pred': oof_preds_scaled, 
        'true_correctness': y_train_val_scaled
    })
    oof_filename = f"oof_preds.{cfg.baseline.model_type}.csv"
    oof_df.to_csv(oof_filename, index=False, float_format='%.8f')
    log.info(f"Saved FULL OOF predictions for blender training to {oof_filename}")

    # =========================================================================
    # --- PHASE 2: Train Final Model ---
    # =========================================================================
    log.info("\n" + "="*50)
    log.info(f"--- STARTING PHASE 2: Training Final Model on ALL data ({model_type.upper()}) ---")
    log.info(f"--- Training for fixed {avg_best_iteration} iterations/epochs ---")
    log.info("="*50)

    if model_type == 'lgbm':
        final_model = train_lgbm_phase2(
            X_train_val_full, y_train_val_full, lgbm_params, avg_best_iteration
        )
        all_predictions = final_model.predict(X_predict_full)
    elif model_type == 'catboost':
        final_model = train_catboost_phase2(
            X_train_val_full, y_train_val_full, catboost_params, avg_best_iteration
        )
        all_predictions = final_model.predict(X_predict_full)
    elif model_type == 'xgboost':
        final_model = train_xgboost_phase2(
            X_train_val_full, y_train_val_full, xgboost_params, avg_best_iteration
        )
        all_predictions = final_model.predict(xgb.DMatrix(X_predict_full))
    elif model_type == 'nn_tower':
        final_model = train_nn_phase2(
            X_train_val_full_nn, y_train_val_full, nn_params, device, avg_best_iteration
        )
        all_predictions = predict_nn(final_model, X_predict_full_nn, device, nn_params)
    else:
        log.error(f"Unknown model_type: {model_type}"); return

    # =========================================================================
    # --- PREDICTION PHASE ---
    # =========================================================================
    log.info("\n" + "="*50)
    log.info(f"--- STARTING PREDICTION on '{predict_split}' split ---")
    log.info("="*50)
    
    predicted_scores = np.clip(np.array(all_predictions), 0.0, 1.0) * 100.0
    if len(predicted_scores) != len(predict_signals): 
        log.error(f"Prediction count mismatch! Expected {len(predict_signals)}, got {len(predicted_scores)}")
        return

    # --- Save Results ---
    results_df = pd.DataFrame({'signal': predict_signals, 'predicted_correctness': predicted_scores})
    output_filename = f"{cfg.data.dataset}.{cfg.baseline.system}_{model_type}_final.{predict_split}.predict.csv"
    output_path = Path(output_filename)
    log.info(f"Saving predictions to {output_path}...")
    try:
        results_df.to_csv(
            output_path, index=False, header=["signal_ID", "intelligibility_score"], mode="w", float_format='%.8f'
        )
        log.info("Predictions saved successfully.")
    except Exception as e: log.error(f"Failed to save predictions: {e}", exc_info=True)


if __name__ == "__main__":
    globals()['ALL_FEATURES'] = ALL_FEATURES
    globals()['SCALAR_FEATURES'] = SCALAR_FEATURES
    globals()['EMBED_FEATURES'] = EMBED_FEATURES
    globals()['ACOUSTIC_FEATURES'] = ACOUSTIC_FEATURES
    run_experiment()