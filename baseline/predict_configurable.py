from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union  # Needed for Python 3.8

import hydra
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap  # --- NEW: Import SHAP ---
from catboost import CatBoostRegressor  # --- NEW: Import CatBoost ---
from omegaconf import DictConfig
from scipy.stats import kendalltau, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # --- NEW ---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Import the necessary function
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
    # "nli_entail",
    "songprep_wer",
    "whisper",
    "lyricwhiz",
]
OPENAI_SAMPLED_FEATURES = [
    "oa_correct_mean", "oa_correct_median", "oa_correct_std",
    "oa_correct_min", "oa_correct_max",
]
WHISPERX_DETERMINISTIC_FEATURE = ["whisperx_correct"]

# --- NEW: Features from features_remaining.jsonl ---
# Note: 'vocal_snr_ratio' is used for gating but can also be a feature
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

# --- NEW: Gated features and the gate itself ---
GATED_FEATURES = [
    "vocal_gate",
    "stoi_vocals_gated",
    "pesq_wb_vocals_gated"
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
    + REMAINING_FEATURES  # Add new features
    + GATED_FEATURES      # Add gated features
)
# Remove potential duplicates if 'vocal_snr_ratio' was in both
ALL_FEATURES = sorted(list(set(ALL_FEATURES)))
SCALAR_FEATURES = sorted(
    list(
        set(ALL_FEATURES) - set(EMBED_FEATURES) - set(ACOUSTIC_FEATURES)
    )
)

def rmse_score(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the root mean squared error between two arrays"""
    return np.sqrt(np.mean((x - y) ** 2))


def ncc_score(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the normalized cross correlation between two arrays"""
    if np.std(x) == 0 or np.std(y) == 0 or np.isnan(x).any() or np.isnan(y).any():
        log.warning("NCC calculation failed (zero variance or NaN). Returning 0.0")
        return 0.0
    return pearsonr(x, y)[0]


# --- Data Loading (MODIFIED to load remaining features and apply gating) ---
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

    # --- NEW: Load the 'features_remaining.jsonl' file ---
    remaining_filename = f"{cfg.data.dataset}.{split}.features_remaining.jsonl"
    remaining_file = Path(remaining_filename)
    if not remaining_file.exists():
        hydra_cwd = Path(hydra.utils.get_original_cwd())
        remaining_file = hydra_cwd / remaining_filename
        if not remaining_file.exists():
            raise FileNotFoundError(f"Remaining features file not found: {remaining_filename}")
    log.info(f"Loading remaining features from {remaining_file}")
    # Load only 'signal' + features we defined in REMAINING_FEATURES
    cols_to_load = ["signal"] + [col for col in REMAINING_FEATURES if col != "vocal_gate"]
    remaining_df = pd.DataFrame(read_jsonl(str(remaining_file)))
    # Ensure we only keep columns that exist in the file
    cols_to_load = [col for col in cols_to_load if col in remaining_df.columns]
    remaining_df = remaining_df[cols_to_load]

    # --- Merge all dataframes ---
    merged_df = pd.merge(main_features_df, metadata_df, on="signal", how="inner")
    merged_df = pd.merge(merged_df, whisper_df, on="signal", how="inner")
    merged_df = pd.merge(merged_df, songprep_alt_df, on="signal", how="inner")
    merged_df = pd.merge(merged_df, zimtohrli_df, on="signal", how="inner")
    merged_df = pd.merge(merged_df, lyricwhiz_df, on="signal", how="inner")
    merged_df = pd.merge(merged_df, whisperx_df, on="signal", how="inner")
    merged_df = pd.merge(merged_df, remaining_df, on="signal", how="inner") # --- NEW MERGE
    log.info(f"Final merged data has {len(merged_df)} samples.")
    if merged_df.empty: raise ValueError("Merging resulted in empty DataFrame.")


    # --- NEW: Confidence gating for Demucs-derived metrics ---
    # Use 'vocal_snr_ratio' from your sample JSONL
    snr_col = 'vocal_snr_ratio' 
    if snr_col in merged_df.columns:
        alpha = float(getattr(cfg.baseline, 'vocal_gate_alpha', 1.0))  # slope
        snr0 = float(getattr(cfg.baseline, 'vocal_gate_snr0', 0.0))    # midpoint
        # Gate between 0 and 1
        merged_df['vocal_gate'] = 1.0 / (1.0 + np.exp(-alpha * (merged_df[snr_col] - snr0)))
    else:
        log.warning(f"{snr_col} not found; setting vocal_gate=1 for all rows")
        merged_df['vocal_gate'] = 1.0

    # Apply gating to Demucs-dependent features
    for col in ['stoi_vocals', 'pesq_wb_vocals']:
        if col in merged_df.columns:
            merged_df[f'{col}_gated'] = merged_df[col] * merged_df['vocal_gate']
        else:
            log.warning(f"{col} not found in merged_df; skipping gating.")
            merged_df[f'{col}_gated'] = np.nan # Will be filled by mean


    # --- Use the complete ALL_FEATURES list ---
    all_needed_columns = ALL_FEATURES

    # Check for missing columns *after* creating gated ones
    missing_cols = [col for col in all_needed_columns if col not in merged_df.columns]
    if missing_cols:
        log.warning(f"Missing expected columns after merge: {missing_cols}. They will be filled with NaN/mean.")
        # Add missing columns with NaN
        for col in missing_cols:
            merged_df[col] = np.nan
            
    # Replace inf and fill NaNs (from merges or nulls in JSONL)
    merged_df[all_needed_columns] = merged_df[all_needed_columns].replace([np.inf, -np.inf], np.nan)
    if merged_df[all_needed_columns].isnull().any().any():
        log.warning("NaNs detected in features! Filling with column means.")
        # Fill NaNs using the mean of the *training* data (if available) or current split
        # For simplicity here, we fill with current split's mean
        merged_df[all_needed_columns] = merged_df[all_needed_columns].fillna(merged_df[all_needed_columns].mean())
        if merged_df[all_needed_columns].isnull().any().any():
             log.error("NaNs still present after mean imputation. Check data.")
             # Fill remaining NaNs (e.g., if a whole column was NaN) with 0
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
        # Else (for prediction)
        return self.X_embed[idx], self.X_acoustic[idx], self.X_scalar[idx]


# --- NEW: Modality Tower MLP Model ---
class ModalityTowerMLP(nn.Module):
    def __init__(self, embed_dim, acoustic_dim, scalar_dim, hidden_dim=128):
        super().__init__()
        # Tower 1: Embeddings
        self.embed_tower = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        # Tower 2: Acoustic Features
        self.acoustic_tower = nn.Sequential(
            nn.Linear(acoustic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        # Tower 3: Scalar Features
        self.scalar_tower = nn.Sequential(
            nn.Linear(scalar_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Combined Head
        combined_dim = hidden_dim + (hidden_dim // 2) + (hidden_dim // 4)
        self.head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output 0-1
        )

    def forward(self, x_embed, x_acoustic, x_scalar):
        out_embed = self.embed_tower(x_embed)
        out_acoustic = self.acoustic_tower(x_acoustic)
        out_scalar = self.scalar_tower(x_scalar)
        
        combined = torch.cat([out_embed, out_acoustic, out_scalar], dim=1)
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
        eval_metric='l2',  # Stop based on RMSE (l2)
        callbacks=[lgb.early_stopping(100, verbose=True)]
    )
    best_iteration = model.best_iteration_
    if best_iteration is None or best_iteration <= 0:
        log.warning(f"LGBM early stopping failed. Defaulting to {lgbm_params.get('n_estimators', 100)}.")
        best_iteration = lgbm_params.get('n_estimators', 100)
    log.info(f"Phase 1 Training (LGBM) finished. Best iteration: {best_iteration}")
    return model, best_iteration


# --- LGBM Final Model Training Function (Phase 2) ---
def train_lgbm_phase2(X_train_full, y_train_full, lgbm_params: dict, best_iteration: int) -> lgb.LGBMRegressor:
    """ Trains the final LGBM model on *all* training data for a *fixed* number of iterations. """
    log.info(f"Starting Phase 2 Training (LGBM - final model): {best_iteration} fixed iterations...")
    final_params = lgbm_params.copy()
    final_params['n_estimators'] = best_iteration
    final_params.pop('metric', None)
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X_train_full, y_train_full)
    log.info("Phase 2 Training (LGBM) finished.")
    return model


# --- NEW: CatBoost Model Training Function (Phase 1) ---
def train_catboost_phase1(X_train_sub, y_train_sub, X_val_sub, y_val_sub, catboost_params: dict) -> tuple[CatBoostRegressor, int]:
    """ Trains CatBoost with early stopping to find the best number of iterations. """
    log.info("Starting Phase 1 Training (CatBoost - find best iteration)...")
    model = CatBoostRegressor(**catboost_params)
    model.fit(
        X_train_sub, y_train_sub,
        eval_set=[(X_val_sub, y_val_sub)],
        early_stopping_rounds=100,
        verbose=100
    )
    best_iteration = model.get_best_iteration()
    if best_iteration is None or best_iteration <= 0:
        log.warning(f"CatBoost early stopping failed. Defaulting to {catboost_params.get('iterations', 1000)}.")
        best_iteration = catboost_params.get('iterations', 1000)
    log.info(f"Phase 1 Training (CatBoost) finished. Best iteration: {best_iteration}")
    return model, best_iteration


def train_nn_phase1(
    X_train_dict, y_train, X_val_dict, y_val, nn_params, device
) -> tuple[ModalityTowerMLP, int]:
    """ Trains NN with early stopping to find the best epoch. """
    log.info("Starting Phase 1 Training (NN Tower - find best epoch)...")
    
    # Get dims
    embed_dim = X_train_dict['embeds'].shape[1]
    acoustic_dim = X_train_dict['acoustic'].shape[1]
    scalar_dim = X_train_dict['scalar'].shape[1]

    model = ModalityTowerMLP(embed_dim, acoustic_dim, scalar_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=nn_params['lr'])
    loss_fn = nn.MSELoss() # RMSE is just sqrt(MSE)
    
    train_dataset = TabularDataset(X_train_dict, y_train)
    val_dataset = TabularDataset(X_val_dict, y_val)
    train_loader = DataLoader(train_dataset, batch_size=nn_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=nn_params['batch_size'] * 2)

    best_val_loss = float('inf')
    best_epoch = 0
    patience = 20  # N epochs to wait for improvement
    epochs_no_improve = 0

    for epoch in range(nn_params['epochs']):
        model.train()
        train_loss = 0.0
        for x_e, x_a, x_s, y_batch in train_loader:
            x_e, x_a, x_s, y_batch = x_e.to(device), x_a.to(device), x_s.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_e, x_a, x_s)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_e, x_a, x_s, y_batch in val_loader:
                x_e, x_a, x_s, y_batch = x_e.to(device), x_a.to(device), x_s.to(device), y_batch.to(device)
                preds = model(x_e, x_a, x_s)
                loss = loss_fn(preds, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        if (epoch + 1) % 10 == 0:
             log.info(f"Epoch {epoch+1}/{nn_params['epochs']}, Val Loss (MSE): {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            # Save the best model state
            torch.save(model.state_dict(), "nn_tower_best_model.pth")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            log.info(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch} (Val Loss: {best_val_loss:.6f})")
            break
            
    # Load the best model state back
    model.load_state_dict(torch.load("nn_tower_best_model.pth"))
    return model, best_epoch


# --- NEW: NN Final Model Training Function (Phase 2) ---
def train_nn_phase2(
    X_full_dict, y_full, nn_params, device, best_epoch: int
) -> ModalityTowerMLP:
    """ Trains final NN model on *all* data for a *fixed* number of epochs. """
    log.info(f"Starting Phase 2 Training (NN Tower - final model): {best_epoch} fixed epochs...")

    # Get dims
    embed_dim = X_full_dict['embeds'].shape[1]
    acoustic_dim = X_full_dict['acoustic'].shape[1]
    scalar_dim = X_full_dict['scalar'].shape[1]
    
    model = ModalityTowerMLP(embed_dim, acoustic_dim, scalar_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=nn_params['lr'])
    loss_fn = nn.MSELoss()
    
    train_dataset = TabularDataset(X_full_dict, y_full)
    train_loader = DataLoader(train_dataset, batch_size=nn_params['batch_size'], shuffle=True)

    for epoch in range(best_epoch): # Train for fixed number of epochs
        model.train()
        for x_e, x_a, x_s, y_batch in train_loader:
            x_e, x_a, x_s, y_batch = x_e.to(device), x_a.to(device), x_s.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_e, x_a, x_s)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            log.info(f"Phase 2, Epoch {epoch+1}/{best_epoch}")
            
    log.info("Phase 2 Training (NN Tower) finished.")
    return model


# --- NEW: NN Prediction Function ---
def predict_nn(model: ModalityTowerMLP, X_predict_dict, device, nn_params) -> np.ndarray:
    """ Makes predictions using the trained NN model. """
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

# --- NEW: CatBoost Final Model Training Function (Phase 2) ---
def train_catboost_phase2(X_train_full, y_train_full, catboost_params: dict, best_iteration: int) -> CatBoostRegressor:
    """ Trains the final CatBoost model on *all* training data for a *fixed* number of iterations. """
    log.info(f"Starting Phase 2 Training (CatBoost - final model): {best_iteration} fixed iterations...")
    final_params = catboost_params.copy()
    final_params['iterations'] = best_iteration
    final_params.pop('eval_metric', None)
    model = CatBoostRegressor(**final_params)
    model.fit(X_train_full, y_train_full, verbose=False)
    log.info("Phase 2 Training (CatBoost) finished.")
    return model


# --- NEW: SHAP Analysis Function ---
def run_shap_analysis(model, X_val_sub, feature_names: list, model_type: str):
    """ Runs SHAP analysis and logs feature importances. """
    log.info("\n" + "="*50)
    log.info(f"--- STARTING SHAP Analysis on Validation Set ({model_type}) ---")
    log.info("="*50)
    try:
        # TreeExplainer works for both LGBM and CatBoost
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_val_sub)

        # Log overall feature importance
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        shap_importance_df = pd.DataFrame(
            list(zip(feature_names, mean_abs_shap)),
            columns=['feature', 'mean_abs_shap']
        ).sort_values(by='mean_abs_shap', ascending=False)
        
        log.info(f"Top 20 Features (Mean Absolute SHAP):\n{shap_importance_df.head(20).to_string()}")

        # Log specific features of interest (gating)
        gated_features_of_interest = [
            'stoi_vocals', 'pesq_wb_vocals',
            'stoi_vocals_gated', 'pesq_wb_vocals_gated',
            'vocal_gate', 'vocal_snr_ratio'
        ]
        # Filter for features that are actually in our list
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
    log.info(f"Starting Configurable Prediction (Two-Phase) - Config:\n{cfg}")

    predict_split = cfg.get("split", "valid")
    train_split = cfg.get("train_split", "train")
    log.info(f"Training on: '{train_split}', Predicting on: '{predict_split}'")

    # --- Load ALL Data ---
    try:
        train_val_df_full, y_train_val_full = load_all_features_and_metadata(cfg, train_split, is_prediction_set=False)
        predict_df_full, _ = load_all_features_and_metadata(cfg, predict_split, is_prediction_set=True)
    except Exception as e: log.error(f"Failed to load data: {e}", exc_info=True); return

    predict_signals = predict_df_full['signal'].copy()

    # --- NEW: Configurable Feature Extraction ---
    # By default, use all features. Config can override this for ablation.
    features_to_use = list(cfg.baseline.get("features_to_use", ALL_FEATURES))
    # Ensure all features requested are actually available
    features_to_use = [f for f in features_to_use if f in train_val_df_full.columns]
    
    log.info(f"Extracting {len(features_to_use)} features...")
    X_train_val_full = train_val_df_full[features_to_use].values
    X_predict_full = predict_df_full[features_to_use].values
    log.info(f"Training data shape: {X_train_val_full.shape}")
    log.info(f"Prediction data shape: {X_predict_full.shape}")

    # --- Get Model Hyperparameters ---
    val_size = 0.1; random_state = 42
    model_type = cfg.baseline.get("model_type", "lgbm")

    # --- Define Model Parameters ---
    lgbm_params = {
        'objective': 'regression_l2',
        'metric': ['l2', 'pearson'],
        'n_estimators': 2000,
        'learning_rate': 0.02,
        'n_jobs': -1,
        'random_state': 42,
        'boosting_type': 'gbdt',
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }
    
    catboost_params = {
        'iterations': 4000, # High number for early stopping
        'learning_rate': 0.03,
        'depth': 8,
        'loss_function': 'RMSE',
        # 'eval_metric': 'Pearson', # Use Pearson for logging, but early stop on RMSE
        'random_seed': 42,
        'thread_count': -1,
        'verbose': False,
        'early_stopping_rounds': 100,
    }

    nn_params = {
        'epochs': 200,      # High number for early stopping
        'batch_size': 128,
        'lr': 1e-4,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # =========================================================================
    # --- PHASE 1: Find Best Iteration ---
    # =========================================================================
    log.info("\n" + "="*50)
    log.info(f"--- STARTING PHASE 1: Finding Best Iteration (Model: {model_type.upper()}) ---")
    log.info("="*50)

    try:
        train_val_signals = train_val_df_full['signal'].values
        X_train_sub, X_val_sub, y_train_sub, y_val_sub, signals_train_sub, signals_val_sub = train_test_split(
            X_train_val_full, y_train_val_full, train_val_signals, # <-- Add signals here
            test_size=val_size, random_state=random_state
        )
        log.info(f"Train subset size: {len(X_train_sub)}, Val subset size: {len(X_val_sub)}")
    except Exception as e: log.error(f"Failed to split: {e}. Aborting.", exc_info=True); return

    # --- Run Phase 1 Training ---
    if model_type == 'lgbm':
        model_phase1, best_iteration = train_lgbm_phase1(
            X_train_sub, y_train_sub, X_val_sub, y_val_sub, lgbm_params
        )
    elif model_type == 'catboost':
        model_phase1, best_iteration = train_catboost_phase1(
            X_train_sub, y_train_sub, X_val_sub, y_val_sub, catboost_params
        )
    elif model_type == 'nn_tower':
        # --- NN Feature Prep (Dicts + Scaling) ---
        log.info("Preparing data for NN Tower (scaling and splitting)...")
        
        # 1. Create Scalers
        scaler_embed = StandardScaler()
        scaler_acoustic = StandardScaler()
        scaler_scalar = StandardScaler()
        
        # 2. Fit Scalers on FULL training data
        scaler_embed.fit(train_val_df_full[EMBED_FEATURES].values)
        scaler_acoustic.fit(train_val_df_full[ACOUSTIC_FEATURES].values)
        scaler_scalar.fit(train_val_df_full[SCALAR_FEATURES].values)
        
        # 3. Create FULL scaled data dicts
        X_train_val_dict = {
            'embeds': scaler_embed.transform(train_val_df_full[EMBED_FEATURES].values),
            'acoustic': scaler_acoustic.transform(train_val_df_full[ACOUSTIC_FEATURES].values),
            'scalar': scaler_scalar.transform(train_val_df_full[SCALAR_FEATURES].values),
        }
        X_predict_dict = {
            'embeds': scaler_embed.transform(predict_df_full[EMBED_FEATURES].values),
            'acoustic': scaler_acoustic.transform(predict_df_full[ACOUSTIC_FEATURES].values),
            'scalar': scaler_scalar.transform(predict_df_full[SCALAR_FEATURES].values),
        }
        
        # 4. Split train/val subsets
        # We must split the *indices*
        train_indices, val_indices = train_test_split(
            range(len(y_train_val_full)), test_size=val_size, random_state=random_state
        )
        
        X_train_sub_dict = {k: v[train_indices] for k, v in X_train_val_dict.items()}
        y_train_sub = y_train_val_full[train_indices]
        
        X_val_sub_dict = {k: v[val_indices] for k, v in X_train_val_dict.items()}
        y_val_sub = y_train_val_full[val_indices]
        
        signals_val_sub = train_val_df_full['signal'].values[val_indices]
        log.info(f"Train subset size: {len(y_train_sub)}, Val subset size: {len(y_val_sub)}")

        # 5. Run Phase 1
        model_phase1, best_iteration = train_nn_phase1(
            X_train_sub_dict, y_train_sub, X_val_sub_dict, y_val_sub, nn_params, device
        )
        # Store for Phase 2
        X_train_val_full = X_train_val_dict
        X_predict_full = X_predict_dict
    else:
        log.error(f"Unknown model_type: {model_type}")
        return

    # =========================================================================
    # --- NEW: SHAP Analysis Phase ---
    # =========================================================================
    if cfg.baseline.get("run_shap", False):
        if model_type == 'lgbm' or model_type == 'catboost':
            # SHAP for tree models
            run_shap_analysis(model_phase1, X_val_sub, features_to_use, model_type)
        elif model_type == 'nn_tower':
            # --- NEW: SHAP for NN (DeepExplainer) ---
            log.warning("SHAP for NN Tower is complex (DeepExplainer) and not implemented in this script.")
            # Note: Implementing this would require DeepExplainer and batching.
            # explainer = shap.DeepExplainer(model_phase1, [X_val_sub_dict['embeds_batch'], ...])
            pass

    # =========================================================================
    # --- Internal Validation Phase ---
    # =========================================================================
    log.info("\n" + "="*50)
    log.info(f"--- STARTING INTERNAL EVALUATION on {val_size:.0%} hold-out set ---")
    log.info("="*50)
    try:
        if model_type == 'lgbm' or model_type == 'catboost':
            val_preds = model_phase1.predict(X_val_sub)
        elif model_type == 'nn_tower':
            val_preds = predict_nn(model_phase1, X_val_sub_dict, device, nn_params)        
        val_preds_np = np.clip(np.array(val_preds), 0.0, 1.0) * 100.0
        val_targets_np = np.array(y_val_sub) * 100.0
        internal_rmse = rmse_score(val_preds_np, val_targets_np)
        internal_ncc = ncc_score(val_preds_np, val_targets_np)
        log.info("--- Internal Validation Scores (on Phase 1 hold-out) ---")
        log.info(f"--- RMSE: {internal_rmse:.6f}")
        log.info(f"--- NCC:  {internal_ncc:.6f}")
        log.info("---------------------------------------------------------")
        oof_df = pd.DataFrame({
            'signal': signals_val_sub, 
            'oof_pred': val_preds_np, 
            'true_correctness': val_targets_np
        })
        oof_filename = f"oof_preds.{cfg.baseline.model_type}.csv"
        oof_df.to_csv(oof_filename, index=False, float_format='%.8f')
        log.info(f"Saved OOF predictions for blender training to {oof_filename}")
        del model_phase1, val_preds, val_preds_np, val_targets_np
    except Exception as e:
        log.error(f"Failed to run internal validation step: {e}", exc_info=True)
        del model_phase1 # Still try to clean up

    # =========================================================================
    # --- PHASE 2: Train Final Model ---
    # =========================================================================
    log.info("\n" + "="*50)
    log.info(f"--- STARTING PHASE 2: Training Final Model on ALL data ({model_type.upper()}) ---")
    log.info(f"--- Training for fixed {best_iteration} iterations ---")
    log.info("="*50)

    if model_type == 'lgbm':
        final_model = train_lgbm_phase2(
            X_train_val_full, y_train_val_full, lgbm_params, best_iteration
        )
    elif model_type == 'catboost':
        final_model = train_catboost_phase2(
            X_train_val_full, y_train_val_full, catboost_params, best_iteration
        )
    elif model_type == 'nn_tower':
        final_model = train_nn_phase2(
            X_train_val_full, y_train_val_full, nn_params, device, best_iteration
        )
    else:
        log.error(f"Unknown model_type: {model_type}")
        return

    # =========================================================================
    # --- PREDICTION PHASE ---
    # =========================================================================
    log.info("\n" + "="*50)
    log.info(f"--- STARTING PREDICTION on '{predict_split}' split ---")
    log.info("="*50)

    if model_type == 'lgbm' or model_type == 'catboost':
        all_predictions = final_model.predict(X_predict_full)
    elif model_type == 'nn_tower':
        all_predictions = predict_nn(final_model, X_predict_full, device, nn_params)    
    predicted_scores = np.clip(np.array(all_predictions), 0.0, 1.0) * 100.0
    if len(predicted_scores) != len(predict_signals): 
        log.error(f"Prediction count mismatch! Expected {len(predict_signals)}, got {len(predicted_scores)}")
        return

    # --- Save Results ---
    results_df = pd.DataFrame({'signal': predict_signals, 'predicted_correctness': predicted_scores})
    # --- NEW: Filename includes model_type ---
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
    # Define ALL_FEATURES as a global list so it's accessible
    # This is a bit of a hack for the script context but makes it work
    globals()['ALL_FEATURES'] = ALL_FEATURES
    run_experiment()