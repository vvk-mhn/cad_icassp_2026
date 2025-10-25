# train_predict_multimodal_pca.py
"""
Make intelligibility predictions using PCA on high-dimensional features
combined with selected low-dimensional features, trained via MLP.

This script implements a two-phase training strategy:
1.  Phase 1: Split the training set into train/validation subsets.
    Train with early stopping to find the optimal number of epochs.
2.  Phase 2: Re-initialize the model and train it on the *entire*
    training set for the 'best_epoch' number found in Phase 1.
3.  Prediction: Use the model from Phase 2 to make final predictions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union # Needed for Python 3.8 Union type hint

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Import the necessary function
from clarity.utils.file_io import read_jsonl

log = logging.getLogger(__name__)


# --- Feature Groups ---
MFCC_MEAN_FEATURES = [f"mfcc_mean_{i}" for i in range(40)]
MFCC_STD_FEATURES = [f"mfcc_std_{i}" for i in range(40)]
SPECTRAL_CONTRAST_FEATURES = [f"spectral_contrast_mean_{i}" for i in range(7)]
ACOUSTIC_SCALAR_FEATURES = [
    "spectral_centroid_mean", "spectral_centroid_std",
    "stoi", "pesq_wb",
    # "fwSNRseg"
]
PHONEME_FEATURES = ["phoneme_feat_dist", "phoneme_weighted_dist"]
WHISPER_EMBED_FEATURES = [f"whisper_embed_mean_{i}" for i in range(512)]
W2V_EMBED_FEATURES = [f"w2v_embed_mean_{i}" for i in range(1024)]
TEXT_SCALAR_FEATURES = ["whisper_correct", "semantic_sim", "nli_entail", "lyricwhiz"]

PCA_TARGET_FEATURES = (
    MFCC_MEAN_FEATURES + MFCC_STD_FEATURES +
    WHISPER_EMBED_FEATURES + W2V_EMBED_FEATURES
)
DIRECT_FEATURES = (
    SPECTRAL_CONTRAST_FEATURES + ACOUSTIC_SCALAR_FEATURES +
    PHONEME_FEATURES + TEXT_SCALAR_FEATURES
)

# --- Custom Loss Functions ---
def pearson_correlation_loss(pred, target):
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    covariance = (pred_centered * target_centered).mean()
    pred_std = torch.sqrt((pred_centered**2).mean()) + 1e-8
    target_std = torch.sqrt((target_centered**2).mean()) + 1e-8
    correlation = covariance / (pred_std * target_std)
    correlation = torch.clamp(correlation, -1.0, 1.0)
    return 1 - correlation

def combined_loss(pred, target, alpha=0.5):
    mse = F.mse_loss(pred, target); corr = pearson_correlation_loss(pred, target)
    return alpha * mse + (1 - alpha) * corr

# --- MLP Model ---
class PCAPredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.layers(x).squeeze(-1)

# --- Dataset ---
class CombinedFeatureDataset(Dataset):
    def __init__(self, combined_features, targets=None):
        self.features = torch.FloatTensor(np.array(combined_features))
        self.targets = torch.FloatTensor(np.array(targets)) if targets is not None else None
    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        if self.targets is not None: return self.features[idx], self.targets[idx]
        else: return self.features[idx]

# --- Data Loading ---
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

    main_features_filename = f"{cfg.data.dataset}.{split}.{cfg.baseline.system}.jsonl"
    main_features_file = Path(main_features_filename)
    if not main_features_file.exists():
        hydra_cwd = Path(hydra.utils.get_original_cwd())
        main_features_file = hydra_cwd / main_features_filename
        if not main_features_file.exists(): raise FileNotFoundError(f"Main features file not found: {main_features_filename}")
    log.info(f"Loading main features from {main_features_file}")
    main_features_df = pd.DataFrame(read_jsonl(str(main_features_file)))
    log.info(f"Loaded main features ({len(main_features_df)} rows)")

    lyric_features_filename = f"{cfg.data.dataset}.{split}.lyricwhiz.jsonl"
    lyric_features_file = Path(lyric_features_filename)
    if not lyric_features_file.exists():
        hydra_cwd = Path(hydra.utils.get_original_cwd())
        lyric_features_file = hydra_cwd / lyric_features_filename
        if not lyric_features_file.exists(): raise FileNotFoundError(f"Lyricwhiz features file not found: {lyric_features_filename}")
    log.info(f"Loading lyricwhiz features from {lyric_features_file}")
    lyric_features_df = pd.DataFrame(read_jsonl(str(lyric_features_file)))[["signal", "lyricwhiz"]]
    log.info(f"Loaded lyricwhiz features ({len(lyric_features_df)} rows)")


    wav2vec_alt_filename = f"{cfg.data.dataset}.{split}.wav2vec_alt.jsonl"
    wav2vec_alt_file = Path(wav2vec_alt_filename)
    if not wav2vec_alt_file.exists():
        hydra_cwd = Path(hydra.utils.get_original_cwd())
        wav2vec_alt_file = hydra_cwd / wav2vec_alt_filename
        if not wav2vec_alt_file.exists(): raise FileNotFoundError(f"wav2vec_alt features file not found: {wav2vec_alt_filename}")
    log.info(f"Loading wav2vec_alt features from {wav2vec_alt_file}")
    wav2vec_alt_df = pd.DataFrame(read_jsonl(str(wav2vec_alt_file)))[["signal", "wav2vec_alt"]]
    log.info(f"Loaded wav2vec_alt features ({len(wav2vec_alt_df)} rows)")

    merged_df = pd.merge(main_features_df, metadata_df, on="signal", how="inner")
    merged_df = pd.merge(merged_df, lyric_features_df, on="signal", how="inner")
    merged_df = pd.merge(merged_df, wav2vec_alt_df, on="signal", how="inner")
    log.info(f"Final merged data has {len(merged_df)} samples.")
    if merged_df.empty: raise ValueError("Merging resulted in empty DataFrame.")

    all_needed_columns = PCA_TARGET_FEATURES + DIRECT_FEATURES
    missing_cols = [col for col in all_needed_columns if col not in merged_df.columns]
    if missing_cols: raise ValueError(f"Missing expected columns after merge: {missing_cols}")
    merged_df[all_needed_columns] = merged_df[all_needed_columns].replace([np.inf, -np.inf], np.nan)
    if merged_df[all_needed_columns].isnull().any().any():
        log.warning("NaNs detected in features! Filling with column means.")
        merged_df[all_needed_columns] = merged_df[all_needed_columns].fillna(merged_df[all_needed_columns].mean())
        if merged_df[all_needed_columns].isnull().any().any():
             log.error("NaNs still present after mean imputation. Check data.")
             raise ValueError("Unfillable NaNs detected.")

    final_targets = targets_col.loc[merged_df.index].values if targets_col is not None else None
    return merged_df, final_targets


# --- MODIFIED: Model Training Function (Phase 1) ---
def train_model_pca(model, train_loader, val_loader, device, num_epochs=200, lr=0.001) -> tuple[nn.Module, int]:
    """
    Trains the model with early stopping and returns the best model state
    AND the epoch number at which it was found.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    best_val_loss = float('inf'); patience_counter = 0; patience = 20
    best_model_state = model.state_dict().copy()
    best_epoch = 0 # --- NEW: Track the best epoch ---

    log.info(f"Starting Phase 1 Training (find best epoch): {num_epochs} epochs, LR={lr}, Patience={patience}")

    for epoch in range(num_epochs):
        model.train(); train_loss_epoch = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = combined_loss(predictions, y_batch, alpha=0.5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_epoch += loss.item() * x_batch.size(0)
        train_loss_epoch /= len(train_loader.dataset)

        model.eval(); val_loss_epoch = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                predictions = model(x_batch)
                loss = combined_loss(predictions, y_batch, alpha=0.5)
                val_loss_epoch += loss.item() * x_batch.size(0)
        val_loss_epoch /= len(val_loader.dataset)
        scheduler.step(val_loss_epoch)

        if (epoch + 1) % 10 == 0 or epoch == 0: log.info(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {train_loss_epoch:.6f}, Val Loss: {val_loss_epoch:.6f}")

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch; patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1 # --- NEW: Store the best epoch number ---
            log.debug(f"Epoch {epoch+1}: Val loss improved to {best_val_loss:.6f}.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.6f}")
                break

    log.info(f"Phase 1 Training finished. Best epoch found: {best_epoch} (Val Loss: {best_val_loss:.6f})")
    model.load_state_dict(best_model_state)
    return model, best_epoch # --- MODIFIED: Return model and best epoch ---


# --- NEW: Final Model Training Function (Phase 2) ---
def train_final_model(model, train_loader, device, num_epochs, lr=0.001) -> nn.Module:
    """
    Trains the model on the *full* training set for a *fixed*
    number of epochs (no validation, no early stopping).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    log.info(f"Starting Phase 2 Training (final model): {num_epochs} fixed epochs, LR={lr}")

    for epoch in range(num_epochs):
        model.train(); train_loss_epoch = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = combined_loss(predictions, y_batch, alpha=0.5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_epoch += loss.item() * x_batch.size(0)
        train_loss_epoch /= len(train_loader.dataset)

        if (epoch + 1) % 10 == 0 or (epoch+1) == num_epochs or epoch == 0:
            log.info(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {train_loss_epoch:.6f}")

    log.info("Phase 2 Training finished.")
    return model # Return the fully trained model


# --- MODIFIED: Main Prediction Function (Implements Two-Phase Strategy) ---

@hydra.main(config_path="configs", config_name="config", version_base=None)
def predict_with_pca(cfg: DictConfig):
    log.info(f"Starting PCA + MLP Prediction (Two-Phase) - Config:\n{cfg}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    predict_split = cfg.get("split", "valid")
    train_split = cfg.get("train_split", "train")
    n_components_pca = cfg.baseline.mlp.get("n_pca_components", 116)
    log.info(f"Target PCA components: {n_components_pca}")
    log.info(f"Training on: '{train_split}', Predicting on: '{predict_split}'")

    # --- Load ALL Data ---
    try:
        train_val_df_full, y_train_val_full = load_all_features_and_metadata(cfg, train_split, is_prediction_set=False)
        predict_df_full, _ = load_all_features_and_metadata(cfg, predict_split, is_prediction_set=True)
    except Exception as e: log.error(f"Failed to load data: {e}", exc_info=True); return

    predict_signals = predict_df_full['signal'].copy()

    # --- Separate Features ---
    X_train_val_pca_target = train_val_df_full[PCA_TARGET_FEATURES].values
    X_train_val_direct = train_val_df_full[DIRECT_FEATURES].values
    X_predict_pca_target = predict_df_full[PCA_TARGET_FEATURES].values
    X_predict_direct = predict_df_full[DIRECT_FEATURES].values

    # --- Fit Scalers and PCA (on FULL Training set) ---
    log.info(f"Fitting Scalers and PCA on *full* '{train_split}' set ({len(X_train_val_pca_target)} samples)...")
    scaler_pca = StandardScaler().fit(X_train_val_pca_target)
    scaler_direct = StandardScaler().fit(X_train_val_direct)
    pca = PCA(n_components=n_components_pca).fit(scaler_pca.transform(X_train_val_pca_target))
    log.info(f"PCA fitted. Explained variance with {pca.n_components_} components: {pca.explained_variance_ratio_.sum():.4f}")

    # --- Transform ALL Splits ---
    log.info("Transforming full train set and prediction set...")
    X_train_val_pca_comp = pca.transform(scaler_pca.transform(X_train_val_pca_target))
    X_train_val_direct_scaled = scaler_direct.transform(X_train_val_direct)
    X_predict_pca_comp = pca.transform(scaler_pca.transform(X_predict_pca_target))
    X_predict_direct_scaled = scaler_direct.transform(X_predict_direct)

    # --- Combine Features for ALL Splits ---
    X_train_val_combined = np.hstack((X_train_val_pca_comp, X_train_val_direct_scaled))
    X_predict_combined = np.hstack((X_predict_pca_comp, X_predict_direct_scaled))
    combined_dim = X_train_val_combined.shape[1]
    log.info(f"Combined feature dimension: {combined_dim} ({pca.n_components_} PCA + {X_train_val_direct_scaled.shape[1]} Direct)")

    # --- Get Model Hyperparameters ---
    try:
        num_epochs = cfg.baseline.mlp.num_epochs
        learning_rate = cfg.baseline.mlp.learning_rate
        batch_size = cfg.baseline.mlp.get("batch_size", 128)
        num_workers = cfg.baseline.mlp.get("num_workers", 0)
        val_size = 0.1; random_state = 42
    except Exception as e: log.error(f"MLP hyperparameters missing: {e}", exc_info=True); return


    # =========================================================================
    # --- PHASE 1: Find Best Epoch ---
    # =========================================================================
    log.info("\n" + "="*50)
    log.info(f"--- STARTING PHASE 1: Finding Best Epoch (Split: {1-val_size:.0%}/{val_size:.0%}) ---")
    log.info("="*50)

    # --- Split FULL training data into train/val subsets ---
    try:
        X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
            X_train_val_combined, y_train_val_full, test_size=val_size, random_state=random_state
        )
        log.info(f"Train subset size: {len(X_train_sub)}, Val subset size: {len(X_val_sub)}")
    except Exception as e:
        log.error(f"Failed to split: {e}. Using full train set for train/val.", exc_info=True)
        X_train_sub, y_train_sub = X_train_val_combined, y_train_val_full
        X_val_sub, y_val_sub = X_train_val_combined, y_train_val_full # Not ideal

    # --- Create DataLoaders for Phase 1 ---
    train_dataset_sub = CombinedFeatureDataset(X_train_sub, y_train_sub)
    val_dataset_sub = CombinedFeatureDataset(X_val_sub, y_val_sub)
    train_loader_sub = DataLoader(train_dataset_sub, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader_sub = DataLoader(val_dataset_sub, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # --- Initialize Model for Phase 1 ---
    log.info("Initializing MLP model for Phase 1...")
    model_phase1 = PCAPredictor(input_dim=combined_dim).to(device)
    log.info(f"Model architecture:\n{model_phase1}")

    # --- Run Phase 1 Training ---
    _, best_epoch = train_model_pca(
        model_phase1, train_loader_sub, val_loader_sub, device,
        num_epochs=num_epochs, lr=learning_rate
    )

    if best_epoch == 0:
        log.warning("Best epoch was 0, something might be wrong. Defaulting to 1.")
        best_epoch = 1 # Safety check
    
    # Clear memory
    del model_phase1, train_dataset_sub, val_dataset_sub, train_loader_sub, val_loader_sub
    torch.cuda.empty_cache()

    # =========================================================================
    # --- PHASE 2: Train Final Model ---
    # =========================================================================
    log.info("\n" + "="*50)
    log.info(f"--- STARTING PHASE 2: Training Final Model on ALL data ---")
    log.info(f"--- Training for fixed {best_epoch} epochs ---")
    log.info("="*50)

    # --- Create DataLoader for FULL Training Set ---
    train_dataset_full = CombinedFeatureDataset(X_train_val_combined, y_train_val_full)
    train_loader_full = DataLoader(train_dataset_full, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # --- Initialize NEW Model for Phase 2 ---
    log.info("Initializing NEW MLP model for Phase 2...")
    final_model = PCAPredictor(input_dim=combined_dim).to(device)
    
    # --- Run Phase 2 Training ---
    final_model = train_final_model(
        final_model, train_loader_full, device,
        num_epochs=best_epoch, lr=learning_rate
    )

    # =========================================================================
    # --- PREDICTION PHASE ---
    # =========================================================================
    log.info("\n" + "="*50)
    log.info(f"--- STARTING PREDICTION on '{predict_split}' split ---")
    log.info("="*50)

    # --- Create Prediction DataLoader ---
    predict_dataset = CombinedFeatureDataset(X_predict_combined) # No targets
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # --- Make Predictions with Final Model ---
    final_model.eval(); all_predictions = []
    with torch.no_grad():
        for x_batch in predict_loader:
            x_batch = x_batch.to(device)
            predictions_batch = final_model(x_batch)
            all_predictions.extend(predictions_batch.cpu().numpy())

    predicted_scores = np.clip(np.array(all_predictions), 0.0, 1.0) * 100.0
    if len(predicted_scores) != len(predict_signals): log.error(f"Prediction count mismatch!"); return

    # --- Save Results ---
    results_df = pd.DataFrame({'signal': predict_signals, 'predicted_correctness': predicted_scores})
    output_filename = f"{cfg.data.dataset}.{cfg.baseline.system}_pca_final.{predict_split}.predict.csv" # Added _final tag
    output_path = Path(output_filename)
    log.info(f"Saving predictions to {output_path}...")
    try:
        results_df.to_csv(
            output_path, index=False, header=["signal_ID", "intelligibility_score"], mode="w", float_format='%.8f'
        )
        log.info("Predictions saved successfully.")
    except Exception as e: log.error(f"Failed to save predictions: {e}", exc_info=True)


if __name__ == "__main__":
    predict_with_pca()