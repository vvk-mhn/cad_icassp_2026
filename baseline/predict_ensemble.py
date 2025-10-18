# predict_ensemble.py

"""Make intelligibility predictions from an ensemble of scores using XGBoost."""

from __future__ import annotations

import logging

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
import xgboost as xgb
# from sklearn.preprocessing import MinMaxScaler # Optional: for feature normalization

# IMPORTANT: Import the new function
from recipes.cad_icassp_2026.baseline.shared_predict_utils import (
    load_ensemble_features, # This is the new function
)

log = logging.getLogger(__name__)

# Define the features to be used in the ensemble
# These should match the 'system' field in your YAMLs and the score key in your JSONL files
# You MUST ensure your compute_whisper.py output is keyed by "whisper"
ENSEMBLE_FEATURES = ["songprep_wer", "whisper"] 


# pylint: disable = no-value-for-parameter
@hydra.main(config_path="configs", config_name="config", version_base=None)
def predict_ensemble(cfg: DictConfig):
    """Predict intelligibility using an XGBoost ensemble model.

    The model maps multiple baseline scores onto the sentence correctness values.
    """

    log.info(f"Starting XGBoost Ensemble Prediction using features: {ENSEMBLE_FEATURES}")

    # 1. Load the features and ground truth for training and validation
    log.info("Loading ensemble features...")
    # The 'systems' list is passed to load_ensemble_features
    records_train_df = load_ensemble_features(cfg, "train", ENSEMBLE_FEATURES)
    records_valid_df = load_ensemble_features(cfg, "valid", ENSEMBLE_FEATURES)

    # 2. Prepare data for XGBoost
    X_train = records_train_df[ENSEMBLE_FEATURES]
    y_train = records_train_df["correctness"] # Ground truth (0.0 to 1.0)
    X_valid = records_valid_df[ENSEMBLE_FEATURES]
    
    # Optional: Feature Scaling
    # scaler = MinMaxScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_valid_scaled = scaler.transform(X_valid)

    # 3. Compute the XGBoost fit (The "Magic" ✨)
    log.info("Fitting XGBoost Regressor model...")
    
    # XGBoost Parameters: Tweak these for best results!
    # For a small dataset, keep n_estimators low and max_depth small to prevent overfitting.
    model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=150,               # Number of boosting rounds
        learning_rate=0.08,             # Step size shrinkage
        max_depth=4,                    # Maximum depth of a tree
        min_child_weight=1,             # Minimum sum of instance weight needed in a child
        subsample=0.8,                  # Subsample ratio of the training instances
        colsample_bytree=0.8,           # Subsample ratio of columns when constructing each tree
        random_state=42,                # For reproducibility
        n_jobs=-1                       # Use all cores
    )
    
    model.fit(X_train, y_train)

    # 4. Make predictions for the validation data
    log.info("Starting ensemble predictions...")
    
    # Use the fitted model to predict correctness (0.0 to 1.0)
    predictions = model.predict(X_valid)

    # 5. Post-process and save results
    
    # Convert prediction to percentage (0.0 to 100.0) and clip to enforce bounds
    records_valid_df["predicted_correctness"] = np.clip(predictions, 0.0, 1.0) * 100.0
    
    # Ensure all predictions are float/int type before saving
    records_valid_df["predicted_correctness"] = records_valid_df["predicted_correctness"].astype(float)
    
    # Save results to CSV file in the submission format
    output_file = f"{cfg.data.dataset}.ensemble_xgboost.valid.predict.csv"
    records_valid_df[["signal", "predicted_correctness"]].to_csv(
        output_file,
        index=False,
        header=["signal_ID", "intelligibility_score"],
        mode="w",
    )
    log.info(f"Ensemble predictions saved to {output_file}")


if __name__ == "__main__":
    predict_ensemble()