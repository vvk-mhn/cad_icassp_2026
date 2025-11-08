# ensemble_blender_v2.py

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize # --- NEW (Suggestion #1) ---
# from sklearn.linear_model import Ridge # --- REMOVED ---

print("--- Starting 4-Model Ensemble with Constrained Optimization ---")

# --- 1. Define File Names (Added XGB) ---
OOF_LGBM_FILE = "exp/oof_preds.lgbm.csv"
OOF_CATBOOST_FILE = "exp/oof_preds.catboost.csv"
OOF_NN_FILE = "exp/oof_preds.nn_tower.csv"
OOF_XGB_FILE = "exp/oof_preds.xgboost.csv" # --- NEW (Suggestion #3) ---

FINAL_LGBM_PREDS = "exp/cadenza_data.multimodal_features_lgbm_final.valid.predict.csv"
FINAL_CATBOOST_PREDS = "exp/cadenza_data.multimodal_features_catboost_final.valid.predict.csv"
FINAL_NN_PREDS = "exp/cadenza_data.multimodal_features_nn_tower_final.valid.predict.csv"
FINAL_XGB_PREDS = "exp/cadenza_data.multimodal_features_xgboost_final.valid.predict.csv" # --- NEW (Suggestion #3) ---

# --- No model file needed, but we save weights ---
WEIGHTS_FILE = "blender_weights.npy"
FINAL_SUBMISSION_FILE = "final_ensemble_submission_4_optim.csv"

# --- NEW (Suggestion #1): Optimization Function ---
def optimize_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Find optimal non-negative weights that sum to 1
    to minimize RMSE.
    """
    n_models = X.shape[1]
    
    # Loss function (RMSE)
    def loss(weights):
        y_pred = X @ weights
        return np.sqrt(np.mean((y_pred - y) ** 2))

    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    # Bounds: weights are between 0 and 1
    bounds = [(0, 1)] * n_models
    
    # Initial guess: equal weights
    x0 = np.ones(n_models) / n_models
    
    print("Finding optimal weights using 'SLSQP'...")
    result = minimize(loss, x0, method='SLSQP', 
                      bounds=bounds, constraints=constraints)
    
    if not result.success:
        print(f"WARNING: Optimization failed: {result.message}")
    
    return result.x


# --- 2. Train the Blender (Find Weights) ---
try:
    oof_lgbm_df = pd.read_csv(OOF_LGBM_FILE)
    oof_catboost_df = pd.read_csv(OOF_CATBOOST_FILE)
    oof_nn_df = pd.read_csv(OOF_NN_FILE)
    oof_xgb_df = pd.read_csv(OOF_XGB_FILE) # --- NEW ---

    # Merge all four
    meta_train_df = pd.merge(oof_lgbm_df, oof_catboost_df, on="signal", suffixes=('_lgbm', '_catboost'))
    meta_train_df = pd.merge(meta_train_df, oof_nn_df, on="signal", suffixes=('', '_nn'))
    meta_train_df = pd.merge(meta_train_df, oof_xgb_df, on="signal", suffixes=('', '_xgb')) # --- NEW ---
    
    # Rename nn/xgb columns
    meta_train_df = meta_train_df.rename(columns={
        'oof_pred': 'oof_pred_nn', 'true_correctness': 'true_correctness_nn',
        'oof_pred_xgb': 'oof_pred_xgb', 'true_correctness_xgb': 'true_correctness_xgb'
    })

    # Create X (predictions) and y (true labels)
    X_meta_train = meta_train_df[['oof_pred_lgbm', 'oof_pred_catboost', 'oof_pred_nn', 'oof_pred_xgb']] # --- NEW ---
    y_meta_train = meta_train_df['true_correctness_lgbm'] # Use any 'true' col, they're all the same

    print(f"Optimizing weights on {len(meta_train_df)} full OOF samples...")

    # --- NEW (Suggestion #1): Find weights instead of fitting Ridge ---
    optimal_weights = optimize_weights(X_meta_train.values, y_meta_train.values)
    
    np.save(WEIGHTS_FILE, optimal_weights)
    print(f"Optimal weights saved to {WEIGHTS_FILE}")
    print(f"Learned weights:")
    print(f"  LGBM:     {optimal_weights[0]:.6f}")
    print(f"  CatBoost: {optimal_weights[1]:.6f}")
    print(f"  NN Tower: {optimal_weights[2]:.6f}")
    print(f"  XGBoost:  {optimal_weights[3]:.6f}") # --- NEW ---
    print(f"  SUM:      {np.sum(optimal_weights):.6f}")

except FileNotFoundError as e:
    print(f"Error: Missing OOF file: {e}. Did you run `predict_configurable_v2.py` for all models?")
    exit()
except Exception as e:
    print(f"An error occurred during weight optimization: {e}")
    exit()

# --- 3. Make Final Predictions ---
try:
    final_lgbm_df = pd.read_csv(FINAL_LGBM_PREDS)
    final_catboost_df = pd.read_csv(FINAL_CATBOOST_PREDS)
    final_nn_df = pd.read_csv(FINAL_NN_PREDS)
    final_xgb_df = pd.read_csv(FINAL_XGB_PREDS) # --- NEW ---

    # Merge all four
    meta_test_df = pd.merge(final_lgbm_df, final_catboost_df, on="signal_ID", suffixes=('_lgbm', '_catboost'))
    meta_test_df = pd.merge(meta_test_df, final_nn_df, on="signal_ID", suffixes=('', '_nn'))
    meta_test_df = pd.merge(meta_test_df, final_xgb_df, on="signal_ID", suffixes=('', '_xgb')) # --- NEW ---

    meta_test_df = meta_test_df.rename(columns={
        'intelligibility_score': 'intelligibility_score_nn',
        'intelligibility_score_xgb': 'intelligibility_score_xgb'
    })

    # Get the predictions from the four models
    X_meta_test = meta_test_df[[
        'intelligibility_score_lgbm', 
        'intelligibility_score_catboost', 
        'intelligibility_score_nn',
        'intelligibility_score_xgb' # --- NEW ---
    ]]

    print(f"Making final ensemble predictions on {len(X_meta_test)} samples...")
    
    # --- NEW (Suggestion #1): Predict using weighted average ---
    final_predictions = X_meta_test.values @ optimal_weights

    # CRITICAL: Clip predictions to be in the valid [0, 100] range
    final_predictions_clipped = np.clip(final_predictions, 0.0, 100.0)

    # --- 4. Save Final Submission File ---
    submission_df = pd.DataFrame({
        'signal_ID': meta_test_df['signal_ID'],
        'intelligibility_score': final_predictions_clipped
    })

    submission_df.to_csv(FINAL_SUBMISSION_FILE, index=False, float_format='%.8f')
    print("\n--- SUCCESS! ---")
    print(f"Final ensemble submission file created: {FINAL_SUBMISSION_FILE}")
    print("You can now upload this file to EvalAI.")

except FileNotFoundError as e:
    print(f"Error: Missing final prediction file: {e}. Did you run `predict_configurable_v2.py` for all models?")
except Exception as e:
    print(f"An error occurred during final prediction: {e}")