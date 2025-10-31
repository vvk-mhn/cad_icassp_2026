import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# --- 1. Define File Names ---
OOF_LGBM_FILE = "exp/oof_preds.lgbm.csv"
OOF_CATBOOST_FILE = "exp/oof_preds.catboost.csv"
OOF_NN_FILE = "exp/oof_preds.nn_tower.csv"  # --- NEW ---

FINAL_LGBM_PREDS = "exp/cadenza_data.multimodal_features_lgbm_final.valid.predict.csv"
FINAL_CATBOOST_PREDS = "exp/cadenza_data.multimodal_features_catboost_final.valid.predict.csv"
FINAL_NN_PREDS = "exp/cadenza_data.multimodal_features_nn_tower_final.valid.predict.csv" # --- NEW ---

BLENDER_MODEL_FILE = "blender_model_3.joblib"
FINAL_SUBMISSION_FILE = "final_ensemble_submission_3.csv"

print("--- Starting 3-Model Ensemble ---")

# --- 2. Train the Blender Model ---
oof_lgbm_df = pd.read_csv(OOF_LGBM_FILE)
oof_catboost_df = pd.read_csv(OOF_CATBOOST_FILE)
oof_nn_df = pd.read_csv(OOF_NN_FILE) # --- NEW ---

# Merge all three
meta_train_df = pd.merge(oof_lgbm_df, oof_catboost_df, on="signal", suffixes=('_lgbm', '_catboost'))
meta_train_df = pd.merge(meta_train_df, oof_nn_df, on="signal", suffixes=('', '_nn')) # --- NEW ---
# Rename nn columns
meta_train_df = meta_train_df.rename(columns={'oof_pred': 'oof_pred_nn', 'true_correctness': 'true_correctness_nn'})


# Create X (predictions) and y (true labels)
X_meta_train = meta_train_df[['oof_pred_lgbm', 'oof_pred_catboost', 'oof_pred_nn']] # --- NEW ---
y_meta_train = meta_train_df['true_correctness_lgbm'] 

print(f"Training blender model on {len(meta_train_df)} OOF samples...")
blender = Ridge(alpha=1.0) 
blender.fit(X_meta_train, y_meta_train)

joblib.dump(blender, BLENDER_MODEL_FILE)
print(f"Blender model trained and saved to {BLENDER_MODEL_FILE}")
print(f"Learned weights: LGBM={blender.coef_[0]:.4f}, CatBoost={blender.coef_[1]:.4f}, NN={blender.coef_[2]:.4f}") # --- NEW ---


# --- 3. Make Final Predictions ---
final_lgbm_df = pd.read_csv(FINAL_LGBM_PREDS)
final_catboost_df = pd.read_csv(FINAL_CATBOOST_PREDS)
final_nn_df = pd.read_csv(FINAL_NN_PREDS) # --- NEW ---

# Merge all three
meta_test_df = pd.merge(final_lgbm_df, final_catboost_df, on="signal_ID", suffixes=('_lgbm', '_catboost'))
meta_test_df = pd.merge(meta_test_df, final_nn_df, on="signal_ID", suffixes=('', '_nn')) # --- NEW ---
meta_test_df = meta_test_df.rename(columns={'intelligibility_score': 'intelligibility_score_nn'})


# Get the predictions from the three models
X_meta_test_raw = meta_test_df[['intelligibility_score_lgbm', 'intelligibility_score_catboost', 'intelligibility_score_nn']] # --- NEW ---

# Rename columns to match .fit()
X_meta_test = X_meta_test_raw.rename(columns={
    'intelligibility_score_lgbm': 'oof_pred_lgbm',
    'intelligibility_score_catboost': 'oof_pred_catboost',
    'intelligibility_score_nn': 'oof_pred_nn' # --- NEW ---
})

print(f"Making final ensemble predictions on {len(X_meta_test)} samples...")
final_predictions = blender.predict(X_meta_test)

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