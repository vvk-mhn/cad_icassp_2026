# recipes/cad_icassp_2026/baseline/ensemble_blender_v2.py

import argparse
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Tuple

print("--- Starting 4-Model Ensemble with Constrained Optimization ---")

# Default file locations (OOF are split-agnostic; final preds use {split})
OOF_FILES = {
    "lgbm": "exp/oof_preds.lgbm.csv",
    "catboost": "exp/oof_preds.catboost.csv",
    "nn": "exp/oof_preds.nn_tower.csv",
    "xgb": "exp/oof_preds.xgboost.csv",
}

FINAL_FILES_TPL = {
    "lgbm": "exp/cadenza_data.multimodal_features_lgbm_final.{split}.predict.csv",
    "catboost": "exp/cadenza_data.multimodal_features_catboost_final.{split}.predict.csv",
    "nn": "exp/cadenza_data.multimodal_features_nn_tower_final.{split}.predict.csv",
    "xgb": "exp/cadenza_data.multimodal_features_xgboost_final.{split}.predict.csv",
}


def optimize_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Find optimal non-negative weights that sum to 1 to minimize RMSE.
    """
    n_models = X.shape[1]

    def loss(weights):
        y_pred = X @ weights
        return np.sqrt(np.mean((y_pred - y) ** 2))

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * n_models
    x0 = np.ones(n_models) / n_models

    print("Finding optimal weights using 'SLSQP'...")
    result = minimize(loss, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        print(f"WARNING: Optimization failed: {result.message}")

    return result.x


def read_oof(file: str, model_key: str) -> pd.DataFrame:
    """
    Read an OOF CSV and rename columns to be model-specific.
    Expected columns in each OOF file: ['signal', 'oof_pred', 'true_correctness'].
    """
    df = pd.read_csv(file)
    # Defensive checks
    required_cols = {'signal', 'oof_pred', 'true_correctness'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"OOF file {file} missing required columns {required_cols}. Found: {df.columns.tolist()}")

    df = df.rename(columns={
        'oof_pred': f'oof_pred_{model_key}',
        'true_correctness': f'true_correctness_{model_key}',
    })
    return df[['signal', f'oof_pred_{model_key}', f'true_correctness_{model_key}']]


def read_final(file: str, model_key: str) -> pd.DataFrame:
    """
    Read a final prediction CSV and rename columns to be model-specific.
    Expected columns: ['signal_ID', 'intelligibility_score'].
    """
    df = pd.read_csv(file)
    required_cols = {'signal_ID', 'intelligibility_score'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Final file {file} missing required columns {required_cols}. Found: {df.columns.tolist()}")

    df = df.rename(columns={'intelligibility_score': f'intelligibility_score_{model_key}'})
    return df[['signal_ID', f'intelligibility_score_{model_key}']]


def load_meta_train(oof_files: Dict[str, str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Merge OOF predictions from all models into a single meta-training DataFrame.
    Returns:
      - meta_train_df
      - X_meta_train (predictions matrix)
      - y_meta_train (ground truth array)
    """
    dfs = []
    for key, file in oof_files.items():
        try:
            dfs.append(read_oof(file, key))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing OOF file for {key}: {file}") from e

    # Inner merge on 'signal'
    meta_train_df = dfs[0]
    for df in dfs[1:]:
        meta_train_df = pd.merge(meta_train_df, df, on='signal', how='inner')

    # Build X from model-specific preds (consistent order)
    model_order = ['lgbm', 'catboost', 'nn', 'xgb']
    pred_cols = [f'oof_pred_{m}' for m in model_order]
    true_cols = [f'true_correctness_{m}' for m in model_order]

    # Sanity: all true_correctness columns should be equal; use the first
    y_cols_equal = meta_train_df[true_cols].nunique(axis=1).max() == 1
    if not y_cols_equal:
        print("WARNING: 'true_correctness' columns differ across OOF files. Using the first one.")

    X_meta_train = meta_train_df[pred_cols].values
    y_meta_train = meta_train_df[true_cols[0]].values

    print(f"Optimizing weights on {len(meta_train_df)} full OOF samples...")
    return meta_train_df, X_meta_train, y_meta_train


def load_meta_test(final_tpl: Dict[str, str], split: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Merge final predictions from all models for the given split into a single meta-test DataFrame.
    Returns:
      - meta_test_df
      - X_meta_test (predictions matrix)
    """
    dfs = []
    for key, tpl in final_tpl.items():
        path = tpl.format(split=split)
        try:
            dfs.append(read_final(path, key))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing final prediction file for {key} and split '{split}': {path}") from e

    meta_test_df = dfs[0]
    for df in dfs[1:]:
        meta_test_df = pd.merge(meta_test_df, df, on='signal_ID', how='inner')

    model_order = ['lgbm', 'catboost', 'nn', 'xgb']
    pred_cols = [f'intelligibility_score_{m}' for m in model_order]
    X_meta_test = meta_test_df[pred_cols].values

    print(f"Making final ensemble predictions on {len(X_meta_test)} samples...")
    return meta_test_df, X_meta_test


def run(split: str,
        weights_out: str,
        submission_out: str,
        oof_files: Dict[str, str],
        final_tpl: Dict[str, str]) -> None:

    # 1) Train blender (find weights)
    try:
        meta_train_df, X_meta_train, y_meta_train = load_meta_train(oof_files)
        optimal_weights = optimize_weights(X_meta_train, y_meta_train)

        np.save(weights_out, optimal_weights)
        print(f"Optimal weights saved to {weights_out}")
        print("Learned weights:")
        print(f"  LGBM:     {optimal_weights[0]:.6f}")
        print(f"  CatBoost: {optimal_weights[1]:.6f}")
        print(f"  NN Tower: {optimal_weights[2]:.6f}")
        print(f"  XGBoost:  {optimal_weights[3]:.6f}")
        print(f"  SUM:      {np.sum(optimal_weights):.6f}")
    except FileNotFoundError as e:
        print(f"Error: {e}. Did you run `predict_configurable_v2.py` for all models?")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during weight optimization: {e}")
        sys.exit(1)

    # 2) Make final predictions for the requested split
    try:
        meta_test_df, X_meta_test = load_meta_test(final_tpl, split)
        final_predictions = X_meta_test @ optimal_weights
        final_predictions_clipped = np.clip(final_predictions, 0.0, 100.0)

        submission_df = pd.DataFrame({
            'signal_ID': meta_test_df['signal_ID'],
            'intelligibility_score': final_predictions_clipped
        })
        submission_df.to_csv(submission_out, index=False, float_format='%.8f')

        print("\n--- SUCCESS! ---")
        print(f"Final ensemble submission file created: {submission_out}")
        print("You can now upload this file to EvalAI.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Did you run `predict_configurable_v2.py` for all models and split '{split}'?")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during final prediction: {e}")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="4-Model Ensemble Blender with constrained optimization."
    )
    parser.add_argument(
        "--split", choices=["valid", "eval"], default="valid",
        help="Which split to ensemble for final predictions."
    )
    parser.add_argument(
        "--weights-out", default="blender_weights.npy",
        help="Where to save the optimized weights."
    )
    parser.add_argument(
        "--submission-out",
        help="Output submission CSV. Default: optim_4_{split}.csv"
    )
    # Optional: allow overriding default file templates/paths
    parser.add_argument("--oof-lgbm", default=OOF_FILES["lgbm"])
    parser.add_argument("--oof-catboost", default=OOF_FILES["catboost"])
    parser.add_argument("--oof-nn", default=OOF_FILES["nn"])
    parser.add_argument("--oof-xgb", default=OOF_FILES["xgb"])

    parser.add_argument("--final-lgbm-tpl", default=FINAL_FILES_TPL["lgbm"])
    parser.add_argument("--final-catboost-tpl", default=FINAL_FILES_TPL["catboost"])
    parser.add_argument("--final-nn-tpl", default=FINAL_FILES_TPL["nn"])
    parser.add_argument("--final-xgb-tpl", default=FINAL_FILES_TPL["xgb"])

    args = parser.parse_args()

    # Derive default submission filename if not provided
    if args.submission_out is None:
        args.submission_out = f"optim_4_{args.split}.csv"

    return args


def main():
    args = parse_args()

    oof_files = {
        "lgbm": args.oof_lgbm,
        "catboost": args.oof_catboost,
        "nn": args.oof_nn,
        "xgb": args.oof_xgb,
    }
    final_tpl = {
        "lgbm": args.final_lgbm_tpl,
        "catboost": args.final_catboost_tpl,
        "nn": args.final_nn_tpl,
        "xgb": args.final_xgb_tpl,
    }

    run(
        split=args.split,
        weights_out=args.weights_out,
        submission_out=args.submission_out,
        oof_files=oof_files,
        final_tpl=final_tpl,
    )


if __name__ == "__main__":
    main()
