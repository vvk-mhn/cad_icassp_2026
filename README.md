# ICASSP 2026 Cadenza Challenge (CLIP1): Predicting Lyric Intelligibility

Modified baseline by Team T071 (Aalto University)

For CLIP1 details: https://cadenzachallenge.org/

This repository contains:
- A modified baseline pipeline for feature extraction, prediction, and ensembling.
- Scripts to evaluate predictions against the challenge metrics.

See also: `baseline/README.md` for additional baseline details.

---

## Quick Start

### 1) Prerequisites
- Linux/macOS recommended
- Python environment as specified by the Clarity Challenge repository
- (Optional) GPU for Whisper-based features

### 2) Clone and set up the Clarity environment

Follow the environment setup instructions from the Clarity Challenge repository:
```bash
git clone https://github.com/claritychallenge/clarity
cd clarity
# Follow the Clarity repo instructions to create/activate the environment
# Example:
# conda env create -f environment.yml
# conda activate clarity
```

### 3) Clone this repository into `recipes/` inside `clarity/`
```bash
cd clarity/recipes
git clone <THIS_REPO_URL> cad_icassp_2026
```

Your structure should look like:
```
clarity/
└── recipes/
    └── cad_icassp_2026/
        ├── baseline/
        ├── ... (code)
        └── README.md
```

### 4) Install this repo's Python requirements

After activating the conda environment and cloning this repository, install the requirements from within the `cad_icassp_2026` directory:
```bash
cd cad_icassp_2026
pip install -r requirements.txt
```

---

## Running the Pipeline

Run from the root of the `clarity` repository (with the environment activated). The pipeline has three stages: feature computation, prediction, and ensembling.

We recommend computing features for all splits (`train`, `valid`, `eval`) before prediction.

### A) Compute Features

Run the following scripts for each split. Replace `split=` with `train`, `valid`, or `eval`. You can also loop over splits.
```bash
# Multimodal feature set (core)
python -m recipes.cad_icassp_2026.baseline.compute_all_features split=train baseline=multimodal_features
python -m recipes.cad_icassp_2026.baseline.compute_all_features split=valid baseline=multimodal_features
python -m recipes.cad_icassp_2026.baseline.compute_all_features split=eval  baseline=multimodal_features

# Advanced/last-final features
python -m recipes.cad_icassp_2026.baseline.compute_last_final_features split=train baseline=multimodal_advanced
python -m recipes.cad_icassp_2026.baseline.compute_last_final_features split=valid baseline=multimodal_advanced
python -m recipes.cad_icassp_2026.baseline.compute_last_final_features split=eval  baseline=multimodal_advanced

# Remaining/auxiliary features
python -m recipes.cad_icassp_2026.baseline.compute_remaining_features split=train baseline=features_remaining
python -m recipes.cad_icassp_2026.baseline.compute_remaining_features split=valid baseline=features_remaining
python -m recipes.cad_icassp_2026.baseline.compute_remaining_features split=eval  baseline=features_remaining

# Whisper features (may require GPU; downloads models on first run)
python -m recipes.cad_icassp_2026.baseline.compute_whisper split=train baseline=whisper
python -m recipes.cad_icassp_2026.baseline.compute_whisper split=valid baseline=whisper
python -m recipes.cad_icassp_2026.baseline.compute_whisper split=eval  baseline=whisper

# Combined WhisperX multi-channel features
python -m recipes.cad_icassp_2026.baseline.combined_whisper split=train baseline=whisperx_multi
python -m recipes.cad_icassp_2026.baseline.combined_whisper split=valid baseline=whisperx_multi
python -m recipes.cad_icassp_2026.baseline.combined_whisper split=eval  baseline=whisperx_multi

# Zimtohrli: signal-based features only
python -m recipes.cad_icassp_2026.baseline.compute_zimtohrli_only_signals split=train baseline=zimtorhli
python -m recipes.cad_icassp_2026.baseline.compute_zimtohrli_only_signals split=valid baseline=zimtorhli
python -m recipes.cad_icassp_2026.baseline.compute_zimtohrli_only_signals split=eval  baseline=zimtorhli

# STOI features
python -m recipes.cad_icassp_2026.baseline.compute_stoi split=train baseline=stoi
python -m recipes.cad_icassp_2026.baseline.compute_stoi split=valid baseline=stoi
python -m recipes.cad_icassp_2026.baseline.compute_stoi split=eval  baseline=stoi

# LyricWhiz features
python -m recipes.cad_icassp_2026.baseline.compute_lyricwhiz split=train baseline=lyricwhiz
python -m recipes.cad_icassp_2026.baseline.compute_lyricwhiz split=valid baseline=lyricwhiz
python -m recipes.cad_icassp_2026.baseline.compute_lyricwhiz split=eval  baseline=lyricwhiz
```

Tip: To loop in bash/zsh:
```bash
for SPLIT in train valid eval; do
  python -m recipes.cad_icassp_2026.baseline.compute_all_features            split=$SPLIT baseline=multimodal_features
  python -m recipes.cad_icassp_2026.baseline.compute_last_final_features     split=$SPLIT baseline=multimodal_advanced
  python -m recipes.cad_icassp_2026.baseline.compute_remaining_features      split=$SPLIT baseline=features_remaining
  python -m recipes.cad_icassp_2026.baseline.compute_whisper                 split=$SPLIT baseline=whisper
  python -m recipes.cad_icassp_2026.baseline.combined_whisper                split=$SPLIT baseline=whisperx_multi
  python -m recipes.cad_icassp_2026.baseline.compute_zimtohrli_only_signals  split=$SPLIT baseline=zimtorhli
  python -m recipes.cad_icassp_2026.baseline.compute_stoi                    split=$SPLIT baseline=stoi
  python -m recipes.cad_icassp_2026.baseline.compute_lyricwhiz               split=$SPLIT baseline=lyricwhiz
done
```

---

### B) Train and Predict

Run the configurable intrusive predictors on `valid` and/or `eval`. You can use multiple model backends and ensemble later.
```bash
# Gradient boosting variants
python -m recipes.cad_icassp_2026.baseline.predict_configurable_intrusive split=valid baseline=multimodal_features baseline.model_type=lgbm
python -m recipes.cad_icassp_2026.baseline.predict_configurable_intrusive split=eval  baseline=multimodal_features baseline.model_type=lgbm

python -m recipes.cad_icassp_2026.baseline.predict_configurable_intrusive split=valid baseline=multimodal_features baseline.model_type=catboost
python -m recipes.cad_icassp_2026.baseline.predict_configurable_intrusive split=eval  baseline=multimodal_features baseline.model_type=catboost

python -m recipes.cad_icassp_2026.baseline.predict_configurable_intrusive split=valid baseline=multimodal_features baseline.model_type=xgboost
python -m recipes.cad_icassp_2026.baseline.predict_configurable_intrusive split=eval  baseline=multimodal_features baseline.model_type=xgboost

# Neural tower variant (optionally include scalar features)
python -m recipes.cad_icassp_2026.baseline.predict_configurable_intrusive split=valid baseline=multimodal_features baseline.model_type=nn_tower baseline.nn_use_scalars=true
python -m recipes.cad_icassp_2026.baseline.predict_configurable_intrusive split=eval  baseline=multimodal_features baseline.model_type=nn_tower baseline.nn_use_scalars=true
```

Outputs are CSV prediction files stored under the baseline's output directory (see `baseline/README.md` for exact paths).

---

### C) Ensemble the Predictions

Blend predictions from multiple models:
```bash
python -m recipes.cad_icassp_2026.baseline.ensemble_blender_v2 --split=valid
python -m recipes.cad_icassp_2026.baseline.ensemble_blender_v2 --split=eval
```

This produces ensembled CSVs for the specified splits.

---

## Evaluation

Use the evaluator in the baseline directory to score CSVs:
```bash
python baseline/evaluate.py --split valid --pred_csv <path_to_predictions.csv>
python baseline/evaluate.py --split eval  --pred_csv <path_to_predictions.csv>
```

See `baseline/README.md` for CSV format and metrics.

---

## Notes and Tips

- **Data and paths:** Ensure your dataset path and config files match the Clarity repository's expected structure.
- **First runs** may download external models (e.g., Whisper/WhisperX). Ensure internet access or cache ahead of time.
- **GPU/CPU:** Whisper-related features run faster on GPU but can run on CPU with increased time.
- **Reproducibility:** Set RNG seeds in configs if deterministic behavior is required.
- **Troubleshooting:**
  - ImportErrors: Confirm you're running from the `clarity` root with the environment activated.
  - Missing models/assets: Re-run the feature scripts after ensuring dependencies are installed.
  - File not found: Verify you cloned this repo into `clarity/recipes/cad_icassp_2026/`.

---

## License and Acknowledgements

- Baseline adapted by Team T071 (Aalto University)
- Built on Clarity Challenge infrastructure and datasets
- See the Clarity repository and this project for respective licenses and citations

For questions or issues, please open an issue in this repository.
