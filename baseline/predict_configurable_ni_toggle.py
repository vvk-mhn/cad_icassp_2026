# predict_configurable_ni_toggle.py script
from __future__ import annotations

import json
import logging
import sys
import os
import warnings
from pathlib import Path
from typing import Union, List, Dict

import hydra
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
import xgboost as xgb
from omegaconf import DictConfig
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from clarity.utils.file_io import read_jsonl

# --- Suppress Warnings ---
warnings.filterwarnings('ignore', message='.*torch.load.*weights_only.*')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

log = logging.getLogger(__name__)

# =========================================================================
# --- 1. FEATURE CONTROL CENTER ---
# =========================================================================

# --- A. Acoustic Scalars (Non-Intrusive) ---
MFCC_STAT_FEATURES = [f"mfcc_mean_{i}" for i in range(40)] + [f"mfcc_std_{i}" for i in range(40)]
SPECTRAL_FEATURES = [f"spectral_contrast_mean_{i}" for i in range(7)] + ["spectral_centroid_mean", "spectral_centroid_std"]
BASIC_QUALITY_FEATURES = ["stoi_vocals", "pesq_wb_vocals"] 
ZIMTOHRLI_FEATURES = ["zimtohrli"]

# --- B. New Extended Acoustic/Whisper Stats (Non-Intrusive) ---
WHISPER_TECH_FEATURES = [
    "whisper_avg_logprob", "whisper_prob_drop_max", "whisper_no_speech_prob", 
    "whisper_stability_avg_dist", "whisper_stability_std_dist", "whisper_stability_max_dist"
]
PITCH_FEATURES = [
    "pitch_median", "pitch_p95", "pitch_var", "vocal_band_ratio_1k_4k"
]

# --- C. Text Features (Hypothesis Only - Non-Intrusive) ---
HYP_TEXT_FEATURES = [
    "hyp_n_chars", "hyp_avg_word_len", "hyp_syllables_per_word",
    "hyp_pos_nouns", "hyp_pos_verbs", "hyp_pos_adj", "hyp_pos_adv",
    "hyp_content_ratio"
]

# --- D. Text Features (Reference Based - INTRUSIVE) ---
# (Commented out per your non-intrusive setup)
REF_TEXT_FEATURES = [
    "whisper_correct", "semantic_sim", "songprep_wer", "whisper", "lyricwhiz",
    "bertscore_f1", "canary_correct", "whisperx_correct",
    "ref_n_chars", "ref_avg_word_len", "ref_syllables_per_word",
    "ref_pos_nouns", "ref_pos_verbs", "ref_pos_adj", "ref_pos_adv",
    "ref_content_ratio"
]
RATIO_FEATURES = [
    "ratio_n_chars", "ratio_avg_word_len", "ratio_syllables_per_word",
    "ratio_pos_nouns", "ratio_pos_verbs", "ratio_pos_adj", "ratio_pos_adv",
    "ratio_content_ratio"
]

# --- E. Audio Flamingo ---
AUDIO_FLAMINGO_FEATURES = [
    "af_reasoning_score"      
]

# --- F. Advanced Computed Features ---
ADVANCED_INTERACTION_FEATURES = [
    'stoi_pesq_ratio', 
    'vocal_quality_score', 
]

# --- MASTER LIST ---
ACTIVE_SCALAR_FEATURES = (
    []
    + MFCC_STAT_FEATURES
    + SPECTRAL_FEATURES
    + BASIC_QUALITY_FEATURES       
    + ZIMTOHRLI_FEATURES           
    + WHISPER_TECH_FEATURES        
    + PITCH_FEATURES               
    + HYP_TEXT_FEATURES            
    
    # Intrusive blocks commented out
    # + REF_TEXT_FEATURES
    # + RATIO_FEATURES
    + AUDIO_FLAMINGO_FEATURES
    + ADVANCED_INTERACTION_FEATURES
)
ACTIVE_SCALAR_FEATURES = sorted(list(set(ACTIVE_SCALAR_FEATURES)))

def rmse_score(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(np.mean((x - y) ** 2))

def ncc_score(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0 or np.isnan(x).any() or np.isnan(y).any(): return 0.0
    return pearsonr(x, y)[0]

def add_advanced_features(df):
    df_out = df.copy()
    cols = df_out.columns
    def has(c): return c in cols
    
    if has('stoi_vocals_gated') and has('pesq_wb_vocals_gated'):
        df_out['stoi_pesq_ratio'] = df_out['stoi_vocals_gated'] / (df_out['pesq_wb_vocals_gated'] + 1e-6)
    if has('whisper_correct') and has('songprep_wer'):
        df_out['whisper_wer_ratio'] = df_out['whisper_correct'] / (df_out['songprep_wer'] + 1e-6)
    if has('whisper_correct'): df_out['whisper_correct_sq'] = df_out['whisper_correct'] ** 2
    if has('semantic_sim'): df_out['semantic_sim_sq'] = df_out['semantic_sim'] ** 2
    if has('vocal_gate') and has('stoi_vocals') and has('pesq_wb_vocals'):
        df_out['vocal_quality_score'] = (df_out['vocal_gate'] * df_out['stoi_vocals'] * df_out['pesq_wb_vocals'])
    if has('ref_syllables_per_word') and has('ref_avg_word_len'):
        df_out['text_complexity'] = (df_out['ref_syllables_per_word'] * df_out['ref_avg_word_len'])
    if has('whisper_correct') and has('semantic_sim') and has('bertscore_f1'):
        df_out['text_match_quality'] = (df_out['whisper_correct'] * df_out['semantic_sim'] * df_out['bertscore_f1'])
    if has('zimtohrli') and has('whisper_correct'):
        df_out['acoust_text_align'] = (df_out['zimtohrli'] * df_out['whisper_correct'])
    return df_out

# =========================================================================
# --- 2. DATA LOADING ---
# =========================================================================
def load_enhanced_data(cfg: DictConfig, split: str) -> tuple[pd.DataFrame, np.ndarray | None, list]:
    dataroot = Path(cfg.data.cadenza_data_root) / cfg.data.dataset
    log.info(f"Loading data for split: {split}")
    
    meta_file = dataroot / "metadata" / f"{split}_metadata.json"
    with open(meta_file) as f: meta = json.load(f)
    df = pd.DataFrame(meta)
    
    if "correctness" in df.columns:
        if df["correctness"].max() > 1.01: df["correctness"] = df["correctness"] / 100.0
        df["correctness"] = np.clip(df["correctness"], 0.0, 1.0)
    
    feature_files = [
        (f"{cfg.baseline.system}_updated.jsonl", None),
        ("zimtohrli.jsonl", ["signal", "zimtohrli"]),
        ("lyricwhiz.jsonl", ["signal", "lyricwhiz"]),
        ("whisperx_multi.jsonl", None),
        ("songprep_wer.jsonl", ["signal", "songprep_wer"]),
        ("whisper.jsonl", ["signal", "whisper"]),
        ("features_remaining.jsonl", None),
        ("non_intrusive_extended.jsonl", None), 
        ("audio_flamingo_dual.jsonl", None)    
    ]

    for suffix, cols in feature_files:
        fname = f"{cfg.data.dataset}.{split}.{suffix}"
        fpath = Path(fname)
        if not fpath.exists(): fpath = Path(hydra.utils.get_original_cwd()) / fname
        
        if fpath.exists():
            log.info(f"Merging: {fname}")
            sub_df = pd.DataFrame(read_jsonl(str(fpath)))
            if "vocal_gate" in sub_df.columns and "vocal_gate" in df.columns:
                sub_df = sub_df.drop(columns=["vocal_gate"])
            if cols:
                valid_cols = [c for c in cols if c in sub_df.columns]
                sub_df = sub_df[valid_cols]
            if "signal" in sub_df.columns:
                df = pd.merge(df, sub_df, on="signal", how="inner") 

    if 'vocal_snr_ratio' in df.columns:
        alpha = float(cfg.baseline.get('vocal_gate_alpha', 1.0))
        snr0 = float(cfg.baseline.get('vocal_gate_snr0', 0.0))
        df['vocal_gate'] = 1.0 / (1.0 + np.exp(-alpha * (df['vocal_snr_ratio'] - snr0)))
        for col in ['stoi_vocals', 'pesq_wb_vocals']:
            if col in df.columns: df[f'{col}_gated'] = df[col] * df['vocal_gate']
    
    df = add_advanced_features(df)

    tensor_dir = dataroot / cfg.baseline.get("embedding_dir_name", "embeddings_tensors") / split
    if not tensor_dir.exists():
        tensor_dir = Path(hydra.utils.get_original_cwd()) / ".." / "cadenza_data" / "embeddings_tensors" / split
    df['tensor_path'] = df['signal'].apply(lambda x: str(tensor_dir / f"{x}.pt") if tensor_dir.exists() else "")

    # Filter Columns
    # available_features = [f for f in ACTIVE_SCALAR_FEATURES if f in df.columns]
    # missing_features = [f for f in ACTIVE_SCALAR_FEATURES if f not in df.columns]
    
    # if missing_features:
    #     log.warning(f"Requested features missing from dataframe: {len(missing_features)}")
    #     # Fill missing with 0 so the model doesn't crash
    #     for f in missing_features: df[f] = 0.0
    #     available_features.extend(missing_features)

    # # Clean Numerics
    # df[available_features] = df[available_features].replace([np.inf, -np.inf], np.nan)
    # df[available_features] = df[available_features].fillna(df[available_features].mean()).fillna(0.0)
    
    # targets = df["correctness"].values if "correctness" in df.columns else None
    # return df, targets, available_features
        # ===== REPLACE THIS SECTION =====
    # Get ALL numeric columns that actually exist (like v2 does)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'correctness' in numeric_cols:
        numeric_cols.remove('correctness')
    
    # Clean them
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean()).fillna(0.0)
    
    log.info(f"Using {len(numeric_cols)} numeric features from data")
    
    targets = df["correctness"].values if "correctness" in df.columns else None
    return df, targets, numeric_cols  # Return actual columns, not hardcoded list

# =========================================================================
# --- 3. NN COMPONENTS ---
# =========================================================================
class HybridDataset(Dataset):
    def __init__(self, scalar_data, tensor_paths, targets=None):
        self.scalars = torch.tensor(scalar_data, dtype=torch.float32)
        self.tensor_paths = tensor_paths
        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        self.valid_paths = [Path(p).exists() for p in tensor_paths]

    def __len__(self): return len(self.scalars)

    def __getitem__(self, idx):
        path = self.tensor_paths[idx]
        emb_seq = torch.zeros((10, 768)) 
        if self.valid_paths[idx]:
            try:
                data = torch.load(path, map_location="cpu") 
                if 'wavlm' in data: emb_seq = data['wavlm'].float()
                elif 'wav2vec' in data: emb_seq = data['wav2vec'].float()
            except Exception: pass 
        scalar = self.scalars[idx]
        if self.targets is not None: return emb_seq, scalar, self.targets[idx]
        return emb_seq, scalar

def collate_hybrid(batch):
    has_targets = len(batch[0]) == 3
    if has_targets: embs, scalars, targets = zip(*batch)
    else: embs, scalars = zip(*batch)
    embs_padded = pad_sequence(embs, batch_first=True, padding_value=0.0)
    mask = (embs_padded.sum(dim=-1) == 0) 
    scalars_stack = torch.stack(scalars)
    if has_targets:
        targets_stack = torch.stack(targets)
        return embs_padded, mask, scalars_stack, targets_stack
    return embs_padded, mask, scalars_stack

class HybridTransformer(nn.Module):
    def __init__(self, time_dim=768, scalar_dim=100, hidden_dim=256, n_heads=4, n_layers=2, use_scalars=True):
        super().__init__()
        self.use_scalars = use_scalars
        self.input_proj = nn.Linear(time_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4, 
            dropout=0.2, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.attn_pool_dense = nn.Linear(hidden_dim, 1)
        fusion_dim = hidden_dim
        
        if self.use_scalars and scalar_dim > 0:
            self.scalar_mlp = nn.Sequential(
                nn.Linear(scalar_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_dim, hidden_dim)
            )
            fusion_dim = hidden_dim * 2 
        else:
            self.scalar_mlp = None
        
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

    def forward(self, x_emb, mask, x_scalar):
        x = self.input_proj(x_emb) 
        x_trans = self.transformer(x, src_key_padding_mask=mask)
        attn_scores = self.attn_pool_dense(x_trans).masked_fill(mask.unsqueeze(-1), -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1) 
        pooled_emb = torch.sum(x_trans * attn_weights, dim=1) 
        if self.scalar_mlp is not None:
            scalar_feat = self.scalar_mlp(x_scalar) 
            combined = torch.cat([pooled_emb, scalar_feat], dim=1)
        else:
            combined = pooled_emb
        return self.fusion_head(combined).squeeze(1)

# --- Train Functions ---
def train_hybrid_phase1(X_dict, y, val_dict, y_val, params, device, use_scalars=True):
    log.info(f"Training Hybrid Transformer (Phase 1) - Use Scalars: {use_scalars}...")
    train_ds = HybridDataset(X_dict['scalars'], X_dict['paths'], y)
    val_ds = HybridDataset(val_dict['scalars'], val_dict['paths'], y_val)
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_hybrid, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_hybrid, num_workers=4)
    
    model = HybridTransformer(scalar_dim=X_dict['scalars'].shape[1], use_scalars=use_scalars).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_val_loss = float('inf'); best_epoch = 0; patience = 20 if use_scalars else 30; no_improve = 0
    
    for epoch in range(params['epochs']):
        model.train()
        for x_emb, mask, x_scal, y_batch in train_loader:
            x_emb, mask, x_scal, y_batch = x_emb.to(device), mask.to(device), x_scal.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_emb, mask, x_scal), y_batch)
            loss.backward(); optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_emb, mask, x_scal, y_batch in val_loader:
                x_emb, mask, x_scal, y_batch = x_emb.to(device), mask.to(device), x_scal.to(device), y_batch.to(device)
                val_loss += criterion(model(x_emb, mask, x_scal), y_batch).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss; best_epoch = epoch + 1; no_improve = 0
            torch.save(model.state_dict(), "best_hybrid_temp.pth")
        else:
            no_improve += 1
            if no_improve >= patience: break
    
    model.load_state_dict(torch.load("best_hybrid_temp.pth"))
    return model, best_epoch

def train_hybrid_phase2(X_dict, y, params, device, epochs, use_scalars=True):
    ds = HybridDataset(X_dict['scalars'], X_dict['paths'], y)
    loader = DataLoader(ds, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_hybrid, num_workers=4, drop_last=True)
    model = HybridTransformer(scalar_dim=X_dict['scalars'].shape[1], use_scalars=use_scalars).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-4)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for x_emb, mask, x_scal, y_batch in loader:
            x_emb, mask, x_scal, y_batch = x_emb.to(device), mask.to(device), x_scal.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_emb, mask, x_scal), y_batch)
            loss.backward(); optimizer.step()
    return model

def predict_hybrid(model, X_dict, device, params):
    model.eval()
    ds = HybridDataset(X_dict['scalars'], X_dict['paths'], None)
    loader = DataLoader(ds, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_hybrid, num_workers=4)
    all_preds = []
    with torch.no_grad():
        for x_emb, mask, x_scal in loader:
            x_emb, mask, x_scal = x_emb.to(device), mask.to(device), x_scal.to(device)
            all_preds.append(model(x_emb, mask, x_scal).cpu().numpy())
    return np.concatenate(all_preds)

def train_lgbm_phase1(X_train, y_train, X_val, y_val, params):
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='l2', callbacks=[lgb.early_stopping(100, verbose=False)])
    return model, model.best_iteration_ or params.get('n_estimators', 100)
def train_lgbm_phase2(X_full, y_full, params, best_iter):
    p = params.copy(); p['n_estimators'] = best_iter; p.pop('metric', None)
    model = lgb.LGBMRegressor(**p)
    model.fit(X_full, y_full)
    return model

# --- FIX: Added Missing Tree Functions ---
def train_catboost_phase1(X_train, y_train, X_val, y_val, params):
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    return model, model.get_best_iteration() or params.get('iterations', 1000)
def train_catboost_phase2(X_full, y_full, params, best_iter):
    p = params.copy(); p['iterations'] = best_iter; p.pop('eval_metric', None); p['verbose'] = False
    model = CatBoostRegressor(**p)
    model.fit(X_full, y_full, verbose=False)
    return model

def train_xgboost_phase1(X_train, y_train, X_val, y_val, params):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(params, dtrain, num_boost_round=4000, evals=[(dval, 'valid')], early_stopping_rounds=100, verbose_eval=False)
    return model, model.best_iteration
def train_xgboost_phase2(X_full, y_full, params, best_iter):
    dtrain = xgb.DMatrix(X_full, label=y_full)
    return xgb.train(params, dtrain, num_boost_round=best_iter)

# =========================================================================
# --- 5. MAIN EXECUTION FLOW ---
# =========================================================================
@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_experiment(cfg: DictConfig):
    model_type = cfg.baseline.get("model_type", "lgbm")
    use_scalars = cfg.baseline.get("nn_use_scalars", True)
    
    log.info(f"Starting V5 Pipeline | Model: {model_type} | Use Scalars: {use_scalars}")
    
    train_split = cfg.get("train_split", "train")
    predict_split = cfg.get("split", "valid")
    
    df_train, y_train, active_features = load_enhanced_data(cfg, train_split)
    df_pred, _, _ = load_enhanced_data(cfg, predict_split)
    
    common_cols = [c for c in active_features if c in df_pred.columns]
    X_train_scalar = df_train[common_cols].values
    X_pred_scalar = df_pred[common_cols].values
    
    # --- DEBUG: Check Feature Variance ---
    stds = np.std(X_train_scalar, axis=0)
    zero_variance_cols = [common_cols[i] for i in range(len(common_cols)) if stds[i] == 0]
    if len(zero_variance_cols) > 0:
        log.warning(f"Found {len(zero_variance_cols)} features with ZERO variance (likely missing). Examples: {zero_variance_cols[:5]}")
    else:
        log.info("All features have non-zero variance.")

    scaler = StandardScaler()
    if use_scalars and X_train_scalar.shape[1] > 0:
        X_train_scalar = scaler.fit_transform(X_train_scalar)
        X_pred_scalar = scaler.transform(X_pred_scalar)
        if model_type == 'nn_tower':
            pca = PCA(n_components=0.98, random_state=42)
            X_train_scalar = pca.fit_transform(X_train_scalar)
            X_pred_scalar = pca.transform(X_pred_scalar)
    else:
        X_train_scalar = np.zeros((len(df_train), 1))
        X_pred_scalar = np.zeros((len(df_pred), 1))

    train_paths = df_train['tensor_path'].tolist()
    pred_paths = df_pred['tensor_path'].tolist()
    train_signals = df_train['signal'].values
    pred_signals = df_pred['signal'].values
    
    lgbm_params = {
        'objective': 'regression_l2', 'metric': ['l2', 'pearson'], 'n_estimators': 2000,
        'learning_rate': 0.02, 'n_jobs': -1, 'random_state': 42, 'boosting_type': 'gbdt',
        'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'force_col_wise': True,
    }
    catboost_params = {
        'iterations': 4000, 'learning_rate': 0.03, 'depth': 8, 'loss_function': 'RMSE',
        'random_seed': 42, 'thread_count': -1, 'verbose': False, 'early_stopping_rounds': 100,
    }
    xgboost_params = {
        'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'learning_rate': 0.02,
        'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 3,
        'alpha': 0.1, 'lambda': 0.1, 'seed': 42, 'nthread': -1,
    }
    nn_params = {'epochs': 100, 'batch_size': 48, 'lr': 5e-4}
    
    n_folds = cfg.baseline.get("n_folds", 5)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y_train))
    best_iters = []
    
    for fold, (t_idx, v_idx) in enumerate(kf.split(X_train_scalar, y_train)):
        log.info(f"Fold {fold+1}/{n_folds}")
        
        if model_type == 'nn_tower':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            t_dict = {'scalars': X_train_scalar[t_idx], 'paths': [train_paths[i] for i in t_idx]}
            v_dict = {'scalars': X_train_scalar[v_idx], 'paths': [train_paths[i] for i in v_idx]}
            model, best_iter = train_hybrid_phase1(t_dict, y_train[t_idx], v_dict, y_train[v_idx], nn_params, device, use_scalars=use_scalars)
            oof_preds[v_idx] = predict_hybrid(model, v_dict, device, nn_params)
            best_iters.append(best_iter)
            
        else: # Tree Models
            X_tr, X_val = X_train_scalar[t_idx], X_train_scalar[v_idx]
            y_tr, y_val = y_train[t_idx], y_train[v_idx]
            
            if model_type == 'lgbm':
                model, best_iter = train_lgbm_phase1(X_tr, y_tr, X_val, y_val, lgbm_params)
                oof_preds[v_idx] = model.predict(X_val)
            elif model_type == 'catboost':
                model, best_iter = train_catboost_phase1(X_tr, y_tr, X_val, y_val, catboost_params)
                oof_preds[v_idx] = model.predict(X_val)
            elif model_type == 'xgboost':
                model, best_iter = train_xgboost_phase1(X_tr, y_tr, X_val, y_val, xgboost_params)
                oof_preds[v_idx] = model.predict(xgb.DMatrix(X_val))
                
            best_iters.append(best_iter)

    avg_best_iter = int(np.mean(best_iters))
    oof_scaled = np.clip(oof_preds, 0, 1) * 100.0
    y_scaled = y_train * 100.0
    log.info(f"--- CV Results ({model_type}) ---")
    log.info(f"RMSE: {rmse_score(oof_scaled, y_scaled):.6f} | NCC: {ncc_score(oof_scaled, y_scaled):.6f} | Avg Iter: {avg_best_iter}")
    
    Path("exp").mkdir(parents=True, exist_ok=True)
    suffix = "_only_embed" if (model_type == 'nn_tower' and not use_scalars) else ""
    pd.DataFrame({'signal': train_signals, 'oof_pred': oof_scaled, 'true_correctness': y_scaled}).to_csv(f"exp/oof_preds.{model_type}{suffix}.csv", index=False)
    
    log.info(f"--- Phase 2: Final Training ({avg_best_iter} iters) ---")
    if model_type == 'nn_tower':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        full_train_dict = {'scalars': X_train_scalar, 'paths': train_paths}
        full_pred_dict = {'scalars': X_pred_scalar, 'paths': pred_paths}
        final_model = train_hybrid_phase2(full_train_dict, y_train, nn_params, device, avg_best_iter, use_scalars=use_scalars)
        all_preds = predict_hybrid(final_model, full_pred_dict, device, nn_params)
    elif model_type == 'lgbm':
        final_model = train_lgbm_phase2(X_train_scalar, y_train, lgbm_params, avg_best_iter)
        all_preds = final_model.predict(X_pred_scalar)
    elif model_type == 'catboost':
        final_model = train_catboost_phase2(X_train_scalar, y_train, catboost_params, avg_best_iter)
        all_preds = final_model.predict(X_pred_scalar)
    elif model_type == 'xgboost':
        final_model = train_xgboost_phase2(X_train_scalar, y_train, xgboost_params, avg_best_iter)
        all_preds = final_model.predict(xgb.DMatrix(X_pred_scalar))

    final_preds = np.clip(all_preds, 0, 1) * 100.0
    res_df = pd.DataFrame({'signal_ID': pred_signals, 'intelligibility_score': final_preds})
    out_file = f"exp/cadenza_data.multimodal_features_{model_type}{suffix}_final.{predict_split}.predict.csv"
    res_df.to_csv(out_file, index=False)
    log.info(f"Saved predictions to {out_file}")

if __name__ == "__main__":
    run_experiment()