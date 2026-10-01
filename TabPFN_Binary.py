# tabpfn_binary_5fold_cv.py
# -----------------------------------------------------
# TabPFN classifier: 5-fold CV for binary toxins
# ** Google Colab version **
#
# Setup steps before running:
#   1. Runtime -> Change runtime type -> GPU
#   2. Run the mount cell to connect Google Drive
#   3. Upload x.csv, y_bin.csv, fold_assignments.csv to /content/
#   4. Set DRIVE_OUT to your preferred Google Drive output folder
#
# Outputs (saved to Google Drive):
#   tabpfn_bin_cv_all_folds_metrics.csv
#   tabpfn_bin_cv_per_variable_metrics.csv
#   tabpfn_bin_cv_overall_metrics.csv
# -----------------------------------------------------

import os, random
import numpy as np
import pandas as pd
from collections import Counter
from tabpfn import TabPFNClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import torch

# ── Config ─────────────────────────────────────────────────────────────
SEED      = 1701
N_FOLDS   = 5
BASE_DIR  = "/content"
OUT_DIR   = "/content"          # results saved here — download before session ends
os.makedirs(OUT_DIR, exist_ok=True)

TOXINS_BIN = [
    "DON_bin", "D3G_bin", "Nivalenol_bin", "3-AC-DON_bin", "15-AC-DON_bin",
    "T-2_toxin_bin", "HT-2_toxin_bin", "T2G_bin", "Neos_bin", "ENN_A1_bin",
    "ENN_A_bin", "ENN_B_bin", "ENN_B1_bin", "BEAU_bin", "ZEN_bin",
    "Apicidin_bin", "STER_bin", "DAS_bin", "Quest_bin", "AOH_bin",
    "AME_bin", "MON_bin", "Ergocristine_bin", "EGT_bin"
]

# ── Reproducibility ────────────────────────────────────────────────────
def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(SEED)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEV}")

# ── Helpers ────────────────────────────────────────────────────────────
def safe_to_numeric_01(arr):
    s = pd.Series(arr).astype(str).str.strip().str.lower()
    mapping = {"0":0,"1":1,"false":0,"true":1,"neg":0,"pos":1,"no":0,"yes":1}
    s = s.map(lambda x: mapping.get(x, x))
    s = pd.to_numeric(s, errors="coerce")
    return s.values.astype(float)

# ── Load full data ─────────────────────────────────────────────────────
print("Loading data...")
X_df    = pd.read_csv(os.path.join(BASE_DIR, "x.csv"))
Yb_df   = pd.read_csv(os.path.join(BASE_DIR, "y_bin.csv"))
fold_df = pd.read_csv(os.path.join(BASE_DIR, "fold_assignments.csv"))

X_np  = X_df.select_dtypes(include=[np.number]).values.astype(np.float32)
Yb_np = Yb_df.reindex(columns=TOXINS_BIN).values.astype(float)

assert len(fold_df) == len(X_np), \
    f"fold_assignments.csv has {len(fold_df)} rows but data has {len(X_np)}"

fold_assignments = fold_df["fold"].values
print(f"Data loaded: {X_np.shape[0]} rows, {X_np.shape[1]} features, {len(TOXINS_BIN)} toxins")

if np.isnan(X_np).any():
    raise ValueError("Predictor matrix contains NaNs; TabPFN requires finite inputs.")

# ── 5-Fold CV loop ─────────────────────────────────────────────────────
all_fold_metrics = []

for fold_k in range(1, N_FOLDS + 1):
    print(f"\n{'='*60}")
    print(f"  Fold {fold_k} / {N_FOLDS}")
    print(f"{'='*60}")

    set_seeds(SEED + fold_k)

    test_mask  = (fold_assignments == fold_k)
    train_mask = ~test_mask

    X_train = X_np[train_mask]
    X_test  = X_np[test_mask]
    Yb_train = Yb_np[train_mask]
    Yb_test  = Yb_np[test_mask]

    print(f"  Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    for j, tox in enumerate(TOXINS_BIN):
        base_name = tox.replace("_bin", "")
        y_tr = safe_to_numeric_01(Yb_train[:, j])
        y_te = safe_to_numeric_01(Yb_test[:, j])

        tr_mask = ~np.isnan(y_tr)
        te_mask = ~np.isnan(y_te)

        y_tr_f = y_tr[tr_mask].astype(int)
        y_te_f = y_te[te_mask].astype(int)
        X_tr_f = X_train[tr_mask]
        X_te_f = X_test[te_mask]

        counts_tr = Counter(y_tr_f.tolist())
        if len(counts_tr) < 2 or min(counts_tr.values()) < 2:
            print(f"  Skipping {tox}: training class issue {dict(counts_tr)}")
            all_fold_metrics.append(dict(
                fold=fold_k, Variable=base_name,
                RMSE=np.nan, R2=np.nan, F1_or_Acc=np.nan, AUC=np.nan
            ))
            continue

        clf = TabPFNClassifier(device=DEV)
        clf.fit(X_tr_f, y_tr_f)
        proba = clf.predict_proba(X_te_f)
        pred_prob = proba[:, 1] if proba.shape[1] > 1 else np.full(len(y_te_f), np.nan)

        counts_te = Counter(y_te_f.tolist())
        if len(counts_te) == 2:
            y_pred    = (pred_prob >= 0.5).astype(int)
            f1_or_acc = float(f1_score(y_te_f, y_pred))
            auc       = float(roc_auc_score(y_te_f, pred_prob))
        else:
            y_pred    = (pred_prob >= 0.5).astype(int)
            f1_or_acc = float(accuracy_score(y_te_f, y_pred))
            auc       = np.nan

        all_fold_metrics.append(dict(
            fold=fold_k, Variable=base_name,
            RMSE=np.nan, R2=np.nan, F1_or_Acc=f1_or_acc, AUC=auc
        ))
        print(f"  {tox}: F1={f1_or_acc:.4f}  AUC={auc:.4f}" if not np.isnan(auc)
              else f"  {tox}: Acc={f1_or_acc:.4f}  AUC=N/A")

    # Save incrementally after each fold
    pd.DataFrame(all_fold_metrics).to_csv(
        os.path.join(OUT_DIR, "tabpfn_bin_cv_all_folds_metrics.csv"), index=False
    )
    print(f"  ✓ Fold {fold_k} results saved")

# ── Aggregate ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  Aggregating CV results")
print("="*60)

all_folds_df = pd.DataFrame(all_fold_metrics)

per_variable_cv = (
    all_folds_df
    .groupby("Variable", sort=True)
    .agg(
        F1_mean =("F1_or_Acc", "mean"), F1_var =("F1_or_Acc", "var"),
        AUC_mean=("AUC",       "mean"), AUC_var=("AUC",       "var"),
    )
    .reset_index()
)

per_fold_means = (
    all_folds_df
    .groupby("fold")
    .agg(F1_or_Acc=("F1_or_Acc", "mean"), AUC=("AUC", "mean"))
    .reset_index()
)
overall_cv = pd.DataFrame([{
    "F1_mean":  per_fold_means["F1_or_Acc"].mean(),
    "F1_var":   per_fold_means["F1_or_Acc"].var(),
    "AUC_mean": per_fold_means["AUC"].mean(),
    "AUC_var":  per_fold_means["AUC"].var(),
}])

print("\nPer-variable CV results:")
print(per_variable_cv.to_string(index=False))
print("\nOverall CV results:")
print(overall_cv.to_string(index=False))

# ── Save final results to Drive ────────────────────────────────────────
all_folds_df.to_csv(   os.path.join(OUT_DIR, "tabpfn_bin_cv_all_folds_metrics.csv"),    index=False)
per_variable_cv.to_csv(os.path.join(OUT_DIR, "tabpfn_bin_cv_per_variable_metrics.csv"), index=False)
overall_cv.to_csv(     os.path.join(OUT_DIR, "tabpfn_bin_cv_overall_metrics.csv"),      index=False)

print("\n✓ All results saved to /content/ — download them before your session ends:")
print(f"  tabpfn_bin_cv_all_folds_metrics.csv")
print(f"  tabpfn_bin_cv_per_variable_metrics.csv")
print(f"  tabpfn_bin_cv_overall_metrics.csv")