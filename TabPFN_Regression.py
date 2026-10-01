# tabpfn_regression_5fold_cv.py
# -----------------------------------------------------
# TabPFN regressor: 5-fold CV for continuous toxins
# ** Google Colab version **
#
# Setup steps before running:
#   1. Runtime -> Change runtime type -> GPU
#   2. Run the mount cell below to connect Google Drive
#   3. Upload x.csv, y_cont.csv, fold_assignments.csv to /content/
#   4. Set DRIVE_OUT to your preferred Google Drive output folder
#
# Outputs (saved to Google Drive):
#   tabpfn_cont_cv_all_folds_metrics.csv
#   tabpfn_cont_cv_per_variable_metrics.csv
#   tabpfn_cont_cv_overall_metrics.csv
# -----------------------------------------------------

import os, random
import numpy as np
import pandas as pd
from tabpfn import TabPFNRegressor
from sklearn.metrics import r2_score

# ── Config ─────────────────────────────────────────────────────────────
SEED      = 1701
N_FOLDS   = 5
BASE_DIR  = "/content"          # where you uploaded the CSVs
OUT_DIR   = "/content"          # results saved here — download before session ends
os.makedirs(OUT_DIR, exist_ok=True)

TOXINS = [
    "DON", "D3G", "Nivalenol", "3-AC-DON", "15-AC-DON", "T-2_toxin",
    "HT-2_toxin", "T2G", "Neos", "ENN_A1", "ENN_A", "ENN_B",
    "ENN_B1", "BEAU", "ZEN", "Apicidin", "STER", "DAS",
    "Quest", "AOH", "AME", "MON", "Ergocristine", "EGT"
]

# ── Reproducibility ────────────────────────────────────────────────────
def set_seeds(seed):
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(SEED)

# ── Device ─────────────────────────────────────────────────────────────
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEV}")

# ── Metric helpers ─────────────────────────────────────────────────────
def rmse_safe(y_true, y_pred):
    try:
        from sklearn.metrics import root_mean_squared_error
        return float(root_mean_squared_error(y_true, y_pred))
    except Exception:
        from sklearn.metrics import mean_squared_error
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# ── Load full data ─────────────────────────────────────────────────────
print("Loading data...")
X_df    = pd.read_csv(os.path.join(BASE_DIR, "x.csv"))
Yc_df   = pd.read_csv(os.path.join(BASE_DIR, "y_cont.csv"))
fold_df = pd.read_csv(os.path.join(BASE_DIR, "fold_assignments.csv"))

X_np  = X_df.select_dtypes(include=[np.number]).values.astype(np.float32)
Yc_np = Yc_df.reindex(columns=TOXINS).values.astype(np.float32)

assert len(fold_df) == len(X_np), \
    f"fold_assignments.csv has {len(fold_df)} rows but data has {len(X_np)}"

fold_assignments = fold_df["fold"].values   # 1-indexed
print(f"Data loaded: {X_np.shape[0]} rows, {X_np.shape[1]} features, {len(TOXINS)} toxins")

if np.isnan(X_np).any():
    raise ValueError("Predictor matrix contains NaNs; TabPFN requires finite inputs.")

# ── 5-Fold CV loop ─────────────────────────────────────────────────────
all_fold_metrics = []

for fold_k in range(1, N_FOLDS + 1):
    print(f"\n{'='*60}")
    print(f"  Fold {fold_k} / {N_FOLDS}")
    print(f"{'='*60}")

    set_seeds(SEED + fold_k)

    # Split
    test_mask  = (fold_assignments == fold_k)
    train_mask = ~test_mask

    X_train = X_np[train_mask]
    X_test  = X_np[test_mask]
    Yc_train = Yc_np[train_mask]
    Yc_test  = Yc_np[test_mask]

    print(f"  Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # Per-toxin loop
    for j, tox in enumerate(TOXINS):
        y_tr = Yc_train[:, j]
        y_te = Yc_test[:, j]

        tr_mask = ~np.isnan(y_tr)
        te_mask = ~np.isnan(y_te)

        if tr_mask.sum() < 10:
            print(f"  Skipping {tox}: <10 training rows")
            all_fold_metrics.append(dict(
                fold=fold_k, Variable=tox,
                RMSE=np.nan, R2=np.nan, F1_or_Acc=np.nan, AUC=np.nan
            ))
            continue

        X_tr_f = X_train[tr_mask];  y_tr_f = y_tr[tr_mask]
        X_te_f = X_test[te_mask];   y_te_f = y_te[te_mask]

        reg = TabPFNRegressor(device=DEV, n_estimators=8)
        reg.fit(X_tr_f, y_tr_f)
        preds = reg.predict(X_te_f)

        rmse = rmse_safe(y_te_f, preds)
        r2   = float(r2_score(y_te_f, preds)) if len(np.unique(y_te_f)) > 1 else np.nan

        all_fold_metrics.append(dict(
            fold=fold_k, Variable=tox,
            RMSE=rmse, R2=r2, F1_or_Acc=np.nan, AUC=np.nan
        ))
        print(f"  {tox}: RMSE={rmse:.4f}  R2={r2:.4f}")

    # Save incrementally after each fold in case of session timeout
    pd.DataFrame(all_fold_metrics).to_csv(
        os.path.join(OUT_DIR, "tabpfn_cont_cv_all_folds_metrics.csv"), index=False
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
        RMSE_mean=("RMSE", "mean"), RMSE_var=("RMSE", "var"),
        R2_mean  =("R2",   "mean"), R2_var  =("R2",   "var"),
    )
    .reset_index()
)

per_fold_means = (
    all_folds_df
    .groupby("fold")
    .agg(RMSE=("RMSE", "mean"), R2=("R2", "mean"))
    .reset_index()
)
overall_cv = pd.DataFrame([{
    "RMSE_mean": per_fold_means["RMSE"].mean(),
    "RMSE_var":  per_fold_means["RMSE"].var(),
    "R2_mean":   per_fold_means["R2"].mean(),
    "R2_var":    per_fold_means["R2"].var(),
}])

print("\nPer-variable CV results:")
print(per_variable_cv.to_string(index=False))
print("\nOverall CV results:")
print(overall_cv.to_string(index=False))

# ── Save final results to Drive ────────────────────────────────────────
all_folds_df.to_csv(   os.path.join(OUT_DIR, "tabpfn_cont_cv_all_folds_metrics.csv"),    index=False)
per_variable_cv.to_csv(os.path.join(OUT_DIR, "tabpfn_cont_cv_per_variable_metrics.csv"), index=False)
overall_cv.to_csv(     os.path.join(OUT_DIR, "tabpfn_cont_cv_overall_metrics.csv"),      index=False)

print("\n✓ All results saved to /content/ — download them before your session ends:")
print(f"  tabpfn_cont_cv_all_folds_metrics.csv")
print(f"  tabpfn_cont_cv_per_variable_metrics.csv")
print(f"  tabpfn_cont_cv_overall_metrics.csv")