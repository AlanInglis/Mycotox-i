# tabnet_binary_5fold_cv_full.py
# -------------------------------------------------------
# TabNet classifier: 5-fold CV for binary toxins
# Pretraining variant: FULL
#   - Encoder pretrained ONCE on all X (train+test) before the CV loop
#   - Pretrained weights cloned into a fresh classifier each fold
#   - Matches the "full autoencoder" approach in the R transfer learning scripts
#
# Outputs:
#   tabnet_bin_full_cv_all_folds_metrics.csv
#   tabnet_bin_full_cv_per_variable_metrics.csv
#   tabnet_bin_full_cv_overall_metrics.csv
# -------------------------------------------------------

import os, random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer

# ── Config (inlined from tabnet_config.py) ────────────────────────────────────
BASE_DIR   = "/Users/alaninglis/Desktop/Transfer Learning models/TabNET"
DATA_DIR   = os.path.join(BASE_DIR, "data")
RESULTS_DIR= os.path.join(BASE_DIR, "tabnet_results_cv")
OUT_DIR    = os.path.join(RESULTS_DIR, "binary_full")
os.makedirs(OUT_DIR, exist_ok=True)

FOLD_CSV  = "/Users/alaninglis/Desktop/Mycotoxin Paper Final/fold_assignments.csv"
X_CSV     = "/Users/alaninglis/Desktop/Mycotoxin Paper Final/x.csv"
YBIN_CSV  = "/Users/alaninglis/Desktop/Mycotoxin Paper Final/y_bin.csv"

SEED     = 1701
N_FOLDS  = 5
VAL_FRAC = 0.15

BINARY_TOXINS = [
    "DON_bin", "D3G_bin", "Nivalenol_bin", "3-AC-DON_bin", "15-AC-DON_bin",
    "T-2_toxin_bin", "HT-2_toxin_bin", "T2G_bin", "Neos_bin", "ENN_A1_bin",
    "ENN_A_bin", "ENN_B_bin", "ENN_B1_bin", "BEAU_bin", "ZEN_bin",
    "Apicidin_bin", "STER_bin", "DAS_bin", "Quest_bin", "AOH_bin",
    "AME_bin", "MON_bin", "Ergocristine_bin", "EGT_bin",
]

TABNET_ARCH = dict(
    n_d           = 32,
    n_a           = 32,
    n_steps       = 5,
    gamma         = 1.3,
    lambda_sparse = 1e-4,
    momentum      = 0.02,
)

FIT_PARAMS = dict(
    max_epochs         = 200,
    patience           = 20,
    batch_size         = 256,
    virtual_batch_size = 128,
    num_workers        = 0,
)

PRETRAIN_PARAMS = dict(
    max_epochs         = 100,
    patience           = 15,
    batch_size         = 256,
    virtual_batch_size = 128,
    num_workers        = 0,
    pretraining_ratio  = 0.5,
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def set_seeds(seed):
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def to_binary(arr):
    mapping = {"0":0.,"1":1.,"false":0.,"true":1.,"neg":0.,"pos":1.,"no":0.,"yes":1.}
    s = pd.Series(arr).astype(str).str.strip().str.lower().map(
        lambda x: mapping.get(x, x))
    return pd.to_numeric(s, errors="coerce").values.astype(np.float32)

def resolve_batch_size(n, requested=256, virtual=128):
    bs  = min(requested, max(32, n // 4))
    vbs = min(virtual, bs // 2)
    vbs = max(vbs, 1)
    return bs, vbs

set_seeds(SEED)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
X_df    = pd.read_csv(X_CSV)
Yb_df   = pd.read_csv(YBIN_CSV)
fold_df = pd.read_csv(FOLD_CSV)

X_np  = X_df.select_dtypes(include=[np.number]).values.astype(np.float32)
Yb_np = Yb_df.reindex(columns=BINARY_TOXINS).values.astype(float)
fold_assignments = fold_df["fold"].values

assert len(fold_df) == len(X_np), \
    f"fold_assignments.csv has {len(fold_df)} rows but X has {len(X_np)}"

if np.isnan(X_np).any():
    raise ValueError("Predictor matrix contains NaNs — TabNet requires finite inputs.")

print(f"Data: {X_np.shape[0]} rows x {X_np.shape[1]} features, {len(BINARY_TOXINS)} toxins")

# ── Step 1: Pretrain encoder ONCE on full X ───────────────────────────────────
print("\n" + "="*60)
print("  Pretraining encoder on full X (all rows)...")
print("="*60)

set_seeds(SEED)
pretrainer = TabNetPretrainer(
    **TABNET_ARCH,
    seed        = SEED,
    verbose     = 1,
    device_name = "cpu",
)
pretrainer.fit(X_np, eval_set=[], **PRETRAIN_PARAMS)

pretrain_save_path = os.path.join(OUT_DIR, "tabnet_pretrained_full")
pretrainer.save_model(pretrain_save_path)
print(f"  ✓ Pretrained encoder saved → {pretrain_save_path}.zip")

# ── Step 2: 5-Fold CV ─────────────────────────────────────────────────────────
all_fold_metrics = []

for fold_k in range(1, N_FOLDS + 1):
    print(f"\n{'='*60}")
    print(f"  Fold {fold_k} / {N_FOLDS}")
    print(f"{'='*60}")

    set_seeds(SEED + fold_k)

    test_mask  = (fold_assignments == fold_k)
    train_mask = ~test_mask

    X_train_fold = X_np[train_mask]
    X_test_fold  = X_np[test_mask]
    Yb_train     = Yb_np[train_mask]
    Yb_test      = Yb_np[test_mask]

    print(f"  Train: {X_train_fold.shape[0]} rows | Test: {X_test_fold.shape[0]} rows")

    for j, tox in enumerate(BINARY_TOXINS):
        base_name = tox.replace("_bin", "")

        y_tr_raw = to_binary(Yb_train[:, j])
        y_te_raw = to_binary(Yb_test[:, j])

        tr_mask = ~np.isnan(y_tr_raw)
        te_mask = ~np.isnan(y_te_raw)

        y_tr_all = y_tr_raw[tr_mask].astype(int)
        y_te_obs = y_te_raw[te_mask].astype(int)
        X_tr_all = X_train_fold[tr_mask]
        X_te_obs = X_test_fold[te_mask]

        counts_tr = Counter(y_tr_all.tolist())
        if len(counts_tr) < 2 or min(counts_tr.values()) < 2:
            print(f"  Skipping {tox}: training class issue {dict(counts_tr)}")
            all_fold_metrics.append(dict(
                fold=fold_k, Variable=base_name,
                F1_or_Acc=np.nan, AUC=np.nan))
            continue

        # Val split from training fold only
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_tr_all, y_tr_all,
                test_size=VAL_FRAC, random_state=SEED + fold_k,
                stratify=y_tr_all)
        except ValueError:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_tr_all, y_tr_all,
                test_size=VAL_FRAC, random_state=SEED + fold_k)

        bs, vbs = resolve_batch_size(len(y_tr),
                                     FIT_PARAMS["batch_size"],
                                     FIT_PARAMS["virtual_batch_size"])

        # Load fresh pretrained encoder for this toxin/fold
        fresh_pretrainer = TabNetPretrainer(
            **TABNET_ARCH, seed=SEED + fold_k, device_name="cpu")
        fresh_pretrainer.load_model(pretrain_save_path + ".zip")

        clf = TabNetClassifier(
            **TABNET_ARCH,
            seed        = SEED + fold_k,
            verbose     = 0,
            device_name = "cpu",
        )
        clf.fit(
            X_train           = X_tr,
            y_train           = y_tr,
            eval_set          = [(X_val, y_val)],
            eval_metric       = ["auc"],
            max_epochs        = FIT_PARAMS["max_epochs"],
            patience          = FIT_PARAMS["patience"],
            batch_size        = bs,
            virtual_batch_size= vbs,
            num_workers       = FIT_PARAMS["num_workers"],
            from_unsupervised = fresh_pretrainer,
        )

        proba     = clf.predict_proba(X_te_obs)
        pred_prob = proba[:, 1] if proba.shape[1] > 1 else np.full(len(y_te_obs), np.nan)

        counts_te = Counter(y_te_obs.tolist())
        if len(counts_te) == 2:
            y_pred    = (pred_prob >= 0.5).astype(int)
            f1_or_acc = float(f1_score(y_te_obs, y_pred))
            auc       = float(roc_auc_score(y_te_obs, pred_prob))
        else:
            y_pred    = (pred_prob >= 0.5).astype(int)
            f1_or_acc = float(accuracy_score(y_te_obs, y_pred))
            auc       = np.nan

        all_fold_metrics.append(dict(
            fold=fold_k, Variable=base_name,
            F1_or_Acc=f1_or_acc, AUC=auc))

        msg = f"  {tox}: F1={f1_or_acc:.4f}"
        if not np.isnan(auc):
            msg += f"  AUC={auc:.4f}"
        print(msg)

    # Save incrementally after each fold
    pd.DataFrame(all_fold_metrics).to_csv(
        os.path.join(OUT_DIR, "tabnet_bin_full_cv_all_folds_metrics.csv"), index=False)
    print(f"  ✓ Fold {fold_k} saved")

# ── Aggregate ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  Aggregating results...")
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
    all_folds_df.groupby("fold")
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

all_folds_df.to_csv(   os.path.join(OUT_DIR, "tabnet_bin_full_cv_all_folds_metrics.csv"),    index=False)
per_variable_cv.to_csv(os.path.join(OUT_DIR, "tabnet_bin_full_cv_per_variable_metrics.csv"), index=False)
overall_cv.to_csv(     os.path.join(OUT_DIR, "tabnet_bin_full_cv_overall_metrics.csv"),      index=False)

print(f"\n✓ All results saved to: {OUT_DIR}")