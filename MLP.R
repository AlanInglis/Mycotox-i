########################################################################
## Transfer Learning MLP: frozen / unfrozen encoder                   ##
## Alan Inglis                                                         ##
## ** 5-Fold Cross-Validation version **                               ##
## ** AUTOENCODER PRETRAINED ON FULL DATASET (non-strict) **          ##
##                                                                     ##
## Design choice: the autoencoder is trained once on the full x       ##
## matrix before the CV loop begins. Because it is unsupervised       ##
## (no labels used), this is analogous to using an externally         ##
## pretrained feature extractor. The task model (heads) is still      ##
## trained and evaluated strictly within each fold.                   ##
##                                                                     ##
## USAGE: Set FREEZE_ENCODER below, then run the full script.         ##
##   FREEZE_ENCODER <- TRUE   →  frozen transfer learning             ##
##   FREEZE_ENCODER <- FALSE  →  unfrozen transfer learning           ##
########################################################################

# ── TOGGLE THIS FLAG ───────────────────────────────────────────────────
FREEZE_ENCODER <- FALSE     # <-- change to FALSE for the unfrozen run
# ──────────────────────────────────────────────────────────────────────

library(keras3)
library(tensorflow)
library(reticulate)
library(tidyverse)
library(caret)
library(purrr)
library(tibble)
library(pROC)
library(dplyr)

# ── Paths ──────────────────────────────────────────────────────────────
# All data files and fold_assignments.csv must live in the working directory.
# Confirm with getwd(); change with setwd("your/path") if needed.
WORK_DIR <- getwd()


# ── Seeds ──────────────────────────────────────────────────────────────
np     <- import("numpy",       convert = TRUE)
random <- import("random",      convert = TRUE)
tf     <- import("tensorflow",  convert = TRUE)
use_backend("tensorflow")

set_all_seeds <- function(seed = 1701L) {
  set.seed(seed)
  np$random$seed(seed)
  random$seed(seed)
  tf$random$set_seed(seed)
}

set_all_seeds()
tf$config$experimental$enable_op_determinism()


######################## 1.  Data import ################################

# Full predictor and response matrices (used to reconstruct folds)
x      <- readRDS("x.rds")
y_cont <- readRDS("y_cont.rds")
y_bin  <- readRDS("y_bin.rds")

to_numeric_matrix <- function(m) {
  m <- apply(m, 2, as.numeric)
  as.matrix(m)
}

x      <- to_numeric_matrix(x)
y_cont <- to_numeric_matrix(y_cont)
y_bin  <- to_numeric_matrix(y_bin)

cont_vars <- colnames(y_cont)
bin_vars  <- colnames(y_bin)


######################## 2.  Load fold assignments ######################
# Generated once by the baseline script — guarantees identical folds
# across all 6 models.

fold_df <- read.csv(file.path(WORK_DIR, "fold_assignments.csv"))
n_folds <- max(fold_df$fold)

# Reconstruct fold_list: list of test-set row indices per fold
fold_list <- lapply(seq_len(n_folds), function(k) {
  fold_df$row_index[fold_df$fold == k]
})


######################## 3.  Masked losses ##############################

masked_mse <- function(y_true, y_pred) {
  keep        <- tf$math$logical_not(tf$math$is_nan(y_true))
  keep_f32    <- tf$cast(keep, tf$float32)
  y_true_safe <- tf$where(keep, y_true, tf$zeros_like(y_true))
  se          <- tf$math$square(y_pred - y_true_safe) * keep_f32
  tf$math$reduce_sum(se) /
    (tf$math$reduce_sum(keep_f32) + 1e-7)
}

masked_bce <- function(y_true, y_pred) {
  keep        <- tf$math$logical_not(tf$math$is_nan(y_true))
  keep_f32    <- tf$cast(keep, tf$float32)
  y_true_safe <- tf$where(keep, y_true, tf$zeros_like(y_true))
  y_pred_clip <- tf$clip_by_value(y_pred, 1e-7, 1 - 1e-7)
  ce_elem     <- -(y_true_safe * tf$math$log(y_pred_clip) +
                     (1 - y_true_safe) * tf$math$log(1 - y_pred_clip))
  ce_masked   <- ce_elem * keep_f32
  tf$math$reduce_sum(ce_masked) /
    (tf$math$reduce_sum(keep_f32) + 1e-7)
}


######################## 4.  Metric helpers #############################

metrics_cont <- function(obs, pred) {
  ok <- !is.na(obs);  n <- sum(ok)
  if (n == 0) return(c(RMSE = NA_real_, R2 = NA_real_))
  resid <- pred[ok] - obs[ok]
  rmse  <- sqrt(mean(resid^2))
  r2 <- if (n > 1 && var(obs[ok]) > 0) {
    1 - sum(resid^2) / sum((obs[ok] - mean(obs[ok]))^2)
  } else { NA_real_ }
  c(RMSE = rmse, R2 = r2)
}

metrics_bin <- function(obs, prob, threshold = 0.5) {
  ok <- !is.na(obs)
  if (sum(ok) == 0) return(c(F1_or_Acc = NA_real_, AUC = NA_real_))
  obs_ok  <- obs[ok];  prob_ok <- prob[ok]
  pred_ok <- as.numeric(prob_ok >= threshold)
  if (length(unique(obs_ok)) == 2) {
    tp <- sum(pred_ok == 1 & obs_ok == 1)
    fp <- sum(pred_ok == 1 & obs_ok == 0)
    fn <- sum(pred_ok == 0 & obs_ok == 1)
    precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
    recall    <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
    f1        <- ifelse(precision + recall == 0, 0,
                        2 * precision * recall / (precision + recall))
    auc <- as.numeric(pROC::auc(obs_ok, prob_ok))
    return(c(F1_or_Acc = f1, AUC = auc))
  }
  acc <- mean(pred_ok == obs_ok)
  c(F1_or_Acc = acc, AUC = 0.5)
}

build_metrics_tbl <- function(y_cont_test, y_cont_pred,
                              y_bin_test,  y_bin_pred) {
  cont_tbl <- map2_dfc(
    as.data.frame(y_cont_test),
    as.data.frame(y_cont_pred),
    metrics_cont
  ) |> t() |> as_tibble(rownames = "Variable") |> rename(RMSE = V1, R2 = V2)
  
  bin_tbl <- map2_dfc(
    as.data.frame(y_bin_test),
    as.data.frame(y_bin_pred),
    metrics_bin
  ) |> t() |> as_tibble(rownames = "Variable") |>
    rename(F1_or_Acc = V1, AUC = V2) |>
    mutate(Variable = str_remove(Variable, "_bin$"))
  
  cont_tbl |> left_join(bin_tbl, by = "Variable") |> arrange(Variable)
}


######################## 5.  Pretrain autoencoder on FULL dataset ########
# The autoencoder is unsupervised (no labels) so training on the full x
# is analogous to using an externally pretrained feature extractor.
# The encoder weights are fixed here; only the task heads are trained
# inside the CV loop (frozen run), or the whole network is fine-tuned
# (unfrozen run) — but always evaluated strictly within each fold.

set_all_seeds(1701L)

input_dim  <- ncol(x)
output_dim <- ncol(y_cont)

message("\n===== Pretraining autoencoder on full x =====")

input_layer <- layer_input(shape = input_dim, name = "autoencoder_input")

encoded <- input_layer |>
  layer_dense(units = 512, activation = "relu") |>
  layer_dropout(rate = 0.2) |>
  layer_dense(units = 256, activation = "relu") |>
  layer_dropout(rate = 0.2) |>
  layer_dense(units = 128, activation = "relu", name = "latent_space")

decoded <- encoded |>
  layer_dense(units = 256, activation = "relu") |>
  layer_dropout(rate = 0.2) |>
  layer_dense(units = 512, activation = "relu") |>
  layer_dropout(rate = 0.2) |>
  layer_dense(units = input_dim, activation = "linear",
              name = "reconstructed_output")

autoencoder <- keras_model(inputs = input_layer, outputs = decoded)

autoencoder |> compile(
  loss      = "mse",
  optimizer = optimizer_adam(learning_rate = 1e-3),
  metrics   = list("mse")
)

autoencoder |> fit(
  x, x,                          # <-- full dataset, both times
  epochs           = 100,
  batch_size       = 32,
  validation_split = 0.2,
  verbose          = 2,
  shuffle          = FALSE,
  callbacks        = list(
    callback_early_stopping(patience = 10, restore_best_weights = TRUE)
  )
)

# Extract encoder once — weights are shared across all folds
encoder_pretrained <- keras_model(inputs = input_layer, outputs = encoded)

message("Autoencoder pretraining complete. Encoder extracted.")


######################## 6.  5-Fold CV loop #############################

fold_results <- vector("list", n_folds)

for (k in seq_len(n_folds)) {
  
  message("\n===== Fold ", k, " / ", n_folds,
          "  [encoder ", ifelse(FREEZE_ENCODER, "FROZEN", "UNFROZEN"), "] =====")
  
  set_all_seeds(1701L + k)
  
  # ── Split ────────────────────────────────────────────────────────────
  test_idx  <- fold_list[[k]]
  train_idx <- setdiff(seq_len(nrow(x)), test_idx)
  
  x_train      <- x[train_idx, ];  x_test      <- x[test_idx, ]
  y_cont_train <- y_cont[train_idx, ]; y_cont_test <- y_cont[test_idx, ]
  y_bin_train  <- y_bin[train_idx, ];  y_bin_test  <- y_bin[test_idx, ]
  
  # ── Step 1: Clone pretrained encoder so each fold starts from the
  #            same pretrained weights (not fine-tuned weights from a
  #            previous fold's unfrozen run) ───────────────────────────
  encoder <- keras_model(inputs = input_layer,
                         outputs = encoded)
  encoder$set_weights(encoder_pretrained$get_weights())
  
  # ── Step 2: Optionally freeze ────────────────────────────────────────
  if (FREEZE_ENCODER) {
    freeze_weights(encoder)
    message("  Encoder weights FROZEN.")
  } else {
    message("  Encoder weights left UNFROZEN (will fine-tune end-to-end).")
  }
  
  # ── Step 3: Attach task heads ────────────────────────────────────────
  encoded_output <- encoder$output
  
  cont_out <- encoded_output |>
    layer_dense(units = output_dim, activation = "relu",    name = "cont_out")
  bin_out  <- encoded_output |>
    layer_dense(units = output_dim, activation = "sigmoid", name = "bin_out")
  
  model <- keras_model(inputs = encoder$input,
                       outputs = list(cont_out, bin_out))
  
  # Unfreeze before compiling if unfrozen run
  if (!FREEZE_ENCODER) unfreeze_weights(encoder)
  
  model |> compile(
    optimizer = optimizer_adam(learning_rate = 1e-3),
    loss      = list(cont_out = masked_mse, bin_out = masked_bce),
    metrics   = list(cont_out = masked_mse, bin_out = masked_bce)
  )
  
  # ── Step 4: Fit task model ───────────────────────────────────────────
  model |> fit(
    x_train,
    list(cont_out = y_cont_train, bin_out = y_bin_train),
    epochs           = 500,
    batch_size       = 32,
    validation_split = 0.2,
    callbacks        = list(
      callback_early_stopping(monitor = "val_loss", patience = 25,
                              restore_best_weights = TRUE)
    ),
    shuffle = FALSE,
    verbose = 2
  )
  
  # ── Step 5: Predict & compute metrics ────────────────────────────────
  pred        <- model |> predict(x_test)
  y_cont_pred <- pred[[1]]; colnames(y_cont_pred) <- cont_vars
  y_bin_pred  <- pred[[2]]; colnames(y_bin_pred)  <- bin_vars
  
  fold_metrics <- build_metrics_tbl(y_cont_test, y_cont_pred,
                                    y_bin_test,  y_bin_pred) |>
    mutate(fold = k)
  
  fold_results[[k]] <- fold_metrics
  
  message("Fold ", k, " summary:")
  print(
    fold_metrics |>
      summarise(RMSE = mean(RMSE, na.rm = TRUE),
                R2   = mean(R2,   na.rm = TRUE),
                F1   = mean(F1_or_Acc, na.rm = TRUE),
                AUC  = mean(AUC,  na.rm = TRUE))
  )
}


######################## 7.  Aggregate results ##########################

all_folds <- bind_rows(fold_results)

per_variable_cv <- all_folds |>
  group_by(Variable) |>
  summarise(
    RMSE_mean = mean(RMSE, na.rm = TRUE), RMSE_var = var(RMSE,  na.rm = TRUE),
    R2_mean   = mean(R2,   na.rm = TRUE), R2_var   = var(R2,    na.rm = TRUE),
    F1_mean   = mean(F1_or_Acc, na.rm = TRUE), F1_var = var(F1_or_Acc, na.rm = TRUE),
    AUC_mean  = mean(AUC,  na.rm = TRUE), AUC_var  = var(AUC,   na.rm = TRUE),
    .groups = "drop"
  ) |>
  arrange(Variable)

overall_cv <- all_folds |>
  group_by(fold) |>
  summarise(
    RMSE      = mean(RMSE,      na.rm = TRUE),
    R2        = mean(R2,        na.rm = TRUE),
    F1_or_Acc = mean(F1_or_Acc, na.rm = TRUE),
    AUC       = mean(AUC,       na.rm = TRUE),
    .groups = "drop"
  ) |>
  summarise(
    RMSE_mean = mean(RMSE),      RMSE_var  = var(RMSE),
    R2_mean   = mean(R2),        R2_var    = var(R2),
    F1_mean   = mean(F1_or_Acc), F1_var    = var(F1_or_Acc),
    AUC_mean  = mean(AUC),       AUC_var   = var(AUC)
  )

message("\n===== Per-variable CV results =====")
print(per_variable_cv, n = Inf)

message("\n===== Overall CV results (mean & variance across 5 folds) =====")
print(overall_cv)


######################## 8.  Save results ###############################


# tag encodes both the freeze strategy AND the autoencoder training scope
# so filenames are unambiguous across all four variants:
#   frozen_full  /  unfrozen_full  (autoencoder on full dataset)
tag <- paste0(ifelse(FREEZE_ENCODER, "frozen", "unfrozen"), "_full")

saveRDS(all_folds,       paste0("cv_all_folds_metrics_",    tag, ".rds"))
saveRDS(per_variable_cv, paste0("cv_per_variable_metrics_", tag, ".rds"))
saveRDS(overall_cv,      paste0("cv_overall_metrics_",      tag, ".rds"))

write.csv(all_folds,
          file.path(WORK_DIR, paste0("cv_all_folds_metrics_",    tag, ".csv")),
           row.names = FALSE)
write.csv(per_variable_cv,
          file.path(WORK_DIR, paste0("cv_per_variable_metrics_", tag, ".csv")),
          row.names = FALSE)
write.csv(overall_cv,
          file.path(WORK_DIR, paste0("cv_overall_metrics_",      tag, ".csv")),
          row.names = FALSE)

message("\nAll CV results saved with tag: '", tag, "'")
