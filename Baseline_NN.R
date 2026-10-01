########################################################################
## Multi-task network: joint continuous + binary mycotoxin prediction ##
## Alan Inglis                                                        ##
## ** 5-Fold Cross-Validation version with raw predictions saving **  ##
########################################################################

library(tidyverse)
library(keras3)
library(reticulate)
library(caret)       # for createFolds()
library(purrr)
library(tibble)
library(pROC)
library(dplyr)

# ── Paths ──────────────────────────────────────────────────────────────
TRANSFER_DIR <- getwd()

# ── Backend / seed helpers ─────────────────────────────────────────────
tf     <- import("tensorflow")
np     <- import("numpy",     convert = TRUE)
random <- import("random",    convert = TRUE)
keras3::use_backend("tensorflow")

set_all_seeds <- function(seed = 1701L) {
  set.seed(seed)
  np$random$seed(seed)
  random$seed(seed)
  tf$random$set_seed(seed)
}

set_all_seeds()
tf$config$experimental$enable_op_determinism()


######################## 1.  Data import ################################

dat <- readRDS("dat_nn.rds")

dat <- dat %>%
  select(-replicate_number) %>%
  select(-Year_X2022) %>%
  select(-Year_X2023)

start_resp <- match("DON", names(dat))
resp_cols  <- names(dat)[start_resp:length(dat)]
bin_vars   <- resp_cols[str_detect(resp_cols, "_bin$")]
cont_vars  <- setdiff(resp_cols, bin_vars)
pred_vars  <- setdiff(names(dat), resp_cols)

dat <- as.matrix(dat)

######################## 2.  Predictors and responses ###################

non_response_vars <- setdiff(colnames(dat), c(cont_vars, bin_vars))

x      <- dat[, non_response_vars]
y_cont <- dat[, cont_vars]
y_bin  <- dat[, bin_vars]

######################## 3.  Generate & save fold assignments ###########

set.seed(1701)
n_folds <- 5
fold_list <- createFolds(1:nrow(x), k = n_folds, list = TRUE, returnTrain = FALSE)

# Build a vector: fold_assignments[i] = which fold row i belongs to (as test)
fold_assignments <- integer(nrow(x))
for (k in seq_len(n_folds)) {
  fold_assignments[fold_list[[k]]] <- k
}

fold_df <- data.frame(row_index = seq_len(nrow(x)),
                      fold      = fold_assignments)

fold_path <- file.path(TRANSFER_DIR, "fold_assignments.csv")
write.csv(fold_df, fold_path, row.names = FALSE)
message("Fold assignments saved to: ", fold_path)


######################## 4.  Masked losses ##############################

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


######################## 5.  Metric helpers #############################

metrics_cont <- function(obs, pred) {
  ok <- !is.na(obs)
  n  <- sum(ok)
  if (n == 0) return(c(RMSE = NA_real_, R2 = NA_real_))
  resid <- pred[ok] - obs[ok]
  rmse  <- sqrt(mean(resid^2))
  r2 <- if (n > 1 && var(obs[ok]) > 0) {
    ss_tot <- sum((obs[ok] - mean(obs[ok]))^2)
    1 - sum(resid^2) / ss_tot
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
  ) |>
    t() |>
    as_tibble(rownames = "Variable") |>
    rename(RMSE = V1, R2 = V2)
  
  bin_tbl <- map2_dfc(
    as.data.frame(y_bin_test),
    as.data.frame(y_bin_pred),
    metrics_bin
  ) |>
    t() |>
    as_tibble(rownames = "Variable") |>
    rename(F1_or_Acc = V1, AUC = V2) |>
    mutate(Variable = str_remove(Variable, "_bin$"))
  
  cont_tbl |>
    left_join(bin_tbl, by = "Variable") |>
    arrange(Variable)
}


######################## 6.  Model builder ##############################

build_model <- function(input_dim, output_dim) {
  inputs <- layer_input(shape = input_dim, name = "predictors")
  
  shared <- inputs |>
    layer_dense(units = 128, activation = "relu") |>
    layer_dropout(rate = 0.20) |>
    layer_dense(units = 64,  activation = "relu") |>
    layer_dropout(rate = 0.20)
  
  cont_out <- shared |>
    layer_dense(units = output_dim, activation = "relu",    name = "cont_out")
  bin_out  <- shared |>
    layer_dense(units = output_dim, activation = "sigmoid", name = "bin_out")
  
  model <- keras_model(inputs = inputs, outputs = list(cont_out, bin_out))
  
  model |> compile(
    optimizer = optimizer_adam(learning_rate = 1e-3),
    loss      = list(cont_out = masked_mse, bin_out = masked_bce),
    metrics   = list(cont_out = masked_mse, bin_out = masked_bce)
  )
  model
}


######################## 7.  5-Fold CV loop #############################

input_dim  <- ncol(x)
output_dim <- length(cont_vars)

fold_results <- vector("list", n_folds)   # per-fold metric tables
all_predictions <- vector("list", n_folds) # raw predictions for plotting

for (k in seq_len(n_folds)) {
  
  message("\n===== Fold ", k, " / ", n_folds, " =====")
  
  # Re-seed everything before each fold for determinism
  set_all_seeds(1701L + k)
  
  # ── Split ────────────────────────────────────────────────────────────
  test_idx  <- fold_list[[k]]
  train_idx <- setdiff(seq_len(nrow(x)), test_idx)
  
  x_train      <- x[train_idx, ];  x_test      <- x[test_idx, ]
  y_cont_train <- y_cont[train_idx, ]; y_cont_test <- y_cont[test_idx, ]
  y_bin_train  <- y_bin[train_idx, ];  y_bin_test  <- y_bin[test_idx, ]
  
  # ── Build fresh model ────────────────────────────────────────────────
  model <- build_model(input_dim, output_dim)
  
  early_stop <- callback_early_stopping(
    monitor              = "val_loss",
    patience             = 25,
    restore_best_weights = TRUE
  )
  
  # ── Fit ──────────────────────────────────────────────────────────────
  model |> fit(
    x_train,
    list(cont_out = y_cont_train, bin_out = y_bin_train),
    epochs           = 500,
    batch_size       = 32,
    validation_split = 0.2,
    callbacks        = list(early_stop),
    verbose          = 2,
    shuffle          = FALSE
  )
  
  # ── Predict & evaluate ───────────────────────────────────────────────
  pred        <- model |> predict(x_test)
  y_cont_pred <- pred[[1]]; colnames(y_cont_pred) <- cont_vars
  y_bin_pred  <- pred[[2]]; colnames(y_bin_pred)  <- bin_vars
  
  fold_metrics <- build_metrics_tbl(y_cont_test, y_cont_pred,
                                    y_bin_test,  y_bin_pred) |>
    mutate(fold = k)
  
  fold_results[[k]] <- fold_metrics
  
  # ── Save raw predictions for plotting ────────────────────────────────
  # Continuous predictions
  cont_predictions <- data.frame(
    toxin = rep(cont_vars, each = nrow(y_cont_test)),
    actual = as.vector(y_cont_test),
    predicted = as.vector(y_cont_pred),
    fold = k,
    type = "continuous"
  )
  
  # Binary predictions  
  bin_predictions <- data.frame(
    toxin = rep(str_remove(bin_vars, "_bin$"), each = nrow(y_bin_test)),
    actual = as.vector(y_bin_test), 
    predicted = as.vector(y_bin_pred),
    fold = k,
    type = "binary"
  )
  
  # Combine and store
  fold_predictions <- bind_rows(cont_predictions, bin_predictions)
  all_predictions[[k]] <- fold_predictions
  
  message("Fold ", k, " summary:")
  print(
    fold_metrics |>
      summarise(RMSE = mean(RMSE, na.rm = TRUE),
                R2   = mean(R2,   na.rm = TRUE),
                F1   = mean(F1_or_Acc, na.rm = TRUE),
                AUC  = mean(AUC,  na.rm = TRUE))
  )
}

######################## 8.  Aggregate results ##########################

all_folds <- bind_rows(fold_results)

# ── Per-variable: mean & variance across folds ──────────────────────
per_variable_cv <- all_folds |>
  group_by(Variable) |>
  summarise(
    RMSE_mean = mean(RMSE, na.rm = TRUE),
    RMSE_var  = var(RMSE,  na.rm = TRUE),
    R2_mean   = mean(R2,   na.rm = TRUE),
    R2_var    = var(R2,    na.rm = TRUE),
    F1_mean   = mean(F1_or_Acc, na.rm = TRUE),
    F1_var    = var(F1_or_Acc,  na.rm = TRUE),
    AUC_mean  = mean(AUC,  na.rm = TRUE),
    AUC_var   = var(AUC,   na.rm = TRUE),
    .groups = "drop"
  ) |>
  arrange(Variable)

# ── Overall: mean & variance across folds (averaged over variables) ──
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

# ── Combine all predictions ──────────────────────────────────────────
all_predictions_df <- bind_rows(all_predictions)

message("\n===== Per-variable CV results =====")
print(per_variable_cv, n = Inf)

message("\n===== Overall CV results (mean ± variance across 5 folds) =====")
print(overall_cv)


######################## 9.  Save results ###############################

# Save metrics as before
saveRDS(all_folds,        "cv_all_folds_metrics.rds")
saveRDS(per_variable_cv,  "cv_per_variable_metrics.rds")
saveRDS(overall_cv,       "cv_overall_metrics.rds")

write.csv(all_folds,       file.path(TRANSFER_DIR, "cv_all_folds_metrics.csv"),       row.names = FALSE)
write.csv(per_variable_cv, file.path(TRANSFER_DIR, "cv_per_variable_metrics.csv"),    row.names = FALSE)
write.csv(overall_cv,      file.path(TRANSFER_DIR, "cv_overall_metrics.csv"),         row.names = FALSE)

# Save raw predictions for plotting
write.csv(all_predictions_df, file.path(TRANSFER_DIR, "baseline_nn_raw_predictions.csv"), row.names = FALSE)

message("\nAll CV results and raw predictions saved.")