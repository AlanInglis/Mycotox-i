

permutation_importance <- function(model, x_test, y_test, loss_function, 
                                   output_index, n_rounds = 5, task_name = "task") {
  
  cat("Calculating permutation importance for", task_name, "...\n")
  
  # Get baseline performance
  baseline_pred <- predict(model, x_test, verbose = 0)
  # Convert prediction to matrix with same structure as y_test
  baseline_pred_matrix <- as.matrix(baseline_pred[[output_index]])
  baseline_loss <- as.numeric(loss_function(
    tf$constant(y_test, dtype = tf$float32), 
    tf$constant(baseline_pred_matrix, dtype = tf$float32)
  ))
  
  cat("Baseline", task_name, "loss:", round(baseline_loss, 6), "\n")
  
  # Initialize results
  feature_names <- colnames(x_test)
  n_features <- length(feature_names)
  importance_scores <- numeric(n_features)
  names(importance_scores) <- feature_names
  
  # Create progress bar
  pb <- txtProgressBar(min = 0, max = n_features, style = 3, width = 50)
  cat("Progress:\n")
  
  # Calculate importance for each feature
  for (i in seq_along(feature_names)) {
    feature_name <- feature_names[i]
    
    round_losses <- numeric(n_rounds)
    
    # Multiple permutation rounds for stability
    for (round in 1:n_rounds) {
      # Create copy of test data
      x_test_perm <- x_test
      
      # Permute the current feature
      set.seed(1701 + round + i)  # Reproducible but different for each round/feature
      x_test_perm[, i] <- sample(x_test_perm[, i])
      
      # Get predictions with permuted feature
      perm_pred <- predict(model, x_test_perm, verbose = 0)
      
      # Convert prediction to matrix and ensure same tensor types
      perm_pred_matrix <- as.matrix(perm_pred[[output_index]])
      
      # Calculate loss with permuted feature
      round_losses[round] <- as.numeric(loss_function(
        tf$constant(y_test, dtype = tf$float32), 
        tf$constant(perm_pred_matrix, dtype = tf$float32)
      ))
    }
    
    # Average loss across rounds
    avg_perm_loss <- mean(round_losses)
    
    # Calculate percentage increase in loss
    importance_scores[i] <- ((avg_perm_loss - baseline_loss) / baseline_loss) * 100
    
    # Update progress bar
    setTxtProgressBar(pb, i)
  }
  
  # Close progress bar
  close(pb)
  cat("\n")
  
  # Create results dataframe
  results <- data.frame(
    feature = feature_names,
    importance_pct = importance_scores,
    stringsAsFactors = FALSE
  ) %>%
    arrange(desc(importance_pct))
  
  cat("Completed", task_name, "permutation importance!\n\n")
  return(results)
}





set.seed(1701)

# Calculate permutation importance for regression task
cat("=== REGRESSION TASK IMPORTANCE ===\n")
regression_importance <- permutation_importance(
  model = model,
  x_test = x_test,
  y_test = y_cont_test,
  loss_function = masked_mse,
  output_index = 1,  # cont_out is first output
  n_rounds = 5,
  task_name = "regression"
)

# Calculate permutation importance for classification task  
cat("=== CLASSIFICATION TASK IMPORTANCE ===\n")
classification_importance <- permutation_importance(
  model = model,
  x_test = x_test,
  y_test = y_bin_test,
  loss_function = masked_bce,
  output_index = 2,  # bin_out is second output
  n_rounds = 5,
  task_name = "classification"
)



# Optional: Save results
write.csv(regression_importance, "regression_feature_importance.csv", row.names = FALSE)
write.csv(classification_importance, "classification_feature_importance.csv", row.names = FALSE)
# load csv with read.csv("regression_feature_importance.csv")
read.csv("regression_feature_importance.csv")
read.csv("classification_feature_importance.csv")

# Function to clean and relabel feature names
clean_feature_names <- function(df) {
  df |>
    mutate(feature_clean = feature |>
             # replace "rh_minusXX"
             str_replace_all("^rh_minus(\\d+)$", "Relative humidity lag \\1") |>
             # replace "temp_minusXX"
             str_replace_all("^temp_minus(\\d+)$", "Temperature lag \\1") |>
             # replace "rain_minusXX"
             str_replace_all("^rain_minus(\\d+)$", "Rainfall lag \\1") |>
             # replace underscores generally (e.g. "Soil_pH" -> "Soil pH")
             str_replace_all("_", " "))
}

classification_importance <- clean_feature_names(classification_importance)
regression_importance     <- clean_feature_names(regression_importance)



ggplot(data = classification_importance[1:5,], 
       aes(x = reorder(feature_clean, importance_pct), y = importance_pct)) +
  geom_col(fill = "coral", col = 'black', alpha = 0.8, width = 0.7) +
  coord_flip() +
  labs(
    x = "Features",
    y = "Importance (% increase in loss)") +
  theme_bw() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        axis.text.y = element_text(size = 11),
        axis.text.x = element_text(size = 10),
        panel.grid.minor = element_blank())

# change Year_X2022 to Year 2022 in regression_importance data
regression_importance <- regression_importance %>%
  mutate(feature_clean = str_replace_all(feature_clean, "Year X(\\d+)", "Year \\1"))


ggplot(data = regression_importance[1:5,], 
       aes(x = reorder(feature_clean, importance_pct), y = importance_pct)) +
  geom_col(fill = "steelblue", col = 'black', alpha = 0.8, width = 0.7) +
  coord_flip() +
  labs(
    x = "Features",
    y = "Importance (% increase in loss)") +
  theme_bw() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        axis.text.y = element_text(size = 11),
        axis.text.x = element_text(size = 10),
        panel.grid.minor = element_blank())




# -------------------------------------------------------------------------


# Function to create feature groups
create_feature_groups <- function(feature_names) {
  
  groups <- list()
  ungrouped <- character(0)
  
  # One-hot encoded categorical groups
  categorical_prefixes <- c(
    "Year_", "County_", "Crop_", "Sowing_Ideotype_", "Variety_", 
    "Rotation_", "Establishment_system_", "Cropping_system_", "Soil_type_",
    "Previous_year_1_", "Previous_year_2_", "Previous_year_3_", 
    "Previous_year_4_", "Previous_year_5_",
    "Fertiliser_product_", "Micronutrients_product_",
    "Herbicide_product_",
    "Fungicide_product_",
    "Growth_regulator__product_"
  )
  
  # Create groups for categorical variables
  for (prefix in categorical_prefixes) {
    matching_features <- feature_names[startsWith(feature_names, prefix)]
    if (length(matching_features) > 0) {
      group_name <- gsub("_$", "", prefix)  # Remove trailing underscore
      groups[[group_name]] <- matching_features
    }
  }
  
  # Weather variable groups
  groups[["rainfall_history"]] <- feature_names[startsWith(feature_names, "rain_minus")]
  groups[["temperature_history"]] <- feature_names[startsWith(feature_names, "temp_minus")]
  groups[["humidity_history"]] <- feature_names[startsWith(feature_names, "rh_minus")]
  
  # Cyclic encoding pairs (sin/cos)
  cyclic_bases <- c("Sowing_date_yday", "Fertiliser_application_time_1_yday", 
                    "Fungicide_application_time_1_yday", "Harvest_yday")
  for (base in cyclic_bases) {
    sin_var <- paste0(base, "_sin")
    cos_var <- paste0(base, "_cos")
    if (all(c(sin_var, cos_var) %in% feature_names)) {
      groups[[base]] <- c(sin_var, cos_var)
    }
  }
  
  # Find ungrouped features
  grouped_features <- unlist(groups)
  ungrouped <- setdiff(feature_names, grouped_features)
  
  # Add ungrouped features as individual groups
  for (feature in ungrouped) {
    groups[[feature]] <- feature
  }
  
  return(groups)
}

# Modified permutation importance function with grouping
grouped_permutation_importance <- function(model, x_test, y_test, loss_function, 
                                           output_index, n_rounds = 5, task_name = "task") {
  
  cat("Calculating grouped permutation importance for", task_name, "...\n")
  
  # Get baseline performance
  baseline_pred <- predict(model, x_test, verbose = 0)
  baseline_pred_matrix <- as.matrix(baseline_pred[[output_index]])
  baseline_loss <- as.numeric(loss_function(
    tf$constant(y_test, dtype = tf$float32), 
    tf$constant(baseline_pred_matrix, dtype = tf$float32)
  ))
  
  cat("Baseline", task_name, "loss:", round(baseline_loss, 6), "\n")
  
  # Create feature groups
  feature_names <- colnames(x_test)
  feature_groups <- create_feature_groups(feature_names)
  
  cat("Created", length(feature_groups), "feature groups\n")
  cat("Groups with multiple features:\n")
  multi_groups <- feature_groups[sapply(feature_groups, length) > 1]
  for (i in seq_along(multi_groups)) {
    cat("  ", names(multi_groups)[i], ":", length(multi_groups[[i]]), "features\n")
  }
  cat("\n")
  
  # Initialize results
  group_names <- names(feature_groups)
  n_groups <- length(group_names)
  importance_scores <- numeric(n_groups)
  names(importance_scores) <- group_names
  
  # Create progress bar
  pb <- txtProgressBar(min = 0, max = n_groups, style = 3, width = 50)
  cat("Progress:\n")
  
  # Calculate importance for each group
  for (i in seq_along(group_names)) {
    group_name <- group_names[i]
    group_features <- feature_groups[[group_name]]
    
    round_losses <- numeric(n_rounds)
    
    # Multiple permutation rounds for stability
    for (round in 1:n_rounds) {
      # Create copy of test data
      x_test_perm <- x_test
      
      # Permute all features in the group together
      set.seed(1701 + round + i)  # Reproducible but different for each round/group
      perm_indices <- sample(nrow(x_test_perm))
      
      # Apply the same permutation to all features in the group
      for (feature in group_features) {
        feature_col <- which(colnames(x_test_perm) == feature)
        x_test_perm[, feature_col] <- x_test_perm[perm_indices, feature_col]
      }
      
      # Get predictions with permuted group
      perm_pred <- predict(model, x_test_perm, verbose = 0)
      perm_pred_matrix <- as.matrix(perm_pred[[output_index]])
      
      # Calculate loss with permuted group
      round_losses[round] <- as.numeric(loss_function(
        tf$constant(y_test, dtype = tf$float32), 
        tf$constant(perm_pred_matrix, dtype = tf$float32)
      ))
    }
    
    # Average loss across rounds
    avg_perm_loss <- mean(round_losses)
    
    # Calculate percentage increase in loss
    importance_scores[i] <- ((avg_perm_loss - baseline_loss) / baseline_loss) * 100
    
    # Update progress bar
    setTxtProgressBar(pb, i)
  }
  
  # Close progress bar
  close(pb)
  cat("\n")
  
  # Create results dataframe
  results <- data.frame(
    group = group_names,
    n_features = sapply(feature_groups, length),
    importance_pct = importance_scores,
    stringsAsFactors = FALSE
  ) %>%
    arrange(desc(importance_pct))
  
  cat("Completed", task_name, "grouped permutation importance!\n\n")
  return(results)
}

# Calculate grouped permutation importance for regression task
cat("=== GROUPED REGRESSION TASK IMPORTANCE ===\n")
regression_grouped_importance <- grouped_permutation_importance(
  model = model,
  x_test = x_test,
  y_test = y_cont_test,
  loss_function = masked_mse,
  output_index = 1,  # cont_out is first output
  n_rounds = 5,
  task_name = "regression"
)

# Calculate grouped permutation importance for classification task  
cat("=== GROUPED CLASSIFICATION TASK IMPORTANCE ===\n")
classification_grouped_importance <- grouped_permutation_importance(
  model = model,
  x_test = x_test,
  y_test = y_bin_test,
  loss_function = masked_bce,
  output_index = 2,  # bin_out is second output
  n_rounds = 5,
  task_name = "classification"
)


library(dplyr)
library(stringr)

# Function to clean group names
clean_group_names <- function(df) {
  df |>
    mutate(group_clean = group |>
             str_replace_all("_", " ") |>
             str_replace_all("\\bYear\\b", "Year") |>
             str_replace_all("\\bVariety\\b", "Variety") |>
             str_replace_all("\\bRotation\\b", "Rotation") |>
             str_replace_all("\\bSoil pH\\b", "Soil pH") |>
             str_replace_all("\\bFungicide dose\\b", "Fungicide dose") |>
             str_replace_all("\\bHerbicide dose\\b", "Herbicide dose") |>
             str_replace_all("\\bFungicide product\\b", "Fungicide product") |>
             str_replace_all("\\bMicronutrients product\\b", "Micronutrients product") |>
             str_replace_all("\\bFertiliser product\\b", "Fertiliser product") |>
             str_replace_all("\\bPrevious year (\\d+)\\b", "Previous year \\1") |>
             str_replace_all("\\brainfall history\\b", "Rainfall history") |>
             str_replace_all("\\btemperature history\\b", "Temperature history") |>
             str_replace_all("\\bhumidity history\\b", "Relative humidity history") )
}

classification_grouped_importance <- clean_group_names(classification_grouped_importance)
regression_grouped_importance     <- clean_group_names(regression_grouped_importance)

# remoce Crop row from both datasets
classification_grouped_importance <- classification_grouped_importance %>%
  filter(group_clean != "Crop")
regression_grouped_importance <- regression_grouped_importance %>%
  filter(group_clean != "Crop")

# Classification grouped importance plot
ggplot(data = classification_grouped_importance[1:10,], 
       aes(x = reorder(group_clean, importance_pct), y = importance_pct)) +
  geom_col(fill = "coral", col = 'black', alpha = 0.8, width = 0.7) +
  coord_flip() +
  labs(
    x = "Feature Groups",
    y = "Importance (% increase in loss)") +
  theme_bw() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        axis.text.y = element_text(size = 11),
        axis.text.x = element_text(size = 10),
        panel.grid.minor = element_blank())

# Regression grouped importance plot
ggplot(data = regression_grouped_importance[1:10,], 
       aes(x = reorder(group_clean, importance_pct), y = importance_pct)) +
  geom_col(fill = "steelblue", col = 'black', alpha = 0.8, width = 0.7) +
  coord_flip() +
  labs(
    x = "Feature Groups",
    y = "Importance (% increase in loss)") +
  theme_bw() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        axis.text.y = element_text(size = 11),
        axis.text.x = element_text(size = 10),
        panel.grid.minor = element_blank())

# find vars common to both and plot on same plot
vars_common <- intersect(
  regression_grouped_importance$group_clean,
  classification_grouped_importance$group_clean
)

# plot top 10
regression_common <- regression_grouped_importance %>%
  filter(group_clean %in% vars_common) %>%
  top_n(10, importance_pct)

classification_common <- classification_grouped_importance %>%
  filter(group_clean %in% vars_common) %>%
  top_n(10, importance_pct)

combined_common <- merge(regression_common, classification_common, 
                         by = "group_clean", 
                         suffixes = c("_regression", "_classification"))
ggplot(data = combined_common,
       aes(x = reorder(group_clean, importance_pct_regression))) +
  geom_col(aes(y = importance_pct_regression, fill = "Regression"), 
           position = position_nudge(x = -0.1), width = 0.2, col = 'black', alpha = 0.8) +
  geom_col(aes(y = importance_pct_classification, fill = "Classification"), 
           position = position_nudge(x = 0.1), width = 0.2, col = 'black', alpha = 0.8) +
  coord_flip() +
  labs(
    x = "Feature Groups",
    y = "Importance (% increase in loss)",
    fill = "Task") +
  scale_fill_manual(values = c("Regression" = "steelblue", "Classification" = "coral")) +
  theme_bw() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        axis.text.y = element_text(size = 12),
        axis.text.x = element_text(size = 10),
        panel.grid.minor = element_blank())


