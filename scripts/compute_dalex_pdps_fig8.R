suppressPackageStartupMessages({
  library(reticulate)
  library(DALEX)
  library(jsonlite)
  library(ggplot2)
  library(patchwork)
  library(dplyr)
})

use_python("/opt/homebrew/bin/python3", required = TRUE)
h2o <- import("h2o")
h2o$init()
h2o$no_progress()

meta <- fromJSON("outputs/real_world/rashomon_model_ids.json")

# Helper function to compute PDPs with DALEX::model_profile
run_dalex_profiles <- function(csv_path, target_col, feature_col, model_info, grid_points, N_sample = 250) {
  df <- read.csv(csv_path, stringsAsFactors = FALSE)
  df_x <- df[, setdiff(names(df), target_col)]
  y_raw <- df[[target_col]]
  
  pred_fun <- function(m, newdata) {
    hf_new <- h2o$H2OFrame(r_to_py(newdata))
    p <- m$predict(hf_new)$as_data_frame()
    if ("p1" %in% colnames(p)) {
      return(as.numeric(p[["p1"]]))
    }
    return(as.numeric(p[, ncol(p)]))
  }
  
  # Best model profile
  cat(paste("Computing DALEX model_profile for Best Model:", model_info$best_model_id, "\n"))
  best_m <- h2o$get_model(model_info$best_model_id)
  exp_best <- explain.default(
    model = best_m,
    data = df_x,
    y = as.numeric(as.factor(y_raw)) - 1,
    predict_function = pred_fun,
    label = "Best Model",
    verbose = FALSE
  )
  
  split_list <- list()
  split_list[[feature_col]] <- grid_points
  
  prof_best <- model_profile(
    exp_best,
    variables = feature_col,
    variable_splits = split_list,
    N = N_sample,
    type = "partial"
  )
  best_pdp <- prof_best$agr_profiles$`_yhat_`
  
  # Rashomon set models
  cat(paste("Computing DALEX model_profile for", length(model_info$rashomon_model_ids), "Rashomon models...\n"))
  rash_pdps <- list()
  for (i in seq_along(model_info$rashomon_model_ids)) {
    mid <- model_info$rashomon_model_ids[i]
    m <- h2o$get_model(mid)
    exp_m <- explain.default(
      model = m,
      data = df_x,
      y = as.numeric(as.factor(y_raw)) - 1,
      predict_function = pred_fun,
      label = paste0("Model_", i),
      verbose = FALSE
    )
    prof_m <- model_profile(
      exp_m,
      variables = feature_col,
      variable_splits = split_list,
      N = N_sample,
      type = "partial"
    )
    rash_pdps[[i]] <- prof_m$agr_profiles$`_yhat_`
  }
  
  P <- do.call(rbind, rash_pdps) # shape K x length(grid_points)
  
  # Rashomon PDP is pointwise mean
  rash_mean <- colMeans(P)
  
  # 95% Confidence Interval across Rashomon models (Equation 9 in Cavus et al. 2026)
  ci_lower <- apply(P, 2, quantile, probs = 0.025)
  ci_upper <- apply(P, 2, quantile, probs = 0.975)
  
  # Summary uncertainty metrics
  mwci <- mean(ci_upper - ci_lower)
  cr <- mean(best_pdp >= ci_lower & best_pdp <= ci_upper)
  
  cat(paste("  -> MWCI =", round(mwci, 4), "CR =", round(cr, 4), "\n"))
  
  return(list(
    grid = grid_points,
    best_pdp = best_pdp,
    rash_mean = rash_mean,
    ci_lower = ci_lower,
    ci_upper = ci_upper,
    mwci = mwci,
    cr = cr,
    P = P,
    K = nrow(P)
  ))
}

# 1. Adult (feature: age)
grid_adult <- seq(18, 75, length.out = 25)
res_adult <- run_dalex_profiles(
  csv_path = "outputs/real_world/adult_subsample.csv",
  target_col = "class",
  feature_col = "age",
  model_info = meta$adult,
  grid_points = grid_adult,
  N_sample = 250
)

# 2. PC3 (feature: DESIGN_COMPLEXITY)
grid_pc3 <- seq(1, 20, length.out = 20)
res_pc3 <- run_dalex_profiles(
  csv_path = "outputs/real_world/pc3_subsample.csv",
  target_col = "c",
  feature_col = "DESIGN_COMPLEXITY",
  model_info = meta$pc3,
  grid_points = grid_pc3,
  N_sample = 250
)

saveRDS(list(adult = res_adult, pc3 = res_pc3), "outputs/real_world/dalex_fig8_results.rds")
cat("Successfully computed all DALEX partial dependence profiles!\n")
