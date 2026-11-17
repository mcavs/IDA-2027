library(ggplot2)
library(reticulate)
library(patchwork)

np <- import("numpy")
py <- import_builtins()
pickle <- import("pickle")

f <- py$open("outputs/real_world_pdp_results.pkl", "rb")
d <- pickle$load(f)
f$close()

make_df <- function(res, dataset_name, feature_label, y_label) {
  grid <- as.numeric(res$grid_vals)
  best <- as.numeric(res$best_pdp)
  
  df_list <- list()
  
  # Best model
  df_list[[1]] <- data.frame(
    x = grid,
    y = best,
    Method = "Single Best Model",
    Type = "Baseline",
    Dataset = dataset_name
  )
  
  # Aggregations
  aggs <- res$aggregations
  method_names <- names(aggs)
  
  for (m in method_names) {
    df_list[[length(df_list) + 1]] <- data.frame(
      x = grid,
      y = as.numeric(aggs[[m]]),
      Method = m,
      Type = "Aggregation",
      Dataset = dataset_name
    )
  }
  
  df <- do.call(rbind, df_list)
  return(df)
}

df_ee <- make_df(d$energy_efficiency, "Energy Efficiency (surface_area)", "Surface Area", "Partial Dependence (Heating Load)")
df_ff <- make_df(d$forest_fires, "Forest Fires (temp)", "Temperature", "Partial Dependence (Burned Area)")

# Method colors matching academic palette
palette <- c(
  "Single Best Model"    = "#000000",
  "Simple Average"       = "#1F77B4",
  "Performance-Weighted" = "#2CA02C",
  "Softmax BMA"          = "#FF7F0E",
  "Pointwise Median"     = "#9467BD",
  "Trimmed Mean"         = "#8C564B",
  "Functional Median"    = "#D62728",
  "Alignment Consensus"  = "#17BECF"
)

linetypes <- c(
  "Single Best Model"    = "dashed",
  "Simple Average"       = "solid",
  "Performance-Weighted" = "solid",
  "Softmax BMA"          = "solid",
  "Pointwise Median"     = "solid",
  "Trimmed Mean"         = "solid",
  "Functional Median"    = "solid",
  "Alignment Consensus"  = "solid"
)

sizes <- c(
  "Single Best Model"    = 1.1,
  "Simple Average"       = 0.85,
  "Performance-Weighted" = 0.85,
  "Softmax BMA"          = 0.85,
  "Pointwise Median"     = 0.85,
  "Trimmed Mean"         = 0.85,
  "Functional Median"    = 1.0,
  "Alignment Consensus"  = 0.85
)

# Plot A: Energy Efficiency
p1 <- ggplot(df_ee, aes(x = x, y = y, color = Method, linetype = Method, size = Method)) +
  geom_line() +
  scale_color_manual(values = palette) +
  scale_linetype_manual(values = linetypes) +
  scale_size_manual(values = sizes) +
  labs(
    title = "(a) High Consensus: Energy Efficiency",
    subtitle = "Surface Area effect on Heating Load",
    x = "Surface Area",
    y = "Partial Dependence (Heating Load)"
  ) +
  theme_bw(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(size = 11, color = "gray30"),
    panel.grid.minor = element_blank(),
    legend.position = "none"
  )

# Plot B: Forest Fires
p2 <- ggplot(df_ff, aes(x = x, y = y, color = Method, linetype = Method, size = Method)) +
  geom_line() +
  scale_color_manual(values = palette) +
  scale_linetype_manual(values = linetypes) +
  scale_size_manual(values = sizes) +
  labs(
    title = "(b) High Multiplicity Disagreement: Forest Fires",
    subtitle = "Temperature effect on Burned Area",
    x = "Temperature (C)",
    y = "Partial Dependence (Area)"
  ) +
  theme_bw(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(size = 11, color = "gray30"),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.title = element_blank()
  )

combined_plot <- p1 + p2 + plot_layout(ncol = 2, widths = c(1, 1.35))

ggsave("latex/figures/real_world_figure6_reproduced.pdf", combined_plot, width = 11, height = 4.5, device = cairo_pdf)
ggsave("latex/figures/real_world_figure6_reproduced.png", combined_plot, width = 11, height = 4.5, dpi = 300)
ggsave("figures/real_world_figure6_reproduced.pdf", combined_plot, width = 11, height = 4.5, device = cairo_pdf)
ggsave("figures/real_world_figure6_reproduced.png", combined_plot, width = 11, height = 4.5, dpi = 300)

cat("Successfully generated and saved real_world_figure6_reproduced.pdf in latex/figures and figures!\n")
