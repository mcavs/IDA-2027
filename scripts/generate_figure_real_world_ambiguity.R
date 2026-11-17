suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
})

clean_method_name <- function(m) {
  recode(m,
    "B1 (Single Best)"        = "Single Best (B1)",
    "B0 (Simple Avg)"         = "Simple Average (B0)",
    "B0 (Simple Average)"     = "Simple Average (B0)",
    "B2 (Perf Weighted)"      = "Performance-Weighted (B2)",
    "M1 (Softmax BMA)"        = "Softmax BMA (M1)",
    "M2 (Pointwise Median)"   = "Pointwise Median (M2)",
    "M3 (Trimmed Mean)"       = "Trimmed Mean (M3)",
    "M4 (Fréchet Median)"     = "Fréchet Median (M4)",
    "M5 (Pairwise Alignment)" = "Pairwise Alignment (M5)"
  )
}

# Panel (a): Customer Churn (total_day_calls)
m5_data <- fromJSON("outputs/real_world_5datasets/real_world_5datasets_master.json")
low_data <- m5_data$churn$features$total_day_calls
low_grid <- low_data$grid_vals

low_df_list <- list()
for (m in names(low_data$curves)) {
  low_df_list[[m]] <- data.frame(
    FeatureVal = low_grid,
    Prob = low_data$curves[[m]],
    Method = clean_method_name(m),
    stringsAsFactors = FALSE
  )
}
df_low <- bind_rows(low_df_list)

# Panel (b): Blood Transfusion (V2 / Frequency)
final_curves <- fromJSON("outputs/real_world/final_figure3_curves.json")
high_data <- final_curves$panel_a
high_grid <- high_data$grid

high_df_list <- list()
for (m in names(high_data$curves)) {
  high_df_list[[m]] <- data.frame(
    FeatureVal = high_grid,
    Prob = high_data$curves[[m]],
    Method = clean_method_name(m),
    stringsAsFactors = FALSE
  )
}
df_high <- bind_rows(high_df_list)

# Distinct 8-color palette in Açık Yöntem İsmi (Kısaltma) format
method_colors <- c(
  "Single Best (B1)"          = "#000000", # Solid Black
  "Simple Average (B0)"       = "#1f77b4", # Blue
  "Performance-Weighted (B2)" = "#2ca02c", # Green
  "Softmax BMA (M1)"          = "#ff7f0e", # Orange
  "Pointwise Median (M2)"     = "#9467bd", # Purple
  "Trimmed Mean (M3)"         = "#8c564b", # Brown
  "Fréchet Median (M4)"       = "#e377c2", # Pink
  "Pairwise Alignment (M5)"   = "#17becf"  # Cyan
)

# Plot Panel (a)
p_low <- ggplot(df_low, aes(x = FeatureVal, y = Prob, color = Method, group = Method)) +
  geom_line(linewidth = 1.2, linetype = "solid") +
  scale_color_manual(values = method_colors) +
  scale_y_continuous(limits = c(0.12, 0.16), breaks = seq(0.12, 0.16, by = 0.01)) +
  labs(
    title = "(a) Customer Churn",
    x = "Total Day Calls",
    y = "Predicted Probability"
  ) +
  theme_bw(base_size = 20) +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    plot.subtitle = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "none",
    axis.title.x = element_text(face = "plain", size = 17),
    axis.title.y = element_text(face = "plain", size = 17),
    axis.text = element_text(size = 15)
  )

# Plot Panel (b)
p_high <- ggplot(df_high, aes(x = FeatureVal, y = Prob, color = Method, group = Method)) +
  geom_line(linewidth = 1.2, linetype = "solid") +
  scale_color_manual(values = method_colors) +
  scale_y_continuous(limits = c(0.15, 0.82), breaks = seq(0.2, 0.8, by = 0.1)) +
  labs(
    title = "(b) Blood Transfusion",
    x = "Total Number of Donations",
    y = "Predicted Probability"
  ) +
  theme_bw(base_size = 20) +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    plot.subtitle = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "none",
    axis.title.x = element_text(face = "plain", size = 17),
    axis.title.y = element_text(face = "plain", size = 17),
    axis.text = element_text(size = 15)
  )

# Combine with unified legend at bottom
p_combined <- (p_low | p_high) + 
  plot_layout(guides = "collect") & 
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 16),
    legend.key.width = unit(2.2, "cm"),
    legend.box.margin = margin(t = 10, r = 0, b = 4, l = 0)
  ) &
  guides(color = guide_legend(nrow = 2, byrow = TRUE, override.aes = list(linewidth = 2.5)))

# Save to all target locations
pdf_targets <- c(
  "figures/real_world_ambiguity_figure.pdf",
  "figures/real_world_classification_figure.pdf",
  "figures/real_world_ambiguity_figure.pdf",
  "latex/figures/real_world_ambiguity_figure.pdf",
  "latex/figures/real_world_classification_figure.pdf"
)

png_targets <- c(
  "figures/real_world_ambiguity_figure.png",
  "figures/real_world_classification_figure.png",
  "figures/real_world_ambiguity_figure.png",
  "latex/figures/real_world_ambiguity_figure.png",
  "latex/figures/real_world_classification_figure.png"
)

for (p in pdf_targets) {
  ggsave(p, plot = p_combined, width = 16.0, height = 6.8, device = cairo_pdf)
}

for (p in png_targets) {
  ggsave(p, plot = p_combined, width = 16.0, height = 6.8, dpi = 300)
}

cat("Successfully generated cleaned figure with minimalist titles, plain axis labels, and no subtitles!\n")
