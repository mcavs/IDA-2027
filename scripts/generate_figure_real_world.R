suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
})

data <- fromJSON("outputs/real_world/final_figure3_curves.json")

method_colors <- c(
  "B1 (Single Best)"        = "#111111",
  "B0 (Simple Average)"     = "#1f77b4",
  "B2 (Perf Weighted)"      = "#2ca02c",
  "M1 (Softmax BMA)"        = "#ff7f0e",
  "M2 (Pointwise Median)"   = "#9467bd",
  "M3 (Trimmed Mean)"       = "#8c564b",
  "M4 (Fréchet Median)"     = "#e377c2",
  "M5 (Pairwise Alignment)" = "#17becf"
)

# Extract Panel A: Blood Transfusion
pa_data <- data$panel_a
pa_grid <- pa_data$grid
pa_df_list <- list()
for (m in names(pa_data$curves)) {
  pa_df_list[[m]] <- data.frame(
    FeatureVal = pa_grid,
    Prob = pa_data$curves[[m]],
    Method = m,
    stringsAsFactors = FALSE
  )
}
df_pa <- bind_rows(pa_df_list)

# Extract Panel B: Bank Marketing
pb_data <- data$panel_b
pb_grid <- pb_data$grid
pb_df_list <- list()
for (m in names(pb_data$curves)) {
  pb_df_list[[m]] <- data.frame(
    FeatureVal = pb_grid,
    Prob = pb_data$curves[[m]],
    Method = m,
    stringsAsFactors = FALSE
  )
}
df_pb <- bind_rows(pb_df_list)

# Plot Panel A
p_a <- ggplot(df_pa, aes(x = FeatureVal, y = Prob, color = Method, group = Method)) +
  geom_line(aes(linetype = (Method == "B1 (Single Best)"), linewidth = (Method == "B1 (Single Best)"))) +
  geom_point(size = 2.2, alpha = 0.85) +
  scale_color_manual(values = method_colors) +
  scale_linetype_manual(values = c("TRUE" = "dashed", "FALSE" = "solid"), guide = "none") +
  scale_linewidth_manual(values = c("TRUE" = 1.7, "FALSE" = 1.1), guide = "none") +
  scale_y_continuous(limits = c(0.12, 0.82), breaks = seq(0.2, 0.8, by = 0.1)) +
  labs(
    title = expression(bold("(a) Blood Transfusion: Donation Frequency (") * bolditalic(V2) * bold(")")),
    subtitle = "Single best model (B1, dashed) diverges sharply; all aggregations establish a stable plateau",
    x = "Total Number of Donations (Frequency)",
    y = expression(paste("Predicted Probability ", P(Y == 1 ~ "|" ~ x)))
  ) +
  theme_bw(base_size = 15) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11.5, color = "#444444"),
    panel.grid.minor = element_blank(),
    legend.position = "none",
    axis.title = element_text(face = "bold", size = 13)
  )

# Plot Panel B
p_b <- ggplot(df_pb, aes(x = FeatureVal, y = Prob, color = Method, group = Method)) +
  geom_line(aes(linetype = (Method == "B1 (Single Best)"), linewidth = (Method == "B1 (Single Best)"))) +
  geom_point(size = 2.2, alpha = 0.85) +
  scale_color_manual(values = method_colors) +
  scale_linetype_manual(values = c("TRUE" = "dashed", "FALSE" = "solid"), guide = "none") +
  scale_linewidth_manual(values = c("TRUE" = 1.7, "FALSE" = 1.1), guide = "none") +
  scale_y_continuous(limits = c(0.10, 0.20), breaks = seq(0.10, 0.20, by = 0.02)) +
  labs(
    title = expression(bold("(b) Bank Marketing: Average Yearly Balance (") * bolditalic(V6) * bold(")")),
    subtitle = "Best model (B1) and Simple Average (B0) pull away from robust medians at high balance tail",
    x = "Yearly Balance (EUR)",
    y = expression(paste("Predicted Probability ", P(Y == 1 ~ "|" ~ x)))
  ) +
  theme_bw(base_size = 15) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11.5, color = "#444444"),
    panel.grid.minor = element_blank(),
    legend.position = "none",
    axis.title = element_text(face = "bold", size = 13)
  )

p_combined <- (p_a | p_b) + 
  plot_layout(guides = "collect") & 
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 12.5),
    legend.key.width = unit(1.6, "cm"),
    legend.box.margin = margin(t = 8, r = 0, b = 2, l = 0)
  ) &
  guides(color = guide_legend(nrow = 2, byrow = TRUE, override.aes = list(linewidth = 2.2, size = 4.0)))

out_pdf <- "figures/real_world_classification_figure.pdf"
out_png <- "figures/real_world_classification_figure.png"
ggsave(out_pdf, plot = p_combined, width = 15.0, height = 6.2, device = cairo_pdf)
ggsave(out_png, plot = p_combined, width = 15.0, height = 6.2, dpi = 300)
ggsave("latex/figures/real_world_classification_figure.pdf", plot = p_combined, width = 15.0, height = 6.2, device = cairo_pdf)

cat("Successfully generated updated Figure 3 (Blood Transfusion + Bank Marketing)!\n")
