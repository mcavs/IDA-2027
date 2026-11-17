suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
})

data <- fromJSON("outputs/real_world/real_world_classification_results.json")

method_colors <- c(
  "B1 (Single Best)"        = "#333333",
  "B0 (Simple Avg)"         = "#1f77b4",
  "B2 (Perf Weighted)"      = "#2ca02c",
  "M1 (Softmax BMA)"        = "#ff7f0e",
  "M2 (Pointwise Median)"   = "#9467bd",
  "M3 (Trimmed Mean)"       = "#8c564b",
  "M4 (Fréchet Median)"     = "#e377c2",
  "M5 (Pairwise Alignment)" = "#17becf"
)

# Extract curves for Adult
adult_data <- data$adult
adult_grid <- adult_data$grid_vals
adult_df_list <- list()
for (m in names(adult_data$curves)) {
  adult_df_list[[m]] <- data.frame(
    Dataset = "Panel~(a):~Adult~(Income > 50K)",
    FeatureVal = adult_grid,
    Prob = adult_data$curves[[m]],
    Method = m,
    stringsAsFactors = FALSE
  )
}
df_adult <- bind_rows(adult_df_list)

# Extract curves for Churn
churn_data <- data$churn
churn_grid <- churn_data$grid_vals
churn_df_list <- list()
for (m in names(churn_data$curves)) {
  churn_df_list[[m]] <- data.frame(
    Dataset = "Panel~(b):~Customer~Churn~(Total~Day~Minutes)",
    FeatureVal = churn_grid,
    Prob = churn_data$curves[[m]],
    Method = m,
    stringsAsFactors = FALSE
  )
}
df_churn <- bind_rows(churn_df_list)

# Combine datasets for 1x2 plot
df_all <- bind_rows(df_adult, df_churn)

# Plot
p_adult <- ggplot(df_adult, aes(x = FeatureVal, y = Prob, color = Method, group = Method)) +
  geom_line(aes(linetype = (Method == "B1 (Single Best)"), linewidth = (Method == "B1 (Single Best)"))) +
  geom_point(size = 2.5) +
  scale_color_manual(values = method_colors) +
  scale_linetype_manual(values = c("TRUE" = "dashed", "FALSE" = "solid"), guide = "none") +
  scale_linewidth_manual(values = c("TRUE" = 1.6, "FALSE" = 1.1), guide = "none") +
  labs(
    title = expression(bold("(a) Adult Dataset: Effect of Age on Income > 50K")),
    subtitle = "Single best model (B1) diverges at retirement age (>55); aggregation methods provide robust consensus",
    x = "Age (years)",
    y = expression(paste("Predicted Probability ", P(Y == 1 ~ "|" ~ x)))
  ) +
  theme_bw(base_size = 15) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(size = 12, color = "#444444"),
    panel.grid.minor = element_blank(),
    legend.position = "none",
    axis.title = element_text(face = "bold", size = 13)
  )

p_churn <- ggplot(df_churn, aes(x = FeatureVal, y = Prob, color = Method, group = Method)) +
  geom_line(aes(linetype = (Method == "B1 (Single Best)"), linewidth = (Method == "B1 (Single Best)"))) +
  geom_point(size = 2.5) +
  scale_color_manual(values = method_colors) +
  scale_linetype_manual(values = c("TRUE" = "dashed", "FALSE" = "solid"), guide = "none") +
  scale_linewidth_manual(values = c("TRUE" = 1.6, "FALSE" = 1.1), guide = "none") +
  labs(
    title = expression(bold("(b) Churn Dataset: Effect of Total Day Minutes on Churn")),
    subtitle = "High day minutes trigger sharp divergence; BMA and Fréchet median dampen edge fluctuations",
    x = "Total Day Minutes",
    y = expression(paste("Predicted Probability ", P(Y == 1 ~ "|" ~ x)))
  ) +
  theme_bw(base_size = 15) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(size = 12, color = "#444444"),
    panel.grid.minor = element_blank(),
    legend.position = "none",
    axis.title = element_text(face = "bold", size = 13)
  )

suppressPackageStartupMessages(library(patchwork))

p_combined <- (p_adult | p_churn) + 
  plot_layout(guides = "collect") & 
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 13),
    legend.key.width = unit(1.8, "cm"),
    legend.box.margin = margin(t = 10, r = 0, b = 4, l = 0)
  ) &
  guides(color = guide_legend(nrow = 2, byrow = TRUE, override.aes = list(linewidth = 2.2, size = 4.5)))

out_pdf <- "figures/real_world_classification_figure.pdf"
out_png <- "figures/real_world_classification_figure.png"
ggsave(out_pdf, plot = p_combined, width = 15.0, height = 6.2, device = cairo_pdf)
ggsave(out_png, plot = p_combined, width = 15.0, height = 6.2, dpi = 300)
ggsave("latex/figures/real_world_classification_figure.pdf", plot = p_combined, width = 15.0, height = 6.2, device = cairo_pdf)

cat("Successfully generated real-world classification Figure 3 in PDF and PNG!\n")
