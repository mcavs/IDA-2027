suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(dplyr)
})

# Load the DALEX computed profiles
res <- readRDS("outputs/real_world/dalex_fig8_results.rds")

# Prepare Panel (a): Adult (age)
df_adult_lines <- data.frame(
  x = rep(res$adult$grid, 2),
  y = c(res$adult$rash_mean, res$adult$best_pdp),
  Method = rep(c("Rashomon PDP", "Best Model PDP"), each = length(res$adult$grid)),
  stringsAsFactors = FALSE
)

df_adult_ribbon <- data.frame(
  x = res$adult$grid,
  ymin = res$adult$ci_lower,
  ymax = res$adult$ci_upper,
  Ribbon = "95% Confidence Interval"
)

# Prepare Panel (b): PC3 (DESIGN_COMPLEXITY)
df_pc3_lines <- data.frame(
  x = rep(res$pc3$grid, 2),
  y = c(res$pc3$rash_mean, res$pc3$best_pdp),
  Method = rep(c("Rashomon PDP", "Best Model PDP"), each = length(res$pc3$grid)),
  stringsAsFactors = FALSE
)

df_pc3_ribbon <- data.frame(
  x = res$pc3$grid,
  ymin = res$pc3$ci_lower,
  ymax = res$pc3$ci_upper,
  Ribbon = "95% Confidence Interval"
)

# Professional academic palette matching Cavus et al. (2026) / IDA style:
# Rashomon PDP: Dark Orange / Amber (#D95F02 or #E66101)
# Best Model PDP: Solid Dark Grey / Black (#1A1A1A or #2B83BA)
# Ribbon: Soft Amber / Orange fill (#FEE0B6 or #FDB863 with alpha = 0.35)

line_colors <- c(
  "Rashomon PDP"   = "#E66101",
  "Best Model PDP" = "#1A1A1A"
)

line_types <- c(
  "Rashomon PDP"   = "solid",
  "Best Model PDP" = "dashed"
)

line_widths <- c(
  "Rashomon PDP"   = 1.4,
  "Best Model PDP" = 1.3
)

# Plot Panel (a): Adult
p_a <- ggplot() +
  geom_ribbon(
    data = df_adult_ribbon,
    aes(x = x, ymin = ymin, ymax = ymax, fill = Ribbon),
    alpha = 0.28
  ) +
  geom_line(
    data = df_adult_lines,
    aes(x = x, y = y, color = Method, linetype = Method, linewidth = Method)
  ) +
  geom_point(
    data = df_adult_lines,
    aes(x = x, y = y, color = Method),
    size = 2.0
  ) +
  scale_color_manual(values = line_colors) +
  scale_linetype_manual(values = line_types) +
  scale_linewidth_manual(values = line_widths) +
  scale_fill_manual(values = c("95% Confidence Interval" = "#FDB863")) +
  scale_y_continuous(
    limits = c(0.0, 0.42),
    breaks = seq(0.0, 0.40, by = 0.10)
  ) +
  labs(
    title = expression(bold("(a) Adult Dataset: Effect of ") * bolditalic("age") * bold(" on Income > 50K")),
    subtitle = "Divergence between Best Model and Rashomon PDP (low coverage, wide uncertainty)",
    x = "Age (years)",
    y = expression(paste("Predicted Probability ", P(Y == 1 ~ "|" ~ x)))
  ) +
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 13.5),
    plot.subtitle = element_text(size = 11, color = "#444444"),
    panel.grid.minor = element_blank(),
    legend.position = "none",
    axis.title = element_text(face = "bold", size = 12)
  )

# Plot Panel (b): PC3
p_b <- ggplot() +
  geom_ribbon(
    data = df_pc3_ribbon,
    aes(x = x, ymin = ymin, ymax = ymax, fill = Ribbon),
    alpha = 0.28
  ) +
  geom_line(
    data = df_pc3_lines,
    aes(x = x, y = y, color = Method, linetype = Method, linewidth = Method)
  ) +
  geom_point(
    data = df_pc3_lines,
    aes(x = x, y = y, color = Method),
    size = 2.0
  ) +
  scale_color_manual(values = line_colors) +
  scale_linetype_manual(values = line_types) +
  scale_linewidth_manual(values = line_widths) +
  scale_fill_manual(values = c("95% Confidence Interval" = "#FDB863")) +
  scale_y_continuous(
    limits = c(0.03, 0.16),
    breaks = seq(0.04, 0.16, by = 0.02)
  ) +
  labs(
    title = expression(bold("(b) PC3 Dataset: Effect of ") * bolditalic("DESIGN_COMPLEXITY") * bold(" on Defect")),
    subtitle = "Consistent explanations: Best Model PDP lies fully inside the Rashomon confidence interval",
    x = "Design Complexity (McCabe)",
    y = expression(paste("Predicted Probability ", P(Y == 1 ~ "|" ~ x)))
  ) +
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 13.5),
    plot.subtitle = element_text(size = 11, color = "#444444"),
    panel.grid.minor = element_blank(),
    legend.position = "none",
    axis.title = element_text(face = "bold", size = 12)
  )

# Combine with unified legend at bottom
p_combined <- (p_a | p_b) +
  plot_layout(guides = "collect") &
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 12.5),
    legend.key.width = unit(1.8, "cm"),
    legend.box.margin = margin(t = 6, r = 0, b = 2, l = 0)
  ) &
  guides(
    color = guide_legend(order = 1, override.aes = list(linewidth = 2.0, size = 3.5)),
    linetype = guide_legend(order = 1),
    linewidth = "none",
    fill = guide_legend(order = 2, override.aes = list(alpha = 0.35))
  )

# Output paths
out_pdf <- "figures/real_world_classification_figure.pdf"
out_png <- "figures/real_world_classification_figure.png"
ggsave(out_pdf, plot = p_combined, width = 14.5, height = 5.8, device = cairo_pdf)
ggsave(out_png, plot = p_combined, width = 14.5, height = 5.8, dpi = 300)
ggsave("latex/figures/real_world_classification_figure.pdf", plot = p_combined, width = 14.5, height = 5.8, device = cairo_pdf)

cat("Successfully generated Figure 3 reproducing Cavus et al. (2026) Figure 8 with DALEX!\n")
