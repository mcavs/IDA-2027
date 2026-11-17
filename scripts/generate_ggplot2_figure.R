#!/usr/bin/env Rscript
# Generate publication-quality 2x2 line plot using ggplot2 in R
# Combining Tables 3-4-5-6 (MD(GT) across 4 epsilon thresholds and 9 scenarios)
# User updates:
# 1. Siyah renkli çizgi (B1: Best-model) GRİ yapıldı (#666666 / #555555).
# 2. Y ekseni sıfırdan başlayacak şekilde (0 altta, 0.38 üstte) eski haline döndürüldü.
# 3. Setup etiket kutucukları panelin üst kısmına taşındı.

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
})

# Load master results
data_path <- "outputs/master_analysis_results_2880s_160m.json"
data <- fromJSON(data_path)

eps_list <- c("eps_0.01", "eps_0.05", "eps_0.1", "eps_0.2")
eps_labels <- c(
  "eps_0.01" = "epsilon == 0.01",
  "eps_0.05" = "epsilon == 0.05",
  "eps_0.1"  = "epsilon == 0.10",
  "eps_0.2"  = "epsilon == 0.20"
)

# Ordered methods
methods <- c(
  "B1:Best-model", "B0:SimpleAvg", "B2:PerfWeighted",
  "B3:AGGrobust",  "B4:AVA-PDP",    "B8:AGREE-Rank",
  "Ö1:AGGfaith",   "Ö2:AGGopt"
)

method_labels <- c(
  "B1:Best-model"   = "B1: Best-model",
  "B0:SimpleAvg"    = "B0: SimpleAvg",
  "B2:PerfWeighted" = "B2: PerfWeighted",
  "B3:AGGrobust"    = "B3: AGGrobust",
  "B4:AVA-PDP"      = "B4: AVA-PDP",
  "B8:AGREE-Rank"   = "B8: AGREE-Rank",
  "Ö1:AGGfaith"     = "Ö1: AGGfaith",
  "Ö2:AGGopt"       = "Ö2: AGGopt"
)

# Yüksek Kontrastlı Renk Skalası (B1 Gri):
method_colors <- c(
  "B1: Best-model"   = "#5A5A5A", # Zarif Koyu Gri (Slate Grey)
  "B0: SimpleAvg"    = "#0055D4", # Derin Canlı Mavi
  "B2: PerfWeighted" = "#009E49", # Zümrüt Yeşili
  "B3: AGGrobust"    = "#FF6F00", # Parlak Sıcak Turuncu
  "B4: AVA-PDP"      = "#8A2BE2", # Canlı Mor (BlueViolet)
  "B8: AGREE-Rank"   = "#7B3F00", # Koyu Kahve / Çikolata
  "Ö1: AGGfaith"     = "#00AEEF", # Parlak Turkuaz / Camgöbeği
  "Ö2: AGGopt"       = "#E01A22"  # Parlak Ateş Kırmızısı
)

# Distinct shapes for each method
method_shapes <- c(
  "B1: Best-model"   = 15, # Solid Square
  "B0: SimpleAvg"    = 16, # Solid Circle
  "B2: PerfWeighted" = 17, # Solid Triangle Up
  "B3: AGGrobust"    = 18, # Solid Diamond
  "B4: AVA-PDP"      = 25, # Solid Triangle Down
  "B8: AGREE-Rank"   = 8,  # Star
  "Ö1: AGGfaith"     = 3,  # Plus
  "Ö2: AGGopt"       = 4   # Cross
)

# Extract data into a long data frame
df_list <- list()

scenarios <- c()
for (s in 1:3) {
  for (nv in c(1, 4, 9)) {
    scenarios <- c(scenarios, paste0("S", s, "_nv", nv))
  }
}

for (eps in eps_list) {
  for (sc_idx in seq_along(scenarios)) {
    sc <- scenarios[sc_idx]
    s_num <- as.integer(substr(sc, 2, 2))
    nv_val <- as.integer(sub(".*nv", "", sc))
    
    for (m in methods) {
      val <- data[[sc]][["by_epsilon"]][[eps]][["results"]][[m]][["md"]]
      df_list[[length(df_list) + 1]] <- data.frame(
        epsilon = eps,
        scenario = sc,
        x_idx = sc_idx,
        setup = paste0("Setup ", s_num),
        method_raw = m,
        method = method_labels[m],
        md = val,
        stringsAsFactors = FALSE
      )
    }
  }
}

df <- bind_rows(df_list)
df$method <- factor(df$method, levels = unname(method_labels))
df$is_best <- (df$method == "B1: Best-model")

# Shading rectangles data for background
rect_data <- data.frame(
  xmin = c(0.5, 3.5, 6.5),
  xmax = c(3.5, 6.5, 9.5),
  shading = c("A", "B", "A"),
  setup_label = c("Setup 1 (Linear/Poly)", "Setup 2 (Trigonometric)", "Setup 3 (Nonlinear)"),
  x_label = c(2, 5, 8)
)

# X tick labels: only numeric values 1, 4, 9 repeated across setups
x_labels <- rep(c("1", "4", "9"), 3)

p <- ggplot(df, aes(x = x_idx, y = md, group = interaction(method, setup), color = method, shape = method)) +
  # Background shading for setups
  geom_rect(data = rect_data, aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf, fill = shading),
            inherit.aes = FALSE, show.legend = FALSE) +
  scale_fill_manual(values = c("A" = "#F4F7F9", "B" = "#FFFFFF")) +
  # Vertical dashed separator lines between setups
  geom_vline(xintercept = c(3.5, 6.5), linetype = "dashed", color = "#708090", alpha = 0.45, linewidth = 0.6) +
  # Setup headers placed at the TOP of the y-axis (y = 0.365)
  geom_label(data = rect_data, aes(x = x_label, y = 0.365, label = setup_label),
             inherit.aes = FALSE, size = 4.8, fontface = "bold", color = "#1A365D",
             fill = "white", linewidth = 0.3, label.padding = unit(0.25, "lines")) +
  # Solid lines with transparency for aggregation methods (alpha = 0.65)
  geom_line(data = filter(df, !is_best), linetype = "solid", linewidth = 1.4, alpha = 0.65) +
  geom_point(data = filter(df, !is_best), size = 3.3, stroke = 1.2, alpha = 0.80) +
  # Gri Best-model çizgisi (alpha = 0.65)
  geom_line(data = filter(df, is_best), linetype = "solid", linewidth = 1.6, alpha = 0.65) +
  geom_point(data = filter(df, is_best), size = 3.6, alpha = 0.80) +
  # Scales
  scale_color_manual(values = method_colors) +
  scale_shape_manual(values = method_shapes) +
  scale_x_continuous(
    breaks = 1:9,
    labels = x_labels,
    expand = expansion(mult = c(0.02, 0.02))
  ) +
  # Standard Y scale starting from 0 (0 to 0.39)
  scale_y_continuous(
    limits = c(0.0, 0.39),
    breaks = seq(0.0, 0.35, by = 0.05),
    expand = expansion(mult = c(0.0, 0.02))
  ) +
  facet_wrap(~ epsilon, ncol = 2, labeller = as_labeller(eps_labels, default = label_parsed)) +
  labs(
    x = expression(bold("Noise Variance") ~ (sigma^2)),
    y = "Mean Deviation from Ground Truth, MD(GT)",
    color = NULL,
    shape = NULL
  ) +
  theme_bw(base_size = 20, base_family = "serif") +
  theme(
    panel.grid.major = element_line(linetype = "dotted", color = "#D0D0D0", linewidth = 0.45),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "#444444", fill = NA, linewidth = 0.8),
    strip.background = element_rect(fill = "#ECEFF1", color = "#444444", linewidth = 0.8),
    strip.text = element_text(size = 20, face = "bold"),
    axis.text.x = element_text(size = 18, color = "#222222", angle = 0, hjust = 0.5),
    axis.text.y = element_text(size = 17, color = "#222222"),
    axis.title.x = element_text(size = 20, face = "bold", margin = margin(t = 12)),
    axis.title.y = element_text(size = 20, face = "bold", margin = margin(r = 14)),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.text = element_text(size = 16),
    legend.key = element_blank(),
    legend.key.width = unit(2.2, "lines"),
    legend.background = element_rect(fill = "white", color = "#CCCCCC", linewidth = 0.5),
    plot.margin = margin(t = 12, r = 18, b = 12, l = 12)
  ) +
  guides(
    color = guide_legend(nrow = 2, byrow = TRUE),
    shape = guide_legend(nrow = 2, byrow = TRUE)
  )

# Save plot to PDF and PNG
pdf_out <- "latex/figures/combined_md_all_epsilons.pdf"
png_out <- "latex/figures/combined_md_all_epsilons.png"

ggsave(pdf_out, plot = p, width = 15, height = 10.5, device = cairo_pdf)
ggsave(png_out, plot = p, width = 15, height = 10.5, dpi = 300)

cat(sprintf("Successfully generated ggplot2 figures with grey B1, standard Y from 0, and top Setup labels:\n- %s\n- %s\n", pdf_out, png_out))
