suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(ggh4x)
})

data <- fromJSON("outputs/master_analysis_results_classification_27scenarios.json")

records <- list()
counter <- 1

methods_list <- c("B0:SimpleAvg", "B1:Best-model", "B2:PerfWeighted", 
                  "M1:SoftmaxBMA", "M2:PointMedian", "M3:TrimMean", 
                  "M4:FuncMedian", "M5:PairwiseAlign")

for (scen_id in names(data)) {
  scen <- data[[scen_id]]
  setup_val <- paste0("Setup~", scen$setup)
  sigma_val <- paste0("sigma^2 == ", scen$noise_var)
  
  for (eps_name in names(scen$by_epsilon)) {
    eps_val <- as.numeric(gsub("eps_", "", eps_name))
    res <- scen$by_epsilon[[eps_name]]$results
    
    for (m in methods_list) {
      records[[counter]] <- data.frame(
        Scenario = scen_id,
        Setup = setup_val,
        Sigma = sigma_val,
        NoiseVar = as.numeric(scen$noise_var),
        Epsilon = eps_val,
        EpsilonLabel = paste0("epsilon == ", sprintf("%.2f", eps_val)),
        Method = m,
        MD = res[[m]]$md,
        stringsAsFactors = FALSE
      )
      counter <- counter + 1
    }
  }
}

df <- bind_rows(records)

df$MethodLabel <- recode(df$Method,
  "B0:SimpleAvg"     = "Simple Average (B0)",
  "B1:Best-model"    = "Single Best (B1)",
  "B2:PerfWeighted"  = "Performance-Weighted (B2)",
  "M1:SoftmaxBMA"    = "Softmax BMA (M1)",
  "M2:PointMedian"   = "Pointwise Median (M2)",
  "M3:TrimMean"      = "Trimmed Mean (M3)",
  "M4:FuncMedian"    = "Fréchet Median (M4)",
  "M5:PairwiseAlign" = "Pairwise Alignment (M5)"
)

method_colors <- c(
  "Single Best (B1)"           = "#333333",
  "Simple Average (B0)"        = "#1f77b4",
  "Performance-Weighted (B2)"  = "#2ca02c",
  "Softmax BMA (M1)"           = "#ff7f0e",
  "Pointwise Median (M2)"      = "#9467bd",
  "Trimmed Mean (M3)"          = "#8c564b",
  "Fréchet Median (M4)"        = "#e377c2",
  "Pairwise Alignment (M5)"    = "#17becf"
)

# Multi-panel layout across epsilon thresholds:
# - Y ekseni dikey boyutu daha da uzatıldı (height = 22.0 inç)
# - Eksen değerleri (axis tick text) daha da büyütüldü:
#     axis.text.x: 24pt
#     axis.text.y: 22pt
# - Panel başlıkları (strip.text x ve y) daha da büyütüldü:
#     strip.text.x (Setup): 28pt, bold
#     strip.text.y (epsilon): 28pt, bold
# - Eksen başlıkları (axis.title): 28pt, plain
# - Lejant metinleri: 22pt, nesneler: linewidth = 3.6, size = 9.0
p <- ggplot(df, aes(x = NoiseVar, y = MD, color = MethodLabel, group = MethodLabel)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1.4, aes(linetype = (MethodLabel == "Single Best (B1)"))) +
  stat_summary(fun = mean, geom = "point", size = 3.6) +
  scale_color_manual(values = method_colors) +
  scale_linetype_manual(values = c("TRUE" = "dashed", "FALSE" = "solid"), guide = "none") +
  scale_x_continuous(breaks = 1:9) +
  facet_nested(
    EpsilonLabel ~ Setup, 
    scales = "free_y", 
    independent = "y",
    labeller = label_parsed
  ) +
  labs(
    x = "Noise Variance",
    y = "Deviation from Ground Truth",
    color = NULL
  ) +
  theme_bw(base_size = 22) +
  theme(
    strip.background = element_rect(fill = "#f2f4f8", color = "#cccccc"),
    strip.text.x = element_text(face = "bold", size = 28),
    strip.text.y = element_text(face = "bold", size = 28),
    panel.grid.minor = element_blank(),
    panel.spacing.y = unit(0.9, "cm"),
    panel.spacing.x = unit(0.6, "cm"),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 17),
    legend.key.width = unit(2.0, "cm"),
    legend.key.height = unit(0.9, "cm"),
    legend.spacing.x = unit(0.5, "cm"),
    legend.spacing.y = unit(0.3, "cm"),
    legend.box.margin = margin(t = 12, r = 0, b = 6, l = 0),
    axis.title.x = element_text(face = "plain", size = 28, margin = margin(t = 12)),
    axis.title.y = element_text(face = "plain", size = 28, margin = margin(r = 14)),
    axis.text.x = element_text(size = 24, color = "#222222"),
    axis.text.y = element_text(size = 22, color = "#222222")
  ) +
  guides(color = guide_legend(
    nrow = 2, 
    byrow = TRUE, 
    override.aes = list(linewidth = 3.6, size = 9.0)
  ))

dir.create("latex/figures", showWarnings = FALSE, recursive = TRUE)
dir.create("figures", showWarnings = FALSE, recursive = TRUE)

# height daha da uzatıldı: 18.0 -> 22.0 inç (y ekseni aralıkları belirgin şekilde ferahlatıldı)
ggsave("latex/figures/combined_md_all_epsilons.pdf", plot = p, width = 17.5, height = 22.0, device = cairo_pdf)
ggsave("latex/figures/combined_md_all_epsilons.png", plot = p, width = 17.5, height = 22.0, dpi = 300)
ggsave("figures/combined_md_all_epsilons.pdf", plot = p, width = 17.5, height = 22.0, device = cairo_pdf)

cat("Updated Figure 1 with height=22.0 in, strip=28pt, axis.text.x=24pt, axis.text.y=22pt generated successfully!\n")
