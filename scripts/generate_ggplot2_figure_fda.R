suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(ggh4x)
})

data <- fromJSON("outputs/master_analysis_results_fda_27scenarios.json")

records <- list()
counter <- 1

for (scen_id in names(data)) {
  scen <- data[[scen_id]]
  setup_val <- paste0("Setup~", scen$setup)
  sigma_val <- paste0("sigma^2 == ", scen$noise_var)
  
  for (eps_name in names(scen$by_epsilon)) {
    eps_val <- as.numeric(gsub("eps_", "", eps_name))
    res <- scen$by_epsilon[[eps_name]]$results
    
    for (m in c("B0:SimpleAvg", "B1:Best-model", "B2:PerfWeighted", 
               "M1:SoftmaxBMA", "M2:PointMedian", "M3:TrimMean", 
               "M4:FuncMedian", "M5:AVA-PDP")) {
      records[[counter]] <- data.frame(
        Scenario = scen_id,
        Setup = setup_val,
        Sigma = sigma_val,
        NoiseVar = as.numeric(scen$noise_var),
        Epsilon = eps_val,
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
  "B0:SimpleAvg"    = "B0 (SimpleAvg)",
  "B1:Best-model"   = "B1 (Best-model)",
  "B2:PerfWeighted" = "B2 (PerfWeighted)",
  "M1:SoftmaxBMA"   = "M1 (Softmax-BMA)",
  "M2:PointMedian"  = "M2 (Pointwise Median)",
  "M3:TrimMean"     = "M3 (Trimmed Mean)",
  "M4:FuncMedian"   = "M4 (Functional Median)",
  "M5:AVA-PDP"      = "M5 (AVA-PDP)"
)

method_colors <- c(
  "B1 (Best-model)"       = "#333333",
  "B0 (SimpleAvg)"        = "#1f77b4",
  "B2 (PerfWeighted)"     = "#2ca02c",
  "M1 (Softmax-BMA)"      = "#ff7f0e",
  "M2 (Pointwise Median)" = "#9467bd",
  "M3 (Trimmed Mean)"     = "#8c564b",
  "M4 (Functional Median)"= "#e377c2",
  "M5 (AVA-PDP)"          = "#17becf"
)

# Figure 1: base_size = 20, legend.text = 20, well-proportioned keys
p_noise <- ggplot(df, aes(x = NoiseVar, y = MD, color = MethodLabel, group = MethodLabel)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1.0, aes(linetype = (MethodLabel == "B1 (Best-model)"))) +
  stat_summary(fun = mean, geom = "point", size = 2.8) +
  scale_color_manual(values = method_colors) +
  scale_linetype_manual(values = c("TRUE" = "dashed", "FALSE" = "solid"), guide = "none") +
  scale_x_continuous(breaks = 1:9) +
  facet_grid2(
    factor(Epsilon, labels = c("epsilon == 0.01", "epsilon == 0.05", "epsilon == 0.10", "epsilon == 0.20")) ~ Setup, 
    labeller = label_parsed,
    scales = "free_y",
    independent = "y"
  ) +
  labs(
    x = "noise level",
    y = "mean distance to ground truth",
    color = NULL
  ) +
  theme_bw(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 20, face = "plain"),
    legend.key.width = unit(2.2, "cm"),
    legend.key.height = unit(0.8, "cm"),
    strip.background = element_rect(fill = "#f2f2f2", color = "gray70"),
    strip.text = element_text(face = "plain", size = 18),
    panel.grid.minor = element_blank(),
    axis.title = element_text(face = "plain", size = 20),
    axis.text = element_text(face = "plain", size = 15)
  ) +
  guides(color = guide_legend(
    nrow = 2,
    override.aes = list(linewidth = 2.0, size = 5.0)
  ))

dir.create("latex/figures", showWarnings = FALSE, recursive = TRUE)
ggsave("latex/figures/combined_md_all_epsilons.pdf", plot = p_noise, width = 15.0, height = 12.0, device = cairo_pdf)
ggsave("latex/figures/combined_md_all_epsilons.png", plot = p_noise, width = 15.0, height = 12.0, dpi = 300)

cat("Successfully generated updated combined_md_all_epsilons.pdf and .png with base_size=20!\n")
