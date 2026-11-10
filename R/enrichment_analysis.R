#!/usr/bin/env Rscript
# Functional enrichment of TE families enriched in immunogenic peptides.
# Inputs:
#   data/processed/immunogenic_tep_by_family.tsv (te_family, n_immunogenic, n_total)
# Outputs:
#   results/reports/te_family_enrichment.tsv
#   results/figures/te_family_enrichment_bar.png

suppressPackageStartupMessages({
  library(tidyverse)
})

IN_FILE <- commandArgs(trailingOnly = TRUE)[1] %||% "data/processed/immunogenic_tep_by_family.tsv"
OUT_DIR <- "results/reports"
FIG_DIR <- "results/figures"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

df <- read_tsv(IN_FILE, show_col_types = FALSE)

# Fisher's exact test per family
total_immunogenic <- sum(df$n_immunogenic)
total_non_immunogenic <- sum(df$n_total - df$n_immunogenic)

results <- df %>%
  rowwise() %>%
  mutate(
    a = n_immunogenic,
    b = n_total - n_immunogenic,
    c = total_immunogenic - n_immunogenic,
    d = total_non_immunogenic - (n_total - n_immunogenic),
    fisher_p = fisher.test(matrix(c(a, b, c, d), nrow = 2))$p.value,
    odds_ratio = (a * d + 0.5) / (b * c + 0.5),
    log2_or = log2(odds_ratio)
  ) %>%
  ungroup() %>%
  mutate(adj_p = p.adjust(fisher_p, method = "BH")) %>%
  arrange(adj_p)

write_tsv(results, file.path(OUT_DIR, "te_family_enrichment.tsv"))

# Top 20 families bar plot
top <- results %>%
  slice_min(adj_p, n = 20) %>%
  mutate(significant = adj_p < 0.05)

png(file.path(FIG_DIR, "te_family_enrichment_bar.png"),
    width = 2400, height = 2000, res = 300)
top %>%
  mutate(te_family = factor(te_family, levels = te_family[order(log2_or)])) %>%
  ggplot(aes(x = te_family, y = log2_or, fill = significant)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c(`TRUE` = "#C0504D", `FALSE` = "#A6A6A6")) +
  labs(x = "TE family", y = "log2 odds ratio",
       title = "TE families enriched in immunogenic peptides",
       fill = "FDR < 0.05") +
  theme_bw(base_size = 11) +
  theme(legend.position = "right",
        panel.grid.minor = element_blank())
dev.off()

message("Done.")
