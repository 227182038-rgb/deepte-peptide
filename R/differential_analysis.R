#!/usr/bin/env Rscript
# Differential TE expression analysis across cancer types
# Inputs:
#   data/processed/te_expression.tsv  (rows = TEs, cols = samples)
#   data/processed/sample_metadata.tsv (sample_id, cancer_type, ...)
# Outputs:
#   results/reports/differential_te_expression.tsv
#   results/figures/volcano_diff_te.png

suppressPackageStartupMessages({
  library(tidyverse)
  library(edgeR)
  library(limma)
})

# --- Configuration ---------------------------------------------------------
INPUT_FILE  <- commandArgs(trailingOnly = TRUE)[1] %||% "data/processed/te_expression.tsv"
META_FILE   <- commandArgs(trailingOnly = TRUE)[2] %||% "data/processed/sample_metadata.tsv"
OUT_DIR     <- "results/reports"
FIG_DIR     <- "results/figures"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

# --- Load data -------------------------------------------------------------
message("Loading expression matrix...")
expr <- read_tsv(INPUT_FILE, show_col_types = FALSE)
meta <- read_tsv(META_FILE, show_col_types = FALSE)

# Ensure sample IDs match
stopifnot(all(colnames(expr)[-1] %in% meta$sample_id))

# Build DGEList
gene_ids <- expr[[1]]
mat <- as.matrix(expr[, -1])
rownames(mat) <- gene_ids
y <- DGEList(counts = mat, group = meta$cancer_type)
y <- calcNormFactors(y, method = "TMM")

# Filter low-count TEs
keep <- filterByExpr(y)
y <- y[keep, , keep.lib.sizes = FALSE]
message(sprintf("Retained %d / %d TEs after filtering", sum(keep), length(keep)))

# Build design matrix
design <- model.matrix(~ 0 + cancer_type, data = meta)
colnames(design) <- levels(factor(meta$cancer_type))

# Voom transformation
v <- voom(y, design, plot = FALSE)

# Per-cancer-type pairwise comparison vs all others
cancer_types <- unique(meta$cancer_type)
all_results <- list()

for (ct in cancer_types) {
  others <- setdiff(cancer_types, ct)
  contrast <- makeContrasts(contrasts = paste0(ct, " - (", paste(others, collapse = " + "), ")/", length(others)),
                            levels = design)
  fit <- lmFit(v, design)
  fit2 <- contrasts.fit(fit, contrast)
  fit2 <- eBayes(fit2, trend = TRUE)
  tt <- topTable(fit2, number = Inf, sort.by = "P") %>%
    rownames_to_column("te_id") %>%
    mutate(cancer_type = ct)
  all_results[[ct]] <- tt
}

results <- bind_rows(all_results)
write_tsv(results, file.path(OUT_DIR, "differential_te_expression.tsv"))

# Volcano plot for top cancer type
top_ct <- results %>% count(cancer_type, sort = TRUE) %>% slice(1) %>% pull(cancer_type)
top_results <- results %>% filter(cancer_type == top_ct)

png(file.path(FIG_DIR, "volcano_diff_te.png"), width = 1800, height = 1500, res = 300)
top_results %>%
  mutate(sig = ifelse(adj.P.Val < 0.05 & abs(logFC) > 1, "Significant", "NS")) %>%
  ggplot(aes(x = logFC, y = -log10(adj.P.Val), color = sig)) +
  geom_point(alpha = 0.5, size = 1) +
  scale_color_manual(values = c("Significant" = "#C0504D", "NS" = "#A6A6A6")) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", alpha = 0.5) +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", alpha = 0.5) +
  labs(x = "log2 fold change", y = "-log10(adjusted P)",
       title = sprintf("Differential TE expression: %s vs others", top_ct)) +
  theme_bw(base_size = 11) +
  theme(legend.position = "none",
        panel.grid.minor = element_blank())
dev.off()

message("Done.")
