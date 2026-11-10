#!/usr/bin/env Rscript
# Survival analysis: TE-peptide burden vs overall survival in TCGA
# Inputs:
#   data/processed/te_peptide_burden.tsv (sample_id, tep_count, immunogenic_tep_count)
#   data/processed/clinical.tsv (sample_id, OS, OS_event, ...)
# Outputs:
#   results/reports/survival_te_burden.tsv
#   results/figures/km_high_vs_low_immunogenic_tep.png

suppressPackageStartupMessages({
  library(tidyverse)
  library(survival)
  library(survminer)
})

BURDEN_FILE <- commandArgs(trailingOnly = TRUE)[1] %||% "data/processed/te_peptide_burden.tsv"
CLINICAL_FILE <- commandArgs(trailingOnly = TRUE)[2] %||% "data/processed/clinical.tsv"
OUT_DIR <- "results/reports"
FIG_DIR <- "results/figures"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

message("Loading data...")
burden <- read_tsv(BURDEN_FILE, show_col_types = FALSE)
clinical <- read_tsv(CLINICAL_FILE, show_col_types = FALSE)

# Merge
df <- inner_join(burden, clinical, by = "sample_id")
message(sprintf("Merged %d samples", nrow(df)))

# Stratify by median immunogenic TE-peptide count
df <- df %>%
  mutate(group = ifelse(immunogenic_tep_count > median(immunogenic_tep_count, na.rm = TRUE),
                        "High", "Low"))

# Kaplan-Meier
surv_obj <- Surv(time = df$OS, event = df$OS_event)
km_fit <- survfit(surv_obj ~ group, data = df)
p_km <- ggsurvplot(km_fit, data = df,
                   pval = TRUE, conf.int = TRUE,
                   risk.table = TRUE,
                   palette = c("#1F4E79", "#C0504D"),
                   legend.labs = c("Low immunogenic TEP burden", "High immunogenic TEP burden"),
                   xlab = "Overall survival (days)",
                   ylab = "Survival probability",
                   title = "TCGA pan-cancer: TE-peptide burden stratification")
png(file.path(FIG_DIR, "km_high_vs_low_immunogenic_tep.png"),
    width = 1800, height = 2200, res = 300)
print(p_km)
dev.off()

# Cox proportional hazards
cox_fit <- coxph(surv_obj ~ immunogenic_tep_count + age + sex + cancer_type,
                 data = df)
cox_summary <- summary(cox_fit)
cox_df <- data.frame(
  variable = rownames(cox_summary$coefficients),
  HR = cox_summary$coefficients[, "exp(coef)"],
  CI_low = cox_summary$conf.int[, "lower .95"],
  CI_high = cox_summary$conf.int[, "upper .95"],
  p_value = cox_summary$coefficients[, "Pr(>|z|)"]
)
write_tsv(cox_df, file.path(OUT_DIR, "survival_te_burden.tsv"))

# Forest plot
png(file.path(FIG_DIR, "forest_cox_te_burden.png"),
    width = 2000, height = 1200, res = 300)
ggplot(cox_df, aes(x = HR, y = variable)) +
  geom_point(size = 3) +
  geom_errorbarh(aes(xmin = CI_low, xmax = CI_high), height = 0.2) +
  geom_vline(xintercept = 1, linetype = "dashed", alpha = 0.5) +
  scale_x_log10() +
  labs(x = "Hazard ratio (95% CI, log scale)", y = "",
       title = "Cox regression: TE-peptide burden and overall survival") +
  theme_bw(base_size = 11)
dev.off()

message("Done.")
