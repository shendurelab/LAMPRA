### module load pcre2/10.39; module load R/4.3.1

library(tidyverse)
library(ggplot2)
library(cowplot)
library(ggrastr)
library(castor)
library(reshape2)
library(gtools)
library(patchwork)

# source the adapted tablemaker script
source("mpra_tablemaker_20250911_v9.R")

p_count <- 0.05

# Load count table
df_counts_w_CREs2 <- read.table("data/longMPRA_count_table_20260105.txt.gz",
                                header=TRUE)


# # # # # # # # # # # # #
# Saturation calculations
# # # # # # # # # # # # #
coverage_lib <- calculate_MPRA_coverage(df_counts_w_CREs2)
write.table(coverage_lib, 'tables/longMPRA_saturation_calculations_20260105.txt', 
            sep = '\t', row.names = FALSE, quote = FALSE)

# Generate wide MPRA table
df_wide <- generate_MPRA_wide_table(df_counts_w_CREs2 %>% filter(material != 'pool'), coverage_lib)

write.table(
  df_wide,
  file = "tables/longMPRA_wide_table_20260105.txt",
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

# Generate MPRA activity table
DNA_umi_thresh <- 2
df_mpra_act <- generate_winsorize_MPRA_table(df_wide, DNA_UMI_thresh_oi = DNA_umi_thresh)

# Plot DNA barcode recovery CDF
bc.cdf <- ggplot(df_mpra_act) + 
  stat_ecdf(aes(x = n_BC)) +
  facet_wrap(~biol_rep, ncol = 3) +
  scale_x_log10() +
  theme_cowplot()

ggsave('plots/longMPRA_dna_barcode_recovery_cdf_20260105.pdf', plot = bc.cdf, height = 6, width = 10, dpi = 300, device = 'pdf')

# Optional: Generic replicate correlation plot
# Pivot for generic replicate correlation check
rep_table <- df_mpra_act %>%
  pivot_wider(id_cols = insert_combo, names_from = biol_rep, values_from = MPRA_act)

# Generate replicate correlation plots by integration_status
plots_by_status <- generate_DNA_BC_R2_plots_by_integration(df_wide)

# Save each integration status plot if there are at least 2 replicates
for(status in names(plots_by_status)) {
  # Count replicates in this status
  n_reps <- df_wide %>% filter(integration_status == status) %>% 
    distinct(biol_rep) %>% nrow()
  
  if(n_reps >= 2) {
    ggsave(
      filename = paste0('plots/longMPRA_replicate_correlation_', status, '_20260105.pdf'),
      plot = plots_by_status[[status]],
      height = 4, width = 10, dpi = 300
    )
  } else {
    message("Skipping ", status, " because less than 2 replicates.")
  }
}

# Final MPRA activity table
write.table(
  df_mpra_act,
  "tables/longMPRA_act_with_integration_status_20260105.txt",
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
