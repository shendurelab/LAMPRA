### module load pcre2/10.39; module load R/4.3.1

library(tidyverse)
library(ggplot2)
library(cowplot)
library(ggrastr)
library(castor)
library(reshape2)
library(DescTools)

# -------------------------------
# Calculate MPRA coverage per library
# -------------------------------
calculate_MPRA_coverage <- function(data){
  # Include integration_status
  df_metad <- data %>% distinct(lib_name, material, biol_rep, integration_status)
  
  df_BC_recovery_per_lib_simple <- data %>%
    group_by(lib_name, integration_status, material, biol_rep) %>%
    summarize(
      frac_BC_recovered = mean(!is.na(UMI)), 
      on_target_reads = sum(reads, na.rm = TRUE),
      on_target_UMI = sum(UMI, na.rm = TRUE),
      frac_BC_recovered_2plus = sum(UMI >= 2, na.rm = TRUE) / length(UMI),
      .groups = "drop"
    )
  
  df_BC_recovery_per_lib_simple <- df_BC_recovery_per_lib_simple %>%
    mutate(
      on_target_reads = on_target_reads / 1e6,
      on_target_UMI = on_target_UMI / 1e6,
      saturation = 1 - (on_target_UMI / on_target_reads)
    ) %>%
    left_join(df_metad, by = c("lib_name", "material", "biol_rep", "integration_status"))
  
  return(df_BC_recovery_per_lib_simple)
}


# -------------------------------
# Generate wide MPRA table
# -------------------------------
generate_MPRA_wide_table <- function(data, coverage_lib) {
  
  # -----------------------------
  # 1 Prepare coverage table
  # -----------------------------
  df_read_coverage_fixed <- coverage_lib %>%
    filter(material != "plasmid") %>%
    pivot_wider(
      id_cols = c(biol_rep, integration_status),  # keep integration_status
      names_from = material,
      values_from = on_target_UMI,
      names_prefix = "on_target_UMI_",
      values_fill = 0  # fill missing libraries with 0
    )
  
  # -----------------------------
  # 2 Pivot main MPRA count data to wide format
  # -----------------------------
  df_wide <- data %>%
    filter(material != "plasmid") %>%
    pivot_wider(
      id_cols = c(BC, insert_combo, biol_rep, integration_status),  # keep integration_status
      names_from = material,
      values_from = c(UMI, reads),
      values_fill = 0  # fill missing DNA/RNA combinations with 0
    )
  
  # -----------------------------
  # 3 Join coverage information
  # -----------------------------
  df_wide <- df_wide %>%
    filter(!is.na(insert_combo)) %>%
    left_join(df_read_coverage_fixed, by = c("biol_rep", "integration_status"))
  
  return(df_wide)
}


# -------------------------------
# Generate winsorized MPRA activity table
# -------------------------------
generate_winsorize_MPRA_table <- function(data_wide, DNA_UMI_thresh_oi = 2, winsor_cut = 0.01){
  
  df_mpra_act <- data_wide %>%
    # Only consider DNA UMIs above threshold
    filter(UMI_DNA >= DNA_UMI_thresh_oi) %>%
    
    # Group by biol_rep, insert_combo, AND integration_status
    group_by(biol_rep, insert_combo, integration_status) %>%
    
    # Summarize MPRA activity
    summarize(
      n_BC = length(BC),
      MPRA_act = sum(
        Winsorize(UMI_RNA / on_target_UMI_RNA,
                  val = quantile(UMI_RNA / on_target_UMI_RNA,
                                 probs = c(0, 1 - winsor_cut),
                                 na.rm = TRUE))
      ) /
        sum(
          Winsorize(UMI_DNA / on_target_UMI_DNA,
                    val = quantile(UMI_DNA / on_target_UMI_DNA,
                                   probs = c(0, 1 - winsor_cut),
                                   na.rm = TRUE))
        ),
      sum_DNA_umi = sum(UMI_DNA),
      sum_RNA_umi = sum(UMI_RNA),
      std_DNA_umi = sd(UMI_DNA),
      std_RNA_umi = sd(UMI_RNA),
      .groups = "drop"  # avoid carrying over the grouping
    )
  
  return(df_mpra_act)
}
# -------------------------------
# Generate replicate correlation plots
# -------------------------------
generate_DNA_BC_R2_plots_by_integration <- function(data) {
  library(reshape2)
  library(ggplot2)
  library(cowplot)
  library(ggrastr)
  
  plots_list <- list()
  
  for(status in unique(data$integration_status)) {
    df_sub <- data %>% filter(integration_status == status)
    
    # Pivot to wide: BC x biol_rep
    bc_table_cast <- dcast(df_sub, BC ~ biol_rep, value.var = 'UMI_DNA', fun.aggregate = mean)
    bc_table_cast <- bc_table_cast[complete.cases(bc_table_cast), ]
    bc_table_cast <- log(bc_table_cast[,-1] + 1)
    
    rep_names <- colnames(bc_table_cast)
    
    # Generate all pairwise combinations of replicates
    rep_pairs <- combn(rep_names, 2, simplify = FALSE)
    
    corr_sq_list <- sapply(rep_pairs, function(pair) {
      cor(bc_table_cast[[pair[1]]], bc_table_cast[[pair[2]]], method = 'spearman')^2
    })
    
    rep_labels <- sapply(rep_pairs, function(pair) paste(pair, collapse='-'))
    rep.corrs <- data.frame(replicates = rep_labels, corr_sq = corr_sq_list)
    
    # Generate scatter plots for each pair
    plot_list <- lapply(rep_pairs, function(pair) {
      corr_val <- rep.corrs$corr_sq[rep.corrs$replicates == paste(pair, collapse='-')]
      ggplot(bc_table_cast, aes_string(x = pair[1], y = pair[2])) +
        geom_point_rast(alpha = 0.5) +
        geom_smooth(method = "lm", col = "blue", se = FALSE) +
        labs(
          x = paste(pair[1], "\nlog(DNA UMI reads)"),
          y = paste(pair[2], "\nlog(DNA UMI reads)"),
          subtitle = paste("R2 =", round(corr_val, 3), "\n", status)
        ) +
        theme_classic() + theme(text = element_text(size = 14))
    })
    
    # Combine plots in a grid (adjust ncol based on number of pairs)
    ncol_grid <- ceiling(sqrt(length(plot_list)))
    corr.plots <- plot_grid(plotlist = plot_list, ncol = ncol_grid, align = 'vh')
    
    plots_list[[status]] <- corr.plots
  }
  
  return(plots_list)
}
