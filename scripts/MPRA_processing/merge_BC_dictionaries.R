library(dplyr)
library(stringr)
library(tidyr)

print("Merging BC dictionaries...")

# Load minimal association table (BC + insert_combo)
df_all_CREs <- read.delim(
  gzfile("seq008_longMPRA_v1_association_minimal.txt.gz"),
  header = TRUE,
  stringsAsFactors = FALSE
)

# Remove duplicate BCs if any
df_all_CREs2 <- df_all_CREs %>%
  mutate(is_duplicate = duplicated(BC) | duplicated(BC, fromLast = TRUE)) %>%
  filter(!is_duplicate)

print("Merging BC count information...")

# Get all BC quant files
files_oi <- Sys.glob("data/BC_quant/*_BC_quant_20260105.txt.gz")

all_list <- list()

for(name_oi in files_oi){
  
  message("Processing: ", name_oi)
  
  # Read gzipped BC count file safely
  df_BC_counts <- tryCatch({
    read.delim(gzfile(name_oi), header = TRUE, stringsAsFactors = FALSE)
  }, error = function(e){
    message("Error reading file: ", name_oi, " Skipping.")
    return(NULL)
  })
  
  # Skip empty files or files with no rows
  if(is.null(df_BC_counts) || nrow(df_BC_counts) == 0){
    message("Skipping empty file: ", name_oi)
    next
  }
  
  # Filter out invalid BCs
  df_BC_counts <- df_BC_counts %>%
    filter(!is.na(mBC) & mBC != "")
  
  if(nrow(df_BC_counts) == 0){
    message("No valid BCs in file: ", name_oi)
    next
  }
  
  # Extract library info from filename
  filename <- basename(name_oi)
  parts <- strsplit(filename, "_")[[1]]
  prefix <- parts[1]   # e.g., "RepA-yesPB-DNA"
  subparts <- strsplit(prefix, "-")[[1]]
  biol_rep <- subparts[1]
  integration_status <- subparts[2]
  material <- subparts[3]
  
  # Keep relevant columns and add library info
  df_BC_counts2 <- df_BC_counts %>%
    dplyr::select(BC = mBC, UMI = n_UMI_per_mBC, reads = n_reads_per_mBC) %>%
    mutate(
      lib_name = paste(biol_rep, integration_status, material, sep="_"),
      biol_rep = biol_rep,
      integration_status = integration_status,
      material = material
    )
  
  # Merge with minimal association table
  merged_df <- df_all_CREs2 %>%
    left_join(df_BC_counts2, by = "BC")
  
  # Skip if no matches
  if(nrow(merged_df) == 0){
    message("No matching BCs in association table for file: ", name_oi)
    next
  }
  
  all_list[[length(all_list)+1]] <- merged_df
}

# Combine all libraries
df_all_CREs_w_counts <- bind_rows(all_list)

# Write out merged count table
write.table(
  df_all_CREs_w_counts,
  gzfile("data/longMPRA_count_table_20260105.txt.gz"),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

print("Calculating UMI recovery per library...")

# Ensure every BC is represented for each library
df_all_CREs_w_counts_full <- df_all_CREs2 %>%
  crossing(lib_name = unique(df_all_CREs_w_counts$lib_name)) %>%
  left_join(df_all_CREs_w_counts, by = c("BC", "lib_name"))

# Calculate recovery fractions
df_BC_recovery_per_lib <- df_all_CREs_w_counts_full %>%
  group_by(lib_name) %>%
  summarize(
    frac_BC_recovered = mean(!is.na(UMI)),          # fraction of all BCs detected
    frac_BC_recovered_2plus = sum(UMI >= 2, na.rm = TRUE) / n()  # fraction with ≥2 UMIs
  )

# Write out recovery summary
write.table(
  df_BC_recovery_per_lib,
  gzfile("data/longMPRA_recovery_per_lib_20260105_corrected.txt.gz"),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

print("All done.")
