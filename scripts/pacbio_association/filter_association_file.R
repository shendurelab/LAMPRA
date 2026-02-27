# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)


full_BCs_seq008 <- read.table("collapsed_inserts_and_bc_counts.txt.gz", 
                                  header = TRUE, 
                                  sep = "\t", 
                                  stringsAsFactors = FALSE)

#remove things with unknown or NA 
full_BCs_seq008_filtered <- full_BCs_seq008[
  complete.cases(full_BCs_seq008) &
    !apply(full_BCs_seq008 == "unknown", 1, any),
]

######################################
#plot BC counts
######################################

# Set your threshold and title
read_thresh <- 10 
file_BC_in <- "Filtered Insert Combinations"

# Plot log-log histogram of counts
plt_pdf_BC_counts <- ggplot(full_BCs_seq008_filtered) + 
  stat_bin(aes(x = count), geom = "step", bins = 50) +
  geom_vline(aes(xintercept = read_thresh), linetype = "dashed", color = "orange") +
  scale_x_log10() + 
  scale_y_log10() +
  labs(x = "Read counts per insert combination", 
       y = "Number of insert combinations") +
  ggtitle(file_BC_in) +
  theme_minimal() +
  theme(plot.title = element_text(size = 10))

print(plt_pdf_BC_counts)

######################################
#filter for unique
######################################

# Create the insert_combo column
full_BCs_seq008_filtered <- full_BCs_seq008_filtered %>%
  mutate(insert_combo = paste(insert_1, insert_2, insert_3, insert_4, insert_5, sep = ";"))

# Count number of unique insert combinations
n_unique_combos <- n_distinct(full_BCs_seq008_filtered$insert_combo)

# Show the result
print(n_unique_combos)

# Put it into new association df 
association_df <- full_BCs_seq008_filtered %>%
  mutate(insert_combo = paste(insert_1, insert_2, insert_3, insert_4, insert_5, sep = ";")) %>%
  select(insert_combo, BC, count)

######################################
#filter out BCs that repeat and BCs that are not the right length
######################################
# Keep only BCs that are 16 bases
association_df_filtered <- association_df %>%
  filter(nchar(BC) == 16)

# Keep only BCs that are associated with a single insert_combo
association_df_unique <- association_df_filtered %>%
  group_by(BC) %>%
  filter(n_distinct(insert_combo) == 1) %>%
  ungroup()


######################################
# write out association table - one with counts and one without 
######################################

write.table(
  association_df_unique,
  file = "seq008_longMPRA_v1_association_df.txt.gz",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)


# Minimal association table with only BC and insert_combo
association_df_min <- association_df_unique %>%
  select(BC, insert_combo)

write.table(
  association_df_min,
  file = "seq008_longMPRA_v1_association_minimal.txt.gz",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

