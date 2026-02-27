# LAMPRA

Plasmid maps, amplicon maps, and scripts for: Locus-scAle Massively Parallel Reporter Assays (LAMPRAs).

Note: scripts contain herein do not currently constitute streamlined pipelines and are shared for transparency. Iterations and improvements will be made in the near future.

Scripts are separated in subdirectories organized by categories as detailed below with short descriptive of contents.


# Plasmid Maps

Plasmid maps can be found in "plasmid_maps" (refer to manuscript Methods for further details).
See plasmid_descriptions for a description of all plasmids. 

# Custom Sequencing Amplicons

Structures of custom sequencing amplicons used in this study are listed in "custom_amplicon_structures".

# Scripts
## MPRA Processing Scripts

The bash pipeline (arr_bulk_mBC_UMI_count_MPRA.sh) is first run, generating per-library barcode quantification files. This pipeline performs PEAR-based error correction of barcode reads, UMI extraction, and homopolymer filtering. The per-library barcode counts are then merged with a pre-determined barcode-to-insert association table using merge_BC_dictionaries.R. Finally, the merged count table is processed using first_pass_analysis_MPRA_dictionaries.R to generate winsorized RNA/DNA activity scores.

## Modeling Scripts

*Coming soon.*
