#$ -q shendure-short.q
#$ -cwd
#$ -S /bin/bash
#$ -o /net/gs/vol1/home/amcgee1/sge_logs #where to put output logs 
#$ -e /net/gs/vol1/home/amcgee1/sge_logs #where to put error logs 
#$ -M amcgee1@uw.edu
#$ -l mfree=8G,h_rt=6:00:00:00
#$ -tc 1
#$ -t 1-1:1

#modify to number of slots/samples! 


module load modules modules-init modules-gs
module load samtools/1.19
module load bwa/0.7.17
module load bedtools/2.31.1  
#module load gcc/8.1.0
module load R/4.3.2 
module load python/3.12.1
module load biopython

module load pear/0.9.11 
module load seqtk/1.4

set -e

echo "[INFO] Modules loaded."

# --- Inputs ---
input_table="/net/shendure/vol8/projects/Abby/seq008_pacbio_eNMU_longMPRA_160k_20250529/nobackup/extracted_inserts_and_bc.txt.gz"
reference_table="/net/shendure/vol8/projects/Abby/seq008_pacbio_eNMU_longMPRA_160k_20250529/nobackup/insert_reference_table.txt"
output_table="/net/shendure/vol8/projects/Abby/seq008_pacbio_eNMU_longMPRA_160k_20250529/nobackup/annotated_inserts_and_bc.txt.gz"

# --- Python logic to annotate ---
python3 - <<EOF
import gzip

# Load reference lookup
lookup = {}
with open("$reference_table") as ref:
    for line in ref:
        if line.strip():
            name, seq = line.strip().split()
            lookup[seq] = name

# Process extracted table
with gzip.open("$input_table", "rt") as fin, gzip.open("$output_table", "wt") as fout:
    header = fin.readline().strip().split("\t")
    fout.write("\t".join(header) + "\n")

    for line in fin:
        fields = line.strip().split("\t")
        annotated = []

        for i in range(5):  # insert_1 through insert_5
            val = fields[i]
            if val in lookup:
                annotated.append(lookup[val])
            else:
                annotated.append("unknown")

        annotated.append(fields[5])  # BC stays the same
        fout.write("\t".join(annotated) + "\n")

print("[INFO] Annotation complete. Output written to: $output_table")
EOF
