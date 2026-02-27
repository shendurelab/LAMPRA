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

# MODIFY THESE
input_bam="/net/shendure/vol9/seq/PacBio/smrtlink_data/r21034_20250529_222811/1_A01/hifi_reads/m21034_250529_223018.hifi_reads.bam"
output_dir="/net/shendure/vol8/projects/Abby/seq008_pacbio_eNMU_longMPRA_160k_20250529/nobackup"
temp_fastq="${output_dir}/temp_all_reads.fastq"
output_fastq="${output_dir}/same_orientation_reads_all.fastq.gz"

echo "[INFO] Extracting all reads from BAM..."
samtools fastq "$input_bam" > "$temp_fastq"

echo "[INFO] Processing all reads with Python..."
python3 - <<EOF | gzip > "$output_fastq"
from Bio.Seq import Seq
import sys

motif = "GGGCACGGGCAGCTTGC"
lines = []
read_count = 0
revcomp_count = 0

def print_normalized_read(header, seq, plus, qual):
    global revcomp_count
    if motif in seq:
        rc_seq = str(Seq(seq).reverse_complement())
        rc_qual = qual[::-1]
        revcomp_count += 1
        sys.stdout.write(f"{header}\n{rc_seq}\n{plus}\n{rc_qual}\n")
    else:
        sys.stdout.write(f"{header}\n{seq}\n{plus}\n{qual}\n")

with open("$temp_fastq") as f:
    for line in f:
        lines.append(line.strip())
        if len(lines) == 4:
            read_count += 1
            print_normalized_read(*lines)
            lines = []
            if read_count % 100000 == 0:
                print(f"[PYTHON DEBUG] Processed {read_count} reads...", file=sys.stderr)

print(f"[PYTHON DEBUG] Total reads processed: {read_count}", file=sys.stderr)
print(f"[PYTHON DEBUG] Reads reverse complemented: {revcomp_count}", file=sys.stderr)
EOF

echo "All reads processed and written to: $output_fastq"