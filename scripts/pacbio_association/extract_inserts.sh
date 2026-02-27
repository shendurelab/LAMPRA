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
input_fastq="/net/shendure/vol8/projects/Abby/seq008_pacbio_eNMU_longMPRA_160k_20250529/nobackup/same_orientation_reads_all.fastq.gz"
output_txt="/net/shendure/vol8/projects/Abby/seq008_pacbio_eNMU_longMPRA_160k_20250529/nobackup/extracted_inserts_and_bc.txt.gz"



echo "[INFO] Starting sequence extraction..."

zcat "$input_fastq" | \
awk '
  BEGIN {
    print "insert_1\tinsert_2\tinsert_3\tinsert_4\tinsert_5\tBC"
    count = 0

    labels[1] = "insert_1"; anchors["insert_1"] = "CTAGTCATG"
    labels[2] = "insert_2"; anchors["insert_2"] = "CATGAGGACATG"
    labels[3] = "insert_3"; anchors["insert_3"] = "CATGAGCCCATG"
    labels[4] = "insert_4"; anchors["insert_4"] = "CATGACATCATG"
    labels[5] = "insert_5"; anchors["insert_5"] = "CATGCACCCATG"
    labels[6] = "BC";       anchors["BC"]       = "CTCTTCCGATCT"
  }

  NR % 4 == 2 {
    seq = $0
    out = ""
    for (i = 1; i <= 6; i++) {
      label = labels[i]
      anchor = anchors[label]
      len = (label == "BC") ? 16 : 12
      pos = index(seq, anchor)
      if (pos > 0) {
        out = out ((i == 1) ? "" : "\t") substr(seq, pos + length(anchor), len)
      } else {
        out = out ((i == 1) ? "" : "\t") "NA"
      }
    }
    print out
    count++
  }

  END {
    print "[INFO] Total reads processed: " count > "/dev/stderr"
  }' | gzip > "$output_txt"
