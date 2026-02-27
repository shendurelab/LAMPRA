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

echo "[INFO] Starting summary generation..."

# === Paths ===
input_file="/net/shendure/vol8/projects/Abby/seq008_pacbio_eNMU_longMPRA_160k_20250529/nobackup/annotated_inserts_and_bc.txt.gz"
ref_table="/net/shendure/vol8/projects/Abby/seq008_pacbio_eNMU_longMPRA_160k_20250529/nobackup/insert_reference_table.txt"
output_counts="/net/shendure/vol8/projects/Abby/seq008_pacbio_eNMU_longMPRA_160k_20250529/nobackup/collapsed_inserts_and_bc_counts.txt.gz"
output_stats="/net/shendure/vol8/projects/Abby/seq008_pacbio_eNMU_longMPRA_160k_20250529/nobackup/summary_stats.txt"

# === Step 1: Collapse identical rows and count ===
echo "[INFO] Generating collapsed count table..."
zcat "$input_file" | \
awk 'NR==1 {header=$0; next} {counts[$0]++}
     END {
       print header "\tcount"
       for (line in counts) {
         print line "\t" counts[line]
       }
     }' | gzip > "$output_counts"

echo "[INFO] Collapsed table written to: $output_counts"

# === Step 2: Generate stats ===
echo "[INFO] Generating insert and orientation stats..."
awk -v ref="$ref_table" '
BEGIN {
  i = 0
  while ((getline < ref) > 0) {
    if ($1 != "name") {
      refnames[$1] = 1
      order[++i] = $1
    }
  }
  order[++i] = "unknown"
  order[++i] = "NA"
}
NR == 1 { next }
{
  for (i = 1; i <= 5; i++) {
    val = $i
    if (val in refnames) {
      insert[val]++
      if (val ~ /_forward$/) orientation["forward"]++
      else if (val ~ /_reverse$/) orientation["reverse"]++
      else orientation["unknown"]++
    } else if (val == "NA") {
      insert["NA"]++
      orientation["NA"]++
    } else if (val == "unknown") {
      insert["unknown"]++
      orientation["unknown"]++
    }
  }
}
END {
  print "Insert Counts:"
  for (j = 1; j in order; j++) {
    label = order[j]
    print label "\t" (label in insert ? insert[label] : 0)
  }

  print "\nOrientation Counts:"
  print "forward\t" (orientation["forward"]+0)
  print "reverse\t" (orientation["reverse"]+0)
  print "unknown\t" (orientation["unknown"]+0)
  print "NA\t" (orientation["NA"]+0)
}' < <(zcat "$input_file") > "$output_stats"

echo "[INFO] Stats written to: $output_stats"
echo "Done!"