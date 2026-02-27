#$ -q shendure-short.q
#$ -cwd
#$ -S /bin/bash
#$ -o /net/gs/vol1/home/amcgee1/sge_logs #where to put output logs 
#$ -e /net/gs/vol1/home/amcgee1/sge_logs #where to put error logs 
#$ -M amcgee1@uw.edu
#$ -l mfree=64G,h_rt=6:00:00:00
#$ -t 1-6


module load modules modules-init modules-gs
module load samtools/1.19
module load bedtools/2.31.1
#module load gcc/8.1.0
module load python/3.12.1
module load biopython/1.83
module load R/4.5.1 

module load pear/0.9.11 
module load seqtk/1.4

set -e

UTIL_PATH='/net/shendure/vol10/projects/Abby/MPRA_util_scripts/'

head_dir="/net/shendure/vol12/projects/randomMPRA/2025-12-11-thirdattempt/nobackup/"

lib_array=(0 RepA-yesPB-DNA_S1 RepA-yesPB-RNA_S2 RepB-yesPB-DNA_S3 RepB-yesPB-RNA_S4 RepC-yesPB-DNA_S5 RepC-yesPB-RNA_S6 )

dir_pear_outs=${head_dir}"data/pear_outs/"
dir_merged_file=${head_dir}"data/merged_files/"
dir_final_outs=${head_dir}"data/BC_quant/"
lib_name="${lib_array[$SGE_TASK_ID]}"

fastq_dir=${head_dir}"fastq/"
date_str=$(date "+20%y%m%d")

mkdir -p ${dir_pear_outs}
mkdir -p ${dir_merged_file}
mkdir -p ${dir_final_outs}

# # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # demultiplexing with index read as a fastq: 12/10/2021
# # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # 
# #  trimming the end of the BC read. 
# # # # # # # # # # # # # # # # # # # #

echo "trimming"

# from seq010: 
#R1: 30 cycles: mBC forward (16 bp) 
#R2: 10 cycles: UMI (10 bp)  
#R3: 30 cycles: mBC reverse (16 bp)
len_BC=16

trim_R1=0
trim_R2=0
trim_R3=0 

#change depending on read structure

# NOTHING TO MODIFY BELOW THIS POINT

new_suffix_R1="_R1_001_e"$trim_R1".fastq"
new_suffix_R2="_R2_001_e"$trim_R2".fastq"
new_suffix_R3="_R3_001_e"$trim_R3".fastq"


seqtk trimfq -e $trim_R1 ${fastq_dir}${lib_name}_R1_001.fastq.gz > ${fastq_dir}${lib_name}${new_suffix_R1}
seqtk trimfq -e $trim_R2 ${fastq_dir}${lib_name}_R2_001.fastq.gz > ${fastq_dir}${lib_name}${new_suffix_R2}
seqtk trimfq -e $trim_R3 ${fastq_dir}${lib_name}_R3_001.fastq.gz > ${fastq_dir}${lib_name}${new_suffix_R3}

# note: command does not preserve compressed format. 


# # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # read merging/error correction of BC reads
# # # # # # # # # # # # # # # # # # # # # # # # # # # 

# "merge" the reads, which effectively corresponds to error correction w/ PEAR
# need to combine the two reads that are reverse complement of each other, here R1 and R3. 
 
echo "using pear to combine BC reads"

out_file=${dir_pear_outs}${lib_name}"_pear_BC_"$date_str

parallel_nodes=8
pear -j $parallel_nodes -v $len_BC -m $len_BC -n $len_BC -t $len_BC \
    -f ${fastq_dir}${lib_name}${new_suffix_R1} \
    -r ${fastq_dir}${lib_name}${new_suffix_R3} \
    -o $out_file
	
#updating to trimmed files here 	
# see for pear option description: https://cme.h-its.org/exelixis/web/software/pear/doc.html



# # # # # # # # # # # # # # # # # # # # # #
# # # # processing the pear output fastq
# # # # # # # # # # # # # # # # # # # # # #
echo "processing pear out"

# pear output file must be compressed for SeqIO.
gzip $out_file".assembled.fastq"

# run python script (reformatting the fastq file as: read_id \t pear_corrected_BC)
out_file2=${dir_pear_outs}${lib_name}"_pear_BC_seqs_20260105.txt.gz"

python ${UTIL_PATH}/reformat_pear_outputs.py \
 	-i $out_file".assembled.fastq.gz" \
 	-o $out_file2



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # generating final output file w/ gRNA and BC connected in a single text file
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# echo "compressing fastqs"

# compress the fastq's:
# need to do if trimmed them? 
gzip ${fastq_dir}${lib_name}${new_suffix_R1}
gzip ${fastq_dir}${lib_name}${new_suffix_R2} 
gzip ${fastq_dir}${lib_name}${new_suffix_R3}

# launch python script to generate the final read/BC association file (need to have 3.6.5 for the biopython module)

echo "generating merged BC-UMI file"

#simple output file
out_BC_valid_simple=${dir_merged_file}${lib_name}"_BC_valid_pear_simple_"$date_str".txt.gz"
out_BC_no_pear_simple=${dir_merged_file}${lib_name}"_BC_no_pear_simple_"$date_str".txt.gz"

R1_name="BC"
R2_name="UMI"
R3_name="RC_BC"

python ${UTIL_PATH}/parse_fastqs_w_reformatted_pear.py \
 	--pear_file $out_file2 \
 	--out_valid $out_BC_valid_simple \
 	--out_no_pear_BC $out_BC_no_pear_simple \
	--in_R1 ${fastq_dir}${lib_name}${new_suffix_R1}.gz \
	--in_R2 ${fastq_dir}${lib_name}${new_suffix_R3}.gz \
	--in_UMI ${fastq_dir}${lib_name}${new_suffix_R2}.gz \
 	--R1_name $R1_name \
 	--R2_name $R3_name \
 	--UMI_name $R2_name \
 	--simple_out_bool 1
 	
#make sure read names match here
#changed to trimmed file names and gz versions 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # pile-up of reads for downstream processing
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

echo "condensing outs"

# condensing the output to a pile-up for downstream processing
out_condensed=${dir_final_outs}${lib_name}"_BC_pear_UMI_condensed_"$date_str".txt"

Rscript --vanilla ${UTIL_PATH}/condense_MPRA_BC_file.R \
 	$out_BC_valid_simple \
 	$out_condensed \
 	"BC_pear" \
 	"UMI"

gzip $out_condensed



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # further pile-up of reads for downstream processing
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# condensing the output to a pile-up for downstream processing
out_condensed2=${dir_final_outs}${lib_name}"_BC_pear_UMI_condensed_"$date_str".txt.gz"
out_mBC_only=${dir_final_outs}${lib_name}"_BC_quant_"$date_str".txt"

Rscript --vanilla ${UTIL_PATH}/pileup_MPRA_mBC_noUMI_correction.R \
	$out_condensed2 \
	$out_mBC_only


gzip $out_mBC_only


