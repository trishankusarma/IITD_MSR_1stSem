#!/bin/sh
#PBS -N MLA4Q1
#PBS -P scai
#PBS -q scai_q
#PBS -m bea
#PBS -M $USER@iitd.ac.in
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=5:00:00
# $PBS_O_WORKDIR is the directory from where the job is fired.

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $HOME

cd /home/scai/msr/aiy257590/scratch

source /home/scai/msr/aiy257590/anaconda3/bin/activate

python train_rnn.py 
