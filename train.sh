#!/bin/bash
#$ -N example1
#sbatch --partition=gpu --gres=gpu:1 --time=10:00:00 name.sh
#for 10 hours run name.sh
#sacct : give details of all jobs
#squeue : all jobs running on all machines
#output in slurm-jobid
#tell grid engine to use current directory
#$ -cwd

# Set Email Address where notifications are to be sent , you need to use your stanford.edu address, non Stanford addresses will be blocked
#$ -M ashenoi@stanford.edu

# Tell Grid Engine to notify job owner if job 'b'egins, 'e'nds, 's'uspended is 'a'borted, or 'n'o mail
#$ -m bea

# Tel Grid Engine to join normal output and error output into one file 
#$ -j y


## the "meat" of the script

#just print the name of this machine
module load cudnn/6.0
python LSTM.py 2017_11_17_slow --frac 1 --timesteps 5 --batchsize 100 --epochs 15
python LSTM.py 2017_11_17_slow --frac 1 --timesteps 20 --batchsize 100 --epochs 15