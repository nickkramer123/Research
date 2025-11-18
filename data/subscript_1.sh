#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-23:59:59
#SBATCH --mem=64gb
#SBATCH --job-name=job_

cd /gpfs1/home/n/c/nckramer/Project
source activate python3.7
python EvoOpt_AllB_QUICK.py 1


