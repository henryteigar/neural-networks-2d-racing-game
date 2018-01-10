#!/bin/bash
#SBATCH --p main
#SBATCH -J br_job

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --time=00:10:00

uname -a
cd /gpfs/hpchome/henry300/breakout_script

source activate python-2.7.11
python --version

python br.py