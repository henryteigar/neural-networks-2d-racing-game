#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -J CarRacingGPU

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5

#SBATCH --time=08:00:00

uname -a
cd /gpfs/hpchome/henry300/neural-networks-2d-racing-game/CarRacing-v0
source ~/.bashrc

module load python-2.7.11
python --version

python racing.py