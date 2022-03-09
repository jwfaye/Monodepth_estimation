#!/bin/bash

#SBATCH --partition=insa-gpu
#SBATCH -w crn22
#SBATCH --job-name=diode_l2_l1
#SBATCH --output=/data/jofaye/monodepth/cout.txt
#SBATCH --error=/data/jofaye/monodepth/cerr.txt
#SBATCH --gres=gpu:1

srun singularity run --nv --bind /data/jofaye/monodepth/datasets/ --bind /data/jofaye/monodepth /data/jofaye/monodepth/monodepth.sif /bin/bash -c "python3 train.py"
