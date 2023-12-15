#!/bin/bash
#SBATCH --job-name=retinal_waves
#SBATCH --mem=256G
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --time=4096
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --error=/mnt/home/aligeralde/ceph/slurm/retinal_waves/mftma/%j_%a_%N.err
#SBATCH --output=/mnt/home/aligeralde/ceph/slurm/retinal_waves/mftma/%j_%a_%N.out

# command
YAML_FILE_MFTMA=$1
SEED=$2

cd /mnt/home/aligeralde/ceph/retinal_waves_learning
export PYTHONPATH="/mnt/home/aligeralde/ceph/retinal_waves_learning"
python3 -u scripts/run_experiments.py --cfg_path ${YAML_FILE_MFTMA} --seed ${SEED}
