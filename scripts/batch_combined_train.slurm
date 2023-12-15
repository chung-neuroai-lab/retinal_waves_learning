#!/bin/bash
#SBATCH --job-name=retinal_waves
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --time=4096
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --error=/mnt/home/aligeralde/ceph/slurm/retinal_waves/train/%j_%a_%N.err
#SBATCH --output=/mnt/home/aligeralde/ceph/slurm/retinal_waves/train/%j_%a_%N.out


# command
YAML_FILE_1=$1
YAML_FILE_2=$2
YAML_FILE_3=$3
YAML_FILE_4=$4
SEED=$5

cd /mnt/home/aligeralde/ceph/retinal_waves_learning
export PYTHONPATH="/mnt/home/aligeralde/ceph/retinal_waves_learning"
python3 scripts/run_experiments.py --cfg_path ${YAML_FILE_1} --seed ${SEED}
python3 scripts/run_experiments.py --cfg_path ${YAML_FILE_2} --seed ${SEED}

python3 scripts/run_experiments.py --cfg_path ${YAML_FILE_3} --seed ${SEED}
python3 scripts/run_experiments.py --cfg_path ${YAML_FILE_4} --seed ${SEED}
