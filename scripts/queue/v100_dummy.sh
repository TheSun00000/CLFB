#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=0:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@v100
#SBATCH --qos=qos_gpu-dev
# dummy job (2026-11-02): checks how many V100 dev jobs can run at once. Does nothing but print the GPU and sleep.
echo "dummy v100 job on $(hostname)"; nvidia-smi -L; date; sleep 150; date
