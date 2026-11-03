#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=1:55:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module unload cudnn
module load cudnn/9.8.0.87-cuda
conda activate jax_env

CKPT=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
CMD="python offline_rl_approx.py env_name=cheetah task=vel10Back reward_approx=true num_evals=20 seed=2 & python offline_rl_approx.py env_name=cheetah task=vel10 reward_approx=true num_evals=20 seed=2 & python offline_rl_approx.py env_name=cheetah task=jump reward_approx=true num_evals=20 seed=2 & wait"
echo "Running: $CMD"
mkdir -p models logs
python offline_rl_approx.py env_name=cheetah task=vel10Back reward_approx=true num_evals=20 seed=2 & python offline_rl_approx.py env_name=cheetah task=vel10 reward_approx=true num_evals=20 seed=2 & python offline_rl_approx.py env_name=cheetah task=jump reward_approx=true num_evals=20 seed=2 & wait
