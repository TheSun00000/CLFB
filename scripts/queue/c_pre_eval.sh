#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=0:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@v100
#SBATCH --qos=qos_gpu-dev

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module unload cudnn
module load cudnn/9.8.0.87-cuda
conda activate jax_env

CKPT=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
CMD="python fb_continual_temp.py env_name=cheetah checkpoint=saved_models/fb_agent_classical_cheetah.pickle tasks=vel10 bench=final num_evals=20 eval_ckpt=saved_models/fb_agent_classical_cheetah.pickle"
echo "Running: $CMD"
mkdir -p models logs
python fb_continual_temp.py env_name=cheetah checkpoint=saved_models/fb_agent_classical_cheetah.pickle tasks=vel10 bench=final num_evals=20 eval_ckpt=saved_models/fb_agent_classical_cheetah.pickle
