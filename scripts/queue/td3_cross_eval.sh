#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=1:00:00
#SBATCH --exclude=jean-zay-iam26
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
CMD="python offline_rl_approx.py env_name=walker num_evals=20 eval_ckpt=models/2026-11-01_00-01-28_TD3-walker-stand/td3_agent_latest.pickle,models/2026-11-01_00-01-28_TD3-walker-run/td3_agent_latest.pickle,models/2026-11-01_00-01-28_TD3-walker-flip/td3_agent_latest.pickle,models/2026-11-01_00-01-28_TD3-walker-supine/td3_agent_latest.pickle,models/2026-11-01_00-01-28_TD3-walker-jump_any/td3_agent_latest.pickle,models/2026-11-01_00-01-28_TD3-walker-kick_l/td3_agent_latest.pickle,models/2026-11-01_00-01-28_TD3-walker-split/td3_agent_latest.pickle,models/2026-11-01_00-01-28_TD3-walker-flamingo_r/td3_agent_latest.pickle"
echo "Running: $CMD"
mkdir -p models logs
python offline_rl_approx.py env_name=walker num_evals=20 eval_ckpt=models/2026-11-01_00-01-28_TD3-walker-stand/td3_agent_latest.pickle,models/2026-11-01_00-01-28_TD3-walker-run/td3_agent_latest.pickle,models/2026-11-01_00-01-28_TD3-walker-flip/td3_agent_latest.pickle,models/2026-11-01_00-01-28_TD3-walker-supine/td3_agent_latest.pickle,models/2026-11-01_00-01-28_TD3-walker-jump_any/td3_agent_latest.pickle,models/2026-11-01_00-01-28_TD3-walker-kick_l/td3_agent_latest.pickle,models/2026-11-01_00-01-28_TD3-walker-split/td3_agent_latest.pickle,models/2026-11-01_00-01-28_TD3-walker-flamingo_r/td3_agent_latest.pickle
