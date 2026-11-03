#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=1:30:00
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
CMD="python fb_continual_temp.py env_name=cheetah checkpoint=saved_models/fb_agent_cheetah_pre2.pickle tasks=vel10 bench=final num_evals=20 eval_labels=vel10 eval_ckpt=saved_models/fb_agent_cheetah_pre2.pickle; python fb_continual_temp.py env_name=cheetah checkpoint=saved_models/fb_agent_cheetah_pre2.pickle tasks=vel10 bench=final num_evals=20 eval_labels=vel10 eval_ckpt=models/cheetah_ft/ft1_fa_p100/fb_multi.pickle; python fb_continual_temp.py env_name=cheetah checkpoint=saved_models/fb_agent_cheetah_pre2.pickle tasks=vel10 bench=final num_evals=20 eval_labels=vel10 eval_ckpt=models/cheetah_ft/ft2_r4_p100/fb_multi.pickle; python fb_continual_temp.py env_name=cheetah checkpoint=saved_models/fb_agent_cheetah_pre2.pickle tasks=vel10 bench=final num_evals=20 eval_labels=vel10 eval_ckpt=models/cheetah_ft/ft2_fa_p75/fb_multi.pickle; python fb_continual_temp.py env_name=cheetah checkpoint=saved_models/fb_agent_cheetah_pre2.pickle tasks=vel10 bench=final num_evals=20 eval_labels=vel10 eval_ckpt=models/cheetah_ft/ft2_r4_p50/fb_multi.pickle"
echo "Running: $CMD"
mkdir -p models logs
python fb_continual_temp.py env_name=cheetah checkpoint=saved_models/fb_agent_cheetah_pre2.pickle tasks=vel10 bench=final num_evals=20 eval_labels=vel10 eval_ckpt=saved_models/fb_agent_cheetah_pre2.pickle
python fb_continual_temp.py env_name=cheetah checkpoint=saved_models/fb_agent_cheetah_pre2.pickle tasks=vel10 bench=final num_evals=20 eval_labels=vel10 eval_ckpt=models/cheetah_ft/ft1_fa_p100/fb_multi.pickle
python fb_continual_temp.py env_name=cheetah checkpoint=saved_models/fb_agent_cheetah_pre2.pickle tasks=vel10 bench=final num_evals=20 eval_labels=vel10 eval_ckpt=models/cheetah_ft/ft2_r4_p100/fb_multi.pickle
python fb_continual_temp.py env_name=cheetah checkpoint=saved_models/fb_agent_cheetah_pre2.pickle tasks=vel10 bench=final num_evals=20 eval_labels=vel10 eval_ckpt=models/cheetah_ft/ft2_fa_p75/fb_multi.pickle
python fb_continual_temp.py env_name=cheetah checkpoint=saved_models/fb_agent_cheetah_pre2.pickle tasks=vel10 bench=final num_evals=20 eval_labels=vel10 eval_ckpt=models/cheetah_ft/ft2_r4_p50/fb_multi.pickle
