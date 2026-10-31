#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=1:30:00                   # dev QoS max is 2h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --array=0-5 # 6

module purge


module load arch/h100
module load pytorch-gpu/py3/2.4.0 

module unload cudnn
module load cudnn/9.8.0.87-cuda

conda activate jax_env




# Why is R2 (TD3 from scratch on the fine-tuned-B reward, ~485) better than E3 (finetune=reward_critic, ~438)?
# Same reward (held-out corr 0.996 for both). Differences: E3 warm-starts F and the actor from the pretrained FB, uses
# lr=1e-4 / tau=0.01 (FB values) where TD3 uses lr=3e-4 / tau=0.005. E3 = control (job 1744592): walker run, 10k labels.
# Each variant is run twice (random seeds, logged in info.txt).
CKPT=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
FT="env_name=walker checkpoint=$CKPT tasks=run steps_per_stage=1000000 eval_each_steps=10 eval_baseline=0 eval_fresh=0 n_labeled=10000 finetune=reward_critic"

COMMANDS=(
    "python fb_continual.py $FT reinit_critic=1"
    "python fb_continual.py $FT reinit_critic=1"
    "python fb_continual.py $FT lr=3e-4 fb_target_tau=0.005"
    "python fb_continual.py $FT lr=3e-4 fb_target_tau=0.005"
    "python fb_continual.py $FT reinit_critic=1 lr=3e-4 fb_target_tau=0.005"
    "python fb_continual.py $FT reinit_critic=1 lr=3e-4 fb_target_tau=0.005"
)

CMD="${COMMANDS[$SLURM_ARRAY_TASK_ID]}"
echo "Running: $CMD"

eval "$CMD"