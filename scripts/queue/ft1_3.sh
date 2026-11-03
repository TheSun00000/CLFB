#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=1:55:00
#SBATCH --exclude=jean-zay-iam26
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
CMD="python fb_continual_temp.py checkpoint=saved_models/fb_agent_cheetah_pre2.pickle env_name=cheetah tasks=vel10Back,vel10,flip,jump,kick_b,spread n_labeled=100000 num_evals=5 lr_coef=1 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 finetune=multi steps_per_stage=300000 eval_each_steps=30 z_task_spread=0 eval_stream_only=0 tasks=vel10 multi_task_frac=0.25 multi_freeze_f=1 model_dir=models/cheetah_ft/ft1_act_p25; python fb_continual_temp.py checkpoint=saved_models/fb_agent_cheetah_pre2.pickle env_name=cheetah tasks=vel10Back,vel10,flip,jump,kick_b,spread n_labeled=100000 num_evals=5 lr_coef=1 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 finetune=multi steps_per_stage=300000 eval_each_steps=30 z_task_spread=0 eval_stream_only=0 tasks=vel10 multi_task_frac=0.25 multi_freeze_f=1 multi_task_frac=1 model_dir=models/cheetah_ft/ft1_act_p100"
echo "Running: $CMD"
mkdir -p models logs
python fb_continual_temp.py checkpoint=saved_models/fb_agent_cheetah_pre2.pickle env_name=cheetah tasks=vel10Back,vel10,flip,jump,kick_b,spread n_labeled=100000 num_evals=5 lr_coef=1 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 finetune=multi steps_per_stage=300000 eval_each_steps=30 z_task_spread=0 eval_stream_only=0 tasks=vel10 multi_task_frac=0.25 multi_freeze_f=1 model_dir=models/cheetah_ft/ft1_act_p25
python fb_continual_temp.py checkpoint=saved_models/fb_agent_cheetah_pre2.pickle env_name=cheetah tasks=vel10Back,vel10,flip,jump,kick_b,spread n_labeled=100000 num_evals=5 lr_coef=1 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 finetune=multi steps_per_stage=300000 eval_each_steps=30 z_task_spread=0 eval_stream_only=0 tasks=vel10 multi_task_frac=0.25 multi_freeze_f=1 multi_task_frac=1 model_dir=models/cheetah_ft/ft1_act_p100
