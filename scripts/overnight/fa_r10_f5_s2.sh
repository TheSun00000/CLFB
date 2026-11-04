#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=2:00:00
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

SCRIPT=${SCRIPT:-fb_continual_temp.py}
PRE=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
mkdir -p models logs
echo "Running: python $SCRIPT checkpoint=models/eps/fa5ad2_jump_any_s2/fb_stage_1_jump_any.pickle env_name=walker n_labeled=100000 num_evals=5 lr_coef=1 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 finetune=multi steps_per_stage=200000 eval_each_steps=5 eval_dense_until=50 eval_late_every=20 z_task_spread=0 eval_stream_only=0 tasks=jump_any multi_task_frac=0.5 lr=0.0005 b_adapter=1 adapter_ckpt=$PRE kd_actor=30 kd_const=0.5 model_dir=models/multi/fa_r10_jump_any_f5_s2"
python $SCRIPT checkpoint=models/eps/fa5ad2_jump_any_s2/fb_stage_1_jump_any.pickle env_name=walker n_labeled=100000 num_evals=5 lr_coef=1 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 finetune=multi steps_per_stage=200000 eval_each_steps=5 eval_dense_until=50 eval_late_every=20 z_task_spread=0 eval_stream_only=0 tasks=jump_any multi_task_frac=0.5 lr=0.0005 b_adapter=1 adapter_ckpt=$PRE kd_actor=30 kd_const=0.5 model_dir=models/multi/fa_r10_jump_any_f5_s2
