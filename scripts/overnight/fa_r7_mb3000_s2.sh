#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=2:00:00
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

SCRIPT=${SCRIPT:-fb_continual_temp.py}
PRE=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
mkdir -p models logs
echo "Running: python $SCRIPT env_name=walker checkpoint=$PRE tasks=jump_any n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=0 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 ortho_coef=100 mu_b=3000 mu_q=0 fb_objective=0 z_task_frac=0 actor_z_task_frac=0 seed=202 model_dir=models/eps/fa3mb3000_jump_any_s2"
python $SCRIPT env_name=walker checkpoint=$PRE tasks=jump_any n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=0 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 ortho_coef=100 mu_b=3000 mu_q=0 fb_objective=0 z_task_frac=0 actor_z_task_frac=0 seed=202 model_dir=models/eps/fa3mb3000_jump_any_s2
echo "Running: python tools/b_diag.py --pre $PRE --run fa3mb3000_jump_any_s2:models/eps/fa3mb3000_jump_any_s2/fb_stage_1_jump_any.pickle:202"
python tools/b_diag.py --pre $PRE --run fa3mb3000_jump_any_s2:models/eps/fa3mb3000_jump_any_s2/fb_stage_1_jump_any.pickle:202
echo "Running: python $SCRIPT checkpoint=models/eps/fa3mb3000_jump_any_s2/fb_stage_1_jump_any.pickle env_name=walker n_labeled=100000 num_evals=5 lr_coef=1 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 finetune=multi steps_per_stage=200000 eval_each_steps=5 eval_dense_until=50 eval_late_every=20 z_task_spread=0 eval_stream_only=0 tasks=jump_any multi_task_frac=0.5 lr=0.0003 b_adapter=1 adapter_ckpt=$PRE kd_actor=15 model_dir=models/multi/fa_r7_jump_any_mb3000_lr0.0003_f0.5_ka15_s2"
python $SCRIPT checkpoint=models/eps/fa3mb3000_jump_any_s2/fb_stage_1_jump_any.pickle env_name=walker n_labeled=100000 num_evals=5 lr_coef=1 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 finetune=multi steps_per_stage=200000 eval_each_steps=5 eval_dense_until=50 eval_late_every=20 z_task_spread=0 eval_stream_only=0 tasks=jump_any multi_task_frac=0.5 lr=0.0003 b_adapter=1 adapter_ckpt=$PRE kd_actor=15 model_dir=models/multi/fa_r7_jump_any_mb3000_lr0.0003_f0.5_ka15_s2
