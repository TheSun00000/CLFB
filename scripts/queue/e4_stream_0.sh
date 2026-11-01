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
CMD="python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run,jump_any,flamingo_r,supine n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.5 actor_z_task_frac=0.5 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=10 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 mu_b=0 mu_q=0 model_dir=models/eps/e4_stream_s1_ctrl_r1; python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run,jump_any,flamingo_r,supine n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.5 actor_z_task_frac=0.5 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=10 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 mu_b=0 mu_q=0 model_dir=models/eps/e4_stream_s1_ctrl_r2"
echo "Running: $CMD"
mkdir -p models logs
python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run,jump_any,flamingo_r,supine n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.5 actor_z_task_frac=0.5 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=10 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 mu_b=0 mu_q=0 model_dir=models/eps/e4_stream_s1_ctrl_r1
python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run,jump_any,flamingo_r,supine n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.5 actor_z_task_frac=0.5 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=10 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 mu_b=0 mu_q=0 model_dir=models/eps/e4_stream_s1_ctrl_r2
