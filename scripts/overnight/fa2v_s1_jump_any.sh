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
echo "Running: python $SCRIPT env_name=walker checkpoint=$PRE tasks=jump_any n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=0 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 ortho_coef=100 mu_b=300 mu_q=0 fb_objective=0 z_task_frac=0 actor_z_task_frac=0 seed=201 model_dir=models/eps/fa2v_jump_any_s1"
python $SCRIPT env_name=walker checkpoint=$PRE tasks=jump_any n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=0 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 ortho_coef=100 mu_b=300 mu_q=0 fb_objective=0 z_task_frac=0 actor_z_task_frac=0 seed=201 model_dir=models/eps/fa2v_jump_any_s1
echo "Running: python $SCRIPT env_name=walker checkpoint=$PRE tasks=jump_any n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=0 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 ortho_coef=100 mu_b=300 mu_q=0 fb_objective=0 z_task_frac=0 actor_z_task_frac=0 seed=202 model_dir=models/eps/fa2v_jump_any_s2"
python $SCRIPT env_name=walker checkpoint=$PRE tasks=jump_any n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=0 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 ortho_coef=100 mu_b=300 mu_q=0 fb_objective=0 z_task_frac=0 actor_z_task_frac=0 seed=202 model_dir=models/eps/fa2v_jump_any_s2
echo "Running: python $SCRIPT env_name=walker checkpoint=$PRE tasks=jump_any n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=0 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 ortho_coef=100 mu_b=300 mu_q=0 fb_objective=0 z_task_frac=0 actor_z_task_frac=0 seed=203 model_dir=models/eps/fa2v_jump_any_s3"
python $SCRIPT env_name=walker checkpoint=$PRE tasks=jump_any n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=0 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 ortho_coef=100 mu_b=300 mu_q=0 fb_objective=0 z_task_frac=0 actor_z_task_frac=0 seed=203 model_dir=models/eps/fa2v_jump_any_s3
echo "Running: python tools/b_diag.py --pre $PRE --run fa2v_jump_any_s1:models/eps/fa2v_jump_any_s1/fb_stage_1_jump_any.pickle:201 --run fa2v_jump_any_s2:models/eps/fa2v_jump_any_s2/fb_stage_1_jump_any.pickle:202 --run fa2v_jump_any_s3:models/eps/fa2v_jump_any_s3/fb_stage_1_jump_any.pickle:203"
python tools/b_diag.py --pre $PRE --run fa2v_jump_any_s1:models/eps/fa2v_jump_any_s1/fb_stage_1_jump_any.pickle:201 --run fa2v_jump_any_s2:models/eps/fa2v_jump_any_s2/fb_stage_1_jump_any.pickle:202 --run fa2v_jump_any_s3:models/eps/fa2v_jump_any_s3/fb_stage_1_jump_any.pickle:203
