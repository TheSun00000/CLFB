#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=1:00:00
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
CMD="python fb_continual_temp.py env_name=walker checkpoint=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle tasks=run n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.5 actor_z_task_frac=0.5 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=0 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 mu_b=300 mu_q=0.5 lam=3 model_dir=models/eps/o2_c_lam3; python fb_continual_temp.py env_name=walker checkpoint=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle tasks=run n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.5 actor_z_task_frac=0.5 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=0 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 mu_b=300 mu_q=0.5 lam=1 model_dir=models/eps/o2_c_lam1; python fb_continual_temp.py env_name=walker checkpoint=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle tasks=run n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.5 actor_z_task_frac=0.5 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=0 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 mu_b=300 mu_q=0.5 lam=10 ortho_coef=10 model_dir=models/eps/o2_c_lam10_orth10; python fb_continual_temp.py env_name=walker checkpoint=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle tasks=run n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.5 actor_z_task_frac=0.5 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=0 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 mu_b=300 mu_q=0.5 lam=30 lr_coef=1 model_dir=models/eps/o2_c_lam30_lrc1"
echo "Running: $CMD"
mkdir -p models logs
python fb_continual_temp.py env_name=walker checkpoint=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle tasks=run n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.5 actor_z_task_frac=0.5 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=0 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 mu_b=300 mu_q=0.5 lam=3 model_dir=models/eps/o2_c_lam3
python fb_continual_temp.py env_name=walker checkpoint=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle tasks=run n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.5 actor_z_task_frac=0.5 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=0 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 mu_b=300 mu_q=0.5 lam=1 model_dir=models/eps/o2_c_lam1
python fb_continual_temp.py env_name=walker checkpoint=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle tasks=run n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.5 actor_z_task_frac=0.5 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=0 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 mu_b=300 mu_q=0.5 lam=10 ortho_coef=10 model_dir=models/eps/o2_c_lam10_orth10
python fb_continual_temp.py env_name=walker checkpoint=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle tasks=run n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.5 actor_z_task_frac=0.5 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=0 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 mu_b=300 mu_q=0.5 lam=30 lr_coef=1 model_dir=models/eps/o2_c_lam30_lrc1
