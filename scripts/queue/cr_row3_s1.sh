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
CMD="python fb_continual_temp.py env_name=cheetah checkpoint=saved_models/fb_agent_classical_cheetah.pickle tasks=vel10 n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.5 actor_z_task_frac=0.5 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=0 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 ortho_coef=100 mu_b=0 mu_q=0 fb_objective=0 z_task_frac=0 actor_z_task_frac=0 sigma_report=1 model_dir=models/cheetah_eps/cr_row3_s1; python fb_continual_temp.py checkpoint=models/cheetah_eps/cr_row3_s1/fb_stage_1_run.pickle env_name=cheetah tasks=vel10Back,vel10,flip,jump,kick_b,spread n_labeled=100000 num_evals=5 lr_coef=1 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 finetune=multi steps_per_stage=300000 eval_each_steps=30 z_task_spread=0 eval_stream_only=0 tasks=vel10 multi_task_frac=0.25 model_dir=models/cheetah_multi/cr_row3_s1"
echo "Running: $CMD"
mkdir -p models logs
python fb_continual_temp.py env_name=cheetah checkpoint=saved_models/fb_agent_classical_cheetah.pickle tasks=vel10 n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.5 actor_z_task_frac=0.5 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=0 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 ortho_coef=100 mu_b=0 mu_q=0 fb_objective=0 z_task_frac=0 actor_z_task_frac=0 sigma_report=1 model_dir=models/cheetah_eps/cr_row3_s1
python fb_continual_temp.py checkpoint=models/cheetah_eps/cr_row3_s1/fb_stage_1_run.pickle env_name=cheetah tasks=vel10Back,vel10,flip,jump,kick_b,spread n_labeled=100000 num_evals=5 lr_coef=1 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 finetune=multi steps_per_stage=300000 eval_each_steps=30 z_task_spread=0 eval_stream_only=0 tasks=vel10 multi_task_frac=0.25 model_dir=models/cheetah_multi/cr_row3_s1
