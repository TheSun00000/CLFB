#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=0:25:00
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
CMD="python fb_continual_temp.py checkpoint=models/eps/e3b_mb300_mq0.5_run_r1/fb_stage_1_run.pickle env_name=walker tasks=stand,run,flip,supine,jump_any,kick_l,split,flamingo_r n_labeled=100000 num_evals=5 lr_coef=1 z_task_spread=0.2 multi_task_frac=0.5 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 finetune=multi steps_per_stage=2000 eval_each_steps=1 multi_task_frac=1 z_task_spread=0 eval_stream_only=0 eval_rollouts=0 num_evals=2 tasks=stand,run,flip,supine,jump_any,kick_l,split,flamingo_r multi_new=run kd_actor=1 kd_q=1 model_dir=models/multi/m9_smoke"
echo "Running: $CMD"
mkdir -p models logs
python fb_continual_temp.py checkpoint=models/eps/e3b_mb300_mq0.5_run_r1/fb_stage_1_run.pickle env_name=walker tasks=stand,run,flip,supine,jump_any,kick_l,split,flamingo_r n_labeled=100000 num_evals=5 lr_coef=1 z_task_spread=0.2 multi_task_frac=0.5 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 finetune=multi steps_per_stage=2000 eval_each_steps=1 multi_task_frac=1 z_task_spread=0 eval_stream_only=0 eval_rollouts=0 num_evals=2 tasks=stand,run,flip,supine,jump_any,kick_l,split,flamingo_r multi_new=run kd_actor=1 kd_q=1 model_dir=models/multi/m9_smoke
