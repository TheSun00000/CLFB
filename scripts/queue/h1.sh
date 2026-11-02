#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=1:55:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module unload cudnn
module load cudnn/9.8.0.87-cuda
conda activate jax_env

mkdir -p models logs
echo "Running: python fb_continual_temp.py checkpoint=models/eps/q1_s8_o100_r1/fb_stage_8_flip.pickle env_name=walker tasks=stand,run,flip,supine,jump_any,kick_l,split,flamingo_r n_labeled=100000 num_evals=5 lr_coef=1 z_task_spread=0.2 multi_task_frac=0.5 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 finetune=multi steps_per_stage=300000 eval_each_steps=100 z_task_spread=0 eval_stream_only=0 num_evals=20 tasks=run multi_white=1 multi_task_frac=0.25 model_dir=models/multi/h_s1_run_white_p25_a"
python fb_continual_temp.py checkpoint=models/eps/q1_s8_o100_r1/fb_stage_8_flip.pickle env_name=walker tasks=stand,run,flip,supine,jump_any,kick_l,split,flamingo_r n_labeled=100000 num_evals=5 lr_coef=1 z_task_spread=0.2 multi_task_frac=0.5 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 finetune=multi steps_per_stage=300000 eval_each_steps=100 z_task_spread=0 eval_stream_only=0 num_evals=20 tasks=run multi_white=1 multi_task_frac=0.25 model_dir=models/multi/h_s1_run_white_p25_a
python fb_continual_temp.py checkpoint=models/eps/q1_s8_o100_r1/fb_stage_8_flip.pickle env_name=walker tasks=stand,run,flip,supine,jump_any,kick_l,split,flamingo_r n_labeled=100000 num_evals=5 lr_coef=1 z_task_spread=0.2 multi_task_frac=0.5 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 finetune=multi steps_per_stage=300000 eval_each_steps=100 z_task_spread=0 eval_stream_only=0 num_evals=20 tasks=run multi_white=1 multi_task_frac=0.25 model_dir=models/multi/h_s1_run_white_p25_b
python fb_continual_temp.py checkpoint=models/eps/q1_s8_o100_r1/fb_stage_8_flip.pickle env_name=walker tasks=stand,run,flip,supine,jump_any,kick_l,split,flamingo_r n_labeled=100000 num_evals=5 lr_coef=1 z_task_spread=0.2 multi_task_frac=0.5 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 finetune=multi steps_per_stage=300000 eval_each_steps=100 z_task_spread=0 eval_stream_only=0 num_evals=20 tasks=run multi_white=1 multi_task_frac=0.25 model_dir=models/multi/h_s1_run_white_p25_c
