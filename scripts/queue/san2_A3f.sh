#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=1:20:00
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
CMD="python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run n_labeled=100000 num_evals=20 steps_per_stage=200000 lam=0 ridge=0 z_task_frac=0 actor_z_task_frac=0 ztask_bs=4096 bench=final eval_each_steps=10 eval_baseline=1 eval_stream_only=1 est_same=1 mu_b=100 mu_q=0.1 model_dir=models/ret/san2_A3f; python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run bench=final num_evals=20 eval_ckpt=models/ret/san2_A3f/fb_stage_1_run.pickle"
echo "Running: $CMD"
mkdir -p models logs
python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run n_labeled=100000 num_evals=20 steps_per_stage=200000 lam=0 ridge=0 z_task_frac=0 actor_z_task_frac=0 ztask_bs=4096 bench=final eval_each_steps=10 eval_baseline=1 eval_stream_only=1 est_same=1 mu_b=100 mu_q=0.1 model_dir=models/ret/san2_A3f
python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run bench=final num_evals=20 eval_ckpt=models/ret/san2_A3f/fb_stage_1_run.pickle
