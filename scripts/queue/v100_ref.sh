#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=0:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@v100
#SBATCH --qos=qos_gpu-dev

# timing test on V100 (2026-11-02): the 'ref' run of o1 (run, mu_b 300 + mu_q 0.5, 30k updates, eps only); the same run on A100 is in o1_orth.
module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module unload cudnn
module load cudnn/9.8.0.87-cuda
conda activate jax_env

CKPT=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
echo "Running: v100 timing, ref run of o1"
mkdir -p models logs
nvidia-smi -L
date
python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.5 actor_z_task_frac=0.5 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=0 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 mu_b=300 mu_q=0.5 model_dir=models/eps/v100_ref
date
