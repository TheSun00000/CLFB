#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=0:30:00
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

# Latent drift z (pretrained B) vs z' (B of the best walker/run setup: lam 100, z_task_frac 0.5, actor_z_task_frac 1.0, spread 0.2,
# run folder 2026-10-31_18-57-15_FBcontinual-walker-run_5), all tasks. Eval only.
CMD="python tools/z_drift.py --pre models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle --new models/2026-10-31_18-57-15_FBcontinual-walker-run_5/fb_stage_1_run.pickle"
echo "Running: $CMD"
mkdir -p logs
$CMD
