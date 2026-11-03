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

# DEBUG of diagnostics E and F: 2 episodes, 2 tasks, one seed.
PRE=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
TASKS=stand,run,flip,supine,jump_any,kick_l,split,flamingo_r
mkdir -p logs
for d in E F; do
    CMD="python fb_continual_temp.py env_name=walker checkpoint=$PRE eval_ckpt=models/multi/r2_row4_s2/fb_multi.pickle tasks=$TASKS bench=final n_labeled=100000 num_evals=2 eval_labels=run,stand diag=$d diag_seed=7770304"
    echo "Running: $CMD"
    $CMD
done
