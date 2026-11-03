#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=0:45:00
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

# Latent drift z vs z' for a run WITH retention: stream run,jump_any,flamingo_r,supine (lam 100, alpha 0.9, mu_b 300, mu_q 0.5, 30k updates per stage,
# run folder 2026-11-01_15-44-13_FBcontinual-walker-run+jump_any+flamingo_r+supine, models/eps/e6_stream_s1_a09_mb300_r2). Eval only.
# 1: pretrained -> stage 1 (run, no retention yet: control), 2: pretrained -> stage 4 (after 3 retention stages), 3: stage 1 -> stage 4 (drift during the retention stages).
PRE=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
M=models/eps/e6_stream_s1_a09_mb300_r2
mkdir -p logs
for pair in "$PRE $M/fb_stage_1_run.pickle" "$PRE $M/fb_stage_4_supine.pickle" "$M/fb_stage_1_run.pickle $M/fb_stage_4_supine.pickle"; do
    set -- $pair
    echo "Running: python tools/z_drift.py --pre $1 --new $2"
    python tools/z_drift.py --pre $1 --new $2
done
