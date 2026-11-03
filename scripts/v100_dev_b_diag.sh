#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=1:00:00
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

# Diagnostics A-D (tools/b_diag.py) of row 4 on walker run only: pretrained B vs the B after stage 1 (L_rec + retention mu_b 300 mu_q 0.5, ortho 100, 30k updates),
# 3 seeds. B is bit-identical in fb_stage_1_run.pickle and fb_multi.pickle (checked in job 1910987), so stage 1 is the whole change. The seeds (info.txt of the
# runs) rebuild the fine-tuning labeled set D_k, which every state used here avoids.
PRE=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
CMD="python tools/b_diag.py --pre $PRE --run r1_nofb_uni:models/eps/r1_nofb_uni_run/fb_stage_1_run.pickle:7375394 --run r2_row4_s2:models/eps/r2_row4_s2/fb_stage_1_run.pickle:7770304 --run r2_row4_s3:models/eps/r2_row4_s3/fb_stage_1_run.pickle:6547665"
echo "Running: $CMD"
mkdir -p logs
$CMD
