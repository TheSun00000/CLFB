#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=0:30:00
#SBATCH --exclude=jean-zay-iam26
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

SCRIPT=${SCRIPT:-fb_continual_temp.py}
PRE=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
mkdir -p models logs
echo "Running: python tools/rb_tasks.py --pre $PRE --new fa2v_jump_any_s1:models/eps/fa2v_jump_any_s1/fb_stage_1_jump_any.pickle --new mb1000_jump_any_s1:models/eps/fa3mb1000_jump_any_s1/fb_stage_1_jump_any.pickle --new mb3000_jump_any_s1:models/eps/fa3mb3000_jump_any_s1/fb_stage_1_jump_any.pickle --new fa2v_run_s1:models/eps/fa2v_run_s1/fb_stage_1_run.pickle"
python tools/rb_tasks.py --pre $PRE --new fa2v_jump_any_s1:models/eps/fa2v_jump_any_s1/fb_stage_1_jump_any.pickle --new mb1000_jump_any_s1:models/eps/fa3mb1000_jump_any_s1/fb_stage_1_jump_any.pickle --new mb3000_jump_any_s1:models/eps/fa3mb3000_jump_any_s1/fb_stage_1_jump_any.pickle --new fa2v_run_s1:models/eps/fa2v_run_s1/fb_stage_1_run.pickle
