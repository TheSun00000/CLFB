#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=0:30:00
#SBATCH --exclude=jean-zay-iam26
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

SCRIPT=${SCRIPT:-fb_continual_temp.py}
PRE=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
mkdir -p models logs
echo "Running: python tools/rb_tasks.py --pre $PRE --new fa2v_jump_any_s1:models/eps/fa2v_jump_any_s1/fb_stage_1_jump_any.pickle --new mb1000_jump_any_s1:models/eps/fa3mb1000_jump_any_s1/fb_stage_1_jump_any.pickle --new c3_const_mb300_s1:models/eps/fa4vc3_jump_any_s1/fb_stage_1_jump_any.pickle --new run_tuned_s1:models/eps/fa2v_run_s1/fb_stage_1_run.pickle"
python tools/rb_tasks.py --pre $PRE --new fa2v_jump_any_s1:models/eps/fa2v_jump_any_s1/fb_stage_1_jump_any.pickle --new mb1000_jump_any_s1:models/eps/fa3mb1000_jump_any_s1/fb_stage_1_jump_any.pickle --new c3_const_mb300_s1:models/eps/fa4vc3_jump_any_s1/fb_stage_1_jump_any.pickle --new run_tuned_s1:models/eps/fa2v_run_s1/fb_stage_1_run.pickle
