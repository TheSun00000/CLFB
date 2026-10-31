#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=0:45:00                   # dev QoS max is 2h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --array=0-2 # 3 eval-only jobs

module purge


module load arch/h100
module load pytorch-gpu/py3/2.4.0 

module unload cudnn
module load cudnn/9.8.0.87-cuda

conda activate jax_env

# Eval-only: every benchmark task, zero-shot (latent from fresh labels, 20 episodes), on the final checkpoint of finished runs of arrays 1788307.
# No training. The log folders are logs/<time>_FBeval-*; their info.txt records the source checkpoint. Summary: tools/eval_summary.py.
CKPT=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle

COMMANDS=(
    # job 0: pretrained model (baseline), 1788307.0, 1788307.1, 1788353.0, 1788353.1, 1788354.0, 1788354.1
    "python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run eval_ckpt=$CKPT eval_z=random; python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run eval_ckpt=models/2026-10-31_18-57-15_FBcontinual-walker-run_9/fb_stage_1_run.pickle eval_z=random; python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run eval_ckpt=models/2026-10-31_19-37-02_FBcontinual-walker-run/fb_stage_1_run.pickle eval_z=random; python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run eval_ckpt=models/2026-10-31_18-57-15_FBcontinual-walker-run_3/fb_stage_1_run.pickle eval_z=random; python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run eval_ckpt=models/2026-10-31_19-37-12_FBcontinual-walker-run_1/fb_stage_1_run.pickle eval_z=random; python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run eval_ckpt=models/2026-10-31_18-57-15_FBcontinual-walker-run_2/fb_stage_1_run.pickle eval_z=random; python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run eval_ckpt=models/2026-10-31_19-37-12_FBcontinual-walker-run/fb_stage_1_run.pickle eval_z=random"
    # job 1: 1788355.0, 1788355.1, 1788356.0, 1788356.1, 1788357.0, 1788357.1, 1788358.0
    "python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run eval_ckpt=models/2026-10-31_18-57-15_FBcontinual-walker-run_6/fb_stage_1_run.pickle eval_z=random; python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run eval_ckpt=models/2026-10-31_19-37-29_FBcontinual-walker-run/fb_stage_1_run.pickle eval_z=random; python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run eval_ckpt=models/2026-10-31_18-57-15_FBcontinual-walker-run_5/fb_stage_1_run.pickle eval_z=random; python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run eval_ckpt=models/2026-10-31_19-37-15_FBcontinual-walker-run/fb_stage_1_run.pickle eval_z=random; python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run eval_ckpt=models/2026-10-31_18-57-15_FBcontinual-walker-run_4/fb_stage_1_run.pickle eval_z=random; python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run eval_ckpt=models/2026-10-31_19-33-20_FBcontinual-walker-run/fb_stage_1_run.pickle eval_z=random; python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run eval_ckpt=models/2026-10-31_18-57-15_FBcontinual-walker-run_1/fb_stage_1_run.pickle eval_z=random"
    # job 2: 1788358.1, 1788359.0, 1788359.1, 1788360.0, 1788360.1, 1788361.0, 1788361.1
    "python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run eval_ckpt=models/2026-10-31_19-37-33_FBcontinual-walker-run/fb_stage_1_run.pickle eval_z=random; python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run eval_ckpt=models/2026-10-31_18-57-15_FBcontinual-walker-run/fb_stage_1_run.pickle eval_z=random; python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run eval_ckpt=models/2026-10-31_19-37-27_FBcontinual-walker-run/fb_stage_1_run.pickle eval_z=random; python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run eval_ckpt=models/2026-10-31_18-57-15_FBcontinual-walker-run_7/fb_stage_1_run.pickle eval_z=random; python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run eval_ckpt=models/2026-10-31_19-35-54_FBcontinual-walker-run/fb_stage_1_run.pickle eval_z=random; python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run eval_ckpt=models/2026-10-31_18-57-15_FBcontinual-walker-run_8/fb_stage_1_run.pickle eval_z=random; python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run eval_ckpt=models/2026-10-31_19-37-06_FBcontinual-walker-run/fb_stage_1_run.pickle eval_z=random"
)

CMD="${COMMANDS[$SLURM_ARRAY_TASK_ID]}"
echo "Running: $CMD"

eval "$CMD"
