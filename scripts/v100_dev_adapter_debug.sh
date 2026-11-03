#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=0:20:00
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

# DEBUG of b_adapter: 2000 updates, evaluation of 4 tasks every 1000 updates, 3 episodes, seed r2_row4_s2.
SCRIPT=${SCRIPT:-fb_continual_temp.py}
PRE=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
CMD="python $SCRIPT checkpoint=models/eps/r2_row4_s2/fb_stage_1_run.pickle env_name=walker n_labeled=100000 num_evals=3 lr_coef=1 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 finetune=multi steps_per_stage=2000 eval_each_steps=1 z_task_spread=0 eval_stream_only=0 tasks=run multi_task_frac=0.25 eval_labels=run,stand,flip,split b_adapter=1 adapter_ckpt=$PRE model_dir=models/multi/adapter_debug"
echo "Running: $CMD"
mkdir -p models logs
$CMD
