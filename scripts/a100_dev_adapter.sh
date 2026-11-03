#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --array=0-2 # 3 seeds

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module unload cudnn
module load cudnn/9.8.0.87-cuda
conda activate jax_env

# b_adapter: stage 2 of row 4 (walker, run only) with F and the actor = the PRETRAINED nets seen through W = E[B0 B1^T] Sigma_1^-1 (B1 = the B of the stage-1
# checkpoint of the same seed). Same command as the stage 2 of r1_nofb_uni / r2_row4_s2 / r2_row4_s3 (300k updates, multi_task_frac 0.25, tasks=run, 5 episodes,
# evaluation of the 8 tasks every 30k updates) plus b_adapter=1 adapter_ckpt=<pretrained>. The B is NOT retrained: the existing stage-1 checkpoints are used.
# SCRIPT is the edited fb_continual_temp.py (untracked copy on the cluster until the branch is merged).
SCRIPT=${SCRIPT:-fb_continual_temp.py}
PRE=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
S1=(models/eps/r1_nofb_uni_run models/eps/r2_row4_s2 models/eps/r2_row4_s3)
NAMES=(r1_nofb_uni r2_row4_s2 r2_row4_s3)
i=$SLURM_ARRAY_TASK_ID
CMD="python $SCRIPT checkpoint=${S1[$i]}/fb_stage_1_run.pickle env_name=walker n_labeled=100000 num_evals=5 lr_coef=1 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 finetune=multi steps_per_stage=300000 eval_each_steps=30 z_task_spread=0 eval_stream_only=0 tasks=run multi_task_frac=0.25 b_adapter=1 adapter_ckpt=$PRE model_dir=models/multi/adapter_${NAMES[$i]}"
echo "Running: $CMD"
mkdir -p models logs
$CMD
