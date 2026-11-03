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
#SBATCH --array=0-5 # 3 seeds x (row 4 stage 2, row 5 stage 2)

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module unload cudnn
module load cudnn/9.8.0.87-cuda
conda activate jax_env

# Stage 2 of rows 4 and 5 with multi_task_frac=0: only uniform random latents (no run latent) in the F and actor phase. Everything else as in the row-4 command
# (300k updates, tasks=run, 5 episodes, evaluation of the 8 tasks every 30k updates, the stage-1 checkpoints of the same seeds, B frozen).
# Array tasks 0-2: row 4 style (F / actor of the stage-1 checkpoint); 3-5: row 5 style (b_adapter=1, pretrained F / actor through W). Seeds: r1_nofb_uni, r2_row4_s2, r2_row4_s3.
SCRIPT=${SCRIPT:-fb_continual_temp.py}
PRE=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
S1=(models/eps/r1_nofb_uni_run models/eps/r2_row4_s2 models/eps/r2_row4_s3)
NAMES=(r1_nofb_uni r2_row4_s2 r2_row4_s3)
i=$((SLURM_ARRAY_TASK_ID % 3))
COMMON="checkpoint=${S1[$i]}/fb_stage_1_run.pickle env_name=walker n_labeled=100000 num_evals=5 lr_coef=1 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 finetune=multi steps_per_stage=300000 eval_each_steps=30 z_task_spread=0 eval_stream_only=0 tasks=run multi_task_frac=0"
if [ $SLURM_ARRAY_TASK_ID -lt 3 ]; then
    CMD="python $SCRIPT $COMMON model_dir=models/multi/mtf0_${NAMES[$i]}"
else
    CMD="python $SCRIPT $COMMON b_adapter=1 adapter_ckpt=$PRE model_dir=models/multi/mtf0_adapter_${NAMES[$i]}"
fi
echo "Running: $CMD"
mkdir -p models logs
$CMD
