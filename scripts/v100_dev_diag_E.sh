#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=2:00:00
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

# Diagnostic E (behaviour) of row 4: see _utils/diag_ef.py. 3 seeds, theta_0 = pretrained, theta_1 = fb_multi.pickle of the run.
PRE=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
TASKS=stand,run,flip,supine,jump_any,kick_l,split,flamingo_r
mkdir -p logs
# seed name: fine-tuned checkpoint (theta_1) and seed of the run (rebuilds the fine-tuning labeled set D_k)
RUNS=("models/multi/r1_nofb_uni_run_p25/fb_multi.pickle 7375394" "models/multi/r2_row4_s2/fb_multi.pickle 7770304" "models/multi/r2_row4_s3/fb_multi.pickle 6547665")
for r in "${RUNS[@]}"; do
    set -- $r
    CMD="python ${SCRIPT:-fb_continual_temp.py} env_name=walker checkpoint=$PRE eval_ckpt=$1 tasks=$TASKS bench=final n_labeled=100000 num_evals=20 diag=E diag_seed=$2"
    echo "Running: $CMD"
    $CMD
done
