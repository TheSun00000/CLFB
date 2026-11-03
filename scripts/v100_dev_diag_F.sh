#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=1:30:00
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

# Diagnostic F (consistency of F with B) of row 4: see _utils/diag_ef.py. 3 seeds. Then tools/b_diag.py again (plain-coordinate Procrustes added).
PRE=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
TASKS=stand,run,flip,supine,jump_any,kick_l,split,flamingo_r
mkdir -p logs
# seed name: fine-tuned checkpoint (theta_1) and seed of the run (rebuilds the fine-tuning labeled set D_k)
RUNS=("models/multi/r1_nofb_uni_run_p25/fb_multi.pickle 7375394" "models/multi/r2_row4_s2/fb_multi.pickle 7770304" "models/multi/r2_row4_s3/fb_multi.pickle 6547665")
for r in "${RUNS[@]}"; do
    set -- $r
    CMD="python fb_continual_temp.py env_name=walker checkpoint=$PRE eval_ckpt=$1 tasks=$TASKS bench=final n_labeled=100000 num_evals=20 diag=F diag_seed=$2"
    echo "Running: $CMD"
    $CMD
done
echo "Running: tools/b_diag.py"
python tools/b_diag.py --pre $PRE --run r1_nofb_uni:models/eps/r1_nofb_uni_run/fb_stage_1_run.pickle:7375394 --run r2_row4_s2:models/eps/r2_row4_s2/fb_stage_1_run.pickle:7770304 --run r2_row4_s3:models/eps/r2_row4_s3/fb_stage_1_run.pickle:6547665
