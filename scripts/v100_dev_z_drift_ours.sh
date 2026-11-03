#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=0:40:00
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

# Latent drift z vs z' for "ours" on run only (main table row 4): B tuned with L_rec + retention (mu_b 300, mu_q 0.5, stage 1: the old model is the pretrained one),
# then F (Q loss) and the actor with B frozen (finetune=multi). 3 seeds: r1_nofb_uni (models/eps/r1_nofb_uni_run -> models/multi/r1_nofb_uni_run_p25),
# r2_row4_s2, r2_row4_s3. For each seed: pretrained -> B after stage 1 (before F / actor fine-tuning) and pretrained -> final model (fb_multi.pickle).
# A check prints whether B is identical in the two checkpoints.
PRE=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
mkdir -p logs
python - <<'PY'
import pickle, numpy as np, jax
for name, s1, fin in [('r1_nofb_uni', 'models/eps/r1_nofb_uni_run/fb_stage_1_run.pickle', 'models/multi/r1_nofb_uni_run_p25/fb_multi.pickle'),
                      ('r2_row4_s2', 'models/eps/r2_row4_s2/fb_stage_1_run.pickle', 'models/multi/r2_row4_s2/fb_multi.pickle'),
                      ('r2_row4_s3', 'models/eps/r2_row4_s3/fb_stage_1_run.pickle', 'models/multi/r2_row4_s3/fb_multi.pickle')]:
    a, b = pickle.load(open(s1, 'rb')), pickle.load(open(fin, 'rb'))
    la = jax.tree_util.tree_leaves(a['fb_params']['backward']); lb = jax.tree_util.tree_leaves(b['fb_params']['backward'])
    print('CHECK', name, 'max |B_stage1 - B_final| =', max(float(np.abs(np.asarray(x) - np.asarray(y)).max()) for x, y in zip(la, lb)), flush=True)
PY
for pair in "r1_nofb_uni models/eps/r1_nofb_uni_run/fb_stage_1_run.pickle models/multi/r1_nofb_uni_run_p25/fb_multi.pickle" \
            "r2_row4_s2 models/eps/r2_row4_s2/fb_stage_1_run.pickle models/multi/r2_row4_s2/fb_multi.pickle" \
            "r2_row4_s3 models/eps/r2_row4_s3/fb_stage_1_run.pickle models/multi/r2_row4_s3/fb_multi.pickle"; do
    set -- $pair
    echo "SEED $1 STAGE1 (before F / actor fine-tuning)"
    python tools/z_drift.py --pre $PRE --new $2
    echo "SEED $1 FINAL (after fine-tuning)"
    python tools/z_drift.py --pre $PRE --new $3
done
