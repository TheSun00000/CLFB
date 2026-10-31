#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=2:00:00                   # dev QoS max is 2h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --array=0-9 # 10 jobs x 2 runs

module purge


module load arch/h100
module load pytorch-gpu/py3/2.4.0 

module unload cudnn
module load cudnn/9.8.0.87-cuda

conda activate jax_env

# Round 3 of "make L_rec regress B as well as fit_backward_to_reward (E3 phase 1)": L_FB and the ortho term are ON (fb_objective=1, ortho_coef=1
# unless a run says otherwise), single stage of 500k updates. Free round (array 1783403, L_FB off) showed that L_rec alone matches the reference
# (eps ~0.03) and that what lowers the match is the competition: ortho at coefficient 1 doubles eps, L_FB doubles it again (round 1, ~0.1).
# Question: how low can the held-out eps (run_fresh_eps) go while L_FB and ortho stay active, and at what price in orth_l2 and fb_loss
# (training_log.csv)? The control lam=0 gives the values of orth_l2/fb_loss/return without L_rec.
# F and the actor are trained on uniform z (z_task_frac=0 default), the returns (fresh zero-shot) are evaluated every 25k updates, with the
# pretrained policy at step 0. mu_b=mu_q=0 (nothing to retain with one task). Each array task runs 2 commands in a row: ~40 min each.
#   E3-like L_rec = fixed readout (w_fit), batch 4096, B lr x3 with cosine decay over the 500k updates.
CKPT=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
P="python fb_continual_temp.py"
BASE="env_name=walker checkpoint=$CKPT tasks=run n_labeled=10000 num_evals=10 mu_b=0 mu_q=0"
FB="$BASE steps_per_stage=500000 eval_each_steps=25 eval_baseline=1"

COMMANDS=(
    # 0: ends of the lam sweep
    "$P $FB lam=10 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1; $P $FB lam=1000 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1"
    # 1: middle of the lam sweep
    "$P $FB lam=30 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1; $P $FB lam=300 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1"
    # 2: lam 100 twice (noise at 500k updates)
    "$P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1; $P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1"
    # 3: ortho_coef at lam 100: 0.1 and 0
    "$P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 ortho_coef=0.1; $P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 ortho_coef=0"
    # 4: ortho_coef 0.01 at lam 100; ortho_coef 0.1 at lam 10
    "$P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 ortho_coef=0.01; $P $FB lam=10 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 ortho_coef=0.1"
    # 5: B-only prefit (fit_backward_to_reward, 30k steps) then joint, lam 10 and 100
    "$P $FB lam=10 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 rec_prefit=1; $P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 rec_prefit=1"
    # 6: prefit at lam 1000; the current L_rec form (resolved readout) at lam 100
    "$P $FB lam=1000 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 rec_prefit=1; $P $FB lam=100 rec_bs=4096 lr_coef=3 b_cosine=1"
    # 7: resolved readout at lam 1000; B lr x1 with cosine at lam 100
    "$P $FB lam=1000 rec_bs=4096 lr_coef=3 b_cosine=1; $P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=1 b_cosine=1"
    # 8: B lr x3 constant at lam 100 (does the decay hide a drift); 100k labels at lam 100
    "$P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=0; $P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 n_labeled=100000"
    # 9: a second task (walk) at lam 100; the control without L_rec (lam 0: L_FB and ortho only)
    "$P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 tasks=walk; $P $FB lam=0"
)

CMD="${COMMANDS[$SLURM_ARRAY_TASK_ID]}"
echo "Running: $CMD"

eval "$CMD"
