#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=0:50:00                   # dev QoS max is 2h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --array=0-9 # 10

module purge


module load arch/h100
module load pytorch-gpu/py3/2.4.0 

module unload cudnn
module load cudnn/9.8.0.87-cuda

conda activate jax_env

# Which differences between L_rec (continual) and fit_backward_to_reward (E3 phase 1, finetune=reward_critic) matter for how well
# B regresses onto the reward? fb_continual_temp.py adds flags to the L_rec path (see its docstring). Walker run, 10k labels,
# single stage of 60k updates, mu_b=mu_q=0 (nothing to retain with one task), eval every 10k updates. One seed per variant.
# Metric: run_fresh_eps / run_fresh_rew_corr (re-solved readout) and run_fit_eps / run_fit_rew_corr (fixed readout) in eval_log.csv.
CKPT=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
BASE="env_name=walker checkpoint=$CKPT tasks=run n_labeled=10000 num_evals=10 eval_baseline=0"
REC="$BASE steps_per_stage=60000 eval_each_steps=10 mu_b=0 mu_q=0 lam=10"
ISO="fb_objective=0 ortho_coef=0"                          # B receives gradient from L_rec only
OPT="rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1"        # the optimisation choices of fit_backward_to_reward

COMMANDS=(
    # 0: reference, E3 phase 1 (30k steps of B alone); the ft_start row holds the fresh eps of the fitted B
    "python fb_continual_temp.py $BASE steps_per_stage=1000 finetune=reward_critic"
    # 1: baseline L_rec (current continual code)
    "python fb_continual_temp.py $REC"
    # 2: heavier weight on L_rec
    "python fb_continual_temp.py $REC lam=100"
    # 3-5: one change at a time: fixed readout, minibatch, lr 3e-4 with cosine decay
    "python fb_continual_temp.py $REC rec_fixed_w=1"
    "python fb_continual_temp.py $REC rec_bs=4096"
    "python fb_continual_temp.py $REC lr_coef=3 b_cosine=1"
    # 6: B driven by L_rec only (no L_FB, no ortho)
    "python fb_continual_temp.py $REC $ISO"
    # 7: all the optimisation choices, L_FB still on
    "python fb_continual_temp.py $REC $OPT"
    # 8: all the optimisation choices and B driven by L_rec only (closest to E3 phase 1)
    "python fb_continual_temp.py $REC $OPT $ISO"
    # 9: fixed readout and B driven by L_rec only
    "python fb_continual_temp.py $REC rec_fixed_w=1 $ISO"
)

CMD="${COMMANDS[$SLURM_ARRAY_TASK_ID]}"
echo "Running: $CMD"

eval "$CMD"
