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

# Round 4: the return. Round 3 showed that L_rec (lam 100, 100k labels, fixed readout) fits the reward as well as E3 phase 1 (eps 0.026) while L_FB and the
# ortho term stay active, but F and the actor train on uniform z, so the return stays at the pretrained ~340. Here the policy is pointed at the task:
# z_task_frac of the z of L_FB and of the actor loss is replaced by the current task latent (re-encoded with the current B on a minibatch of D_k,
# ztask_bs) plus angular noise (z_task_spread). Question: how close to E3 (419-438 after 1M updates) / R2 (475-485) can the return get, and does the
# FB side stay intact (orth_l2, fb_loss, eps)? Return = zero-shot with the latent inferred from fresh labels, 20 episodes, every 10k updates,
# only on the stream task (eval_stream_only=1: 5x fewer environments per evaluation than all the benchmark tasks).
# 500k updates, 100k labeled states, walker run, lam 100, E3-like L_rec (fixed readout, batch 4096, B lr x3 with cosine), mu_b=mu_q=0.
# Each array task runs 2 commands in a row: ~45 min each (training ~38 min, evaluations ~6 min).
CKPT=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
P="python fb_continual_temp.py"
BASE="env_name=walker checkpoint=$CKPT tasks=run n_labeled=100000 mu_b=0 mu_q=0 num_evals=20"
FB="$BASE steps_per_stage=500000 eval_each_steps=10 eval_baseline=1 eval_stream_only=1 ztask_bs=4096"

COMMANDS=(
    # 0: task focus at full strength (the base run) and at half strength
    "$P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=1.0 z_task_spread=0.2; $P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.5 z_task_spread=0.2"
    # 1: task focus 0.8; full strength without noise around the latent
    "$P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.8 z_task_spread=0.2; $P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=1.0 z_task_spread=0"
    # 2: wider noise around the latent; F trained without the B(s) latents (mix_ratio 0)
    "$P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=1.0 z_task_spread=0.5; $P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=1.0 z_task_spread=0.2 mix_ratio=0"
    # 3: who sees the task latent: the actor only, with F half-focused or F uniform
    "$P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.5 actor_z_task_frac=1.0 z_task_spread=0.2; $P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0 actor_z_task_frac=1.0 z_task_spread=0.2"
    # 4: controls without L_rec (lam 0): is the reward fit needed for the return
    "$P $FB lam=0 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=1.0 z_task_spread=0.2; $P $FB lam=0 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.5 z_task_spread=0.2"
    # 5: weight of L_rec under task focus
    "$P $FB lam=30 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=1.0 z_task_spread=0.2; $P $FB lam=300 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=1.0 z_task_spread=0.2"
    # 6: weight of the Q loss in the critic (q_loss_coef, default 0.01)
    "$P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=1.0 z_task_spread=0.2 q_loss_coef=0.1; $P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=1.0 z_task_spread=0.2 q_loss_coef=1.0"
    # 7: E3-like critic inside the continual code (Q loss only, no L_FB); B-only prefit then B frozen (z_task re-encoded with the fitted B)
    "$P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=1.0 z_task_spread=0.2 fb_objective=0; $P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=1.0 z_task_spread=0.2 rec_prefit=1 freeze_b=1"
    # 8: base run, two repeats (noise of the return)
    "$P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=1.0 z_task_spread=0.2; $P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=1.0 z_task_spread=0.2"
    # 9: base run, a third repeat; the baseline without task focus (uniform z, same evaluation)
    "$P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=1.0 z_task_spread=0.2; $P $FB lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0"
)

CMD="${COMMANDS[$SLURM_ARRAY_TASK_ID]}"
echo "Running: $CMD"

eval "$CMD"
