#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=1:00:00                   # dev QoS max is 2h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --array=0-9 # 10 jobs x 6 runs

module purge


module load arch/h100
module load pytorch-gpu/py3/2.4.0 

module unload cudnn
module load cudnn/9.8.0.87-cuda

conda activate jax_env

# Round 2 of "make L_rec regress B as well as fit_backward_to_reward (E3 phase 1, finetune=reward_critic)", now without L_FB: B is driven
# by L_rec alone (fb_objective=0 ortho_coef=0), so the only question is the reward reconstruction (held-out eps, run_fresh_eps in eval_log.csv;
# no rollouts). Round 1 (array 1782771) found that this setting reaches the reference (E3-like, run 8: eps 0.030 against 0.034).
# Each array task runs 6 commands in a row (60 runs). 30k updates = the 30k B-only steps of the reference. mu_b=mu_q=0, one stage.
#   E3-like = fixed readout (w_fit), batch 4096, B lr x3 (3e-4) with cosine decay: what fit_backward_to_reward does.
# Seeds are random (info.txt); two runs with the same flags are repeats.
CKPT=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
P="python fb_continual_temp.py"
BASE="env_name=walker checkpoint=$CKPT tasks=run n_labeled=10000 eval_rollouts=0 mu_b=0 mu_q=0"
FREE="$BASE steps_per_stage=30000 eval_each_steps=2 eval_baseline=1 fb_objective=0 ortho_coef=0 lam=1"
REF="$BASE steps_per_stage=1000 finetune=reward_critic eval_baseline=0"      # E3 phase 1 (30k B-only steps), the ft_start row has its eps

COMMANDS=(
    # 0: noise and equivalence on run, 10k labels: 3 E3-like runs and 3 references (the references are fit_backward_to_reward itself)
    "$P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1; $P $REF; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1; $P $REF; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1; $P $REF"
    # 1: readout. Resolved (batch Sigma_theta, as in the current L_rec) x2, + cross-fit, + n_est 16384, + ridge 1e-2, + ridge 1e-1
    "$P $FREE rec_bs=4096 lr_coef=3 b_cosine=1; $P $FREE rec_bs=4096 lr_coef=3 b_cosine=1; $P $FREE rec_bs=4096 lr_coef=3 b_cosine=1 cross_fit=1; $P $FREE rec_bs=4096 lr_coef=3 b_cosine=1 n_est=16384; $P $FREE rec_bs=4096 lr_coef=3 b_cosine=1 ridge=1e-2; $P $FREE rec_bs=4096 lr_coef=3 b_cosine=1 ridge=1e-1"
    # 2: batch and lr of L_rec (E3-like otherwise): batch 1024 x2, whole labeled set x2, lr x1 with cosine x2
    "$P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 rec_bs=1024; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 rec_bs=1024; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 rec_bs=0; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 rec_bs=0; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=1 b_cosine=1; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=1 b_cosine=1"
    # 3: lr: x1 constant x2, x3 constant x2, x10 with cosine x2
    "$P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=1 b_cosine=0; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=1 b_cosine=0; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=0; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=0; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=10 b_cosine=1; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=10 b_cosine=1"
    # 4: few labels: 1k x2 and 3k, each E3-like with its reference
    "$P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 n_labeled=1000; $P $REF n_labeled=1000; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 n_labeled=1000; $P $REF n_labeled=1000; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 n_labeled=3000; $P $REF n_labeled=3000"
    # 5: many labels: 30k, 100k x2, each E3-like with its reference
    "$P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 n_labeled=30000; $P $REF n_labeled=30000; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 n_labeled=100000; $P $REF n_labeled=100000; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 n_labeled=100000; $P $REF n_labeled=100000"
    # 6: other tasks, walk and stand: E3-like x2 and one reference each
    "$P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 tasks=walk; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 tasks=walk; $P $REF tasks=walk; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 tasks=stand; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 tasks=stand; $P $REF tasks=stand"
    # 7: other tasks, flip and vel8: E3-like x2 and one reference each
    "$P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 tasks=flip; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 tasks=flip; $P $REF tasks=flip; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 tasks=vel8; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 tasks=vel8; $P $REF tasks=vel8"
    # 8: ortho term back on, still no L_FB (E3-like otherwise): ortho_coef 0.01, 0.1, 1, two runs each
    "$P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 ortho_coef=0.01; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 ortho_coef=0.01; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 ortho_coef=0.1; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 ortho_coef=0.1; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 ortho_coef=1; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 ortho_coef=1"
    # 9: second runs for the label sweep (3k, 30k, with references) and the resolved readout at 1k and 100k labels
    "$P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 n_labeled=3000; $P $REF n_labeled=3000; $P $FREE rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 n_labeled=30000; $P $REF n_labeled=30000; $P $FREE rec_bs=4096 lr_coef=3 b_cosine=1 n_labeled=1000; $P $FREE rec_bs=4096 lr_coef=3 b_cosine=1 n_labeled=100000"
)

CMD="${COMMANDS[$SLURM_ARRAY_TASK_ID]}"
echo "Running: $CMD"

eval "$CMD"
