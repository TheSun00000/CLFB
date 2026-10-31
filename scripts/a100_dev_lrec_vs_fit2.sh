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

# Round 2 of "make L_rec regress B as well as fit_backward_to_reward (E3 phase 1)". Round 1 (array 1782771) showed that with L_FB and ortho on,
# the held-out reward error eps stays near 0.1 whatever the readout/batch/lr (E3 phase 1: 0.03), and that dropping L_FB/ortho reaches 0.03 but
# destroys the orthonormality of B. Here: all the optimisation choices of round 1 (OPT), and what could let L_rec win against L_FB/ortho:
#   lam sweep, a lower ortho_coef, a B-only prefit before the joint training, 100k labeled states (10k overfit). Returns are not evaluated
#   (eval_rollouts=0); the metrics are run_fresh_eps (eval_log.csv) and orth_l2 (training_log.csv). Walker run, one stage of 60k updates, mu_b=mu_q=0.
CKPT=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
BASE="env_name=walker checkpoint=$CKPT tasks=run n_labeled=10000 num_evals=2 eval_rollouts=0"
REC="$BASE steps_per_stage=60000 eval_each_steps=2 mu_b=0 mu_q=0"
OPT="rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1"        # the optimisation choices of fit_backward_to_reward

COMMANDS=(
    # 0, 1: references, E3 phase 1 (30k steps of B alone) with 10k and 100k labels
    "python fb_continual_temp.py $BASE steps_per_stage=1000 finetune=reward_critic"
    "python fb_continual_temp.py $BASE steps_per_stage=1000 finetune=reward_critic n_labeled=100000"
    # 2-6: weight of L_rec
    "python fb_continual_temp.py $REC $OPT lam=10"
    "python fb_continual_temp.py $REC $OPT lam=30"
    "python fb_continual_temp.py $REC $OPT lam=100"
    "python fb_continual_temp.py $REC $OPT lam=300"
    "python fb_continual_temp.py $REC $OPT lam=1000"
    # 7: weaker orthonormality term, 8: B-only prefit then joint training, 9: 100k labeled states
    "python fb_continual_temp.py $REC $OPT lam=100 ortho_coef=0.1"
    "python fb_continual_temp.py $REC $OPT lam=100 rec_prefit=1"
    "python fb_continual_temp.py $REC $OPT lam=100 n_labeled=100000"
)

CMD="${COMMANDS[$SLURM_ARRAY_TASK_ID]}"
echo "Running: $CMD"

eval "$CMD"
