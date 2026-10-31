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
#SBATCH --array=0-9 # 10

module purge


module load arch/h100
module load pytorch-gpu/py3/2.4.0 

module unload cudnn
module load cudnn/9.8.0.87-cuda

conda activate jax_env



# CK=saved_models/fb_agent_classical_walker.pickle
# COMMON="env_name=walker checkpoint=$CK seed=0 eval_each_steps=10 num_evals=10"
# T="tasks=run steps_per_stage=50000"

# COMMANDS=(
#     "python fb_continual.py $COMMON lam=3 mu_b=1 mu_q=0.01; python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.01; python fb_continual.py $COMMON $T lam=30 mu_b=1 mu_q=0.01; python fb_continual.py $COMMON $T lam=100 mu_b=1 mu_q=0.01; python fb_continual.py $COMMON $T lam=300 mu_b=1 mu_q=0.01; python fb_continual.py $COMMON $T lam=10 mu_b=0 mu_q=0; python fb_continual.py $COMMON $T lam=30 mu_b=0 mu_q=0; python fb_continual.py $COMMON $T lam=100 mu_b=0 mu_q=0; python fb_continual.py $COMMON $T lam=10 mu_b=3 mu_q=0.01; python fb_continual.py $COMMON $T lam=30 mu_b=3 mu_q=0.01"
#     "python fb_continual.py $COMMON $T lam=10 mu_b=3 mu_q=0; python fb_continual.py $COMMON $T lam=10 mu_b=10 mu_q=0; python fb_continual.py $COMMON $T lam=10 mu_b=30 mu_q=0; python fb_continual.py $COMMON $T lam=10 mu_b=0 mu_q=0.001; python fb_continual.py $COMMON $T lam=10 mu_b=0 mu_q=0.01; python fb_continual.py $COMMON $T lam=10 mu_b=0 mu_q=0.1; python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.03; python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.1; python fb_continual.py $COMMON $T lam=30 mu_b=10 mu_q=0; python fb_continual.py $COMMON $T lam=100 mu_b=10 mu_q=0.01"
#     "python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.01 cross_fit=1; python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.01 cross_fit=1 n_labeled=3000; python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.01 n_labeled=3000; python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.01 n_labeled=1000 cross_fit=1; python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.01 ridge=1e-2; python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.01 ridge=1e-1; python fb_continual.py $COMMON tasks=flip steps_per_stage=50000 lam=10 mu_b=1 mu_q=0.01; python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.01 n_est=16384; python fb_continual.py $COMMON tasks=run steps_per_stage=150000 lam=10 mu_b=1 mu_q=0.01; python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.01 lr=3e-5"
# )

# COMMANDS=(
#     "python fb_pretrain.py env_name=walker"
#     "python fb_pretrain.py env_name=walker"
# )



# CK=saved_models/fb_agent_classical_walker.pickle
# COMMON="env_name=walker checkpoint=$CK seed=0 eval_each_steps=10 num_evals=10"
# T="tasks=run steps_per_stage=200000"

# COMMANDS=(
#     "python fb_continual.py $COMMON $T lam=0 mu_b=0 mu_q=0"
#     "python fb_continual.py $COMMON $T lam=0 mu_b=0 mu_q=0"
#     "python fb_continual.py $COMMON $T lam=0 mu_b=0 mu_q=0"
# )

# CK=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
# COMMON="env_name=walker tasks=run checkpoint=$CK seed=0 eval_each_steps=10 num_evals=20 steps_per_stage=200000"


# T="tasks=run steps_per_stage=200000"
# COMMANDS=(
#     "python fb_continual.py $COMMON $T lam=0 mu_b=0 mu_q=0"
#     "python fb_continual.py $COMMON $T lam=0 mu_b=0 mu_q=0"
# )

# COMMANDS=(
#     "python fb_pretrain.py env_name=walker checkpoint=$CK load_target=1"
#     "python fb_pretrain.py env_name=walker checkpoint=$CK load_target=1"
#     "python fb_pretrain.py env_name=walker checkpoint=$CK load_target=0"
#     "python fb_pretrain.py env_name=walker checkpoint=$CK load_target=0"
# )

# COMMANDS=(
#     "python fb_continual.py $COMMON lam=0 mu_b=0 mu_q=0 lr=1e-4"
#     "python fb_continual.py $COMMON $T lam=0 mu_b=0 mu_q=0 lr=1e-5"
#     "python fb_continual.py $COMMON $T lam=0 mu_b=0 mu_q=0 lr=1e-6"
#     "python fb_continual.py $COMMON $T lam=0 mu_b=0 mu_q=0 lr=1e-7"
# )


# CK=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
# COMMON="env_name=walker tasks=run checkpoint=$CK seed=0 eval_each_steps=10 num_evals=20 steps_per_stage=300000"

# COMMANDS=(
#     # LINE 1: how much reconstruction weight (no task latent in the loss). Configs: control lam=0, lam=3, lam=10, lam=30
#     "for i in {1..2}; do python fb_continual.py $COMMON lam=0 mu_b=0 mu_q=0; done; for i in {1..2}; do python fb_continual.py $COMMON lam=3 mu_b=0 mu_q=0; done; for i in {1..2}; do python fb_continual.py $COMMON lam=10 mu_b=0 mu_q=0; done; for i in {1..2}; do python fb_continual.py $COMMON lam=30 mu_b=0 mu_q=0; done"
#     # LINE 2: how much of the z in the loss is the task latent (lam=10, spread 0.5). Configs: frac 0.1, 0.3, 0.5, 0.8
#     "for i in {1..2}; do python fb_continual.py $COMMON lam=10 mu_b=0 mu_q=0 z_task_frac=0.1 z_task_spread=0.5; done; for i in {1..2}; do python fb_continual.py $COMMON lam=10 mu_b=0 mu_q=0 z_task_frac=0.3 z_task_spread=0.5; done; for i in {1..2}; do python fb_continual.py $COMMON lam=10 mu_b=0 mu_q=0 z_task_frac=0.5 z_task_spread=0.5; done; for i in {1..2}; do python fb_continual.py $COMMON lam=10 mu_b=0 mu_q=0 z_task_frac=0.8 z_task_spread=0.5; done"
#     # LINE 3: how wide the cone around the task latent is (lam=10, frac 0.3). Configs: spread 0, 0.2, 1.0, 2.0
#     "for i in {1..2}; do python fb_continual.py $COMMON lam=10 mu_b=0 mu_q=0 z_task_frac=0.3 z_task_spread=0.0; done; for i in {1..2}; do python fb_continual.py $COMMON lam=10 mu_b=0 mu_q=0 z_task_frac=0.3 z_task_spread=0.2; done; for i in {1..2}; do python fb_continual.py $COMMON lam=10 mu_b=0 mu_q=0 z_task_frac=0.3 z_task_spread=1.0; done; for i in {1..2}; do python fb_continual.py $COMMON lam=10 mu_b=0 mu_q=0 z_task_frac=0.3 z_task_spread=2.0; done"
#     # LINE 4: task latent alone (lam=0) and a lower learning rate. Configs: lam=0 frac 0.3, lam=0 frac 0.5, lr=3e-5 control, lr=3e-5 lam=10 frac 0.3
#     "for i in {1..2}; do python fb_continual.py $COMMON lam=0 mu_b=0 mu_q=0 z_task_frac=0.3 z_task_spread=0.5; done; for i in {1..2}; do python fb_continual.py $COMMON lam=0 mu_b=0 mu_q=0 z_task_frac=0.5 z_task_spread=0.5; done; for i in {1..2}; do python fb_continual.py $COMMON lam=0 mu_b=0 mu_q=0 lr=3e-5; done; for i in {1..2}; do python fb_continual.py $COMMON lam=10 mu_b=0 mu_q=0 z_task_frac=0.3 z_task_spread=0.5 lr=3e-5; done"
# )

# COMMANDS=(
#     "python fb_continual.py $COMMON lam=0 mu_b=0 mu_q=0 freeze_fb=1 actor_z_task_frac=0" 
#     "python fb_continual.py $COMMON lam=0 mu_b=0 mu_q=0 freeze_fb=1 actor_z_task_frac=0.5 z_task_spread=0.5"
#     "python fb_continual.py $COMMON lam=0 mu_b=0 mu_q=0 freeze_fb=1 actor_z_task_frac=0.8 z_task_spread=0.5"
#     "python fb_continual.py $COMMON lam=0 mu_b=0 mu_q=0 freeze_fb=1 actor_z_task_frac=1.0 z_task_spread=0.2"
# )


# CK=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
# COMMON="env_name=walker tasks=run checkpoint=$CK seed=0 eval_each_steps=10 num_evals=20 steps_per_stage=500000"

# COMMANDS=(
#     "python fb_continual.py $COMMON lam=0 mu_b=0 mu_q=0 freeze_b=1 z_task_frac=1.0 z_task_spread=0.2 ortho_coef=0"
#     "python fb_continual.py $COMMON lam=0 mu_b=0 mu_q=0 freeze_b=1 z_task_frac=1.0 z_task_spread=0.2 ortho_coef=0"
#     "python fb_continual.py $COMMON lam=0 mu_b=0 mu_q=0 freeze_b=1 z_task_frac=1.0 z_task_spread=0.2 ortho_coef=0"
# )




# COMMANDS=(
#     "python fb_continual.py $COMMON  lam=0 mu_b=0 mu_q=0 ortho_coef=0 freeze_b=1 z_task_frac=1.0 z_task_spread=0"
#     "python fb_continual.py $COMMON  lam=0 mu_b=0 mu_q=0 ortho_coef=0 freeze_b=1 z_task_frac=1.0 z_task_spread=0"
#     "python fb_continual.py $COMMON  lam=0 mu_b=0 mu_q=0 ortho_coef=0 freeze_b=1 z_task_frac=1.0 z_task_spread=0"

#     "python fb_critic_oracle.py $COMMON tasks=run seed=0 steps_per_stage=200000 q_true_reward=1"
#     "python fb_critic_oracle.py $COMMON tasks=run seed=1 steps_per_stage=200000 q_true_reward=1"

#     "python fb_critic_oracle.py $COMMON tasks=run seed=0 steps_per_stage=200000 q_true_reward=0"
#     "python fb_critic_oracle.py $COMMON tasks=run seed=1 steps_per_stage=200000 q_true_reward=0"
# )



# COMMANDS=(
#     # "python offline_rl_approx.py env_name=walker task=walk reward_approx=true"
#     "python offline_rl_approx.py env_name=walker task=walk reward_approx=lrec_full approx_n=10000 n_transitions=100000"
#     "python offline_rl_approx.py env_name=walker task=walk reward_approx=lrec_detach approx_n=10000 n_transitions=100000"
#     "python offline_rl_approx.py env_name=walker task=walk reward_approx=fixed_z approx_n=10000 n_transitions=100000"
#     "python offline_rl_approx.py env_name=walker task=walk reward_approx=mlp approx_n=10000 n_transitions=100000"
# )


CKPT=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
FT="env_name=walker checkpoint=$CKPT tasks=run steps_per_stage=1000000 eval_each_steps=10 eval_baseline=0 eval_fresh=0"

COMMANDS=(
    # R1: pretrained-B reward (= option 2's reward)
    "python offline_rl_approx.py env_name=walker task=run reward_approx=fb_pretrained checkpoint=$CKPT approx_n=100000"
    # R2: pretrained B fine-tuned with fixed readout (= option 3b's reward); approx_lr=3e-4 to match rfit_lr
    "python offline_rl_approx.py env_name=walker task=run reward_approx=fb_finetuned checkpoint=$CKPT approx_n=100000 approx_lr=3e-4"

    "python fb_continual.py $FT n_labeled=100000 finetune=actor"          # E1: option 1
    "python fb_continual.py $FT n_labeled=100000 finetune=critic"         # E2: option 2
    "python fb_continual.py $FT n_labeled=100000 finetune=reward_critic"  # E3: option 3b

    "python offline_rl_approx.py env_name=walker task=run reward_approx=fb_pretrained checkpoint=$CKPT approx_n=10000"
    "python offline_rl_approx.py env_name=walker task=run reward_approx=fb_finetuned checkpoint=$CKPT approx_n=10000 approx_lr=3e-4"
    "python fb_continual.py $FT n_labeled=10000 finetune=actor"
    "python fb_continual.py $FT n_labeled=10000 finetune=critic"
    "python fb_continual.py $FT n_labeled=10000 finetune=reward_critic"
)

CMD="${COMMANDS[$SLURM_ARRAY_TASK_ID]}"
echo "Running: $CMD"
eval "$CMD"