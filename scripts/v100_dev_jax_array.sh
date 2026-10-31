#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@v100
#SBATCH --qos=qos_gpu-dev
#SBATCH --array=0-2  # 3 jobs



module purge


module load arch/h100
module load pytorch-gpu/py3/2.4.0 

module unload cudnn
module load cudnn/9.8.0.87-cuda

conda activate jax_env

nvidia-smi

ls /usr/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/linkhome/rech/genisi01/uyo96iu/.mujoco/mujoco210/bin



# CK=saved_models/fb_agent_classical_walker.pickle
# COMMON="env_name=walker checkpoint=$CK tasks=run steps_per_stage=50000 seed=0"

# COMMANDS=(
#     "python fb_continual.py $COMMON lam=1 mu_b=1 mu_q=0.01"
#     "python fb_continual.py $COMMON lam=1 mu_b=0 mu_q=0"
#     "python fb_continual.py $COMMON lam=1 mu_b=10 mu_q=0.1"
#     "python fb_continual.py $COMMON lam=10 mu_b=1 mu_q=0.01"
# )



CK=saved_models/fb_agent_classical_walker.pickle
COMMON="env_name=walker checkpoint=$CK seed=0 eval_each_steps=10 num_evals=10"
T="tasks=run steps_per_stage=50000"

COMMANDS=(
    "python fb_continual.py $COMMON $T lam=3 mu_b=1 mu_q=0.01; python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.01; python fb_continual.py $COMMON $T lam=30 mu_b=1 mu_q=0.01; python fb_continual.py $COMMON $T lam=100 mu_b=1 mu_q=0.01; python fb_continual.py $COMMON $T lam=300 mu_b=1 mu_q=0.01; python fb_continual.py $COMMON $T lam=10 mu_b=0 mu_q=0; python fb_continual.py $COMMON $T lam=30 mu_b=0 mu_q=0; python fb_continual.py $COMMON $T lam=100 mu_b=0 mu_q=0; python fb_continual.py $COMMON $T lam=10 mu_b=3 mu_q=0.01; python fb_continual.py $COMMON $T lam=30 mu_b=3 mu_q=0.01"
    "python fb_continual.py $COMMON $T lam=10 mu_b=3 mu_q=0; python fb_continual.py $COMMON $T lam=10 mu_b=10 mu_q=0; python fb_continual.py $COMMON $T lam=10 mu_b=30 mu_q=0; python fb_continual.py $COMMON $T lam=10 mu_b=0 mu_q=0.001; python fb_continual.py $COMMON $T lam=10 mu_b=0 mu_q=0.01; python fb_continual.py $COMMON $T lam=10 mu_b=0 mu_q=0.1; python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.03; python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.1; python fb_continual.py $COMMON $T lam=30 mu_b=10 mu_q=0; python fb_continual.py $COMMON $T lam=100 mu_b=10 mu_q=0.01"
    "python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.01 cross_fit=1; python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.01 cross_fit=1 n_labeled=3000; python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.01 n_labeled=3000; python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.01 n_labeled=1000 cross_fit=1; python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.01 ridge=1e-2; python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.01 ridge=1e-1; python fb_continual.py $COMMON tasks=flip steps_per_stage=50000 lam=10 mu_b=1 mu_q=0.01; python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.01 n_est=16384; python fb_continual.py $COMMON tasks=run steps_per_stage=150000 lam=10 mu_b=1 mu_q=0.01; python fb_continual.py $COMMON $T lam=10 mu_b=1 mu_q=0.01 lr=3e-5"
)




CMD="${COMMANDS[$SLURM_ARRAY_TASK_ID]}"
echo "Running: $CMD"
eval "$CMD"