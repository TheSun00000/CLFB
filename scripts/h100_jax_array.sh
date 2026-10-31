#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=4:00:00                   # Adjust as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@h100
#SBATCH -C h100
#SBATCH --array=0-5  # 6 reward functions

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



# COMMANDS=(
#     "python sf.py env_name=cheetah mode=classical; python sf.py env_name=cheetah mode=classical"
#     "python sf.py env_name=cheetah mode=implicit tau=0.9; python sf.py env_name=cheetah mode=implicit tau=0.9"

#     "python sf.py env_name=quadruped mode=classical; python sf.py env_name=quadruped mode=classical"
#     "python sf.py env_name=quadruped mode=implicit tau=0.9; python sf.py env_name=quadruped mode=implicit tau=0.9"
# )


COMMANDS=(
    "python sf.py env_name=cheetah mode=classical n_transitions=1000000; python sf.py env_name=cheetah mode=classical n_transitions=1000000; python sf.py env_name=cheetah mode=classical n_transitions=100000; python sf.py env_name=cheetah mode=classical n_transitions=100000"
    "python sf.py env_name=cheetah mode=implicit tau=0.9 n_transitions=1000000; python sf.py env_name=cheetah mode=implicit tau=0.9 n_transitions=1000000; python sf.py env_name=cheetah mode=implicit tau=0.9 n_transitions=100000; python sf.py env_name=cheetah mode=implicit tau=0.9 n_transitions=100000"

    "python sf.py env_name=walker mode=classical n_transitions=1000000; python sf.py env_name=walker mode=classical n_transitions=1000000; python sf.py env_name=walker mode=classical n_transitions=100000; python sf.py env_name=walker mode=classical n_transitions=100000"
    "python sf.py env_name=walker mode=implicit tau=0.9 n_transitions=1000000; python sf.py env_name=walker mode=implicit tau=0.9 n_transitions=1000000; python sf.py env_name=walker mode=implicit tau=0.9 n_transitions=100000; python sf.py env_name=walker mode=implicit tau=0.9 n_transitions=100000"

    "python sf.py env_name=quadruped mode=classical n_transitions=1000000; python sf.py env_name=quadruped mode=classical n_transitions=1000000; python sf.py env_name=quadruped mode=classical n_transitions=100000; python sf.py env_name=quadruped mode=classical n_transitions=100000"
    "python sf.py env_name=quadruped mode=implicit tau=0.9 n_transitions=1000000; python sf.py env_name=quadruped mode=implicit tau=0.9 n_transitions=1000000; python sf.py env_name=quadruped mode=implicit tau=0.9 n_transitions=100000; python sf.py env_name=quadruped mode=implicit tau=0.9 n_transitions=100000"
)



COMMANDS=(
    "for i in {1..2}; do python offline_rl_fixedz.py env_name=walker task=stand mode=implicit  fb_path=./models/2026-10-07_13-47-10_FB-classical-walker/fb_agent_latest.pickle; done"
    "for i in {1..2}; do python offline_rl_fixedz.py env_name=walker task=stand mode=classical fb_path=./models/2026-10-07_13-47-10_FB-classical-walker/fb_agent_latest.pickle; done"
    "for i in {1..2}; do python offline_rl_fixedz.py env_name=walker task=run mode=implicit  fb_path=./models/2026-10-07_13-47-10_FB-classical-walker/fb_agent_latest.pickle; done"
    "for i in {1..2}; do python offline_rl_fixedz.py env_name=walker task=run mode=classical fb_path=./models/2026-10-07_13-47-10_FB-classical-walker/fb_agent_latest.pickle; done"
)




CMD="${COMMANDS[$SLURM_ARRAY_TASK_ID]}"
echo "Running: $CMD"
eval "$CMD"