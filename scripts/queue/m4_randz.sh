#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module unload cudnn
module load cudnn/9.8.0.87-cuda
conda activate jax_env

CKPT=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
CMD="python fb_continual_temp.py env_name=walker checkpoint=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle tasks=stand,run,flip,supine,jump_any,kick_l,split,flamingo_r bench=final num_evals=20 ridge=0 eval_ckpt=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle eval_z=random model_dir=models/multi/m4_rand_pre; python fb_continual_temp.py env_name=walker checkpoint=models/eps/e7_stream_s8_a099_mb300_r1/fb_stage_8_flip.pickle tasks=stand,run,flip,supine,jump_any,kick_l,split,flamingo_r bench=final num_evals=20 ridge=0 eval_ckpt=models/eps/e7_stream_s8_a099_mb300_r1/fb_stage_8_flip.pickle eval_z=random model_dir=models/multi/m4_rand_mb300; python fb_continual_temp.py env_name=walker checkpoint=models/eps/e7_stream_s8_ctrl_r1/fb_stage_8_flip.pickle tasks=stand,run,flip,supine,jump_any,kick_l,split,flamingo_r bench=final num_evals=20 ridge=0 eval_ckpt=models/eps/e7_stream_s8_ctrl_r1/fb_stage_8_flip.pickle eval_z=random model_dir=models/multi/m4_rand_ctrl"
echo "Running: $CMD"
mkdir -p models logs
python fb_continual_temp.py env_name=walker checkpoint=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle tasks=stand,run,flip,supine,jump_any,kick_l,split,flamingo_r bench=final num_evals=20 ridge=0 eval_ckpt=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle eval_z=random model_dir=models/multi/m4_rand_pre
python fb_continual_temp.py env_name=walker checkpoint=models/eps/e7_stream_s8_a099_mb300_r1/fb_stage_8_flip.pickle tasks=stand,run,flip,supine,jump_any,kick_l,split,flamingo_r bench=final num_evals=20 ridge=0 eval_ckpt=models/eps/e7_stream_s8_a099_mb300_r1/fb_stage_8_flip.pickle eval_z=random model_dir=models/multi/m4_rand_mb300
python fb_continual_temp.py env_name=walker checkpoint=models/eps/e7_stream_s8_ctrl_r1/fb_stage_8_flip.pickle tasks=stand,run,flip,supine,jump_any,kick_l,split,flamingo_r bench=final num_evals=20 ridge=0 eval_ckpt=models/eps/e7_stream_s8_ctrl_r1/fb_stage_8_flip.pickle eval_z=random model_dir=models/multi/m4_rand_ctrl
