#!/usr/bin/env python3
"""Phase 2 of the eps study (2026-11-01): B frozen (checkpoints of the best retention setups), F trained with the Q loss and the actor with TD3 on ALL 8 task latents
(multi_task_frac of the z, the rest uniform), no FB objective, 500k updates, returns evaluated. Script: fb_continual_temp.py finetune=multi.
Usage: python tools/queue_multi.py smoke | <wave>
"""
import subprocess, sys, os
Q = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queue.py')
T8 = 'stand,run,flip,supine,jump_any,kick_l,split,flamingo_r'
BASE = ('env_name=walker tasks=%s n_labeled=100000 num_evals=5 lr_coef=1 z_task_spread=0.2 multi_task_frac=0.5 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 finetune=multi' % T8)
S8 = 'models/eps/e7_stream_s8_a099_mb300_r1/fb_stage_8_flip.pickle'


def add(job, title, ckpt, steps, every, tlimit, extra=''):
    cmd = 'python fb_continual_temp.py checkpoint=%s %s steps_per_stage=%d eval_each_steps=%d %s model_dir=models/multi/%s' % (ckpt, BASE, steps, every, extra, job)
    subprocess.run([sys.executable, Q, 'add', job, '--time', tlimit, '--title', title, '--cmd', cmd], check=True)


if __name__ == '__main__':
    if sys.argv[1] == 'smoke':
        add('m0_smoke', 'Multi-task F + actor training, smoke test (10k updates, speed check)', S8, 10000, 5, '0:40:00')
