#!/usr/bin/env python3
"""Fills the local queue (tools/queue.py) with the experiments of the final task list: stand, run, flip, supine, jump_any, kick_l, split, flamingo_r.

  1. B-only learnability of the 8 tasks (tools/b_only_fit.py)                                        1 job
  2. TD3 on the true reward of each task, 1M updates, the upper bound (offline_rl_approx.py)           8 jobs
  3. the best setup (lam 100, focus 0.5, noise 0.2, actor focus 1.0; 500k updates, 100k labels) fine-tuned on ONE task, the return on that task
     evaluated during training, then a final zero-shot evaluation on all 8 tasks (eval-only) in the same job   7 jobs (run: model of array 1788307)
  4. final evaluation on the 8 tasks of the pretrained model and of the run model                      1 job
Order of the list = order of submission. Run once: python tools/queue_final.py
"""
import os, subprocess, sys

Q = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queue.py')
TASKS = ['supine', 'jump_any', 'kick_l', 'split', 'flamingo_r', 'stand', 'run', 'flip']
RUN_MODEL = 'models/2026-10-31_18-57-15_FBcontinual-walker-run_5/fb_stage_1_run.pickle'      # job 1788356.0 of array 1788307: lam 100, focus 0.5, noise 0.2, actor focus 1.0
SETUP = 'n_labeled=100000 num_evals=20 mu_b=0 mu_q=0 steps_per_stage=500000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.5 actor_z_task_frac=1.0 z_task_spread=0.2 ztask_bs=4096 bench=final'


def add(name, time, cmds, title):
    args = [sys.executable, Q, 'add', name, '--time', time, '--title', title]
    for c in cmds:
        args += ['--cmd', c]
    subprocess.run(args, check=True)


add('final_bonly', '1:30:00', ['python tools/b_only_fit.py --pre $CKPT'], 'B-only learnability of the 8 tasks (10k and 100k labels)')
for t in TASKS:
    add('final_td3_' + t, '1:15:00', ['python offline_rl_approx.py env_name=walker task=%s reward_approx=true num_evals=20' % t], 'TD3 on the true reward, %s (upper bound)' % t)
for t in [x for x in TASKS if x != 'run']:
    add('final_ft_' + t, '1:30:00', [
        'python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=%s %s eval_each_steps=10 eval_baseline=1 eval_stream_only=1 model_dir=models/final8/%s' % (t, SETUP, t),
        'python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=%s bench=final num_evals=20 eval_ckpt=models/final8/%s/fb_stage_1_%s.pickle' % (t, t, t)],
        'Best setup fine-tuned on %s, then zero-shot evaluation on the 8 tasks' % t)
add('final_eval_base', '0:45:00', [
    'python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run bench=final num_evals=20 eval_ckpt=$CKPT',
    'python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run bench=final num_evals=20 eval_ckpt=%s' % RUN_MODEL],
    'Zero-shot evaluation on the 8 tasks of the pretrained model and of the run model')
