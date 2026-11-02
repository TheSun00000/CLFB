#!/usr/bin/env python3
"""Cheetah study (2026-11-02). Tasks (tools/cheetah_design.py): vel10Back (run backward), vel10 (run forward), flip + jump, kick_b, spread (defined on the observation,
|corr| <= 0.23 with each other and with the built-ins on the dataset, random-action return <= 150/1000 except flip 149).
Waves:  td3  = offline TD3 on the true reward of each task, 1M updates, 2 seeds (the upper bound). 12 runs, 3 processes in parallel inside each job (GPU is not saturated), 4 jobs.
Usage: python tools/queue_cheetah.py <wave>
"""
import subprocess, sys, os
Q = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queue.py')
TASKS6 = ['vel10Back', 'vel10', 'flip', 'jump', 'kick_b', 'spread']


def add(job, title, cmds, tlimit='1:55:00'):
    args = [sys.executable, Q, 'add', job, '--time', tlimit, '--title', title]
    for c in cmds:
        args += ['--cmd', c]
    subprocess.run(args, check=True)


if __name__ == '__main__':
    w = sys.argv[1]
    if w == 'td3':
        for seed in (1, 2):
            for k, ts in enumerate((TASKS6[:3], TASKS6[3:])):
                cmd = ' & '.join('python offline_rl_approx.py env_name=cheetah task=%s reward_approx=true num_evals=20 seed=%d' % (t, seed) for t in ts) + ' & wait'
                add('c1_td3_s%d_%d' % (seed, k), 'Cheetah: offline TD3 on the true reward, 1M updates, seed %d: %s (3 runs in parallel)' % (seed, ', '.join(ts)), [cmd])
    if w == 'c2':
        # round 1: one-task fine-tuning of the pretrained cheetah B, eps only (30k updates), the walker setup (tools/queue_eps.py BASE): control, mu_q 0.5, mu_b 100 + mu_q 0.5, mu_b 300 + mu_q 0.5.
        # 6 tasks x 2 seeds = 12 runs per job (about 7 min each), one job per setting.
        BASE = ('n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.5 actor_z_task_frac=0.5 z_task_spread=0.2 '
                'ztask_bs=4096 bench=final eval_each_steps=10 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0')
        CK = 'saved_models/fb_agent_classical_cheetah.pickle'
        for name, mu in (('ctrl', 'mu_b=0 mu_q=0'), ('mq05', 'mu_b=0 mu_q=0.5'), ('mb100mq05', 'mu_b=100 mu_q=0.5'), ('mb300mq05', 'mu_b=300 mu_q=0.5')):
            cmds = ['python fb_continual_temp.py env_name=cheetah checkpoint=%s tasks=%s %s %s model_dir=models/cheetah/c2_%s_%s_r%d' % (CK, t, BASE, mu, name, t, s)
                    for t in TASKS6 for s in (1, 2)]
            add('c2_' + name, 'Cheetah round 1: one-task fine-tuning, eps only, %s (6 tasks x 2 seeds, 12 runs of 30k updates)' % mu, cmds)
