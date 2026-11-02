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
