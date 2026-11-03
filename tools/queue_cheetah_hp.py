#!/usr/bin/env python3
"""Cheetah, row 4 hyper-parameters (2026-11-03). Round 1 = B stage only (one task vel10, 30k updates, L_FB off, uniform latents, ortho_coef 100), eps only (no rollouts),
grid lam x mu_b x mu_q (27 runs, 1 seed, 9 jobs of 3 runs). Metric (tools/cheetah_hp_report.py): eps of vel10 after the stage (learning) and of the 5 other tasks (retention) against step 0.
Round 2 (python tools/queue_cheetah_hp.py full lam mu_b mu_q ...): the whole pipeline (stage 1 + F / actor p = 0.25, 300k updates) for the chosen configs, 3 seeds.
Usage: python tools/queue_cheetah_hp.py sweep | full <lam>,<mu_b>,<mu_q> [...]
"""
import re, subprocess, sys, os
D = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(D, 'queue.py')
sys.path.insert(0, D)
import queue_rows_cheetah as R
BASE = R.B0
CK = R.PRE


def stage1(name, lam, mb, mq):
    return ('python fb_continual_temp.py env_name=cheetah checkpoint=%s tasks=vel10 %s lam=%s ortho_coef=100 mu_b=%s mu_q=%s fb_objective=0 z_task_frac=0 actor_z_task_frac=0 '
            'sigma_report=1 model_dir=models/cheetah_hp/%s' % (CK, BASE, lam, mb, mq, name))


def add(job, title, cmds):
    subprocess.run([sys.executable, Q, 'add', job, '--time', '1:55:00', '--title', title] + sum([['--cmd', c] for c in cmds], []), check=True)


def tag(lam, mb, mq): return 'l%s_b%s_q%s' % (lam, mb, mq)


if __name__ == '__main__':
    if sys.argv[1] == 'sweep':
        for lam in (30, 100, 300):
            for mb in (100, 300, 1000):
                cmds = [stage1('c4_' + tag(lam, mb, mq), lam, mb, mq) for mq in (0.1, 0.5, 2)]
                add('c4_l%s_b%s' % (lam, mb), 'Cheetah hyper-parameters, B stage, eps only: lam %s, mu_b %s, mu_q 0.1 / 0.5 / 2' % (lam, mb), cmds)
    if sys.argv[1] == 'full':
        for cfg in sys.argv[2:]:
            lam, mb, mq = cfg.split(',')
            for s in (1, 2, 3):
                n = 'c5_%s_s%d' % (tag(lam, mb, mq), s)
                cmds = [stage1(n, lam, mb, mq), R.stage2('models/cheetah_hp/%s/fb_stage_1_vel10.pickle' % n, n).replace('models/cheetah_multi/', 'models/cheetah_hpm/')]
                add(n, 'Cheetah row 4 hyper-parameters, full pipeline: lam %s, mu_b %s, mu_q %s, seed %d' % (lam, mb, mq, s), cmds)
