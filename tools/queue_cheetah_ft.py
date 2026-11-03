#!/usr/bin/env python3
"""Cheetah, fine-tuning on vel10 from the NEW pretrained model (2026-11-03: saved_models/fb_agent_cheetah_pre2.pickle, FB pretraining of this repo, 1M updates; the old
fb_agent_classical_cheetah.pickle is no longer used). Round 1 (B frozen, no stage 1): stage 2 only (finetune=multi, 300k updates, tasks= the 6 cheetah tasks evaluated, vel10 trained),
1 seed, varying p (multi_task_frac), lr (F and actor), z_task_spread, bc_coef, and F frozen or not. Jobs ft1_*: two runs each. Read with tools/multi_report.py ft1 .
Usage: python tools/queue_cheetah_ft.py r1
"""
import subprocess, sys, os
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import queue_rows_cheetah as R
Q = os.path.join(D, 'queue.py')
PRE2 = 'saved_models/fb_agent_cheetah_pre2.pickle'


def run(name, extra):
    return R.stage2(PRE2, name, extra + ' ').replace('models/cheetah_multi/', 'models/cheetah_ft/')


def add(job, title, cmds, **kw):
    subprocess.run([sys.executable, Q, 'add', job, '--time', '1:55:00', '--title', title] + sum([['--cmd', c] for c in cmds], []), check=True)


if __name__ == '__main__' and sys.argv[1] == 'r1':
    cfgs = [('ft1_fa_p25', ''), ('ft1_fa_p50', 'multi_task_frac=0.5'), ('ft1_fa_p100', 'multi_task_frac=1'),
            ('ft1_fa_p25_lr3', 'lr=3e-5'), ('ft1_fa_p50_lr3', 'multi_task_frac=0.5 lr=3e-5'), ('ft1_fa_p100_lr3', 'multi_task_frac=1 lr=3e-5'),
            ('ft1_act_p25', 'multi_freeze_f=1'), ('ft1_act_p100', 'multi_freeze_f=1 multi_task_frac=1'),
            ('ft1_fa_p50_sp3', 'multi_task_frac=0.5 z_task_spread=0.3'), ('ft1_fa_p50_bc', 'multi_task_frac=0.5 bc_coef=0.1')]
    for i in range(0, len(cfgs), 2):
        pair = cfgs[i:i + 2]
        add('ft1_%d' % (i // 2), 'Cheetah finetune vel10 (new pretrained model, B frozen): ' + ', '.join(n for n, _ in pair),
            [run(n, e) for n, e in pair])
