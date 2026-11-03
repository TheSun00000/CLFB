#!/usr/bin/env python3
"""Cheetah main table (2026-11-03): single-task fine-tuning of the pretrained cheetah FB model on vel10 (run forward); same four rows as tools/queue_rows.py. Rows (all with ortho_coef 100 where B is trained, p = 0.25 in the F / actor phase, 3 seeds = 3 independent random seeds):
  row 1  actor only (B and F frozen, finetune=multi multi_freeze_f=1)            : cr_row1 (3 runs in one job)
  row 2  F + actor, B frozen                                                       : cr_row2 (3 runs in one job)
  row 3  stage 1 B (L_rec, L_FB off, uniform z, mu_b 0, mu_q 0, 30k) + stage 2 F + actor (300k): cr_row3_s1..s3
  row 4  the same with mu_b 300, mu_q 0.5 (seed 1 = job r1_nofb_uni)               : cr_row4_s2, cr_row4_s3
Stage 2 = finetune=multi, tasks=vel10, multi_task_frac=0.25, spread 0, all 6 tasks of the cheetah list evaluated (5 episodes), as single_run_uni_p25 (m23_f). Read with tools/multi_report.py cr_row.
Usage: python tools/queue_rows.py
"""
import re, subprocess, sys, os
D = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(D, 'queue.py')
src = open(os.path.join(D, 'queue_eps.py')).read()
BASE = eval('(' + re.search(r"BASE = \((.*?)\)\n", src, re.S).group(1) + ')')
B0 = BASE.replace('eval_each_steps=10 ', 'eval_each_steps=0 ')
PRE = 'saved_models/fb_agent_classical_cheetah.pickle'
T6 = 'vel10Back,vel10,flip,jump,kick_b,spread'


def stage1(name, mu_b, mu_q):
    return ('python fb_continual_temp.py env_name=cheetah checkpoint=%s tasks=vel10 %s ortho_coef=100 mu_b=%s mu_q=%s fb_objective=0 z_task_frac=0 actor_z_task_frac=0 '
            'sigma_report=1 model_dir=models/cheetah_eps/%s' % (PRE, B0, mu_b, mu_q, name))


def stage2(ckpt, name, extra=''):
    return ('python fb_continual_temp.py checkpoint=%s env_name=cheetah tasks=%s n_labeled=100000 num_evals=5 lr_coef=1 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 '
            'finetune=multi steps_per_stage=300000 eval_each_steps=30 z_task_spread=0 eval_stream_only=0 tasks=vel10 multi_task_frac=0.25 %smodel_dir=models/cheetah_multi/%s'
            % (ckpt, T6, extra, name))


def add(job, title, cmds):
    subprocess.run([sys.executable, Q, 'add', job, '--time', '1:55:00', '--title', title] + sum([['--cmd', c] for c in cmds], []), check=True)


if __name__ == '__main__':
    add('cr_row1', 'Main table row 1: actor only (B and F frozen), p = 0.25, from the pretrained model on run, 3 seeds: cr_row1_s1, s2, s3',
        [stage2(PRE, 'cr_row1_s%d' % k, 'multi_freeze_f=1 ') for k in (1, 2, 3)])
    add('cr_row2', 'Main table row 2: F + actor, B frozen, p = 0.25, from the pretrained model on run, 3 seeds: cr_row2_s1, s2, s3',
        [stage2(PRE, 'cr_row2_s%d' % k) for k in (1, 2, 3)])
    for k in (1, 2, 3):
        add('cr_row3_s%d' % k, 'Main table row 3 (seed %d): B stage without retention (L_FB off, uniform z, ortho 100, mu_b 0, mu_q 0, 30k) then F + actor p = 0.25 (300k)' % k,
            [stage1('cr_row3_s%d' % k, 0, 0), stage2('models/cheetah_eps/cr_row3_s%d/fb_stage_1_run.pickle' % k, 'cr_row3_s%d' % k)])
    for k in (1, 2, 3):
        add('cr_row4_s%d' % k, 'Main table row 4 (seed %d): B stage with retention (L_FB off, uniform z, ortho 100, mu_b 300, mu_q 0.5, 30k) then F + actor p = 0.25 (300k)' % k,
            [stage1('cr_row4_s%d' % k, 300, 0.5), stage2('models/cheetah_eps/cr_row4_s%d/fb_stage_1_run.pickle' % k, 'cr_row4_s%d' % k)])
