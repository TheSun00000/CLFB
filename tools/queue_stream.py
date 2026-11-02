#!/usr/bin/env python3
"""Second main table: stream of 8 tasks (supine, kick_l, flamingo_r, split, jump_any, stand, run, flip), from the pretrained walker model, p = 0.25, ortho_coef 100, 3 seeds per row.
Per stage k (task t_k): [rows 3, 4: B of stage k from a stream run (L_rec, L_FB off, uniform z, 30k updates per stage, alpha 0.99; mu_b / mu_q = 0 / 0 or 300 / 0.5)], then F + actor
(finetune=multi, B frozen, only the latent of t_k, multi_task_frac 0.25, 200k updates, F and actor warm-started from the previous stage; row 1: F frozen too). All 8 tasks evaluated at the end of every stage (10 episodes).
Rows 1, 2 keep the pretrained B. Run names: s2_row<r>_s<seed>; stage k in models/multi/s2_row<r>_s<seed>_k<k>. Cells: tools/main_table.py with names ..._k8.
Usage: python tools/queue_stream.py
"""
import re, subprocess, sys, os
D = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(D, 'queue.py')
src = open(os.path.join(D, 'queue_eps.py')).read()
BASE = eval('(' + re.search(r"BASE = \((.*?)\)\n", src, re.S).group(1) + ')')
B0 = BASE.replace('eval_each_steps=10 ', 'eval_each_steps=0 ')
PRE = 'models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle'
T8 = 'stand,run,flip,supine,jump_any,kick_l,split,flamingo_r'
S8 = ['supine', 'kick_l', 'flamingo_r', 'split', 'jump_any', 'stand', 'run', 'flip']
STEPS = 200000


def stream_b(name, mu_b, mu_q):
    return ('python fb_continual_temp.py env_name=walker checkpoint=%s tasks=%s %s ortho_coef=100 alpha=0.99 mu_b=%s mu_q=%s fb_objective=0 z_task_frac=0 actor_z_task_frac=0 '
            'sigma_report=1 model_dir=models/eps/%s' % (PRE, ','.join(S8), B0, mu_b, mu_q, name))


def stage(name, k, ckpt, extra=''):
    t = S8[k - 1]
    return ('python fb_continual_temp.py checkpoint=%s env_name=walker tasks=%s n_labeled=100000 num_evals=10 lr_coef=1 bench=final eval_baseline=%d eval_rollouts=1 ridge=0 '
            'finetune=multi steps_per_stage=%d eval_each_steps=%d z_task_spread=0 eval_stream_only=0 tasks=%s multi_task_frac=0.25 %smodel_dir=models/multi/%s_k%d'
            % (ckpt, T8, 1 if k == 1 else 0, STEPS, STEPS // 1000, t, extra, name, k))


def chain(name, bname=None, extra=''):
    cmds = []
    if bname:
        cmds.append(stream_b(bname[0], bname[1], bname[2]))
    prev = PRE
    for k in range(1, 9):
        if bname:
            ck = 'models/multi/%s_ck%d.pickle' % (name, k)
            bk = 'models/eps/%s/fb_stage_%d_%s.pickle' % (bname[0], k, S8[k - 1])
            cmds.append('mkdir -p models/multi && python tools/merge_ckpt.py %s %s %s' % (ck, prev, bk))
            cmds.append(stage(name, k, ck, extra))
        else:
            cmds.append(stage(name, k, prev, extra))
        prev = 'models/multi/%s_k%d/fb_multi.pickle' % (name, k)
    return cmds


def add(job, title, cmds):
    subprocess.run([sys.executable, Q, 'add', job, '--time', '1:55:00', '--title', title] + sum([['--cmd', c] for c in cmds], []), check=True)


if __name__ == '__main__':
    for s in (1, 2, 3):
        add('s2_row1_s%d' % s, 'Stream table row 1 (seed %d): actor only, B and F frozen, 8 tasks in sequence, p = 0.25' % s, chain('s2_row1_s%d' % s, None, 'multi_freeze_f=1 '))
        add('s2_row2_s%d' % s, 'Stream table row 2 (seed %d): F + actor, B frozen, 8 tasks in sequence, p = 0.25' % s, chain('s2_row2_s%d' % s))
        add('s2_row3_s%d' % s, 'Stream table row 3 (seed %d): B stream without retention (mu 0 / 0) then F + actor per task, p = 0.25' % s, chain('s2_row3_s%d' % s, ('s2_row3_s%d_b' % s, 0, 0)))
        add('s2_row4_s%d' % s, 'Stream table row 4 (seed %d): B stream with retention (mu_b 300, mu_q 0.5) then F + actor per task, p = 0.25' % s, chain('s2_row4_s%d' % s, ('s2_row4_s%d_b' % s, 300, 0.5)))
