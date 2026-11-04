#!/usr/bin/env python3
"""Round 5: refine the actor anchor (kd_actor) and retest larger learning rates / task shares with the anchor, tasks run and jump_any, seeds s1 s2, 200k updates.
Configs (lr, task share, kd_actor, kd_q): the anchor strength between 10 and 100 at lr 3e-4, and lr 1e-3 / share 1.0 with the anchor."""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fa_cmds as fc
from fa_round1 import submit

CFGS = [(3e-4, 0.5, 10, 0), (3e-4, 0.5, 100, 0), (1e-3, 0.5, 30, 0), (1e-3, 0.5, 100, 0), (1e-3, 1.0, 100, 0), (3e-4, 1.0, 30, 0)]

if __name__ == '__main__':
    for task in ('run', 'jump_any'):
        runs = [(lr, f, ka, kq, s) for (lr, f, ka, kq) in CFGS for s in ('s1', 's2')]
        for j in range(4):
            mine = runs[j::4]
            cmds = [fc.stage2(task, s, 'r5_%s_lr%g_f%g_ka%g_kq%g_%s' % (task, lr, f, ka, kq, s), lr=lr, frac=f, extra='kd_q=%g kd_actor=%g' % (kq, ka)) for (lr, f, ka, kq, s) in mine]
            submit('fa_r5_%s_%d' % (task, j), cmds, qos='auto')
