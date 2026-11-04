#!/usr/bin/env python3
"""Round 6: around the best setting of round 5 (lr 3e-4, task share 1.0, kd_actor 30): anchor 20 / 50 / 60, lr 5e-4 / 6e-4, and a cosine decay of the lr
(lr 1e-3 over 50k updates, lr 3e-4 over 200k), plus the best setting on seed s3. Tasks run and jump_any, 200k updates."""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fa_cmds as fc
from fa_round1 import submit

# (lr, share, kd_actor, lr_decay_steps)
CFGS = [(3e-4, 1.0, 50, 0), (5e-4, 1.0, 30, 0), (1e-3, 1.0, 30, 50000), (3e-4, 1.0, 30, 200000), (6e-4, 1.0, 60, 0), (3e-4, 1.0, 20, 0)]

if __name__ == '__main__':
    for task in ('run', 'jump_any'):
        runs = [(lr, f, ka, dec, s) for (lr, f, ka, dec) in CFGS for s in ('s1', 's2')] + [(3e-4, 1.0, 30, 0, 's3')]
        for j in range(4):
            mine = runs[j::4]
            cmds = [fc.stage2(task, s, 'r6_%s_lr%g_f%g_ka%g_dec%d_%s' % (task, lr, f, ka, dec, s), lr=lr, frac=f,
                              extra='kd_actor=%g%s' % (ka, ' lr_decay_steps=%d' % dec if dec else '')) for (lr, f, ka, dec, s) in mine]
            submit('fa_r6_%s_%d' % (task, j), cmds, qos='auto')
