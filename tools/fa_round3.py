#!/usr/bin/env python3
"""Round 3: 200k updates (evaluation every 5k until 50k, then every 20k), adapter on, the new stage 1 (mu_q=0, V100 copy fa2v), tasks run and jump_any,
6 (learning rate, task share) settings x seeds s1, s2, packed 3 runs per A100 job."""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fa_cmds as fc
from fa_round1 import submit

CFGS = [(1e-4, 0.25), (3e-4, 0.25), (1e-3, 0.25), (3e-4, 0.5), (1e-3, 0.5), (1e-3, 1.0)]

if __name__ == '__main__':
    for task in ('run', 'jump_any'):
        runs = [(lr, f, s) for (lr, f) in CFGS for s in ('s1', 's2')]
        for j in range(4):
            mine = runs[j::4]
            cmds = [fc.stage2(task, s, 'r3_%s_lr%g_f%g_%s' % (task, lr, f, s), lr=lr, frac=f) for (lr, f, s) in mine]
            submit('fa_r3_%s_%d' % (task, j), cmds, qos='auto')
