#!/usr/bin/env python3
"""Round 10: recover the speed of the best retention setting of round 9 (e1: kd_actor 15, kd_const .5, lr 3e-4, task share .5) on the d2 B (jump_any, models/eps/fa5ad2_*).
Settings (lr, task share, kd_actor, kd_const): f1 (5e-4, .5, 15, .5), f2 (1e-3, .5, 15, .5), f3 (3e-4, .5, 7, .5), f4 (3e-4, .75, 15, .5), f5 (5e-4, .5, 30, .5).
Seed s1 on A100, seed s2 on V100 (one run per job). Report: python tools/fa_report.py r10"""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fa_cmds as fc
from fa_round1 import submit

CFGS = {'f1': (5e-4, .5, 15, .5), 'f2': (1e-3, .5, 15, .5), 'f3': (3e-4, .5, 7, .5), 'f4': (3e-4, .75, 15, .5), 'f5': (5e-4, .5, 30, .5)}

if __name__ == '__main__':
    for c, (lr, f, ka, kc) in CFGS.items():
        for s, qos in (('s1', 'a100'), ('s2', 'v100')):
            cmd = fc.stage2('jump_any', s, 'r10_jump_any_%s_%s' % (c, s), lr=lr, frac=f, extra='kd_actor=%g kd_const=%g' % (ka, kc), s1prefix='fa5ad2')
            submit('fa_r10_%s_%s' % (c, s), [cmd], qos=qos)
