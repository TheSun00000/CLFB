#!/usr/bin/env python3
"""Round 14: stage 2 with NO anchors and no new loss (adapter, Q loss and actor loss of the method, 50-75% task latents / uniform latents), 200k updates, jump_any.
Stage-1 B's trained with a uniform nu only: fa2v (mu_b 300) and fa3mb1000 (mu_b 1000). Only hyper-parameters vary: (stage 1, lr, task share).
Settings: i1 (fa2v, 3e-4, .75), i2 (fa2v, 1e-4, .5), i3 (mb1000, 3e-4, .25), i4 (mb1000, 3e-4, .5), i5 (mb1000, 3e-4, .75), i6 (mb1000, 1e-4, .5).
Seed s1 on A100, seed s2 on V100. Report: python tools/fa_report.py r14"""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fa_cmds as fc
from fa_round1 import submit

CFGS = {'i1': ('fa2v', 3e-4, .75), 'i2': ('fa2v', 1e-4, .5), 'i3': ('fa3mb1000', 3e-4, .25), 'i4': ('fa3mb1000', 3e-4, .5), 'i5': ('fa3mb1000', 3e-4, .75), 'i6': ('fa3mb1000', 1e-4, .5)}

if __name__ == '__main__':
    for c, (pre, lr, f) in CFGS.items():
        for s, qos in (('s1', 'a100'), ('s2', 'v100')):
            cmd = fc.stage2('jump_any', s, 'r14_jump_any_%s_%s' % (c, s), lr=lr, frac=f, s1prefix=pre)
            submit('fa_r14_%s_%s' % (c, s), [cmd], qos=qos)
