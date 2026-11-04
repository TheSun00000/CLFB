#!/usr/bin/env python3
"""Round 7 (A100). (A) stronger retention in stage 1 for jump_any: mu_b 1000 and 3000 (the jump_any B encodes run with fit error 0.5, losing ~100 return on run that stage 2 cannot
fix), 2 seeds, then b_diag and stage 2 (lr 3e-4, task share .5, kd_actor 15) on the new B, in the same job. (B) stage-2 refinements on the existing stage 1 (fa2v): lr 5e-4, share .75,
kd_actor 10, and the best setting (3e-4, .5, 15) on seed s3, both tasks."""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fa_cmds as fc
from fa_round1 import submit

if __name__ == '__main__':
    for mb in (1000, 3000):
        for s in ('s1', 's2'):
            pre = 'fa3mb%d' % mb
            cmds = [fc.stage1('jump_any', s, pre, mu_b=mb), fc.bdiag1('jump_any', s, pre),
                    fc.stage2('jump_any', s, 'r7_jump_any_mb%d_lr0.0003_f0.5_ka15_%s' % (mb, s), lr=3e-4, frac=0.5, extra='kd_actor=15', s1prefix=pre)]
            submit('fa_r7_mb%d_%s' % (mb, s), cmds, qos='auto')
    cfgs = [(5e-4, 0.5, 10 + 5), (3e-4, 0.75, 15), (3e-4, 0.5, 10)]
    runs = [(t, cfg, s) for cfg in cfgs for t in ('run', 'jump_any') for s in ('s1', 's2')] + [(t, (3e-4, 0.5, 15), 's3') for t in ('run', 'jump_any')]
    cmds = [fc.stage2(t, s, 'r7_%s_lr%g_f%g_ka%g_%s' % (t, lr, f, ka, s), lr=lr, frac=f, extra='kd_actor=%g' % ka) for (t, (lr, f, ka), s) in runs]
    n = 4
    for j in range(0, len(cmds), 4):
        submit('fa_r7_s2_%d' % (j // 4), cmds[j:j + 4], qos='auto')
