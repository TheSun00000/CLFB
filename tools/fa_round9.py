#!/usr/bin/env python3
"""Round 9: the noisy constant-reward latent in the stage-2 anchor latents (kd_const), jump_any, stage 1 = the d2 B (models/eps/fa5ad2_*), lr 3e-4, task share .5, 200k updates.
Settings (kd_actor, kd_q, kd_const): e1 (15, 0, .5), e2 (30, 0, .5), e3 (15, 1, .5), e4 (30, 0, .8). Seed s1 on A100, seed s2 on V100 (one run per job)."""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fa_cmds as fc
from fa_round1 import submit

CFGS = {'e1': (15, 0, .5), 'e2': (30, 0, .5), 'e3': (15, 1, .5), 'e4': (30, 0, .8)}

if __name__ == '__main__':
    for c, (ka, kq, kc) in CFGS.items():
        for s, qos in (('s1', 'a100'), ('s2', 'v100')):
            cmd = fc.stage2('jump_any', s, 'r9_jump_any_%s_%s' % (c, s), lr=3e-4, frac=0.5, extra='kd_actor=%g kd_q=%g kd_const=%g' % (ka, kq, kc), s1prefix='fa5ad2')
            submit('fa_r9_%s_%s' % (c, s), [cmd], qos=qos)
