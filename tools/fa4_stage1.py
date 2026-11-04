#!/usr/bin/env python3
"""Stage 1 on jump_any with the latent of the constant reward in nu (nu_const), mu_q=0, 3 seeds, V100 and A100 copies (the first to finish is used).
Settings (mu_b, nu_const, nu_spread): c1 (1000, .5, .5), c2 (1000, .8, .5), c3 (300, .5, .5), c4 (1000, .5, 1.0), c5 (3000, .5, .5).
Model dirs: models/eps/fa4<v|a><c>_jump_any_<s1|s2|s3>. Report: python tools/stage1_report.py"""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fa_cmds as fc
from fa_round1 import submit

CFGS = {'c1': (1000, .5, .5), 'c2': (1000, .8, .5), 'c3': (300, .5, .5), 'c4': (1000, .5, 1.0), 'c5': (3000, .5, .5)}

if __name__ == '__main__':
    for qos, tag in (('v100', 'v'), ('a100', 'a')):
        for c, (mb, nc, sp) in CFGS.items():
            pre = 'fa4%s%s' % (tag, c)
            cmds = [fc.stage1('jump_any', s, pre, mu_b=mb, extra='nu_const=%g nu_spread=%g' % (nc, sp)) for s in ('s1', 's2', 's3')]
            submit('fa4%s_%s' % (tag, c), cmds, qos=qos)
