#!/usr/bin/env python3
"""Stage 1 on jump_any, round 2 of the retention with the latent of the constant reward in nu: more weight on it, other mu_b, lower lam. 3 seeds, V100 and A100 copies.
Settings (mu_b, nu_const, nu_spread, lam): d1 (300, 1.0, .5, 100), d2 (600, .8, .5, 100), d3 (300, .8, .5, 100), d4 (300, .8, .5, 50), d5 (600, 1.0, .75, 100), d6 (100, .8, .5, 100).
Model dirs models/eps/fa5<v|a>d<i>_jump_any_<s>. Report: python tools/stage1_report.py fa5"""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fa_cmds as fc
from fa_round1 import submit

CFGS = {'d1': (300, 1.0, .5, 100), 'd2': (600, .8, .5, 100), 'd3': (300, .8, .5, 100), 'd4': (300, .8, .5, 50), 'd5': (600, 1.0, .75, 100), 'd6': (100, .8, .5, 100)}

if __name__ == '__main__':
    for qos, tag in (('v100', 'v'), ('a100', 'a')):
        for c, (mb, nc, sp, lam) in CFGS.items():
            pre = 'fa5%s%s' % (tag, c)
            cmds = [fc.stage1('jump_any', s, pre, mu_b=mb, extra='nu_const=%g nu_spread=%g lam=%g' % (nc, sp, lam)) for s in ('s1', 's2', 's3')]
            submit('fa5%s_%s' % (tag, c), cmds, qos=qos)
