#!/usr/bin/env python3
"""Round 11: speed through the optimisation of the critic on top of e1 (lr 3e-4, task share .5, kd_actor 15, kd_const .5; jump_any, d2 B): faster target network and larger batch.
Settings: g1 fb_target_tau .03, g2 fb_target_tau .1, g3 batch_size 2048. Seed s1 on A100, seed s2 on V100. Report: python tools/fa_report.py r11"""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fa_cmds as fc
from fa_round1 import submit

CFGS = {'g1': 'fb_target_tau=0.03', 'g2': 'fb_target_tau=0.1', 'g3': 'batch_size=2048'}

if __name__ == '__main__':
    for c, flag in CFGS.items():
        for s, qos in (('s1', 'a100'), ('s2', 'v100')):
            cmd = fc.stage2('jump_any', s, 'r11_jump_any_%s_%s' % (c, s), lr=3e-4, frac=.5, extra='kd_actor=15 kd_const=0.5 %s' % flag, s1prefix='fa5ad2')
            submit('fa_r11_%s_%s' % (c, s), [cmd], qos=qos)
