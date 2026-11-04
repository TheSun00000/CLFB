#!/usr/bin/env python3
"""Round 12: the recipe found on jump_any applied to run. Per seed one job: stage 1 on run (mu_q 0, mu_b 600, nu_const .8, nu_spread .5; models/eps/fa6_run_<s>) then stage 2
(adapter, lr 3e-4, task share .5, kd_actor 15, kd_const .5, 200k). Seeds s1, s3 on A100, s2 on V100. Report: python tools/fa_report.py r12 ; python tools/stage1_report.py fa6"""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fa_cmds as fc
from fa_round1 import submit

if __name__ == '__main__':
    for s, qos in (('s1', 'a100'), ('s2', 'v100'), ('s3', 'a100')):
        cmds = [fc.stage1('run', s, 'fa6', mu_b=600, extra='nu_const=0.8 nu_spread=0.5'),
                fc.stage2('run', s, 'r12_run_e1_%s' % s, lr=3e-4, frac=.5, extra='kd_actor=15 kd_const=0.5', s1prefix='fa6')]
        submit('fa_r12_%s' % s, cmds, qos=qos)
