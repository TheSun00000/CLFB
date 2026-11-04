#!/usr/bin/env python3
"""New stage 1 (mu_q=0) for run and jump_any, 3 seeds each, followed by the quality check tools/b_diag.py in the same job."""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fa_cmds as fc
from fa_round1 import submit

if __name__ == '__main__':
    for task in ('run', 'jump_any'):
        submit('fa2_s1_%s' % task, [fc.stage1(task, s) for s in ('s1', 's2', 's3')] + [fc.bdiag(task)])
