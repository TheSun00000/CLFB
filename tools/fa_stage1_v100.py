#!/usr/bin/env python3
"""Duplicate of the new stage 1 (mu_q=0) on V100 dev, writing to models/eps/fa2v_<task>_<seed>; stage 2 then uses the copy that finishes first."""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fa_cmds as fc
from fa_round1 import submit

if __name__ == '__main__':
    for task in ('run', 'jump_any'):
        submit('fa2v_s1_%s' % task, [fc.stage1(task, s, 'fa2v') for s in ('s1', 's2', 's3')] + [fc.bdiag(task, 'fa2v')], qos='v100')
