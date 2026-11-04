#!/usr/bin/env python3
"""Round 10b: the seed s1 of round 10, whose A100 jobs were cancelled while pending (at 13:11:19 by the cluster account, not by this session), rerun on V100.
Same settings as tools/fa_round10.py."""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fa_cmds as fc
from fa_round1 import submit
from fa_round10 import CFGS

if __name__ == '__main__':
    for c, (lr, f, ka, kc) in CFGS.items():
        cmd = fc.stage2('jump_any', 's1', 'r10_jump_any_%s_s1' % c, lr=lr, frac=f, extra='kd_actor=%g kd_const=%g' % (ka, kc), s1prefix='fa5ad2')
        submit('fa_r10b_%s_s1' % c, [cmd], qos='v100')
