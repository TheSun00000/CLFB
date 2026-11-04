#!/usr/bin/env python3
"""Round 14b: seed s1 of round 14 on V100 (its A100 jobs were cancelled at the user's request: V100 only). Usage: python tools/fa_round14b.py i4 i3 i1 ..."""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fa_cmds as fc
from fa_round1 import submit
from fa_round14 import CFGS

if __name__ == '__main__':
    for c in sys.argv[1:]:
        pre, lr, f = CFGS[c]
        cmd = fc.stage2('jump_any', 's1', 'r14_jump_any_%s_s1' % c, lr=lr, frac=f, s1prefix=pre)
        submit('fa_r14b_%s_s1' % c, [cmd], qos='v100')
