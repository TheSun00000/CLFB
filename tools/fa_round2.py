#!/usr/bin/env python3
"""Round 2: 200k updates (evaluation 5k/20k), task run, adapter on, A100 dev (evaluation every 5k up to 50k, then every 20k). Three learning-rate / task-share settings x seeds s2, s3."""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fa_cmds as fc
from fa_round1 import submit

if __name__ == '__main__':
    for lr, f in ((1e-4, 0.25), (3e-4, 0.5), (1e-3, 0.5)):
        cmds = [fc.stage2('run', s, 'r2_lr%g_f%g_%s' % (lr, f, s), lr=lr, frac=f) for s in ('s2', 's3')]
        submit('fa_r2_lr%g_f%g' % (lr, f), cmds, qos='a100')
