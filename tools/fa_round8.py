#!/usr/bin/env python3
"""Round 8: stage 2 (200k updates, adapter, lr 3e-4, task share .5, kd_actor 15) on the new stage-1 B's of jump_any with the constant-reward latent in nu.
Usage: python tools/fa_round8.py <stage-1 prefix> <label> [<qos> ...]   e.g.  python tools/fa_round8.py fa5ad2 d2 v100 a100
Jobs: one run per job (seeds s1, s2) per partition; tags r8_<label><v|a>_<seed>."""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fa_cmds as fc
from fa_round1 import submit

if __name__ == '__main__':
    prefix, label = sys.argv[1], sys.argv[2]
    qoss = sys.argv[3:] or ['v100', 'a100']
    for qos in qoss:
        tag = qos[0]
        for s in ('s1', 's2'):
            cmd = fc.stage2('jump_any', s, 'r8_jump_any_%s%s_lr0.0003_f0.5_ka15_%s' % (label, tag, s), lr=3e-4, frac=0.5, extra='kd_actor=15', s1prefix=prefix)
            submit('fa_r8_%s%s_%s' % (label, tag, s), [cmd], qos=qos)
