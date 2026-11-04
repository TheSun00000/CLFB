#!/usr/bin/env python3
"""Round 6b (user: no 100% task latents, the random latents must stay in the training distribution): task share 0.5 / 0.25 only, lr 3e-4 unless said, kd_actor 30 unless said.
Configs (lr, share, kd_actor, lr_decay_steps): share .25; anchor 50 and 15; lr 5e-4; cosine decay (lr 1e-3 over 50k, lr 3e-4 over 200k); plus the round-4 setting
(3e-4, .5, 30) on seed s3. Tasks run and jump_any, 200k updates. The first two jobs go to A100, the rest to V100 (A100 slots taken by the pending round 6), 3 runs per V100 job."""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fa_cmds as fc
from fa_round1 import submit

CFGS = [(3e-4, 0.25, 30, 0), (3e-4, 0.5, 50, 0), (5e-4, 0.5, 30, 0), (1e-3, 0.5, 30, 50000), (3e-4, 0.5, 30, 200000), (3e-4, 0.5, 15, 0)]

if __name__ == '__main__':
    runs = []
    for seed in ('s1', 's2'):
        for cfg in CFGS:
            for task in ('run', 'jump_any'):
                runs.append((task, cfg, seed))
    for task in ('run', 'jump_any'):
        runs.append((task, (3e-4, 0.5, 30, 0), 's3'))
    cmds = [fc.stage2(t, s, 'r6b_%s_lr%g_f%g_ka%g_dec%d_%s' % (t, lr, f, ka, dec, s), lr=lr, frac=f,
                      extra='kd_actor=%g%s' % (ka, ' lr_decay_steps=%d' % dec if dec else '')) for (t, (lr, f, ka, dec), s) in runs]
    sizes = [4, 4] + [3] * 6
    assert sum(sizes) == len(cmds), (sum(sizes), len(cmds))
    i = 0
    for j, n in enumerate(sizes):
        submit('fa_r6b_%d' % j, cmds[i:i + n], qos='auto')
        i += n
