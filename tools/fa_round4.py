#!/usr/bin/env python3
"""Round 4: anchors of the actor (kd_actor) and of Q (kd_q) to the starting model (the pretrained nets seen through the adapter) on GENERIC latents (uniform sphere,
a fraction B(s) of shuffled states), lr 3e-4, task share 0.5, 200k updates, tasks run and jump_any, seeds s1 s2 (A100)."""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fa_cmds as fc
from fa_round1 import submit

KD = [(kq, ka) for kq in (0, 1, 10) for ka in (0, 30, 300) if (kq, ka) != (0, 0)]

if __name__ == '__main__':
    for task in ('run', 'jump_any'):
        runs = [(kq, ka, s) for (kq, ka) in KD for s in ('s1', 's2')]
        for j in range(4):
            mine = runs[j::4]
            cmds = [fc.stage2(task, s, 'r4_%s_kq%g_ka%g_%s' % (task, kq, ka, s), lr=3e-4, frac=0.5, extra='kd_q=%g kd_actor=%g' % (kq, ka)) for (kq, ka, s) in mine]
            submit('fa_r4_%s_%d' % (task, j), cmds, qos='auto')
