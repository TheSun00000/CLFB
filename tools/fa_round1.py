#!/usr/bin/env python3
"""Round 1 of the fast-adaptation search, task run (existing stage-1 checkpoints): learning rate x share of task latents, adapter on, plus the no-adapter baseline.
30k updates, evaluation of the 8 tasks every 2k updates, 5 episodes, seeds s2 and s3. Also launches stage 1 on jump_any (3 seeds) for round 2."""
import itertools, subprocess, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import fa_cmds as fc


def submit(name, cmds, qos='a100'):
    args = ['python', __file__.rsplit('/', 1)[0] + '/overnight.py', 'submit', name, '--qos', qos]
    for c in cmds:
        args += ['--cmd', c]
    print(name, subprocess.run(args, capture_output=True, text=True).stdout.strip())


if __name__ == '__main__':
    submit('fa_s1_jump_any', [fc.stage1('jump_any', s) for s in ('s1', 's2', 's3')])
    cfgs = [(True, lr, f) for lr in (1e-4, 3e-4, 1e-3) for f in (0.25, 0.5, 1.0)] + [(False, 1e-4, 0.25)]
    runs = [(ad, lr, f, s) for (ad, lr, f) in cfgs for s in ('s2', 's3')]
    n_jobs = 5
    for j in range(n_jobs):
        mine = runs[j::n_jobs]
        cmds = [fc.stage2('run', s, 'r1_%s_lr%g_f%g_%s' % ('ad' if ad else 'no', lr, f, s), lr=lr, frac=f, adapter=ad) for (ad, lr, f, s) in mine]
        submit('fa_r1_%d' % j, cmds)
