#!/usr/bin/env python3
"""cos between the latents of two tasks before (pretrained B) and after stage 1, from the C_intertask_{pre,new}.csv of a b_diag folder.
Cluster, repo root: python tools/intertask.py <slurm out file of the job that ran b_diag> [task_a task_b]"""
import csv, re, sys

TASKS = ['stand', 'walk', 'vel8', 'run', 'flip', 'supine', 'jump_any', 'kick_l', 'split', 'flamingo_r']
out = open(sys.argv[1]).read()
a, b = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ('run', 'jump_any')
folders = [f for f in re.findall(r'LOGS_FOLDER\s*: \./(\S+)', out) if f.endswith('bdiag')]
for folder in folders:
    rows = {k: list(csv.reader(open('%s/C_intertask_%s.csv' % (folder, k)))) for k in ('pre', 'new')}
    names = []
    for r in rows['pre'][1:]:
        if r[0] not in names:
            names.append(r[0])
    print(folder)
    for n in names:
        res = {}
        for k in ('pre', 'new'):
            block = [r for r in rows[k][1:] if r[0] == n]
            res[k] = float(block[TASKS.index(a)][1 + TASKS.index(b)])
        print('  %-24s cos(z_%s, z_%s): pretrained B %.3f -> B after stage 1 %.3f' % (n, a, b, res['pre'], res['new']))
