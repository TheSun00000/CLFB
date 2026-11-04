#!/usr/bin/env python3
"""Reward-reconstruction correlation (and fit error) of chosen tasks at the start and at the end of a stage-1 run, from the eval_log.csv of the training folders listed in a slurm out file.
Cluster, repo root: python tools/stage1_corr.py <slurm out file> [task ...]   (default tasks: run jump_any)"""
import csv, re, sys

out = open(sys.argv[1]).read()
tasks = sys.argv[2:] or ['run', 'jump_any']
folders = [f for f in re.findall(r'LOGS_FOLDER\s*: \./(\S+)', out) if not f.endswith('bdiag')]
for folder in folders:
    rows = list(csv.DictReader(open('%s/eval_log.csv' % folder)))
    seed = re.search(r"'seed': (\d+)", open('%s/info.txt' % folder).read())
    start, end = rows[0], rows[-1]
    print('%s (seed %s; evals: %s)' % (folder, seed.group(1) if seed else '?', ','.join(r['tag'] for r in rows)))
    for t in tasks:
        g = lambda r, k: float(r['%s_fresh_%s' % (t, k)])
        print('   %-10s rew_corr %.3f -> %.3f    eps %.3f -> %.3f' % (t, g(start, 'rew_corr'), g(end, 'rew_corr'), g(start, 'eps'), g(end, 'eps')))
