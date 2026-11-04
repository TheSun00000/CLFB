#!/usr/bin/env python3
"""Per-task return of fast-adaptation configurations (mean over seeds) at chosen steps, with the change against the pretrained model.
Cluster, repo root: python tools/fa_detail.py <tag prefix of the config, e.g. r3_run_lr0.0003_f0.5> [more prefixes]"""
import csv, glob, os, re, sys
import numpy as np

PRETRAINED = {'run': 345, 'stand': 715, 'flip': 517, 'supine': 985, 'split': 960, 'flamingo_r': 794, 'jump_any': 132, 'kick_l': 77}
TASKS = list(PRETRAINED)
STEPS = [0, 5000, 10000, 20000, 50000, 100000, 200000]
runs = {}
for info in sorted(glob.glob('logs/*/info.txt')):
    m = re.search(r'model_dir=models/multi/fa_(\S+)', open(info).read())
    if m:
        runs[m.group(1)] = os.path.dirname(info)
for prefix in sys.argv[1:]:
    tags = [t for t in runs if re.sub(r'_s[123]$', '', t) == prefix]
    if not tags:
        print('no run for', prefix)
        continue
    per = []
    for t in tags:
        rows = list(csv.DictReader(open(os.path.join(runs[t], 'eval_log.csv'))))
        per.append({int(r['step']): {k: float(r.get('%s_fresh_mean' % k) or 'nan') for k in TASKS} for r in rows})
    print('== %s (%d seeds)' % (prefix, len(per)))
    print('%-8s' % 'step' + ''.join('%11s' % k[:10] for k in TASKS))
    for s in STEPS:
        if all(s in p for p in per):
            print('%-8d' % s + ''.join('%11.0f' % np.mean([p[s][k] for p in per]) for k in TASKS))
    last = [max(p) for p in per]
    fin = {k: np.mean([np.mean([p[s][k] for s in sorted(p)[-3:]]) for p in per]) for k in TASKS}
    print('%-8s' % 'last3' + ''.join('%11.0f' % fin[k] for k in TASKS))
    print('%-8s' % 'vs pre' + ''.join('%+11.0f' % (fin[k] - PRETRAINED[k]) for k in TASKS))
