#!/usr/bin/env python3
"""Stage-1 quality of the fa4 runs: reward-reconstruction correlation (rew_corr) and fit error (eps) at the start and at the end of stage 1, for the target (jump_any), run and the
mean over the six other tasks, per setting (mean over seeds). Cluster, repo root: python tools/stage1_report.py [prefix]  (default fa4)"""
import csv, glob, os, re, sys
import numpy as np

prefix = sys.argv[1] if len(sys.argv) > 1 else 'fa4'
TASKS = ['stand', 'run', 'flip', 'supine', 'jump_any', 'kick_l', 'split', 'flamingo_r']
cfgs = {}
for info in sorted(glob.glob('logs/*/info.txt')):
    txt = open(info).read()
    m = re.search(r'model_dir=models/eps/(%s\w*?)_jump_any_(s\d)' % prefix, txt)
    if not m:
        continue
    rows = list(csv.DictReader(open(os.path.join(os.path.dirname(info), 'eval_log.csv'))))
    if len(rows) < 2:
        continue
    cfgs.setdefault(m.group(1), []).append((rows[0], rows[-1]))
g = lambda r, t, k: float(r['%s_fresh_%s' % (t, k)])
print('%-10s %2s | %-22s | %-22s | %-30s' % ('setting', 'n', 'jump_any corr (eps)', 'run corr (eps)', 'others corr start->end (eps)'))
for c, runs in sorted(cfgs.items()):
    oth = [t for t in TASKS if t not in ('jump_any', 'run')]
    f = lambda t, k, i: np.mean([g(r[i], t, k) for r in runs])
    fo = lambda k, i: np.mean([np.mean([g(r[i], t, k) for t in oth]) for r in runs])
    print('%-10s %2d | %.3f->%.3f (%.3f)  | %.3f->%.3f (%.3f)  | %.3f->%.3f (%.3f->%.3f)' % (
        c, len(runs), f('jump_any', 'rew_corr', 0), f('jump_any', 'rew_corr', 1), f('jump_any', 'eps', 1), f('run', 'rew_corr', 0), f('run', 'rew_corr', 1), f('run', 'eps', 1),
        fo('rew_corr', 0), fo('rew_corr', 1), fo('eps', 0), fo('eps', 1)))
