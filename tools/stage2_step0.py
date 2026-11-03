#!/usr/bin/env python3
"""Return on run at the start and end of stage 2 (finetune=multi) of the row-4 runs. Cluster, repo root: python tools/stage2_step0.py"""
import csv, glob, os

for info in sorted(glob.glob('logs/*/info.txt')):
    txt = open(info).read()
    if not any(('model_dir=models/multi/%s' % n) in txt for n in ('r2_row4_s2', 'r2_row4_s3', 'r1_nofb_uni_run_p25')):
        continue
    if 'finetune=multi' not in txt or 'eval_ckpt' in txt:
        continue
    d = os.path.dirname(info)
    rows = list(csv.DictReader(open(os.path.join(d, 'eval_log.csv'))))
    print('==', d, len(rows), 'evals')
    for r in rows[:4] + rows[-1:]:
        print('  step %-8s tag %-10s run_fresh %-8s stand %-8s flip %-8s split %-8s' % (r['step'], r['tag'], r.get('run_fresh_mean', '')[:6], r.get('stand_fresh_mean', '')[:6],
                                                                              r.get('flip_fresh_mean', '')[:6], r.get('split_fresh_mean', '')[:6]))
