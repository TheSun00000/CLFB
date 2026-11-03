#!/usr/bin/env python3
"""Cells of a main-table row (docs: maintable/single_task): per seed mean of the last 3 evaluations, then mean and std over seeds.
Prints both std conventions (population / sample) for row 4 and for b_adapter, to match the published cells. Cluster, repo root: python tools/table_row.py"""
import csv, glob, os, re
import numpy as np

TASKS = ['run', 'stand', 'flip', 'supine', 'split', 'flamingo_r', 'jump_any', 'kick_l']
NAMES = {'r1_nofb_uni': 'r1_nofb_uni_run_p25', 'r2_row4_s2': 'r2_row4_s2', 'r2_row4_s3': 'r2_row4_s3'}
found = {}
for info in sorted(glob.glob('logs/*/info.txt')):
    txt = open(info).read()
    m = re.search(r'model_dir=models/multi/(\S+)', txt)
    if m and 'finetune=multi' in txt and 'eval_ckpt' not in txt and 'diag=' not in txt:
        found[m.group(1)] = os.path.dirname(info)


def last3(folder):
    rows = list(csv.DictReader(open(os.path.join(folder, 'eval_log.csv'))))[-3:]
    return {t: np.mean([float(r['%s_fresh_mean' % t]) for r in rows]) for t in TASKS}


for label, keys in (('row 4', [NAMES[s] for s in NAMES]), ('row 5 (b_adapter)', ['adapter_' + s for s in NAMES])):
    per = [last3(found[k]) for k in keys]
    print(label, [found[k] for k in keys])
    for conv, ddof in (('population', 0), ('sample', 1)):
        print('  std %-10s' % conv + ' | '.join('%s %.0f ± %.0f' % (t, np.mean([p[t] for p in per]), np.std([p[t] for p in per], ddof=ddof)) for t in TASKS))
