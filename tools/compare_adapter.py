#!/usr/bin/env python3
"""Row 4 (stage 2 from the stage-1 F / actor) against b_adapter=1 (stage 2 from the pretrained F / actor through W), per seed.
Cluster, repo root: python tools/compare_adapter.py. Pairs the runs by model_dir (models/multi/<name> and models/multi/adapter_<name>)."""
import csv, glob, os, re
import numpy as np

NAMES = {'r1_nofb_uni': 'r1_nofb_uni_run_p25', 'r2_row4_s2': 'r2_row4_s2', 'r2_row4_s3': 'r2_row4_s3'}
TASKS = ['run', 'stand', 'flip', 'supine', 'jump_any', 'kick_l', 'split', 'flamingo_r']
found = {}
for info in sorted(glob.glob('logs/*/info.txt')):
    txt = open(info).read()
    m = re.search(r'model_dir=models/multi/(\S+)', txt)
    if not m or 'finetune=multi' not in txt or 'eval_ckpt' in txt or 'diag=' in txt:
        continue
    found[m.group(1)] = os.path.dirname(info)       # the last one wins (latest run of that model_dir)


def curve(folder):
    rows = list(csv.DictReader(open(os.path.join(folder, 'eval_log.csv'))))
    return {int(r['step']): {t: float(r.get('%s_fresh_mean' % t) or 'nan') for t in TASKS} for r in rows}


def show(label, c):
    print('  %-12s' % label + ''.join('%10s' % t[:9] for t in TASKS))
    for s in sorted(c):
        print('  %-12d' % s + ''.join('%10.0f' % c[s][t] for t in TASKS))


res = {'row4': {}, 'adapter': {}}
for seed, base in NAMES.items():
    for kind, key in (('row4', base), ('adapter', 'adapter_' + seed)):
        if key not in found:
            print('MISSING', kind, seed, key)
            continue
        res[kind][seed] = curve(found[key])
        print('==', seed, kind, found[key])
        show(kind, res[kind][seed])
print('\n=== mean over seeds (final = step 300000, last-3 = mean of the evals at 240k, 270k, 300k) ===')
print('%-9s %-9s' % ('', 'point') + ''.join('%10s' % t[:9] for t in TASKS))
for kind in ('row4', 'adapter'):
    cs = list(res[kind].values())
    if not cs:
        continue
    steps = sorted(cs[0])
    for name, sel in (('step 0', [0]), ('30k', [30000]), ('90k', [90000]), ('final', [steps[-1]]), ('last-3', steps[-3:])):
        v = {t: np.mean([[c[s][t] for s in sel] for c in cs]) for t in TASKS}
        print('%-9s %-9s' % (kind, name) + ''.join('%10.0f' % v[t] for t in TASKS))
print('\nper-seed final run return:', {k: {s: round(c[max(c)]['run']) for s, c in res[k].items()} for k in res})
