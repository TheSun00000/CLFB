#!/usr/bin/env python3
"""Report of the finetune=multi runs (read-only). Run on the cluster:  ssh jz 'bash -l -s' <<< "cd \\$WORK/CLFB...; python3 tools/multi_report.py [name-substring]"
For each run (model_dir name): per task, return at step 0, last evaluation, mean of the last 3 evaluations, against the offline TD3 upper bound (1M updates, true reward)."""
import csv, glob, re, sys
UB = {'stand': 971, 'run': 461, 'flip': 939, 'supine': 991, 'jump_any': 396, 'kick_l': 484, 'split': 999, 'flamingo_r': 990}
flt = sys.argv[1] if len(sys.argv) > 1 else ''
for d in sorted(glob.glob('logs/*FBfinetune-multi*')):
    info = open(d + '/info.txt').read()
    m = re.search(r"model_dir': '([^']*)'", info) or re.search(r'model_dir=(\S+)', info)
    name = m.group(1).split('/')[-1] if m else d
    if flt not in name: continue
    try: rows = list(csv.DictReader(open(d + '/eval_log.csv')))
    except Exception: continue
    ts = re.search(r"'tasks': '([^']*)'", info).group(1).split(',')
    steps = [r['step'] for r in rows]
    out = []
    for t in ts:
        v = [float(r[t + '_fresh_mean']) for r in rows if r.get(t + '_fresh_mean')]
        if len(v) < 2: continue
        l3 = sum(v[-3:]) / len(v[-3:])
        out.append('%s %d>%d(%d) /%d' % (t[:6], v[0], v[-1], l3, UB[t]))
    print('%-26s last step %s | ' % (name, steps[-1]) + ' | '.join(out))
