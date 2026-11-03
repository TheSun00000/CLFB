#!/usr/bin/env python3
"""Report of the cheetah B-stage sweep (run on the cluster in the repo): per run eps of the 6 tasks, step 0 > end, and the trace of Sigma^-1. Usage: python3 tools/cheetah_hp_report.py [name-substring]"""
import csv, glob, os, re, sys
flt = sys.argv[1] if len(sys.argv) > 1 else ''
T = ['vel10', 'vel10Back', 'jump', 'kick_b', 'spread', 'flip']
for d in sorted(glob.glob('logs/*FBcontinual-cheetah-vel10*'), key=os.path.getmtime):
    info = open(d + '/info.txt').read()
    m = re.search(r"model_dir': '([^']*)'", info)
    name = m.group(1).split('/')[-1] if m else ''
    if not name.startswith('c4_') or flt not in name: continue
    try: rows = list(csv.DictReader(open(d + '/eval_log.csv')))
    except Exception: continue
    if len(rows) < 2: continue
    a, b = rows[0], rows[-1]
    out = ['%s %.3f>%.3f' % (t[:6], float(a[t + '_fresh_eps']), float(b[t + '_fresh_eps'])) for t in T if a.get(t + '_fresh_eps') and b.get(t + '_fresh_eps')]
    tr = ''
    try:
        sl = list(csv.DictReader(open(d + '/sigma_log.csv'))); tr = ' tr %.0f' % float(sl[-1]['trace_inv'])
    except Exception: pass
    print('%-22s' % name + tr + ' | ' + ' | '.join(out))
