#!/usr/bin/env python3
"""Report of the orthonormality runs (o1, o2, o3, v100): spectrum of Sigma at the end of the stage (sigma_log.csv) and eps of the 8 tasks. Run on the cluster in the repo."""
import csv, glob, os, re, sys
tasks = ['stand', 'run', 'flip', 'supine', 'jump_any', 'kick_l', 'split', 'flamingo_r']
flt = sys.argv[1] if len(sys.argv) > 1 else ''
rows = []
for d in sorted(glob.glob('logs/*FBcontinual-walker-run'), key=os.path.getmtime):
    info = open(d + '/info.txt').read()
    m = re.search(r'model_dir=models/eps/((?:o1|o2_.|o3_.|v100)[\w]*)', info)
    if not m or flt not in m.group(1): continue
    name = m.group(1)
    try: sg = list(csv.DictReader(open(d + '/sigma_log.csv')))[-1]
    except Exception: sg = None
    try: ev = list(csv.DictReader(open(d + '/eval_log.csv')))
    except Exception: ev = []
    if not ev or ev[-1]['tag'] != 'end' or sg is None: continue
    b, z = ev[0], ev[-1]
    e = lambda r, k: float(r[k + '_fresh_eps']) if r.get(k + '_fresh_eps') else float('nan')
    rows.append((name, float(sg['trace_inv']), float(sg['eig_min']), e(z, 'run'), ' '.join('%s %.2f>%.2f' % (k[:4], e(b, k), e(z, k)) for k in tasks if k != 'run')))
for n, tr, em, er, rest in sorted(rows):
    print('%-24s tr(S^-1) %8.0f eig_min %.5f run_eps %.3f | %s' % (n, tr, em, er, rest))
