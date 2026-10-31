#!/usr/bin/env python3
"""Zero-shot return on every benchmark task of the eval-only runs (logs/*FBeval-*), one row per configuration of the source runs.

Usage: tools/eval_summary.py  (after tools/sync_from_hpc.sh and tools/hpc_status.py --diff). Writes hpc/eval_all_tasks.csv and prints a table.
A source run is matched to its configuration by the run folder in the eval folder's info.txt (`eval_only source=models/<run folder>/...`).
"""
import csv, glob, json, os, re, sys, statistics as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
src = open(os.path.join(ROOT, 'tools', 'describe_jobs.py')).read().split('n = 0\nos.makedirs')[0]
sys.argv = ['x']
os.chdir(ROOT)
exec(src)                                    # flag(), classify(), ret_variant(), ... (no file is written)
snap = json.load(open('hpc/status/snapshot.json'))['jobs']
by_folder = {f: j for j in snap.values() for f in (j.get('runFolders') or []) if not j.get('children')}


def variant(folder):
    j = by_folder.get(folder)
    if folder == 'PRETRAINED' or not j:
        return 'pretrained model' if folder == 'PRETRAINED' else folder
    return classify(j)[1]


rows = {}
tasks = None
for d in sorted(glob.glob('hpc/logs/*FBeval-*')):
    try:
        info = open(d + '/info.txt').read()
        ev = list(csv.DictReader(open(d + '/eval_log.csv')))
    except OSError:
        continue
    m = re.search(r'source=models/([^/\s]+)/', info)
    name = variant(m.group(1)) if m else 'PRETRAINED'
    r = ev[-1]
    tasks = tasks or sorted({c[:-len('_fresh_mean')] for c in r if c.endswith('_fresh_mean')}, key=lambda t: ['stand', 'walk', 'vel8', 'run', 'flip'].index(t) if t in ['stand', 'walk', 'vel8', 'run', 'flip'] else 9)
    rows.setdefault(name, []).append({t: float(r['%s_fresh_mean' % t]) for t in tasks if r.get('%s_fresh_mean' % t) not in (None, '')})
if not rows:
    sys.exit('no eval-only run found in hpc/logs (run tools/sync_from_hpc.sh)')
with open('hpc/eval_all_tasks.csv', 'w') as f:
    w = csv.writer(f); w.writerow(['configuration', 'runs'] + tasks)
    for name, rs in rows.items():
        w.writerow([name, len(rs)] + ['%.1f' % st.mean(r[t] for r in rs) for t in tasks])
print('%-62s %4s ' % ('configuration', 'runs') + ' '.join('%7s' % t for t in tasks))
for name, rs in sorted(rows.items(), key=lambda kv: (kv[0] != 'pretrained model', kv[0])):
    print('%-62s %4d ' % (name, len(rs)) + ' '.join('%7.0f' % st.mean(r[t] for r in rs) for t in tasks))
