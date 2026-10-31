#!/usr/bin/env python3
"""Zero-shot return on every benchmark task of the eval-only runs (logs/*FBeval-*), one row per configuration of the source runs.

Usage: tools/eval_summary.py  (after tools/sync_from_hpc.sh and tools/hpc_status.py --diff). Writes hpc/eval_all_tasks.csv, hpc/status/eval_all_tasks.json (the tracker document snapshot/eval_all_tasks) and prints a table.
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
    if folder == 'PRETRAINED' or 'FBpretrain' in folder:
        return 'pretrained model'
    if not j:
        return folder
    return classify(j)[1]


ALL = ['stand', 'walk', 'vel8', 'run', 'flip', 'spin_back', 'kick_r', 'lean_a', 'jump', 'knees_bent', 'split']
rows = {}                       # configuration -> list of (source run folder, {task: return})
for d in sorted(glob.glob('hpc/logs/*FBeval-*')):
    try:
        info = open(d + '/info.txt').read()
        ev = list(csv.DictReader(open(d + '/eval_log.csv')))
    except OSError:
        continue
    m = re.search(r'source=models/([^/\s]+)/', info)
    src = m.group(1) if m else 'PRETRAINED'
    rnd = "'eval_z': 'random'" in info
    r = ev[-1]
    vals = {c[:-len('_fresh_mean')]: float(r[c]) for c in r if c.endswith('_fresh_mean') and r[c] not in (None, '')}
    if vals:
        rows.setdefault(variant(src) + (' [random latent]' if rnd else ''), []).append((src, vals))
tasks = [t for t in ALL if any(t in v for rs in rows.values() for _, v in rs)]
if not rows:
    sys.exit('no eval-only run found in hpc/logs (run tools/sync_from_hpc.sh)')
def cell(rs, t):
    v = [vals[t] for _, vals in rs if t in vals]
    return st.mean(v) if v else None


with open('hpc/eval_all_tasks.csv', 'w') as f:
    w = csv.writer(f); w.writerow(['configuration', 'runs'] + tasks)
    for name, rs in rows.items():
        w.writerow([name, len({s_ for s_, _ in rs})] + ['' if cell(rs, t) is None else '%.1f' % cell(rs, t) for t in tasks])
order = {}
for j in snap.values():
    for f in (j.get('runFolders') or []):
        if not j.get('children') and f in by_folder:
            order[variant(f)] = classify(j)[2]
doc = {'tasks': tasks, 'rows': [{'name': name, 'runs': len({s_ for s_, _ in rs}), 'order': (-1 if name == 'pretrained model' else order.get(name, 99)),
                                 'values': {t: round(cell(rs, t), 1) for t in tasks if cell(rs, t) is not None}} for name, rs in rows.items()]}
doc['rows'].sort(key=lambda r: (r['order'], r['name']))
for r in doc['rows']:
    r['random'] = r['name'].endswith('[random latent]')
# the latent control: same checkpoints, latent from fresh labels vs a random latent
def cfg(name, label):
    f = [r for r in doc['rows'] if r['name'] == name]
    g = [r for r in doc['rows'] if r['name'] == name + ' [random latent]']
    return {'label': label, 'runs': f[0]['runs'] if f else 0, 'fresh': f[0]['values'] if f else {}, 'random': g[0]['values'] if g else {}}
trained = [r['name'] for r in doc['rows'] if not r['random'] and r['name'] != 'pretrained model']
def avg(key):
    out = {}
    for t in tasks:
        v = [r['values'][t] for r in doc['rows'] if r['name'] in trained + [n + ' [random latent]' for n in trained] and r['random'] == (key == 'random') and t in r['values']]
        if v: out[t] = round(st.mean(v), 1)
    return out
doc['control'] = {'groups': [cfg('pretrained model', 'Pretrained model'), cfg('lam 100, focus 1.0, noise 0.2', 'Base run (focus 1.0), 4 runs'),
                             {'label': 'All %d fine-tuned configurations (mean)' % len(trained), 'runs': len(trained), 'fresh': avg('fresh'), 'random': avg('random')}]}
with open('hpc/status/eval_all_tasks.json', 'w') as f:
    json.dump(doc, f)
print('%-62s %4s ' % ('configuration', 'runs') + ' '.join('%7s' % t[:7] for t in tasks))
for name, rs in sorted(rows.items(), key=lambda kv: (kv[0] != 'pretrained model', kv[0])):
    print('%-62s %4d ' % (name, len({s_ for s_, _ in rs})) + ' '.join('%7s' % ('' if cell(rs, t) is None else '%.0f' % cell(rs, t)) for t in tasks))
