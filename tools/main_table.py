#!/usr/bin/env python3
"""Cells of the main tables from the output of tools/multi_report.py (read with `ssh jz ... python3 tools/multi_report.py <name>` piped in).
Usage: python3 tools/main_table.py rows.json < report.txt > cells.json
rows.json: {"<row n>": ["run name", ...]}.  Cell = mean +- std over the runs of the row of the MEAN OF THE LAST 3 EVALUATIONS (rounded); 'pre' row = mean of step 0 over the runs of rows 1 and 2.
A run still going (last step < 300000) is skipped."""
import json, re, sys, statistics as st
rows = json.load(open(sys.argv[1]))
MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 300000       # a run is complete when its last step reaches MIN (stream chains: 200000)
KEY = {'stand': 'stand', 'run': 'run', 'flip': 'flip', 'supine': 'supine', 'jump_a': 'jump_any', 'kick_l': 'kick_l', 'split': 'split', 'flamin': 'flamingo_r', 'vel10B': 'vel10Back', 'vel10': 'vel10', 'kick_b': 'kick_b', 'spread': 'spread'}
runs = {}
for line in sys.stdin:
    m = re.match(r'(\S+)\s+last step (\d+) \|(.*)', line)
    if not m: continue
    name, step, rest = m.group(1), int(m.group(2)), m.group(3)
    vals = {}
    for t in rest.split('|'):
        mm = re.match(r'\s*(\w+) (-?\d+)>(-?\d+)\((-?\d+)\)', t)
        if mm and mm.group(1) in KEY: vals[KEY[mm.group(1)]] = (float(mm.group(2)), float(mm.group(4)))
    runs[name] = (step, vals)
def cell(vs): return '%d ± %d' % (round(st.mean(vs)), round(st.pstdev(vs))) if len(vs) > 1 else ('%d' % round(vs[0]) if vs else '')
out = {}
for n, names in rows.items():
    ok = [runs[x][1] for x in names if x in runs and runs[x][0] >= MIN]
    out[n] = {'n_runs': len(ok), 'cells': {k: cell([r[k][1] for r in ok if k in r]) for k in KEY.values()}}
    if n == '0':
        pre = [runs[x][1] for x in names if x in runs]
        out[n] = {'n_runs': len(pre), 'cells': {k: cell([r[k][0] for r in pre if k in r]) for k in KEY.values()}}
json.dump(out, sys.stdout, indent=1)
