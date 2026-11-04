#!/usr/bin/env python3
"""Pivot of the rb_tasks.csv of a folder: rows = latent, columns = model. Cluster: python tools/show_csv.py logs/<time>_rbtasks"""
import csv, sys

rows = list(csv.DictReader(open(sys.argv[1] + '/rb_tasks.csv')))
names, whats = [], []
for r in rows:
    if r['model'] not in names:
        names.append(r['model'])
    if r['what'] not in whats:
        whats.append(r['what'])
print('%-78s' % 'R^B at the latent of / averaged over' + ''.join('%22s' % n[:21] for n in names) + '   cos w/ const')
for w in whats:
    v = [next(r for r in rows if r['model'] == n and r['what'] == w) for n in names]
    print('%-78s' % w[:77] + ''.join('%22s' % x['R_B_at_latent'] for x in v) + '   ' + v[0]['cos_with_constant_latent'])
