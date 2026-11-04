#!/usr/bin/env python3
"""Run folders of the fast-adaptation runs by model_dir tag. Cluster, repo root: python tools/list_runs.py fa_r14_ [fa_r3_ ...]  (prefix of the tag after models/multi/)"""
import glob, os, re, sys

prefixes = sys.argv[1:]
rows = {}
for info in sorted(glob.glob('logs/*/info.txt')):
    m = re.search(r'model_dir=models/multi/(\S+)', open(info).read())
    if m and any(m.group(1).startswith(p) for p in prefixes):
        rows[m.group(1)] = os.path.basename(os.path.dirname(info))
for k, v in sorted(rows.items()):
    print(k, v)
