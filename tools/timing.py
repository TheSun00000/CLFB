#!/usr/bin/env python3
"""Seconds per 1000 updates and per evaluation of a run folder. Cluster, repo root: python tools/timing.py logs/<run>"""
import csv, sys
import numpy as np

d = sys.argv[1]
tr = list(csv.DictReader(open(d + '/training_log.csv')))
ev = list(csv.DictReader(open(d + '/eval_log.csv')))
print('training: %d chunks, median %.2f s per chunk (1000 updates)' % (len(tr), np.median([float(r['time']) for r in tr])))
print('eval: %d evals, time column median %.1f s' % (len(ev), np.median([float(r['time']) for r in ev if r.get('time')])))
