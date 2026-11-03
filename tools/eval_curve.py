#!/usr/bin/env python3
"""Return curve of a finetune=multi run, from the slurm log. Cluster, repo root: python tools/eval_curve.py .slurm_logs/<script>_<jobid>.out"""
import csv, re, sys

out = open(sys.argv[1]).read()
for m in re.finditer(r'b_adapter:[^\n]*', out):
    print(m.group(0)[:340])
folder = re.search(r'LOGS_FOLDER\s*: \./(\S+)', out).group(1)
print(folder)
rows = list(csv.DictReader(open(folder + '/eval_log.csv')))
tasks = ['run', 'stand', 'flip', 'supine', 'jump_any', 'kick_l', 'split', 'flamingo_r']
print('%-8s %-9s' % ('step', 'tag') + ''.join('%11s' % t for t in tasks))
for r in rows:
    print('%-8s %-9s' % (r['step'], r['tag']) + ''.join('%11s' % (r.get('%s_fresh_mean' % t, '')[:6]) for t in tasks))
