#!/usr/bin/env python3
"""Aggregates the E diagnostic over seeds. Usage (cluster, repo root): python tools/e_report.py .slurm_logs/diag_E_<jobid>.out"""
import collections, csv, re, sys
import numpy as np

out = open(sys.argv[1]).read()
folders = re.findall(r'LOGS_FOLDER\s+: \./(logs/\S+)', out)
print(len(folders), 'folders:', folders)
R = collections.defaultdict(list); S = collections.defaultdict(list); A = collections.defaultdict(list); Bt = []
for f in folders:
    for r in csv.DictReader(open(f + '/E_returns.csv')):
        R[(r['model'], r['variant'], r['task'])].append((float(r['mean']), float(r['eps_centered']), float(r['cos_to_z0'])))
    for r in csv.DictReader(open(f + '/E_sensitivity.csv')):
        S[(r['model'], r['task'], int(r['angle_deg']))].append(float(r['mean']))
    for r in csv.DictReader(open(f + '/E_alignment.csv')):
        A[r['task']].append([float(r[k]) for k in ('cos_raw_z0_z1', 'cos_plain_procrustes', 'cos_white_procrustes', 'cos_backmap')])
    Bt.append([float(r['mean']) for r in csv.DictReader(open(f + '/E_bootstrap.csv'))])
tasks = ['stand', 'run', 'flip', 'supine', 'jump_any', 'kick_l', 'split', 'flamingo_r']
vars_ = [('th0', 'baseline_z0'), ('th1', 'baseline_z1'), ('th1', 'old_z0'), ('th0', 'free_a_raw'), ('th0', 'free_b_plain'), ('th0', 'free_c_white'), ('th0', 'free_d_backmap')]
print('RETURNS (mean over seeds, 20 episodes each); centered eps of the latent fed in brackets')
print('%-16s' % 'variant' + ''.join('%15s' % t for t in tasks))
for m, v in vars_:
    print('%-5s%-11s' % (m, v[:10]) + ''.join('%9.0f (%.2f)' % (np.mean([x[0] for x in R[(m, v, t)]]), np.mean([x[1] for x in R[(m, v, t)]])) for t in tasks))
print('per-seed returns on run:', {v: [round(x[0]) for x in R[(m, v, 'run')]] for m, v in vars_})
print('ALIGNMENT cos(z0, mapped z1): raw / plain / white / backmap')
for t in tasks:
    print('%-11s' % t, np.round(np.mean(A[t], 0), 3))
print('SENSITIVITY: return vs rotation angle (0, 5, 10, 20, 35, 60 deg)')
for m, b in (('th0', 'baseline_z0'), ('th1', 'baseline_z1')):
    for t in tasks:
        print('%s %-11s' % (m, t) + ' '.join('%5d' % (np.mean([x[0] for x in R[(m, b, t)]]) if a == 0 else np.mean(S[(m, t, a)])) for a in (0, 5, 10, 20, 35, 60)))
print('BOOTSTRAP theta_1 on run, per seed (mean of the 10 means, std across the 10 latents):', [(round(float(np.mean(b)), 1), round(float(np.std(b)), 1)) for b in Bt])
