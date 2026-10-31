"""Headline numbers of a finished job, read from the synced logs (hpc/logs/<run>/eval_log.csv).

Three log formats exist, all with one row per evaluation:
    fb_continual.py / fb_critic_oracle.py, continual: <task>_fresh_{mean,std}, row tag 'baseline' (pretrained) and 'end'
    fb_continual.py, finetune=...                   : <task>_ft_{mean,std},    row tag 'ft_start' (before the RL steps) and 'end'
    offline_rl_approx.py (single-reward TD3)        : <task>_{mean,std}
Returns the return on the task of the job: final value (mean, std, step), best value over the run, and the start value.
"""
import csv
import math
import os
import re

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hpc', 'logs')


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _task(cmd):
    m = re.search(r'(?:^|\s)tasks?=(\S+)', cmd or '')
    return m.group(1).split(',')[0] if m else None


def _one(folder, task):
    path = os.path.join(LOGS, folder, 'eval_log.csv')
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None
    if not rows:
        return None
    cols = rows[0].keys()
    for mean_c, std_c, start_tag in ((task + '_ft_mean', task + '_ft_std', 'ft_start'),
                                     (task + '_fresh_mean', task + '_fresh_std', 'baseline'),
                                     (task + '_mean', task + '_std', None)):
        if mean_c in cols and any(_num(r.get(mean_c)) is not None for r in rows):    # the column can exist but be empty
            break
    else:
        return None
    pts = [(r, _num(r[mean_c])) for r in rows if _num(r.get(mean_c)) is not None]
    if not pts:
        return None
    ends = [p for p in pts if p[0].get('tag') == 'end']
    last, final = (ends or pts)[-1]      # TD3 logs have no tag: the last evaluation
    start = next((v for r, v in pts if start_tag and r.get('tag') == start_tag), None)
    best = max(v for _, v in pts)
    step = int(float(last.get('step') or 0))
    out = {'task': task, 'final': round(final, 1), 'std': round(_num(last.get(std_c)) or 0.0, 1),
           'step': step, 'best': round(best, 1), 'start': None if start is None else round(start, 1)}
    for key, col in (('fresh_eps', task + '_fresh_eps'), ('fit_eps', task + '_fit_eps')):     # reward reconstruction at the last evaluation
        v = _num(last.get(col))
        if v is not None:
            out[key] = round(v, 4)
    return out


def curve(job, max_points=150):
    """Learning curve of a finished job: {'steps': [...], 'mean': [...], 'std': [...]} (same series as headline())."""
    if job.get('status') != 'finished' or not job.get('runFolders'):
        return None
    task = _task(job.get('command'))
    if not task:
        return None
    path = os.path.join(LOGS, job['runFolders'][-1], 'eval_log.csv')
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None
    if not rows:
        return None
    for mean_c, std_c in ((task + '_ft_mean', task + '_ft_std'), (task + '_fresh_mean', task + '_fresh_std'), (task + '_mean', task + '_std')):
        if mean_c in rows[0] and any(_num(r.get(mean_c)) is not None for r in rows):
            break
    else:
        return None
    pts = {}
    for r in rows:
        v = _num(r.get(mean_c))
        if v is not None:
            pts[int(float(r.get('step') or 0))] = (v, _num(r.get(std_c)) or 0.0)      # later rows win on a repeated step
    if not pts:
        return None
    steps = sorted(pts)
    if len(steps) > max_points:
        keep = {steps[round(i * (len(steps) - 1) / (max_points - 1))] for i in range(max_points)}
        steps = [s for s in steps if s in keep]
    return {'steps': steps, 'mean': [round(pts[s][0], 1) for s in steps], 'std': [round(pts[s][1], 1) for s in steps]}


def headline(job):
    """dict with a 'text' line for a finished job, or None (not finished, no log yet)."""
    if job.get('status') != 'finished' or not job.get('runFolders'):
        return None
    task = _task(job.get('command'))
    res = [r for r in (_one(f, task) for f in job['runFolders']) if r] if task else []
    if not res:
        return None
    r = res[-1]
    text = '%s return %.1f ± %.1f at %dk steps' % (r['task'], r['final'], r['std'], r['step'] // 1000)
    extra = []
    if r['start'] is not None:
        extra.append('start %.1f' % r['start'])
    extra.append('best %.1f' % r['best'])
    for key, name in (('fresh_eps', 'fresh eps'), ('fit_eps', 'fit eps')):
        if key in r:
            extra.append('%s %.4f' % (name, r[key]))
    return dict(r, text=text, extra=', '.join(extra))
