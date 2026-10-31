"""Headline numbers of a finished job, read from the synced logs (hpc/logs/<run>/eval_log.csv).

Three log formats exist, all with one row per evaluation:
    fb_continual.py / fb_critic_oracle.py, continual: <task>_fresh_{mean,std}, row tag 'baseline' (pretrained) and 'end'
    fb_continual.py, finetune=...                   : <task>_ft_{mean,std},    row tag 'ft_start' (before the RL steps) and 'end'
    offline_rl_approx.py (single-reward TD3)        : <task>_{mean,std}
Runs without rollouts (eval_rollouts=0) have no return: their headline is the reward-reconstruction error (eps) instead.
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
    m = re.findall(r'(?:^|\s)tasks?=(\S+)', cmd or '')      # the last one wins, as in parse_kwargs
    return m[-1].split(',')[0] if m else None


def _orth_last(folder):
    """orth_l2 of the last logged chunk of training_log.csv (None if absent or not logged)."""
    try:
        with open(os.path.join(LOGS, folder, 'training_log.csv')) as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None
    v = _num(rows[-1].get('orth_l2')) if rows else None
    return round(v, 3) if v else None


def _eps_only(folder, task, rows):
    """Headline of a run without returns: reconstruction error of the reward at the last evaluation."""
    last = next((r for r in reversed(rows) if _num(r.get(task + '_fresh_eps')) is not None), None)
    if last is None:
        return None
    out = {'task': task, 'final': None, 'std': None, 'best': None, 'start': None, 'step': int(float(last.get('step') or 0)),
           'fresh_eps': round(_num(last[task + '_fresh_eps']), 4)}
    v = _num(last.get(task + '_fit_eps'))
    if v is not None:
        out['fit_eps'] = round(v, 4)
    first = next((r for r in rows if _num(r.get(task + '_fresh_eps')) is not None), None)
    out['start_eps'] = round(_num(first[task + '_fresh_eps']), 4)
    o = _orth_last(folder)
    if o is not None:
        out['orth'] = o
    return out


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
        return _eps_only(folder, task, rows)
    pts = [(r, _num(r[mean_c])) for r in rows if _num(r.get(mean_c)) is not None]
    if not pts:
        return _eps_only(folder, task, rows)
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
    o = _orth_last(folder)
    if o is not None:
        out['orth'] = o
    return out


def _thin(steps, max_points):
    if len(steps) <= max_points:
        return steps
    keep = {steps[round(i * (len(steps) - 1) / (max_points - 1))] for i in range(max_points)}
    return [s for s in steps if s in keep]


def _read(path):
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def curve(job, max_points=150):
    """Learning curves of a finished job. {'steps','mean','std'}: return (absent without rollouts);
    'eps': {'steps','v'} held-out reward reconstruction error (fresh readout); 'eps_fit' with the fixed readout of the stage;
    'orth': {'steps','v'} orthonormality error of B from training_log.csv. None if the job has none of them."""
    if job.get('status') != 'finished' or not job.get('runFolders'):
        return None
    task = _task(job.get('command'))
    if not task:
        return None
    folder = job['runFolders'][-1]
    rows = _read(os.path.join(LOGS, folder, 'eval_log.csv'))
    if not rows:
        return None
    out = {}
    for mean_c, std_c in ((task + '_ft_mean', task + '_ft_std'), (task + '_fresh_mean', task + '_fresh_std'), (task + '_mean', task + '_std')):
        if mean_c in rows[0] and any(_num(r.get(mean_c)) is not None for r in rows):
            pts = {}
            for r in rows:
                v = _num(r.get(mean_c))
                if v is not None:
                    pts[int(float(r.get('step') or 0))] = (v, _num(r.get(std_c)) or 0.0)      # later rows win on a repeated step
            steps = _thin(sorted(pts), max_points)
            out.update({'steps': steps, 'mean': [round(pts[s][0], 1) for s in steps], 'std': [round(pts[s][1], 1) for s in steps]})
            break
    for key, col in (('eps', task + '_fresh_eps'), ('eps_fit', task + '_fit_eps')):
        pts = {}
        for r in rows:
            v = _num(r.get(col))
            if v is not None:
                pts[int(float(r.get('step') or 0))] = v
        if pts:
            steps = _thin(sorted(pts), max_points)
            out[key] = {'steps': steps, 'v': [round(pts[s], 5) for s in steps]}
    pts = {}
    for r in _read(os.path.join(LOGS, folder, 'training_log.csv')):
        v = _num(r.get('orth_l2'))
        if v:
            pts[int(float(r.get('step') or 0))] = v
    if pts:
        steps = _thin(sorted(pts), max_points)
        out['orth'] = {'steps': steps, 'v': [round(pts[s], 4) for s in steps]}
    return out or None


def headline(job):
    """dict with a 'text' line for a finished job, or None (not finished, no log yet)."""
    if job.get('status') != 'finished' or not job.get('runFolders'):
        return None
    task = _task(job.get('command'))
    res = [r for r in (_one(f, task) for f in job['runFolders']) if r] if task else []
    if not res:
        return None
    r = res[-1]
    if r['final'] is None:
        text = '%s eps %.4f at %dk steps' % (r['task'], r['fresh_eps'], r['step'] // 1000)
        extra = ['start eps %.4f' % r['start_eps']] + (['fit eps %.4f' % r['fit_eps']] if 'fit_eps' in r else []) \
            + (['orth_l2 %.2f' % r['orth']] if 'orth' in r else [])
        return dict(r, text=text, extra=', '.join(extra))
    text = '%s return %.1f ± %.1f at %dk steps' % (r['task'], r['final'], r['std'], r['step'] // 1000)
    extra = []
    if r['start'] is not None:
        extra.append('start %.1f' % r['start'])
    extra.append('best %.1f' % r['best'])
    for key, name in (('fresh_eps', 'fresh eps'), ('fit_eps', 'fit eps')):
        if key in r:
            extra.append('%s %.4f' % (name, r[key]))
    return dict(r, text=text, extra=', '.join(extra))
