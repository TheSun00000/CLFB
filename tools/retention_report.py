#!/usr/bin/env python3
"""Compare the models of the retention study on the 8 tasks (reads the synced logs in hpc/logs; run tools/sync_from_hpc.sh first).

Rows: the pretrained model, FB_run (best setup fine-tuned on run, no retention: the run model of array 1788307), and every retention run
(models/ret/<name>) and fine-tuned model (models/final8/<task>) that has been evaluated by the eval-only job.
For each model: zero-shot return and held-out eps on every task of the list (fresh latent, 20 episodes), the return on run during training
(mean of the last 5 evaluations), and for retention runs the final values of R^B (old_rewards_lost_fraction) and R^Q (old_rewards_value_change).
Prints the table and writes hpc/status/retention.json. Success of a retention run (the targets agreed with the user):
  run          return >= 0.9 x FB_run's
  other tasks  return of the tasks where the pretrained model is good (>= 300) >= 0.85 x pretrained, and eps drift vs pretrained <= +0.03 on the
               tasks that are not aligned with run
"""
import csv, glob, json, os, re, sys, statistics as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
os.chdir(ROOT)
TASKS = ['stand', 'run', 'flip', 'supine', 'jump_any', 'kick_l', 'split', 'flamingo_r']
ALIGNED = {'stand', 'run', 'flip'}
RUN_MODEL = '2026-10-31_18-57-15_FBcontinual-walker-run_5'


def name_of(src):
    if 'FBpretrain' in src: return 'pretrained'
    if src.startswith(RUN_MODEL): return 'FB_run'
    m = re.match(r'ret/([^/]+)', src.replace('models/', ''))
    if m: return m.group(1)
    m = re.match(r'final8/([^/]+)', src.replace('models/', ''))
    if m: return 'FT_' + m.group(1)
    return src


def rd(p):
    try:
        return list(csv.DictReader(open(p)))
    except OSError:
        return []


def main():
    models = {}
    for d in sorted(glob.glob('hpc/logs/*FBeval-*')):
        info = open(d + '/info.txt').read()
        m = re.search(r'source=(\S+)', info)
        if not m or "'eval_z': 'random'" in info:
            continue
        ev = rd(d + '/eval_log.csv')
        if not ev or ev[-1].get('stand_fresh_mean') in (None, ''):
            continue       # an evaluation that does not cover the list
        n = name_of(m.group(1).replace('models/', ''))
        models[n] = {t: (float(ev[-1][t + '_fresh_mean']), float(ev[-1][t + '_fresh_eps'])) for t in TASKS if ev[-1].get(t + '_fresh_mean') not in (None, '')}
    # training logs of the retention runs
    extra = {}
    for d in sorted(glob.glob('hpc/logs/*FBcontinual-walker-run*')):
        info = open(d + '/info.txt').read()
        m = re.search(r'model_dir=models/ret/(\S+)', info)
        if not m:
            continue
        tr, ev = rd(d + '/training_log.csv'), rd(d + '/eval_log.csv')
        if not tr:
            continue
        f = lambda c: st.mean(float(r[c]) for r in tr[-10:] if r.get(c) not in (None, '', 'nan'))
        rets = [float(r['run_fresh_mean']) for r in ev if r.get('run_fresh_mean') not in (None, '', 'nan')]
        extra[m.group(1)] = {'R_B': f('old_rewards_lost_fraction'), 'R_Q': f('old_rewards_value_change'), 'orth': f('orth_l2'),
                             'run_last5': st.mean(rets[-5:]) if rets else None, 'steps': int(float(tr[-1]['step'])), 'n_evals': len(rets)}
    if 'pretrained' not in models:
        sys.exit('the pretrained model has not been evaluated on the list yet (entry final_eval_base)')
    pre, ref = models['pretrained'], models.get('FB_run')
    order = ['pretrained', 'FB_run'] + sorted(n for n in models if n not in ('pretrained', 'FB_run'))
    print('zero-shot return (eps) on the 8 tasks; deltas are vs the pretrained model')
    print('%-18s ' % 'model' + ' '.join('%13s' % t for t in TASKS) + ' | run_last5  R_B    R_Q')
    out = {}
    for n in order:
        if n not in models:
            continue
        r = models[n]
        cells = []
        for t in TASKS:
            if t in r:
                cells.append('%5.0f(%+5.0f)' % (r[t][0], r[t][0] - pre[t][0]) if n != 'pretrained' else '%5.0f       ' % r[t][0])
            else:
                cells.append(' ' * 13)
        e = extra.get(n, {})
        print('%-18s ' % n + ' '.join(cells) + ' | %s %s %s' % (('%7.0f' % e['run_last5']) if e.get('run_last5') else '      -', ('%.3f' % e['R_B']) if e else '    -', ('%.3f' % e['R_Q']) if e else '    -'))
        ok = None
        if ref and n not in ('pretrained', 'FB_run') and 'run' in r:
            good = [t for t in TASKS if t not in ALIGNED and pre[t][0] >= 300 and t in r]
            ok = {'run_return': r['run'][0] >= 0.9 * ref['run'][0],
                  'others_return': all(r[t][0] >= 0.85 * pre[t][0] for t in good),
                  'eps_drift': all(r[t][1] - pre[t][1] <= 0.03 for t in TASKS if t not in ALIGNED and t in r), 'tasks_checked': good}
            print('%-18s   success: %s' % ('', ok))
        out[n] = {'tasks': {t: {'return': r[t][0], 'eps': r[t][1]} for t in r}, 'extra': e, 'success': ok}
    json.dump({'tasks': TASKS, 'models': out}, open('hpc/status/retention.json', 'w'), separators=(',', ':'))


if __name__ == '__main__':
    main()
