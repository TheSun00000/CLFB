#!/usr/bin/env python3
"""Fast retention study (eps only). Reads the synced logs (run tools/sync_from_hpc.sh first) of the runs launched by tools/queue_eps.py
(info.txt has model_dir=models/eps/<config>_r<k>) and prints, per configuration (repeats averaged):
  trained : eps of the task fine-tuned on (end of the run), and at 10k / 20k steps
  others  : change of eps (end - pretrained, same run's baseline row) on the 8 tasks; 'max' = largest increase over the tasks that are NOT aligned with run
Success = trained eps within 0.03 of the no-retention control of the same task ('ctrl' in the config name) AND max increase of the other tasks <= +0.03 (stand/run/flip are exempt only when the trained task is one of them) (tolerance of the evaluation noise: ~0.01-0.02).
Usage: python tools/eps_report.py [prefix]      (prefix filters the configuration names, e.g. e1_)
"""
import csv, glob, os, re, sys, statistics as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
os.chdir(ROOT)
TASKS = ['stand', 'run', 'flip', 'supine', 'jump_any', 'kick_l', 'split', 'flamingo_r']
ALIGNED = {'stand', 'run', 'flip'}


def rd(p):
    try:
        return list(csv.DictReader(open(p)))
    except OSError:
        return []


def fl(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return None


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else ''
    runs = {}
    for d in sorted(glob.glob('hpc/logs/*FBcontinual-*')):
        try:
            info = open(d + '/info.txt').read()
        except OSError:
            continue
        m = re.search(r'model_dir=models/eps/(\S+)', info)
        if not m:
            continue
        name = m.group(1)
        cfg = re.sub(r'_r\d+$', '', name)
        if not cfg.startswith(prefix):
            continue
        ev = rd(d + '/eval_log.csv')
        base = [r for r in ev if r['tag'] == 'baseline']
        end = [r for r in ev if r['tag'] == 'end']
        if not base or not end:
            continue
        task = re.search(r"'tasks': '?([a-z_]+)", info).group(1)
        b, e = base[0], end[-1]
        steps = {int(float(r['step'])): r for r in ev if r['tag'] in ('mid', 'end')}
        runs.setdefault(cfg, []).append({
            'task': task, 'base': {t: fl(b, t + '_fresh_eps') for t in TASKS}, 'end': {t: fl(e, t + '_fresh_eps') for t in TASKS},
            'trained_steps': {s: fl(r, task + '_fresh_eps') for s, r in sorted(steps.items())}})
    if not runs:
        sys.exit('no finished run with model_dir=models/eps/ in hpc/logs')
    refs = {}       # trained task -> mean trained eps of its control ('ctrl' in the configuration name)
    for c, rs in runs.items():
        if 'ctrl' in c:
            refs[rs[0]['task']] = st.mean(r['end'][r['task']] for r in rs)
    any_run = next(iter(runs.values()))[0]
    ALIGNED_NOTE = 'aligned = stand, run, flip only when the trained task is one of them'
    print('pretrained eps:   ' + ' '.join('%s %.3f' % (t[:6], any_run['base'][t]) for t in TASKS))
    print('%-26s %2s %-9s %-20s %s | %s' % ('config', 'n', 'task', 'trained eps 10k/20k/end', 'change of eps on the 8 tasks (end - pretrained)', 'max rise of the others'))
    res = {}
    for cfg in sorted(runs):
        rs = runs[cfg]; t = rs[0]['task']
        tr = st.mean(r['end'][t] for r in rs)
        steps = sorted(rs[0]['trained_steps'])
        mids = ' '.join('%.3f' % st.mean(r['trained_steps'][s] for r in rs) for s in steps)
        dl = {x: st.mean(r['end'][x] - r['base'][x] for r in rs) for x in TASKS}
        al = ALIGNED if t in ALIGNED else set()      # tasks that share their span with the trained one may drop; every other task must not rise
        un = {x: max(r['end'][x] - r['base'][x] for r in rs) for x in TASKS if x not in al and x != t}
        mx = max(un.values())
        ok = (t in refs and tr <= refs[t] + 0.03 and mx <= 0.03)
        res[cfg] = (tr, mx, ok)
        sd = st.pstdev([r['end'][t] for r in rs]) if len(rs) > 1 else 0
        print('%-26s %2d %-9s %s | %s | %+.3f %s' % (cfg, len(rs), t, ('%s (end %.3f+-%.3f)' % (mids, tr, sd)).ljust(40),
              ' '.join('%+.2f' % dl[x] for x in TASKS), mx, 'OK' if ok else ''))
    print('columns of the changes: ' + ' '.join(TASKS))


if __name__ == '__main__':
    main()
