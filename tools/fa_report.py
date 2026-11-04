#!/usr/bin/env python3
"""Report of the fast-adaptation runs (model_dir=models/multi/fa_<prefix>...). Cluster, repo root: python tools/fa_report.py [prefix] [--curves]

Per run: target task = tasks= of the run. Return curves from eval_log.csv (<task>_fresh_mean, 5 episodes, noisy).
  gain       last-3 mean of the target minus its step-0 value (step 0 = the starting model: adapter => the pretrained policy)
  t90        first evaluation (in updates) at which the target reaches step0 + 0.9 * gain (smoothed with a 3-point moving mean)
  v@2k/10k   target return at 2k / 10k updates
  others     mean change of the 7 other tasks, last-3 mean minus the PRETRAINED value (row 0 of the main table), all and without split;
  worst      the largest single drop among the other tasks; split is reported on its own
Configurations (same tag without the trailing _s1/_s2/_s3) are averaged over seeds."""
import csv, glob, os, re, sys
import numpy as np

PRETRAINED = {'run': 345, 'stand': 715, 'flip': 517, 'supine': 985, 'split': 960, 'flamingo_r': 794, 'jump_any': 132, 'kick_l': 77}
TASKS = list(PRETRAINED)
prefix = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else ''
runs = {}
for info in sorted(glob.glob('logs/*/info.txt')):
    txt = open(info).read()
    m = re.search(r'model_dir=models/multi/fa_(\S+)', txt)
    if m and m.group(1).startswith(prefix) and 'finetune=multi' in txt:
        runs[m.group(1)] = os.path.dirname(info)


def load(folder):
    rows = list(csv.DictReader(open(os.path.join(folder, 'eval_log.csv'))))
    steps = np.array([int(r['step']) for r in rows])
    val = {t: np.array([float(r.get('%s_fresh_mean' % t) or 'nan') for r in rows]) for t in TASKS}
    return steps, val


def metrics(folder, target):
    steps, val = load(folder)
    v = val[target]
    sm = np.convolve(np.pad(v, (1, 1), mode='edge'), np.ones(3) / 3, mode='valid')
    r0, rf = v[0], np.mean(v[-3:])
    gain = rf - r0
    t90 = next((int(s) for s, x in zip(steps, sm) if x >= r0 + 0.9 * gain), -1) if gain > 0 else -1
    at = lambda k: float(v[np.argmin(abs(steps - k))])
    others = [t for t in TASKS if t != target]
    d = {t: np.mean(val[t][-3:]) - PRETRAINED[t] for t in others}
    return dict(r0=r0, rf=rf, gain=gain, t90=t90, v2=at(2000), v10=at(10000), oth=np.mean(list(d.values())),
                oth_ns=np.mean([d[t] for t in others if t != 'split']), worst=min(d.values()), split=d.get('split', np.nan), steps=int(steps[-1]))


cfgs = {}
for tag, folder in runs.items():
    target = re.search(r'tasks=(\w+)', open(os.path.join(folder, 'info.txt')).read()).group(1)
    cfg = re.sub(r'_s[123]$', '', tag)
    cfgs.setdefault((cfg, target), []).append(metrics(folder, target))
print('%-34s %-8s %2s | %6s %6s %6s | %6s %6s | %6s %7s %7s %7s | %6s' % ('config', 'target', 'n', 'step0', 'final', 'gain', 'v@2k', 'v@10k', 't90', 'oth', 'oth-spl', 'worst', 'split'))
for (cfg, target), ms in sorted(cfgs.items()):
    f = lambda k: np.mean([m[k] for m in ms])
    print('%-34s %-8s %2d | %6.0f %6.0f %6.0f | %6.0f %6.0f | %6.0f %7.0f %7.0f %7.0f | %6.0f' %
          (cfg, target, len(ms), f('r0'), f('rf'), f('gain'), f('v2'), f('v10'), f('t90'), f('oth'), f('oth_ns'), f('worst'), f('split')))
if '--curves' in sys.argv:
    for tag, folder in sorted(runs.items()):
        target = re.search(r'tasks=(\w+)', open(os.path.join(folder, 'info.txt')).read()).group(1)
        steps, val = load(folder)
        print(tag, target, ' '.join('%d:%.0f' % (s, x) for s, x in zip(steps, val[target])))
