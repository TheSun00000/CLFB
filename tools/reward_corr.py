#!/usr/bin/env python3
"""Evaluate every reward of tools/reward_library.py (plus the 5 built-in tasks and the 6 tasks of the first study) on dataset states and
compute the matrix of |Pearson correlation| between them. No training of anything: rewards on states only.

Usage: python tools/reward_corr.py [--n 100000] [--png out.png] [--json hpc/status/task_corr.json]
Needs the sample of dataset states written by the first study (tools/task_design.py): sample.npz with `obs` (states) and `aux`
(the 4 physics quantities of the built-in rewards, index-aligned with obs).
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reward_library as RL
import task_design as TD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', default=os.path.join(os.environ.get('CLAUDE_JOB_DIR', '.'), 'tmp', 'design', 'sample.npz'))
    ap.add_argument('--n', type=int, default=100_000)
    ap.add_argument('--k', type=int, default=12)
    ap.add_argument('--png', default=None)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    z = np.load(a.sample)
    sel = np.random.default_rng(0).choice(len(z['obs']), a.n, replace=False)
    obs, aux = z['obs'][sel].astype(np.float64), z['aux'][sel]
    cols, group, desc = {}, {}, {}
    for k, p in (('stand', 0), ('walk', 2), ('vel8', 8), ('run', 10), ('flip', 'flip')):
        cols[k], group[k], desc[k] = TD.builtin(aux, p), 'built-in', 'benchmark task'
    prev = ('spin_back', 'kick_r', 'lean_a', 'jump', 'knees_bent', 'split')
    for k, (d, fn) in RL.LIBRARY.items():
        cols[k] = fn(obs)
        group[k] = 'first study' if k in prev else 'new'
        desc[k] = d
    names = list(cols)
    R = np.stack([cols[k] for k in names], 1)
    support = 100 * (R >= 0.5).mean(0)          # share of the states where the reward is >= 0.5 (the task is 'satisfied')
    sd = R.std(0)
    C = np.corrcoef(R.T)
    C = np.nan_to_num(C)
    A = np.abs(C)
    bi = [i for i, k in enumerate(names) if group[k] == 'built-in']
    maxb = A[:, bi].max(1)

    print('%d rewards on %d states' % (len(names), a.n))
    print('%-16s %-11s %6s %9s %9s  %s' % ('reward', 'group', 'sat%', 'max|c|bi', 'nearest', 'description'))
    for i in np.argsort(maxb):
        k = names[i]
        if group[k] == 'built-in':
            continue
        B = A[i].copy(); B[i] = 0
        j = int(np.argmax(B))
        print('%-16s %-11s %6.1f %9.2f %9.2f  %s  [nearest: %s]' % (k, group[k], support[i], maxb[i], B[j], desc[k], names[j]))
    # greedy diverse set among the usable candidates
    ok = [i for i, k in enumerate(names) if group[k] != 'built-in' and 4 <= support[i] <= 60 and maxb[i] <= 0.25]
    chosen = []
    while ok and len(chosen) < a.k:
        best = min(ok, key=lambda i: (max([A[i, j] for j in chosen] + [0.0]), maxb[i]))
        chosen.append(best); ok.remove(best)
    print('\nusable candidates (4 to 60 percent satisfied, max |corr| with the built-ins <= 0.25): %d' % len([i for i, k in enumerate(names) if group[k] != 'built-in' and 4 <= support[i] <= 60 and maxb[i] <= 0.25]))
    print('greedy set of %d (each added to minimise its largest |corr| with the ones already chosen):' % a.k)
    for n_, i in enumerate(chosen):
        print('  %2d. %-16s sat %5.1f%%  max|corr| with the previous: %.2f   %s' % (n_ + 1, names[i], support[i], max([A[i, j] for j in chosen[:n_]] + [0.0]), desc[names[i]]))

    # hierarchical order for the picture
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        D = 1 - A
        np.fill_diagonal(D, 0)
        order = list(leaves_list(linkage(squareform(D, checks=False), 'average')))
    except Exception:
        order = list(np.argsort(maxb))
    if a.png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        n = len(names)
        fig, ax = plt.subplots(figsize=(17, 15))
        M = A[np.ix_(order, order)]
        im = ax.imshow(M, cmap='viridis_r', vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        lab = [names[i] for i in order]
        ax.set_xticklabels(lab, rotation=90, fontsize=7); ax.set_yticklabels(lab, fontsize=7)
        for t, i in zip(ax.get_xticklabels(), order):
            t.set_color({'built-in': 'tab:red', 'first study': 'tab:blue', 'new': 'black'}[group[names[i]]])
        for t, i in zip(ax.get_yticklabels(), order):
            t.set_color({'built-in': 'tab:red', 'first study': 'tab:blue', 'new': 'black'}[group[names[i]]])
        fig.colorbar(im, ax=ax, fraction=0.03, label='|Pearson correlation| of the rewards over %d dataset states' % a.n)
        ax.set_title('Reward correlation matrix (red = built-in tasks, blue = first study, black = new)')
        fig.tight_layout()
        fig.savefig(a.png, dpi=110)
    if a.json:
        out = {'n_states': a.n, 'names': [names[i] for i in order], 'group': [group[names[i]] for i in order],
               'desc': [desc[names[i]] for i in order], 'support': [round(float(support[i]), 1) for i in order],
               'chosen': [names[i] for i in chosen],
               'nearest': [names[int(np.argmax(np.where(np.arange(len(names)) == i, -1, A[i])))] for i in order],
               'nearest_corr': [round(float(np.max(np.where(np.arange(len(names)) == i, -1, A[i]))), 2) for i in order],
               'max_builtin': [round(float(maxb[i]), 2) for i in order],
               'matrix': [[round(float(A[i, j]), 2) for j in order] for i in order]}
        json.dump(out, open(a.json, 'w'), separators=(',', ':'))
    return names, A, group, support, desc, maxb


if __name__ == '__main__':
    main()
