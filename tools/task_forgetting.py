#!/usr/bin/env python3
"""B-only learnability and forgetting of candidate tasks (tools/reward_library.py), in both orders, as in tools/task_design.py.

For each task X (B-only fits: Adam, fixed readout, 10k labels, `--steps` steps, nothing else trained):
  A) start from the B fine-tuned on run (--start): fit X, then look at run again          -> eps_X(fit), eps_run after (forgetting of run)
  B) start from the pretrained B: fit X, then fit run, then look at X again               -> eps_X before / after run (forgetting of X)
Also prints |corr| of the reward with each kept task (--keep) on the dataset states.
Usage: python tools/task_forgetting.py --pre pretrained.pickle --start run_focus.pickle --tasks split knees_bent ... [--keep stand run flip]
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reward_library as RL
import task_design as TD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pre', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--tasks', nargs='+', required=True)
    ap.add_argument('--keep', nargs='*', default=['stand', 'run', 'flip'])
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--sample', default=os.path.join(os.environ.get('CLAUDE_JOB_DIR', '.'), 'tmp', 'design', 'sample.npz'))
    a = ap.parse_args()
    import jax, jax.numpy as jnp
    z = np.load(a.sample)
    n = 200_000
    obs, aux = z['obs'][:n].astype(np.float32), z['aux'][:n]
    R = {'stand': TD.builtin(aux, 0), 'walk': TD.builtin(aux, 2), 'vel8': TD.builtin(aux, 8), 'run': TD.builtin(aux, 10), 'flip': TD.builtin(aux, 'flip')}
    for k, (d, fn) in RL.LIBRARY.items():
        R[k] = fn(obs.astype(np.float64))
    p_pre, p_run = TD.load_params(a.pre), TD.load_params(a.start)
    feat = lambda p: np.asarray(jax.jit(lambda x: TD.B_apply(p, x))(obs))
    F_pre, F_run = feat(p_pre), feat(p_run)
    lab = np.random.default_rng(0).choice(n, 10_000, replace=False)

    def readout(F, r):
        s = np.sqrt((r[lab] ** 2).mean()) + 1e-12
        zz = (F[lab] * (r[lab] / s)[:, None]).mean(0)
        S = F[:100_000].T @ F[:100_000] / 100_000 + 1e-5 * np.eye(F.shape[1])
        return jnp.asarray(np.linalg.solve(S, zz), dtype=jnp.float32), s

    run = R['run']
    e_run0 = TD.eps_of(F_run, run, seeds=2)
    print('B fine-tuned on run: eps(run) = %.3f' % e_run0)
    print('%-14s %s | %9s %9s %9s | %9s %9s | %9s %9s %9s' % ('task', ' '.join('|c|%-5s' % k[:5] for k in a.keep), 'eps_X pre', 'eps_X fit', 'eps_X B0', 'eps_run0', 'eps_run A', 'eps_X(1)', 'eps_X(2)', 'eps_run B'), flush=True)
    for k in a.tasks:
        r = R[k]
        cs = ' '.join('%8.2f' % abs(np.corrcoef(r, R[q])[0, 1]) for q in a.keep)
        e_pre = TD.eps_of(F_pre, r, seeds=2)
        e_runB0 = TD.eps_of(F_run, r, seeds=2)
        # A) from the B of run
        w, s = readout(F_run, r)
        pA = TD.fit_B(p_run, obs[lab], r[lab] / s, w, steps=a.steps)
        FA = feat(pA)
        e_fitA, e_runA = TD.eps_of(FA, r, seeds=2), TD.eps_of(FA, run, seeds=2)
        # B) X first from the pretrained B, then run
        w1, s1 = readout(F_pre, r)
        p1 = TD.fit_B(p_pre, obs[lab], r[lab] / s1, w1, steps=a.steps)
        F1 = feat(p1)
        w2, s2 = readout(F1, run)
        p2 = TD.fit_B(p1, obs[lab], run[lab] / s2, w2, steps=a.steps)
        F2 = feat(p2)
        print('%-14s %s | %9.3f %9.3f %9.3f | %9.3f %9.3f | %9.3f %9.3f %9.3f' % (
            k, cs, e_pre, e_fitA, e_runB0, e_run0, e_runA, TD.eps_of(F1, r, seeds=2), TD.eps_of(F2, r, seeds=2), TD.eps_of(F2, run, seeds=2)), flush=True)


if __name__ == '__main__':
    main()
