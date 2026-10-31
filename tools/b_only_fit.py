#!/usr/bin/env python3
"""B-only learnability of the tasks (runs on the cluster, not locally): can B encode the reward of a task?

For every task and number of labelled states: fit B ALONE, starting from the pretrained B, on the labelled states with the readout FIXED
(w = Sigma_old^-1 mean(rbar B_pretrained), as fit_backward_to_reward does: Adam, cosine decay, lr 3e-4, batch 4096), then measure on held-out states
    eps_pre            reward error of the pretrained B with a readout refitted on the labelled states
    eps_fixed_pre      reward error of the pretrained B with the fixed readout
    eps_fixed_after    reward error of the fitted B with the same fixed readout (this is what the fit minimises)
    eps_resolved_after reward error of the fitted B with a readout refitted on fresh labelled states (what the evaluation of the model uses)
eps = mean((rbar - B w)^2) / mean(rbar^2): 1 = nothing explained, 0 = perfect. Output: logs/<time>_Bonly-learnability/learnability.csv.
Usage: python tools/b_only_fit.py --pre models/.../fb_agent_latest.pickle [--tasks stand run flip supine jump_any kick_l split flamingo_r] [--labels 10000 100000]
"""
import argparse, csv, os, sys, time
from datetime import datetime
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'tools'))
sys.path.insert(0, ROOT)
import reward_library as RL
import task_design as TD

LOCAL = 'nazim' in os.getcwd()
DATASET_FOLDER = '../offline_qd/datasets/d4rl/' if LOCAL else '../fre_experiments/datasets/'
BUILTIN = {'stand': 0, 'walk': 2, 'vel8': 8, 'run': 10, 'flip': 'flip'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pre', required=True)
    ap.add_argument('--tasks', nargs='+', default=['stand', 'run', 'flip', 'supine', 'jump_any', 'kick_l', 'split', 'flamingo_r'])
    ap.add_argument('--labels', nargs='+', type=int, default=[10_000, 100_000])
    ap.add_argument('--steps', type=int, default=30_000)
    ap.add_argument('--n', type=int, default=600_000, help='dataset states used (the last 200k are the held-out set)')
    a = ap.parse_args()
    import jax, jax.numpy as jnp
    print(jax.devices(), flush=True)
    folder = os.path.join(ROOT, 'logs', datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_Bonly-learnability')
    os.makedirs(folder)
    from _utils.logs import run_info
    open(os.path.join(folder, 'info.txt'), 'w').write(run_info() + 'args: %s\n' % vars(a))
    print('LOGS_FOLDER      : ./logs/' + os.path.basename(folder), flush=True)

    d = np.load(f'{DATASET_FOLDER}/walker_rnd.npy', allow_pickle=True).item()
    N = 10000 * 1000
    idx = np.sort(np.random.default_rng(0).choice(N, a.n, replace=False))
    obs = np.asarray(d['observations']).reshape(N, 24)[idx].astype(np.float32)
    del d
    aux = np.concatenate([np.load(f'{DATASET_FOLDER}/aux_walker.npy').reshape(N, 3)[idx], np.load(f'{DATASET_FOLDER}/angmomentum_aux_walker.npy').reshape(N, 1)[idx]], 1)
    R = {k: TD.builtin(aux, p) for k, p in BUILTIN.items()}
    for k, (desc, fn) in RL.LIBRARY.items():
        R[k] = fn(obs.astype(np.float64))
    print('support (reward >= 0.5):', {k: round(100 * float((R[k] >= 0.5).mean()), 1) for k in a.tasks}, flush=True)

    p_pre = TD.load_params(a.pre)
    feat = lambda p: np.asarray(jnp.concatenate([jax.jit(lambda x: TD.B_apply(p, x))(obs[i:i + 100_000]) for i in range(0, len(obs), 100_000)]))
    F_pre = feat(p_pre)
    held = np.arange(a.n - 200_000, a.n)
    pool = np.arange(a.n - 200_000)
    rows = []
    out = os.path.join(folder, 'learnability.csv')
    cols = ['task', 'labels', 'support_pct', 'eps_pre', 'eps_fixed_pre', 'eps_fixed_after', 'eps_resolved_after', 'steps', 'seconds']
    with open(out, 'w', newline='') as f:
        csv.writer(f).writerow(cols)
    print('%-12s %7s | %8s %13s %15s %18s' % ('task', 'labels', 'eps_pre', 'eps_fixed_pre', 'eps_fixed_after', 'eps_resolved_after'), flush=True)
    for k in a.tasks:
        r = R[k]
        for L in a.labels:
            t0 = time.time()
            rng = np.random.default_rng(1)
            lab = rng.choice(pool, L, replace=False)
            sig = rng.choice(pool, 100_000, replace=False)
            s = float(np.sqrt((r[lab] ** 2).mean())) + 1e-12
            z = (F_pre[lab] * (r[lab] / s)[:, None]).mean(0)
            S = F_pre[sig].T @ F_pre[sig] / len(sig) + 1e-5 * np.eye(F_pre.shape[1])
            w = np.linalg.solve(S, z)
            rh = r[held] / s
            fixed = lambda F: float(((rh - F[held] @ w) ** 2).mean() / (rh ** 2).mean())
            e_pre = TD.eps_of(F_pre, r, n_label=L, seeds=2)
            e_fix_pre = fixed(F_pre)
            p1 = TD.fit_B(p_pre, obs[lab], r[lab] / s, jnp.asarray(w, dtype=jnp.float32), steps=a.steps)
            F1 = feat(p1)
            e_fix_after, e_res_after = fixed(F1), TD.eps_of(F1, r, n_label=L, seeds=2)
            row = [k, L, round(100 * float((r >= 0.5).mean()), 1), e_pre, e_fix_pre, e_fix_after, e_res_after, a.steps, round(time.time() - t0)]
            with open(out, 'a', newline='') as f:
                csv.writer(f).writerow(row)
            print('%-12s %7d | %8.3f %13.3f %15.3f %18.3f   (%ds)' % (k, L, e_pre, e_fix_pre, e_fix_after, e_res_after, row[-1]), flush=True)


if __name__ == '__main__':
    main()
