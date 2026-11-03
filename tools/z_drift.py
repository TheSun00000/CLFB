#!/usr/bin/env python3
"""Latent drift between the pretrained B and a fine-tuned B (eval only, no training; runs on the cluster, not locally).

Question: after B is fine-tuned on one task, how far does the latent of every task move? For each task, with the SAME labelled states:
    z     = mean(rbar B_pre)       z' = mean(rbar B_new)         (rbar = r / sqrt(mean r^2), as in the evaluation)
Per task (drift.csv):
    norm_pre, norm_new, norm_ratio     ||z||, ||z'||, ||z'|| / ||z||
    cos                                cos(z, z')                       (1 = same direction)
    rel_err                            ||z' - z|| / ||z||
    rel_err_unit                       ||z'/||z'|| - z/||z|| ||         (the direction only, = sqrt(2 - 2 cos))
    cos_T, cos_rot                     cos(z', T z) with T = A Sigma_pre^-1, A = E[B_new B_pre^T] (the linear map between the two latent spaces),
                                       and cos(z', Q z) with Q the best ORTHOGONAL map (Procrustes) from B_pre to B_new
    cos_zTz                            cos(z, T z): how far T z is from z itself (the quantity a "T close to identity" regulariser would shrink)
    corr_recon                         correlation on held-out states of the reward reconstructed by each model, B_pre w and B_new w'
                                       (w = Sigma^-1 z of the model itself): the drift in reward space
    eps_pre, eps_new                   reward error of each model (1 = nothing explained)
Global (global.csv): ||T - I||_F / sqrt(d), singular values of T (min / median / max), ||T - Q||_F / ||T||_F (distance of T to the closest
orthogonal matrix), mean cos(u, T u) and cos(u, Q u) for random unit u, relative residual ||B_new - B_pre Q^T|| / ||B_new|| (and with Q = I).

Usage: python tools/z_drift.py --pre models/.../fb_agent_latest.pickle --new models/.../fb_stage_1_run.pickle
Output: logs/<time>_zdrift/{drift.csv,global.csv,info.txt}
"""
import argparse, csv, os, sys
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
ALL_TASKS = ['stand', 'walk', 'vel8', 'run', 'flip', 'supine', 'jump_any', 'kick_l', 'split', 'flamingo_r']


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pre', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--tasks', nargs='+', default=ALL_TASKS)
    ap.add_argument('--labels', type=int, default=100_000)
    ap.add_argument('--n', type=int, default=600_000, help='dataset states used (the last 200k are the held-out set)')
    ap.add_argument('--ridge', type=float, default=1e-5)
    a = ap.parse_args()
    import jax, jax.numpy as jnp
    print(jax.devices(), flush=True)
    folder = os.path.join(ROOT, 'logs', datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_zdrift')
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

    def feat(p):
        f = jax.jit(lambda x: TD.B_apply(p, x))
        return np.concatenate([np.asarray(f(obs[i:i + 100_000])) for i in range(0, len(obs), 100_000)]).astype(np.float64)
    Fp, Fn = feat(TD.load_params(a.pre)), feat(TD.load_params(a.new))
    dim = Fp.shape[1]
    held = np.arange(a.n - 200_000, a.n)
    pool = np.arange(a.n - 200_000)
    rng = np.random.default_rng(1)
    lab = rng.choice(pool, a.labels, replace=False)
    sig = rng.choice(pool, 100_000, replace=False)
    cov = lambda F: F[sig].T @ F[sig] / len(sig) + a.ridge * np.eye(dim)
    Sp, Sn = cov(Fp), cov(Fn)
    A = Fn[sig].T @ Fp[sig] / len(sig)                      # E[B_new B_pre^T]
    T = A @ np.linalg.inv(Sp)                               # z' = T z for rewards in the span of B_pre
    U, S, Vt = np.linalg.svd(Fn[sig].T @ Fp[sig] / len(sig))
    Q = U @ Vt                                              # orthogonal Procrustes map B_pre -> B_new
    sv = np.linalg.svd(T, compute_uv=False)
    u = np.random.default_rng(2).normal(size=(1000, dim))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    Uo, _, Vo = np.linalg.svd(T)
    Qt = Uo @ Vo
    g = dict(dim=dim, T_minus_I_rel=np.linalg.norm(T - np.eye(dim)) / np.sqrt(dim), sv_min=sv.min(), sv_median=np.median(sv), sv_max=sv.max(),
             T_to_orthogonal_rel=np.linalg.norm(T - Qt) / np.linalg.norm(T), mean_cos_u_Tu=float(np.mean(np.sum(u * (u @ T.T), 1) / np.linalg.norm(u @ T.T, axis=1))),
             mean_cos_u_Qu=float(np.mean(np.sum(u * (u @ Q.T), 1) / np.linalg.norm(u @ Q.T, axis=1))),
             procrustes_rel_resid=float(np.linalg.norm(Fn[sig] - Fp[sig] @ Q.T) / np.linalg.norm(Fn[sig])),
             identity_rel_resid=float(np.linalg.norm(Fn[sig] - Fp[sig]) / np.linalg.norm(Fn[sig])))
    with open(os.path.join(folder, 'global.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(list(g)); w.writerow([('%.6g' % v) for v in g.values()])
    print('global:', {k: round(float(v), 4) for k, v in g.items()}, flush=True)

    cols = ['task', 'norm_pre', 'norm_new', 'norm_ratio', 'cos', 'rel_err', 'rel_err_unit', 'cos_T', 'cos_rot', 'cos_zTz', 'corr_recon', 'eps_pre', 'eps_new']
    rows = []
    print('%-12s %8s %8s %6s | %6s %8s %9s | %6s %7s %7s | %10s %7s %7s' % tuple(cols[0:1] + ['|z|', "|z'|", 'ratio', 'cos', 'rel_err', 'rel_unit', 'cos_T', 'cos_rot', 'cos_zTz', 'corr_recon', 'eps_pre', 'eps_new']), flush=True)
    for k in a.tasks:
        r = R[k]
        s = float(np.sqrt((r[lab] ** 2).mean())) + 1e-12
        rb = r / s
        z, zn = (Fp[lab] * rb[lab, None]).mean(0), (Fn[lab] * rb[lab, None]).mean(0)
        wp, wn = np.linalg.solve(Sp, z), np.linalg.solve(Sn, zn)
        rp, rn = Fp[held] @ wp, Fn[held] @ wn
        eps = lambda rh: float(((rb[held] - rh) ** 2).mean() / (rb[held] ** 2).mean())
        row = [k, np.linalg.norm(z), np.linalg.norm(zn), np.linalg.norm(zn) / np.linalg.norm(z), cos(z, zn), np.linalg.norm(zn - z) / np.linalg.norm(z),
               np.linalg.norm(zn / np.linalg.norm(zn) - z / np.linalg.norm(z)), cos(zn, T @ z), cos(zn, Q @ z), cos(z, T @ z),
               float(np.corrcoef(rp, rn)[0, 1]), eps(rp), eps(rn)]
        rows.append(row)
        print('%-12s %8.3f %8.3f %6.2f | %6.3f %8.3f %9.3f | %6.3f %7.3f %7.3f | %10.3f %7.3f %7.3f' % tuple(row), flush=True)
    with open(os.path.join(folder, 'drift.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(cols)
        for row in rows:
            w.writerow([row[0]] + [('%.6g' % v) for v in row[1:]])


if __name__ == '__main__':
    main()
