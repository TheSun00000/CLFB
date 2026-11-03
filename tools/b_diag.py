#!/usr/bin/env python3
"""Diagnostics of how B changed between the pretrained model (theta_0) and the B after stage 1 (theta_1). Eval only; runs on the cluster, not locally.

Sections of the test plan (docs/journal.md 2026-11-03):
  A  sanity            : spectrum / trace / condition of Sigma_0, Sigma_1, pointwise norms ||B(s)|| (vs sqrt d), fit-error definition.
  B  subspace change   : A~ = S1^-1/2 E[B1 B0^T] S0^-1/2, its singular values (cosines of the principal angles), counts above 5 / 10 / 30 degrees,
                         R^B = 1 - mean sigma^2, share of sum sin^2 in the top 1 / 3 / 5 angles, polar decomposition A~ = Q P, same SVD on the
                         pre-normalisation output b(s) of the last layer.
  C  per-task latents  : cos of the whitened latents raw and Procrustes-aligned (cos(z0, Q^T z1)), tracked T z0 vs fresh z1, inter-task cosine matrices,
                         load of z0 on the changed directions (sigma < 0.9), held-out fit errors (and on the fine-tuning labeled set D_k), fit error with an intercept.
  D  retention signal  : R^B integrand z^T M z / z^T S0^-1 z, M = S0^-1 - T^T S1^-1 T, T = E[B1 B0^T] S0^-1, on goal latents z = B0(s) (s ~ rho, and s in
                         the high-reward states of the pose tasks) against the uniform sphere.
Sections E (behaviour) and F (consistency of F with B) need rollouts / the FB loss and are not in this script.

All expectations over rho use 200k states of D; latents use 100k labeled states, fit errors 200k held-out states; every state used is DISJOINT from the
fine-tuning labeled set D_k of the run (rebuilt from its seed: valid_idx[default_rng(seed*1000+1).choice(9_990_000, 100000)]).

Usage: python tools/b_diag.py --pre models/.../fb_agent_latest.pickle --run NAME:path/to/fb_stage_1_run.pickle:SEED [--run ...]
Output: logs/<time>_bdiag/{A_sanity,B_angles,B_summary,C_tasks,C_intertask_pre,C_intertask_new,C_fit,D_retention}.csv, B_singular_values.png, info.txt
"""
import argparse, csv, os, sys
from datetime import datetime
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'tools'))
sys.path.insert(0, ROOT)

LOCAL = 'nazim' in os.getcwd()
DATASET_FOLDER = '../offline_qd/datasets/d4rl/' if LOCAL else '../fre_experiments/datasets/'
BUILTIN = {'stand': 0, 'walk': 2, 'vel8': 8, 'run': 10, 'flip': 'flip'}
ALL_TASKS = ['stand', 'walk', 'vel8', 'run', 'flip', 'supine', 'jump_any', 'kick_l', 'split', 'flamingo_r']
POSE_TASKS = ['supine', 'split', 'flamingo_r', 'jump_any', 'kick_l']
N_RHO, N_LAB, N_HELD = 200_000, 100_000, 200_000
DEF_EPS = ('eps = mean((rbar - B w)^2) / mean(rbar^2) on states disjoint from the labels used for z; rbar = r / sqrt(mean_lab r^2); '
           'w = Sigma^-1 z, z = mean_lab(rbar B), Sigma on the rho states (200k); 1 = nothing explained')


def sym_pow(S, p):
    w, v = np.linalg.eigh(S)
    return (v * w ** p) @ v.T


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def subspace_summary(G0, G1):
    """G0, G1: [n, d] features on the SAME rho states. Returns the whitened cross-covariance A~ and its SVD / polar decomposition."""
    n, d = G0.shape
    S0, S1 = G0.T @ G0 / n, G1.T @ G1 / n
    C = G1.T @ G0 / n
    W0, W1 = sym_pow(S0, -0.5), sym_pow(S1, -0.5)
    At = W1 @ C @ W0
    U, s, Vt = np.linalg.svd(At)
    s_c = np.clip(s, 0.0, 1.0)
    ang = np.degrees(np.arccos(s_c))
    sin2 = np.sin(np.radians(ang)) ** 2
    order = np.argsort(-ang)
    Q = U @ Vt
    P = Vt.T @ np.diag(s) @ Vt
    eig_ang = np.sort(np.abs(np.degrees(np.angle(np.linalg.eigvals(Q)))))[::-1]
    out = dict(S0=S0, S1=S1, C=C, W0=W0, W1=W1, At=At, U=U, s=s, Vt=Vt, ang=ang, Q=Q, P=P, eig_ang=eig_ang,
               n5=int((ang > 5).sum()), n10=int((ang > 10).sum()), n30=int((ang > 30).sum()),
               RB=float(1 - np.mean(s ** 2)), Q_minus_I=float(np.linalg.norm(Q - np.eye(d))), P_minus_I=float(np.linalg.norm(P - np.eye(d))),
               top1=float(sin2[order[:1]].sum() / sin2.sum()), top3=float(sin2[order[:3]].sum() / sin2.sum()), top5=float(sin2[order[:5]].sum() / sin2.sum()))
    return out


def fit_error(Fsig, rb_lab_F, rb_lab, Fheld, rb_held):
    """Held-out error of the readout w = Sigma^-1 mean(rbar F) (Sigma from Fsig)."""
    z = (rb_lab_F * rb_lab[:, None]).mean(0)
    S = Fsig.T @ Fsig / len(Fsig)
    w = np.linalg.solve(S, z)
    return float(((rb_held - Fheld @ w) ** 2).mean() / (rb_held ** 2).mean())


def analyse_task_latents(F0, F1, lab, rho, held, R, tasks, sub, Dk0=None, Dk1=None, RDk=None):
    """F0, F1: features of the pool [n_pool, d]; lab / rho / held: index arrays into the pool; R[task]: rewards on the pool; sub: subspace_summary on rho."""
    d = F0.shape[1]
    rows, fit_rows = [], []
    T = sub['C'] @ np.linalg.inv(sub['S0'])
    low = sub['s'] < 0.9
    Vlow = sub['Vt'][low]                                           # rows: right singular vectors with sigma < 0.9 (input = theta_0 whitened coordinates)
    Z0, Z1, Zw0, Zw1 = {}, {}, {}, {}
    ones = lambda n: np.ones((n, 1))
    for k in tasks:
        r = R[k]
        s = float(np.sqrt((r[lab] ** 2).mean())) + 1e-12
        rb = r / s
        z0, z1 = (F0[lab] * rb[lab, None]).mean(0), (F1[lab] * rb[lab, None]).mean(0)
        zw0, zw1 = sub['W0'] @ z0, sub['W1'] @ z1
        Z0[k], Z1[k], Zw0[k], Zw1[k] = z0, z1, zw0, zw1
        load = float(np.sum((Vlow @ zw0) ** 2) / np.sum(zw0 ** 2)) if low.any() else 0.0
        tz = T @ z0
        rows.append(dict(task=k, cos_plain=cosine(z0, z1), cos_whitened=cosine(zw0, zw1), cos_procrustes=cosine(zw0, sub['Q'].T @ zw1),
                         angle_whitened=float(np.degrees(np.arccos(np.clip(cosine(zw0, zw1), -1, 1)))),
                         angle_procrustes=float(np.degrees(np.arccos(np.clip(cosine(zw0, sub['Q'].T @ zw1), -1, 1)))),
                         cos_tracked=cosine(tz, z1), relerr_tracked=float(np.linalg.norm(tz - z1) / np.linalg.norm(z1)),
                         load_changed=load, n_changed=int(low.sum())))
        # fit errors: readout fitted on lab, error on held-out; theta_0 and theta_1; with an intercept; and on D_k (readout from D_k labels)
        rh = rb[held]
        e = {}
        for name, F in (('pre', F0), ('new', F1)):
            e[name] = fit_error(F[rho], F[lab], rb[lab], F[held], rh)
            Frho, Flab, Fheld = [np.concatenate([ones(len(ix)), F[ix]], 1) for ix in (rho, lab, held)]
            e[name + '_int'] = fit_error(Frho, Flab, rb[lab], Fheld, rh)
        row = dict(task=k, eps_pre=e['pre'], eps_new=e['new'], eps_pre_int=e['pre_int'], eps_new_int=e['new_int'])
        if Dk0 is not None:
            rD = RDk[k] / (float(np.sqrt((RDk[k] ** 2).mean())) + 1e-12)
            # readout from D_k labels, error on held-out states
            for name, Dk, F in (('pre', Dk0, F0), ('new', Dk1, F1)):
                zD = (Dk * rD[:, None]).mean(0)
                w = np.linalg.solve(F[rho].T @ F[rho] / len(rho), zD)
                row['eps_%s_Dk' % name] = float(((rb[held] - F[held] @ w) ** 2).mean() / (rb[held] ** 2).mean())
                row['fit_on_Dk_%s' % name] = float(((rD - Dk @ w) ** 2).mean() / (rD ** 2).mean())     # error on the labeled set itself
        fit_rows.append(row)
    mat = lambda Z: np.array([[cosine(Z[a], Z[b]) for b in tasks] for a in tasks])
    return rows, fit_rows, mat(Z0), mat(Z1), mat(Zw0), mat(Zw1), T


def retention_integrand(sub, T, groups, rng):
    """R^B integrand z^T M z / z^T S0^-1 z on groups of latents (dict name -> [n, d]), plus the uniform sphere."""
    d = sub['S0'].shape[0]
    S0i, S1i = np.linalg.inv(sub['S0']), np.linalg.inv(sub['S1'])
    M = S0i - T.T @ S1i @ T
    g = dict(groups)
    u = rng.normal(size=(100_000, d))
    g['uniform_sphere'] = u / np.linalg.norm(u, axis=1, keepdims=True)
    out = {}
    for name, Z in g.items():
        num = np.einsum('nd,de,ne->n', Z, M, Z)
        den = np.einsum('nd,de,ne->n', Z, S0i, Z)
        r = num / den
        out[name] = (len(Z), float(r.mean()), float(np.median(r)), float(np.percentile(r, 90)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pre', required=True)
    ap.add_argument('--run', action='append', required=True, help='NAME:path_of_stage1_pickle:SEED (seed of the run, to rebuild its labeled set D_k)')
    ap.add_argument('--tasks', nargs='+', default=ALL_TASKS)
    a = ap.parse_args()
    import jax, jax.numpy as jnp
    import reward_library as RL
    import task_design as TD
    print(jax.devices(), flush=True)
    folder = os.path.join(ROOT, 'logs', datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_bdiag')
    os.makedirs(folder)
    from _utils.logs import run_info
    open(os.path.join(folder, 'info.txt'), 'w').write(run_info() + 'args: %s\n%s\n' % (vars(a), DEF_EPS))
    print('LOGS_FOLDER      : ./logs/' + os.path.basename(folder), flush=True)

    runs = []
    for spec in a.run:
        name, path, seed = spec.split(':')
        runs.append((name, path, int(seed)))

    d = np.load(f'{DATASET_FOLDER}/walker_rnd.npy', allow_pickle=True).item()
    N = 10000 * 1000
    obs_all = np.asarray(d['observations']).reshape(N, 24).astype(np.float32)
    del d
    aux_all = np.concatenate([np.load(f'{DATASET_FOLDER}/aux_walker.npy').reshape(N, 3), np.load(f'{DATASET_FOLDER}/angmomentum_aux_walker.npy').reshape(N, 1)], 1)
    valid = np.arange(N)[(np.arange(N) % 1000) != 999]
    Dk_idx = {name: valid[np.random.default_rng(seed * 1000 + 1).choice(len(valid), 100_000, replace=False)] for name, _, seed in runs}
    excluded = np.zeros(N, bool)
    for ix in Dk_idx.values():
        excluded[ix] = True
    free = valid[~excluded[valid]]
    pool_idx = np.sort(np.random.default_rng(0).choice(free, N_RHO + N_LAB + N_HELD, replace=False))
    perm = np.random.default_rng(1).permutation(len(pool_idx))
    rho, lab, held = perm[:N_RHO], perm[N_RHO:N_RHO + N_LAB], perm[N_RHO + N_LAB:]
    print('disjoint from every D_k:', not excluded[pool_idx].any(), flush=True)

    def rewards(idx):
        o, x = obs_all[idx], aux_all[idx]
        R = {k: TD.builtin(x, p) for k, p in BUILTIN.items()}
        for k, (desc, fn) in RL.LIBRARY.items():
            R[k] = fn(o.astype(np.float64))
        return {k: np.asarray(v, dtype=np.float64) for k, v in R.items()}
    R = rewards(pool_idx)
    RDk = {name: rewards(ix) for name, ix in Dk_idx.items()}

    def parts(p, x):
        h = x @ p['TorchDense_0']['kernel'] + p['TorchDense_0']['bias']
        ln = p['NTanh_0']['LayerNorm_0']
        h = jnp.tanh((h - h.mean(-1, keepdims=True)) / jnp.sqrt(h.var(-1, keepdims=True) + 1e-5) * ln['scale'] + ln['bias'])
        h = jnp.maximum(h @ p['TorchDense_1']['kernel'] + p['TorchDense_1']['bias'], 0.0)
        y = h @ p['TorchDense_2']['kernel'] + p['TorchDense_2']['bias']
        return jnp.sqrt(y.shape[-1]) * y / jnp.maximum(jnp.linalg.norm(y, axis=-1, keepdims=True), 1e-12), y

    def feats(p, idx):
        f = jax.jit(lambda x: parts(p, x))
        out = [f(obs_all[idx[i:i + 100_000]]) for i in range(0, len(idx), 100_000)]
        return (np.concatenate([np.asarray(o[0]) for o in out]).astype(np.float64), np.concatenate([np.asarray(o[1]) for o in out]).astype(np.float64))
    p0 = TD.load_params(a.pre)
    F0, Y0 = feats(p0, pool_idx)
    dim = F0.shape[1]

    w = lambda name: csv.writer(open(os.path.join(folder, name), 'w', newline=''))
    wA, wAng, wB, wBn, wC, wI0, wI1, wFit, wD = (w('A_sanity.csv'), w('B_angles.csv'), w('B_summary.csv'), w('B_prenorm_summary.csv'), w('C_tasks.csv'),
                                                 w('C_intertask_pre.csv'), w('C_intertask_new.csv'), w('C_fit.csv'), w('D_retention.csv'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    headers = {}

    def put(wr, key, row):
        if key not in headers:
            wr.writerow(['run'] + list(row)); headers[key] = True
        wr.writerow([cur] + [('%.6g' % v if isinstance(v, (float, np.floating)) else v) for v in row.values()])
    wA.writerow(['run', 'model', 'eig_min', 'eig_median', 'eig_max', 'trace', 'cond', 'norm_B_mean', 'norm_B_min', 'norm_B_max', 'sqrt_d', 'definition_of_eps'])
    wI0.writerow(['run'] + list(a.tasks)); wI1.writerow(['run'] + list(a.tasks))
    wD.writerow(['run', 'group', 'n', 'mean_ratio', 'median_ratio', 'p90_ratio'])
    wAng.writerow(['run', 'i', 'singular_value', 'angle_deg'])
    for name, path, seed in runs:
        cur = name
        p1 = TD.load_params(path)
        F1, Y1 = feats(p1, pool_idx)
        for tag, F in (('pre', F0), ('new', F1)):
            S = F[rho].T @ F[rho] / len(rho)
            ev = np.linalg.eigvalsh(S)
            nb = np.linalg.norm(F, axis=1)
            wA.writerow([name, tag, '%.6g' % ev.min(), '%.6g' % np.median(ev), '%.6g' % ev.max(), '%.6g' % ev.sum(), '%.6g' % (ev.max() / ev.min()),
                         '%.6g' % nb.mean(), '%.6g' % nb.min(), '%.6g' % nb.max(), '%.6g' % np.sqrt(dim), DEF_EPS if tag == 'pre' else ''])
        sub = subspace_summary(F0[rho], F1[rho])
        subY = subspace_summary(Y0[rho], Y1[rho])
        for i, (sv, an) in enumerate(zip(sub['s'], sub['ang'])):
            wAng.writerow([name, i, '%.6g' % sv, '%.6g' % an])
        ax.plot(np.sort(sub['s'])[::-1], label=name)
        keys = ['RB', 'n5', 'n10', 'n30', 'top1', 'top3', 'top5', 'Q_minus_I', 'P_minus_I']
        for wr, S, tg in ((wB, sub, 'B'), (wBn, subY, 'b_prenorm')):
            if name == runs[0][0]:
                wr.writerow(['run', 'what'] + keys + ['sv_min', 'sv_median', 'sv_max', 'Q_eig_angle_max', 'Q_eig_angle_median'])
            wr.writerow([name, tg] + [('%.6g' % S[k]) for k in keys] + ['%.6g' % S['s'].min(), '%.6g' % np.median(S['s']), '%.6g' % S['s'].max(),
                                                                         '%.6g' % S['eig_ang'].max(), '%.6g' % np.median(S['eig_ang'])])
        print('[%s] B: R^B %.4f  angles>5/10/30: %d/%d/%d  top1/3/5 share of sin^2: %.2f/%.2f/%.2f  |Q-I| %.3f |P-I| %.3f  | prenorm: R %.4f >5/10/30: %d/%d/%d top1/3/5 %.2f/%.2f/%.2f'
              % (name, sub['RB'], sub['n5'], sub['n10'], sub['n30'], sub['top1'], sub['top3'], sub['top5'], sub['Q_minus_I'], sub['P_minus_I'],
                 subY['RB'], subY['n5'], subY['n10'], subY['n30'], subY['top1'], subY['top3'], subY['top5']), flush=True)
        Dk0, Dk1 = feats(p0, Dk_idx[name])[0], feats(p1, Dk_idx[name])[0]
        rows, fit_rows, M0, M1, Mw0, Mw1, T = analyse_task_latents(F0, F1, lab, rho, held, R, a.tasks, sub, Dk0, Dk1, RDk[name])
        for r in rows:
            put(wC, 'C', r)
        for r in fit_rows:
            put(wFit, 'Fit', r)
        for i, k in enumerate(a.tasks):
            wI0.writerow([name] + ['%.4f' % v for v in M0[i]]); wI1.writerow([name] + ['%.4f' % v for v in M1[i]])
        off = ~np.eye(len(a.tasks), dtype=bool)
        print('[%s] inter-task cos (plain, off-diagonal mean): pre %.3f new %.3f' % (name, M0[off].mean(), M1[off].mean()), flush=True)
        for r in rows:
            print('[%s] %-11s cos plain %.3f whitened %.3f procrustes %.3f | tracked cos %.3f relerr %.3f | load on changed dirs %.3f' %
                  (name, r['task'], r['cos_plain'], r['cos_whitened'], r['cos_procrustes'], r['cos_tracked'], r['relerr_tracked'], r['load_changed']), flush=True)
        for r in fit_rows:
            print('[%s] %-11s eps pre %.3f new %.3f | with intercept pre %.3f new %.3f | readout from D_k: pre %.3f new %.3f, error on D_k itself pre %.3f new %.3f' %
                  (name, r['task'], r['eps_pre'], r['eps_new'], r['eps_pre_int'], r['eps_new_int'], r['eps_pre_Dk'], r['eps_new_Dk'], r['fit_on_Dk_pre'], r['fit_on_Dk_new']), flush=True)
        groups = {'goal_rho_states': F0[rho]}
        for k in POSE_TASKS:
            m = R[k] >= 0.5
            if m.sum() >= 1000:
                groups['goal_%s_highreward' % k] = F0[m]
        dr = retention_integrand(sub, T, groups, np.random.default_rng(3))
        for g, (n_, mean_, med_, p90_) in dr.items():
            wD.writerow([name, g, n_, '%.6g' % mean_, '%.6g' % med_, '%.6g' % p90_])
            print('[%s] R^B integrand %-28s n=%6d mean %.4f median %.4f p90 %.4f' % (name, g, n_, mean_, med_, p90_), flush=True)
    ax.set_xlabel('index (sorted)'); ax.set_ylabel('singular value of A~ = cos(principal angle)'); ax.set_title('B after stage 1 vs pretrained'); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(folder, 'B_singular_values.png'), dpi=130)


if __name__ == '__main__':
    main()
