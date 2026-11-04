#!/usr/bin/env python3
"""Where does the retention term R^B fail? R^B(z) = z^T M z / z^T Sigma_0^-1 z, M = Sigma_0^-1 - T^T Sigma_1^-1 T, T = E[B1 B0^T] Sigma_0^-1, is the fraction of the energy of the reward
represented by B_0 with latent z that B_1 does not reproduce. The training term averages it over latents z UNIFORM on the sphere. Here it is evaluated at
  - the uniform sphere,
  - the latent of every task (z_k = E[rbar_k B_0]),
  - the latent of the constant reward r = 1 (z_c = E[B_0]),
and the cosine of every task latent with z_c. Eval only; cluster. Usage: python tools/rb_tasks.py --pre PRE --new NAME:path [--new ...]"""
import argparse, csv, os, sys
from datetime import datetime
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'tools'))
sys.path.insert(0, ROOT)
import reward_library as RL
import task_design as TD

DATASET_FOLDER = '../fre_experiments/datasets/'
BUILTIN = {'stand': 0, 'walk': 2, 'vel8': 8, 'run': 10, 'flip': 'flip'}
TASKS = ['stand', 'walk', 'vel8', 'run', 'flip', 'supine', 'jump_any', 'kick_l', 'split', 'flamingo_r']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pre', required=True)
    ap.add_argument('--new', action='append', required=True, help='NAME:path of a stage-1 pickle')
    a = ap.parse_args()
    import jax, jax.numpy as jnp
    d = np.load(f'{DATASET_FOLDER}/walker_rnd.npy', allow_pickle=True).item()
    N = 10000 * 1000
    idx = np.sort(np.random.default_rng(0).choice(N, 400_000, replace=False))
    obs = np.asarray(d['observations']).reshape(N, 24)[idx].astype(np.float32)
    del d
    aux = np.concatenate([np.load(f'{DATASET_FOLDER}/aux_walker.npy').reshape(N, 3)[idx], np.load(f'{DATASET_FOLDER}/angmomentum_aux_walker.npy').reshape(N, 1)[idx]], 1)
    R = {k: TD.builtin(aux, p) for k, p in BUILTIN.items()}
    for k, (desc, fn) in RL.LIBRARY.items():
        R[k] = fn(obs.astype(np.float64))
    feat = lambda p: np.concatenate([np.asarray(jax.jit(lambda x: TD.B_apply(p, x))(obs[i:i + 100_000])) for i in range(0, len(obs), 100_000)]).astype(np.float64)
    F0 = feat(TD.load_params(a.pre))
    n = len(F0)
    folder = os.path.join(ROOT, 'logs', datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_rbtasks')
    os.makedirs(folder)
    w = csv.writer(open(os.path.join(folder, 'rb_tasks.csv'), 'w', newline=''))
    w.writerow(['model', 'what', 'R_B_at_latent', 'cos_with_constant_latent'])
    print('LOGS_FOLDER      : ./logs/' + os.path.basename(folder), flush=True)
    S0 = F0.T @ F0 / n
    S0i = np.linalg.inv(S0)
    zc = F0.mean(0)
    for spec in a.new:
        name, path = spec.split(':', 1)
        F1 = feat(TD.load_params(path))
        S1i = np.linalg.inv(F1.T @ F1 / n)
        T = (F1.T @ F0 / n) @ S0i
        M = S0i - T.T @ S1i @ T
        ratio = lambda z: float(z @ M @ z / (z @ S0i @ z))
        rng = np.random.default_rng(0)
        u = rng.normal(size=(100_000, F0.shape[1]))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        uni = float((np.einsum('nd,de,ne->n', u, M, u) / np.einsum('nd,de,ne->n', u, S0i, u)).mean())
        rows = [('uniform sphere (training average)', uni, float('nan')), ('constant reward r=1', ratio(zc), 1.0)]
        for k in TASKS:
            r = R[k] / (np.sqrt((R[k] ** 2).mean()) + 1e-12)
            z = (F0 * r[:, None]).mean(0)
            rows.append((k, ratio(z), float(z @ zc / (np.linalg.norm(z) * np.linalg.norm(zc)))))
        print('== %s' % name)
        for what, rv, cv in rows:
            w.writerow([name, what, '%.4f' % rv, '%.4f' % cv])
            print('   %-36s R^B at that latent %.3f   cos with constant-reward latent %s' % (what, rv, '%.3f' % cv if cv == cv else '-'), flush=True)


if __name__ == '__main__':
    main()
