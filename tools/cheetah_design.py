#!/usr/bin/env python3
"""Cheetah task design (2026-11-02): candidate rewards on the 17-d observation, their correlation with vel10, vel10Back, flip and with each other on the offline
dataset, their density, and the return of a random-action agent (dm_control, 20 episodes). Local, no model involved.
obs[0:8] = position (rootz, pitch (unwrapped), bthigh, bshin, bfoot, fthigh, fshin, ffoot); obs[8:17] = velocity (rootx, rootz, pitch rate, 6 joint velocities)."""
import math, sys
import numpy as np
INF = float('inf')
D = '../offline_qd/datasets/d4rl/'


def tol(x, lo, hi, m):
    inb = (lo <= x) & (x <= hi)
    d = np.where(x < lo, lo - x, x - hi) / m
    return np.where(inb, 1.0, np.clip(1 - d, 0, 1))


def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi


# aux = [speed, angmom, obs(17)] -> o = aux[..., 2:19]
CAND = {
    'vel10':     lambda o, s, a: tol(s, 10.0, INF, 10.0),
    'vel10Back': lambda o, s, a: tol(-s, 10.0, INF, 10.0),
    'flip':      lambda o, s, a: tol(a, 5.0, INF, 5.0),
    'jump':      lambda o, s, a: tol(o[..., 9], 3.0, INF, 3.0),                       # upward velocity >= 3
    'high':      lambda o, s, a: tol(o[..., 0], 0.4, INF, 0.4),                       # body height (rootz) >= 0.4
    'low':       lambda o, s, a: tol(-o[..., 0], 0.1, INF, 0.2),                      # crouching
    'nose_up':   lambda o, s, a: tol(-wrap(o[..., 1]), 0.7, INF, 0.5),                # pitch wrapped >= 40 deg one way
    'nose_down': lambda o, s, a: tol(wrap(o[..., 1]), 0.7, INF, 0.5),
    'bkick':     lambda o, s, a: tol(o[..., 11], 20.0, INF, 20.0),                    # back thigh swinging fast
    'fkick':     lambda o, s, a: tol(o[..., 14], 20.0, INF, 20.0),                    # front thigh swinging fast
    'spread':    lambda o, s, a: tol(np.abs(o[..., 2] - o[..., 5]), 1.2, INF, 0.8),   # thighs far apart
    'hop_still': lambda o, s, a: tol(o[..., 9], 2.0, INF, 2.0) * tol(np.abs(s), 0.0, 1.0, 2.0),   # up while not moving horizontally
    'upside':    lambda o, s, a: tol(-np.cos(o[..., 1]), 0.5, INF, 1.0),              # pitched past 120 deg (on its back)
}


def main():
    d = np.load(D + 'cheetah_rnd.npy', allow_pickle=True).item()
    obs = np.asarray(d['observations']).reshape(-1, 17)
    ang = np.load(D + 'angmomentum_aux_cheetah.npy').reshape(-1)
    rng = np.random.default_rng(0)
    idx = rng.choice(obs.shape[0], 1_000_000, replace=False)
    o, s, a = obs[idx], obs[idx, 8], ang[idx]
    R = {k: f(o, s, a) for k, f in CAND.items()}
    names = list(R)
    print('density (fraction of states with reward > 0.5) and mean reward:')
    for k in names: print('  %-10s %.4f  %.4f' % (k, (R[k] > 0.5).mean(), R[k].mean()))
    M = np.corrcoef(np.stack([R[k] for k in names]))
    print('correlation matrix:')
    print('           ' + ' '.join('%8s' % k[:8] for k in names))
    for i, k in enumerate(names): print('%-10s ' % k + ' '.join('%8.2f' % M[i, j] for j in range(len(names))))
    # random agent
    from dm_control import suite
    env = suite.load('cheetah', 'run', environment_kwargs=dict(flat_observation=True))
    sp = env.action_spec()

    def ob(ts): return np.concatenate([v if v.ndim else v.reshape(-1) for v in ts.observation.values()])
    res = {k: [] for k in names}
    for ep in range(20):
        ts = env.reset(); acc = {k: 0.0 for k in names}
        for t in range(1000):
            ts = env.step(rng.uniform(sp.minimum, sp.maximum)); ox = ob(ts)
            speed = ox[8]; am = float(env.physics.named.data.subtree_angmom['torso'][1])
            for k, f in CAND.items(): acc[k] += float(f(ox, speed, am))
        for k in names: res[k].append(acc[k])
    print('random-action return (20 episodes, 1000 steps, max 1000):')
    for k in names: print('  %-10s %6.0f ± %4.0f' % (k, np.mean(res[k]), np.std(res[k])))


if __name__ == '__main__':
    main()
