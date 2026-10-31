#!/usr/bin/env python3
"""Design walker tasks that are NOT aligned with run, and measure how well B(s) encodes them.

Why: with the five built-in tasks (stand, walk, vel8, run, flip) every reward is a function of the same 4 physics quantities
(horizontal velocity, uprightness, torso height, angular momentum), so training B on run also aligns it with the others and the retention
terms (mu_b, mu_q) have nothing to protect. A good second task is one that (1) overlaps little with run in reward space, (2) is not a function of
the quantities run uses, (3) is encoded by the PRETRAINED B (otherwise there is nothing to forget) and (4) is encoded worse once B has been
fine-tuned on run (forgetting). This script measures all four on the offline dataset; the real test is the zero-shot return (eval-only job).

Reward functions are defined on the 24-d OBSERVATION (what B sees, and available in rollouts):
    obs[0:14]  = (cos, sin) of the absolute angle of 7 bodies: torso, r.thigh, r.leg, r.foot, l.thigh, l.leg, l.foot
    obs[14]    = torso height          obs[15:24] = velocities: z, x, angular (torso), then the 6 joint velocities
`REWARDS` is the single definition used by the analysis and ported to fb_continual_temp.py.

Usage (needs the dataset, a pretrained checkpoint and optional fine-tuned ones; see ARGS below):
    python tools/task_design.py --pre pretrained.pickle --ft run_focus.pickle run_unfocused.pickle [--n 400000]
"""
import argparse, os, pickle, sys
import numpy as np

# ---------------------------------------------------------------------------------------------------------------- rewards
INF = float('inf')


def tolerance(x, lower, upper, margin, sigmoid='linear', value_at_margin=0.0):
    """dm_control style bounded reward: 1 inside [lower, upper], decays to value_at_margin at distance `margin` outside."""
    in_bounds = (lower <= x) & (x <= upper)
    d = np.where(x < lower, lower - x, x - upper) / margin
    if sigmoid == 'linear':
        scaled = d * (1.0 - value_at_margin)
        out = np.where(np.abs(scaled) < 1, 1.0 - scaled, 0.0)
    else:                                                  # gaussian
        scale = np.sqrt(-2 * np.log(value_at_margin))
        out = np.exp(-0.5 * (d * scale) ** 2)
    return np.where(in_bounds, 1.0, out)


def angle(obs, i):               # absolute angle of body i (0 torso, 1 r.thigh, 2 r.leg, 3 r.foot, 4 l.thigh, 5 l.leg, 6 l.foot)
    return np.arctan2(obs[..., 2 * i + 1], obs[..., 2 * i])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def builtin(aux, param):          # the five tasks of the benchmark, from aux = [vel, upright, height, angmom]
    standing = tolerance(aux[..., 2], 1.2, INF, 0.6, 'gaussian', 0.1)
    stand = (3 * standing + (1 + aux[..., 1]) / 2) / 4
    if param == 'flip':
        move = tolerance(aux[..., 3], 5.0, INF, 5.0, 'linear', 0.5)
    elif param == 0:
        move = np.ones_like(stand)
    else:
        move = tolerance(aux[..., 0], float(param), INF, param / 2, 'linear', 0.5)
    return stand * (5 * move + 1) / 6


# candidate tasks: name -> (description, function(obs) -> reward in [0, 1])
D = np.pi / 180
REWARDS = {
    # --- posture of the legs (independent of speed, of uprightness and of height by construction)
    'split':       ('legs apart: |angle(r.thigh) - angle(l.thigh)| >= 60 deg',
                    lambda o: tolerance(np.abs(wrap(angle(o, 1) - angle(o, 4))), 60 * D, INF, 40 * D)),
    'r_knee_bent': ('right knee bent: the leg is rotated >= 90 deg from the thigh',
                    lambda o: tolerance(np.abs(wrap(angle(o, 2) - angle(o, 1))), 90 * D, INF, 60 * D)),
    'l_knee_bent': ('left knee bent: the leg is rotated >= 90 deg from the thigh',
                    lambda o: tolerance(np.abs(wrap(angle(o, 5) - angle(o, 4))), 90 * D, INF, 60 * D)),
    'knees_bent':  ('both knees bent >= 60 deg',
                    lambda o: tolerance(np.abs(wrap(angle(o, 2) - angle(o, 1))), 60 * D, INF, 40 * D)
                    * tolerance(np.abs(wrap(angle(o, 5) - angle(o, 4))), 60 * D, INF, 40 * D)),
    'feet_up':     ('both feet well above the ankle line: foot angle relative to the leg >= 45 deg',
                    lambda o: tolerance(np.abs(wrap(angle(o, 3) - angle(o, 2))), 45 * D, INF, 30 * D)
                    * tolerance(np.abs(wrap(angle(o, 6) - angle(o, 5))), 45 * D, INF, 30 * D)),
    # --- motion that is not forward speed
    'jump':        ('upward velocity >= 3 (and not lying down)',
                    lambda o: tolerance(o[..., 15], 3.0, INF, 3.0) * (1 + o[..., 0]) / 2),
    'kick_r':      ('right hip velocity >= 20 rad/s',
                    lambda o: tolerance(o[..., 18], 20.0, INF, 20.0)),
    'knee_swing':  ('both knees moving fast: |knee velocities| mean >= 25 rad/s',
                    lambda o: tolerance(0.5 * (np.abs(o[..., 19]) + np.abs(o[..., 22])), 25.0, INF, 25.0)),
    'spin_back':   ('torso angular velocity <= -8 rad/s (backward rotation)',
                    lambda o: tolerance(-o[..., 17], 8.0, INF, 8.0)),
    # --- torso configuration the built-ins cannot see (uprightness is symmetric in the lean)
    'lean_a':      ('torso tilted by 25..50 deg to one side (sign of sin of the torso angle)',
                    lambda o: tolerance(-angle(o, 0), 25 * D, 50 * D, 25 * D)),
    'lean_b':      ('torso tilted by 25..50 deg to the other side',
                    lambda o: tolerance(angle(o, 0), 25 * D, 50 * D, 25 * D)),
    # --- things in the same feature space as the built-ins (controls: expected to be aligned)
    'crouch':      ('(aux) torso height 0.5..0.8 and roughly upright',
                    None),
    'backward':    ('(aux) walking backwards: stand reward x velocity <= -2',
                    None),
    'lying':       ('(aux) lying down: low height and torso horizontal',
                    None),
}


def aux_rewards(aux):
    stand = lambda a: (3 * tolerance(a[..., 2], 1.2, INF, 0.6, 'gaussian', 0.1) + (1 + a[..., 1]) / 2) / 4
    return {
        'crouch': tolerance(aux[..., 2], 0.5, 0.8, 0.3) * (1 + aux[..., 1]) / 2,
        'backward': stand(aux) * tolerance(-aux[..., 0], 2.0, INF, 1.0),
        'lying': tolerance(aux[..., 2], 0.0, 0.5, 0.4) * tolerance(np.abs(aux[..., 1]), 0.0, 0.3, 0.4),
    }


# ---------------------------------------------------------------------------------------------------------------- B(s)
def load_B(path):
    import jax.numpy as jnp
    p = pickle.load(open(path, 'rb'))['fb_params']['backward']
    g = lambda a, b: jnp.asarray(p[a][b])
    W = dict(w0=g('TorchDense_0', 'kernel'), b0=g('TorchDense_0', 'bias'), ls=jnp.asarray(p['NTanh_0']['LayerNorm_0']['scale']),
             lb=jnp.asarray(p['NTanh_0']['LayerNorm_0']['bias']), w1=g('TorchDense_1', 'kernel'), b1=g('TorchDense_1', 'bias'),
             w2=g('TorchDense_2', 'kernel'), b2=g('TorchDense_2', 'bias'))

    def B(x):
        h = x @ W['w0'] + W['b0']
        h = (h - h.mean(-1, keepdims=True)) / jnp.sqrt(h.var(-1, keepdims=True) + 1e-5) * W['ls'] + W['lb']
        h = jnp.tanh(h)
        h = jnp.maximum(h @ W['w1'] + W['b1'], 0.0)
        y = h @ W['w2'] + W['b2']
        return jnp.sqrt(y.shape[-1]) * y / jnp.maximum(jnp.linalg.norm(y, axis=-1, keepdims=True), 1e-12)
    return B


def features(B, obs, chunk=100_000):
    import jax
    f = jax.jit(B)
    return np.concatenate([np.asarray(f(obs[i:i + chunk])) for i in range(0, len(obs), chunk)])


def eps_of(F, r, n_label=10_000, n_sigma=100_000, ridge=1e-5, seeds=3, rng=None):
    """Held-out reward error of the least-squares readout w = Sigma^-1 mean(rbar B) fitted on n_label labeled states (as in the eval)."""
    rng = rng or np.random.default_rng(0)
    N, d = F.shape
    out = []
    for _ in range(seeds):
        p = rng.permutation(N)
        lab, sig, held = p[:n_label], p[n_label:n_label + n_sigma], p[n_label + n_sigma:]
        s = np.sqrt((r[lab] ** 2).mean()) + 1e-12
        z = (F[lab] * (r[lab] / s)[:, None]).mean(0)
        S = F[sig].T @ F[sig] / len(sig) + ridge * np.eye(d)
        w = np.linalg.solve(S, z)
        rh = F[held] @ w
        out.append(((r[held] / s - rh) ** 2).mean() / ((r[held] / s) ** 2).mean())
    return float(np.mean(out))


def latent(F, r):
    r = r / (np.sqrt((r ** 2).mean()) + 1e-12)
    z = (F * r[:, None]).mean(0)
    return z / (np.linalg.norm(z) + 1e-12)


def load_params(path):
    import jax.numpy as jnp
    import jax
    return jax.tree_util.tree_map(jnp.asarray, pickle.load(open(path, 'rb'))['fb_params']['backward'])


def B_apply(p, x):
    import jax.numpy as jnp
    h = x @ p['TorchDense_0']['kernel'] + p['TorchDense_0']['bias']
    ln = p['NTanh_0']['LayerNorm_0']
    h = jnp.tanh((h - h.mean(-1, keepdims=True)) / jnp.sqrt(h.var(-1, keepdims=True) + 1e-5) * ln['scale'] + ln['bias'])
    h = jnp.maximum(h @ p['TorchDense_1']['kernel'] + p['TorchDense_1']['bias'], 0.0)
    y = h @ p['TorchDense_2']['kernel'] + p['TorchDense_2']['bias']
    return jnp.sqrt(y.shape[-1]) * y / jnp.maximum(jnp.linalg.norm(y, axis=-1, keepdims=True), 1e-12)


def fit_B(params, obs_lab, r_lab, w, steps=30000, lr=3e-4, bs=4096, seed=0):
    """B alone regressed on rbar with the readout w FIXED (what fit_backward_to_reward does; Adam, cosine decay)."""
    import jax, jax.numpy as jnp, optax
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps))
    st = opt.init(params)
    X, Y = jnp.asarray(obs_lab), jnp.asarray(r_lab)

    @jax.jit
    def step(p, s, key):
        idx = jax.random.randint(key, (bs,), 0, X.shape[0])
        loss, g = jax.value_and_grad(lambda q: jnp.mean((B_apply(q, X[idx]) @ w - Y[idx]) ** 2))(p)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, loss
    keys = jax.random.split(jax.random.PRNGKey(seed), steps)
    for i in range(steps):
        params, st, _ = step(params, st, keys[i])
    return params


def forgetting(a, obs, R, tasks):
    """Stage-2 test without retention: start from B fine-tuned on run, fit B on task X alone, then look at run again."""
    import jax, jax.numpy as jnp
    p0 = load_params(a.start)
    F0 = np.asarray(jax.jit(lambda x: B_apply(p0, x))(obs[:200_000]))
    run = R['run'][:200_000]
    e_run0 = eps_of(F0, run, seeds=2)
    print('start: B of %s, eps(run)=%.3f' % (os.path.basename(a.start), e_run0))
    print('%-12s %9s %9s | %9s %9s %8s | %s' % ('task X', 'eps_X pre', 'eps_X fit', 'eps_run0', 'eps_run', 'forget', 'cos(z_X, z_run) after'))
    rng = np.random.default_rng(0)
    lab = rng.choice(200_000, 10_000, replace=False)
    out = {}
    for k in tasks:
        r = R[k][:200_000]
        s = np.sqrt((r[lab] ** 2).mean()) + 1e-12
        z = (F0[lab] * (r[lab] / s)[:, None]).mean(0)
        S = F0[:100_000].T @ F0[:100_000] / 100_000 + 1e-5 * np.eye(F0.shape[1])
        w = jnp.asarray(np.linalg.solve(S, z), dtype=jnp.float32)
        p1 = fit_B(p0, obs[:200_000][lab], r[lab] / s, w, steps=a.steps)
        F1 = np.asarray(jax.jit(lambda x: B_apply(p1, x))(obs[:200_000]))
        e_x0, e_x1, e_run1 = eps_of(F0, r, seeds=2), eps_of(F1, r, seeds=2), eps_of(F1, run, seeds=2)
        zc = float(latent(F1, r) @ latent(F1, run))
        out[k] = (e_x0, e_x1, e_run0, e_run1, zc)
        print('%-12s %9.3f %9.3f | %9.3f %9.3f %+8.3f | %5.2f' % (k, e_x0, e_x1, e_run0, e_run1, e_run1 - e_run0, zc), flush=True)
    return out


def forgetting_xfirst(a, obs, R, tasks):
    """The other order: X is learned first (from the pretrained B), then run; how much of X survives?"""
    import jax, jax.numpy as jnp
    p0 = load_params(a.pre)
    n = 200_000
    ob = obs[:n]
    F0 = np.asarray(jax.jit(lambda x: B_apply(p0, x))(ob))
    rng = np.random.default_rng(0)
    lab = rng.choice(n, 10_000, replace=False)
    run = R['run'][:n]
    print('%-12s %9s %9s %9s | %9s %9s   (X learned first from the pretrained B, then run; B-only fits)' % ('task X', 'eps_X pre', 'eps_X(1)', 'eps_X(2)', 'eps_run1', 'eps_run2'))

    def readout(F, r):
        s_ = np.sqrt((r[lab] ** 2).mean()) + 1e-12
        z = (F[lab] * (r[lab] / s_)[:, None]).mean(0)
        S = F[:100_000].T @ F[:100_000] / 100_000 + 1e-5 * np.eye(F.shape[1])
        return jnp.asarray(np.linalg.solve(S, z), dtype=jnp.float32), s_
    for k in tasks:
        r = R[k][:n]
        w, s1 = readout(F0, r)
        p1 = fit_B(p0, ob[lab], r[lab] / s1, w, steps=a.steps)
        F1 = np.asarray(jax.jit(lambda x: B_apply(p1, x))(ob))
        w2, s2 = readout(F1, run)
        p2 = fit_B(p1, ob[lab], run[lab] / s2, w2, steps=a.steps)
        F2 = np.asarray(jax.jit(lambda x: B_apply(p2, x))(ob))
        print('%-12s %9.3f %9.3f %9.3f | %9.3f %9.3f' % (k, eps_of(F0, r, seeds=2), eps_of(F1, r, seeds=2), eps_of(F2, r, seeds=2),
                                                         eps_of(F1, run, seeds=2), eps_of(F2, run, seeds=2)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xfirst', nargs='*', default=None, help='tasks to learn first, then run (from the pretrained B)')
    ap.add_argument('--start', default=None, help='checkpoint of B fine-tuned on run: run the forgetting test from it')
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--pre', required=True)
    ap.add_argument('--ft', nargs='*', default=[])
    ap.add_argument('--sample', default=os.path.join(os.environ.get('CLAUDE_JOB_DIR', '.'), 'tmp', 'design', 'sample.npz'))
    ap.add_argument('--tasks', nargs='*', default=None)
    a = ap.parse_args()
    z = np.load(a.sample)
    obs, aux = z['obs'].astype(np.float32), z['aux']
    R = {'stand': builtin(aux, 0), 'walk': builtin(aux, 2), 'vel8': builtin(aux, 8), 'run': builtin(aux, 10), 'flip': builtin(aux, 'flip')}
    ax = aux_rewards(aux)
    for k, (desc, fn) in REWARDS.items():
        R[k] = ax[k] if fn is None else fn(obs)
    if a.tasks:
        R = {k: v for k, v in R.items() if k in a.tasks or k in ('run',)}
    Bs = {'pretrained': load_B(a.pre)}
    for p in a.ft:
        Bs[os.path.splitext(os.path.basename(p))[0]] = load_B(p)
    Fs = {k: features(B, obs) for k, B in Bs.items()}
    from sklearn.ensemble import HistGradientBoostingRegressor
    proxy = np.stack([obs[:, 14], obs[:, 0], obs[:, 16], obs[:, 17]], 1)       # height, upright, forward velocity, angular velocity
    if a.xfirst is not None:
        return forgetting_xfirst(a, obs, R, a.xfirst)
    if a.start:
        return forgetting(a, obs, R, [k for k in R if k != 'run'])
    run = R['run']
    print('%-12s %6s %6s %6s %6s | %s | %s' % ('task', 'supp%', 'corr', 'cos', 'R2run?', 'eps ' + ' / '.join(Fs), 'cos(z,z_run) pre'))
    rows = []
    for k, r in R.items():
        supp = 100 * float((r > 0.1 * r.max()).mean())
        corr = float(np.corrcoef(r, run)[0, 1])
        cos = float((r * run).mean() / np.sqrt((r ** 2).mean() * (run ** 2).mean()))
        idx = np.random.default_rng(1).permutation(len(r))
        tr, te = idx[:100_000], idx[100_000:200_000]
        m = HistGradientBoostingRegressor(max_iter=100).fit(proxy[tr], r[tr])
        r2 = 1 - ((m.predict(proxy[te]) - r[te]) ** 2).mean() / (r[te].var() + 1e-12)
        eps = [eps_of(F, r) for F in Fs.values()]
        zc = float(latent(Fs['pretrained'], r) @ latent(Fs['pretrained'], run))
        rows.append((k, supp, corr, cos, r2, eps, zc))
        print('%-12s %6.1f %6.2f %6.2f %6.2f | %s | %5.2f' % (k, supp, corr, cos, r2, ' / '.join('%.3f' % e for e in eps), zc), flush=True)
    return rows


if __name__ == '__main__':
    main()
