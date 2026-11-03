"""Diagnostics E (behaviour) and F (consistency of F with B) of a fine-tuned FB checkpoint against the pretrained one.

Called from fb_continual_temp.py (eval-only mode):
    python fb_continual_temp.py env_name=walker checkpoint=<pretrained pickle> eval_ckpt=<theta_1 pickle, e.g. models/multi/<run>/fb_multi.pickle>
        tasks=stand,run,flip,supine,jump_any,kick_l,split,flamingo_r bench=final n_labeled=100000 num_evals=20 diag=E|F diag_seed=<seed of the run>
theta_0 = checkpoint= (pretrained), theta_1 = eval_ckpt= (F, actor and B after stage 1 + stage 2; B is the B of stage 1).
Every state used to infer a latent or to fit a reward is disjoint from the fine-tuning labeled set D_k of the run (rebuilt from diag_seed:
valid_idx[default_rng(seed*1000+1).choice(len(valid_idx), n_labeled)]).

E  (zero-shot returns, num_evals episodes each, z of every episode inferred from its own fresh 10k labels, the same labels for B_0 and B_1)
   baseline_z0      theta_0 (F, actor) with z from B_0                        (pretrained zero-shot)
   baseline_z1      theta_1 with z' from B_1                                  (the fine-tuned model, as in the main table)
   free_a_raw       theta_0's actor fed z'                                    (raw)
   free_b_plain     ... fed Q_plain^T z', Q_plain = polar factor of E[B_1 B_0^T] (keeps ||B|| = sqrt d)
   free_c_white     ... fed S_0^{1/2} Q^T S_1^{-1/2} z', Q = polar factor of the whitened cross-covariance (whitened-coordinate Procrustes mapped back)
   free_d_backmap   ... fed E[B_0 B_1^T] S_1^{-1} z'                          (least-squares map from B_1 to B_0)
   old_z0           theta_1's actor fed the OLD latent z (from B_0)
   For every row the centered fit error eps_c of the latent actually fed: held-out reward vs B^T Sigma^-1 z_fed (B and Sigma of the model whose policy gets it),
   with a free offset, divided by the variance of the reward; eps_unc is the uncentered one (second moment).
   Sensitivity: the latent of theta_0 (z) and theta_1 (z') rotated by a random angle (5, 10, 20, 35, 60 deg; an independent random direction per episode).
   Bootstrap: theta_1 on run with z' inferred from 10 bootstrap resamples of one set of 10k labels (num_evals episodes each).
F  (consistency)
   FB TD loss (L_FB of the training step; target = the same online nets, deterministic next action) of the four combinations (F, pi, B):
   (F0,pi0,B0), (F0,pi0,B1), (F1,pi1,B1), (F1,pi1,B0), on uniform latents and on goal latents (z = B_F(s), B_F the B of the model whose F is used).
   FB loss of (F0,pi0,B0) and (F1,pi1,B1) against the angle (0, 15, 30, 60, 90 deg) between z and the latent of run.
   Value accuracy: min_h F_h(s_t, a_t, zbar)^T z against the Monte Carlo discounted return sum_k gamma^k rbar(s_{t+1+k}) along rollouts of the policy pi(., zbar),
   t in 0, 100, ..., 800; for every task and both models.
Outputs in the log folder: E_returns.csv, E_alignment.csv, E_sensitivity.csv, E_bootstrap.csv / F_fb_loss.csv, F_fb_loss_angle.csv, F_value.csv.
"""
import csv, math, os
import numpy as np


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def sym_pow(S, p):
    w, v = np.linalg.eigh(S)
    return (v * w ** p) @ v.T


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def build_maps(B0, B1):
    """Maps z' (latent of B_1) -> a latent in the coordinates of B_0. B0, B1: [n, d] features on the same states."""
    n, d = B0.shape
    S0, S1 = B0.T @ B0 / n, B1.T @ B1 / n
    C = B1.T @ B0 / n                                               # E[B1 B0^T]
    U, _, Vt = np.linalg.svd(C)
    Qp = U @ Vt                                                     # B1 ~ Qp B0   ->  z' ~ Qp z
    W0, W1 = sym_pow(S0, -0.5), sym_pow(S1, -0.5)
    U2, _, Vt2 = np.linalg.svd(W1 @ C @ W0)
    Qw = U2 @ Vt2                                                   # whitened coordinates
    maps = {'free_a_raw': np.eye(d),
            'free_b_plain': Qp.T,
            'free_c_white': sym_pow(S0, 0.5) @ Qw.T @ W1,
            'free_d_backmap': C.T @ np.linalg.inv(S1)}
    return maps, S0, S1


def free_indices(rb, seed, n_labeled):
    valid = np.asarray(rb.valid_idx)
    dk = valid[np.random.default_rng(seed * 1000 + 1).choice(len(valid), n_labeled, replace=False)]
    mask = np.ones(rb.obs.shape[0], bool)
    mask[dk] = False
    return valid[mask[valid]]


def rotate(zunit, angle_deg, rng):
    e = rng.normal(size=zunit.shape)
    e = unit(e - (e * zunit).sum(-1, keepdims=True) * zunit)
    a = math.radians(angle_deg)
    return math.cos(a) * zunit + math.sin(a) * e


def eps_pair(rb, rec):
    e = rb - rec
    e = e - e.mean()
    return float(np.mean(e ** 2) / (np.var(rb) + 1e-12)), float(np.mean((rb - rec) ** 2) / (np.mean(rb ** 2) + 1e-12))


class Ctx:
    def __init__(self, d):
        self.__dict__.update(d)
        self.d = self.Z_DIM
        self.rng = np.random.default_rng(self.diag_seed + 12345)
        self.free = free_indices(self.replay_buffer, self.diag_seed, self.N_LABELED)
        self.p0, self.p1 = self.ts0.fb_params, self.ts1.fb_params
        self.out = self.LOGS_FOLDER

    def states(self, n):
        idx = self.free[self.rng.integers(0, len(self.free), n)]
        return self.replay_buffer.obs[idx], self.replay_buffer.aux[idx]

    def reward(self, b, aux):
        reward_fn, label, param = self.benchmarks[b]
        return reward_fn(aux, param)

    def feats(self, params, obs):
        return np.asarray(self.big_backward(params['backward'], obs), dtype=np.float64)

    def writer(self, name, header):
        f = open(os.path.join(self.out, name), 'w', newline='')
        w = csv.writer(f)
        w.writerow(header)
        return w, f


def run_E(c):
    import jax.numpy as jnp
    d, N, ANG = c.d, c.NUM_EVALS, [5, 10, 20, 35, 60]
    obs_rho, _ = c.states(200_000)
    B0r, B1r = c.feats(c.p0, obs_rho), c.feats(c.p1, obs_rho)
    maps, S0, S1 = build_maps(B0r, B1r)
    S0i, S1i = np.linalg.inv(S0), np.linalg.inv(S1)
    obs_h, aux_h = c.states(100_000)
    B0h, B1h = c.feats(c.p0, obs_h), c.feats(c.p1, obs_h)
    print('E: maps built; held-out features done', flush=True)
    items0, items1, meta, sens_meta = [], [], {}, {}
    wA, fA = c.writer('E_alignment.csv', ['task', 'cos_raw_z0_z1', 'cos_plain_procrustes', 'cos_white_procrustes', 'cos_backmap'])
    zmean = {}
    for b in c.EVAL_BENCH:
        label = c.LABELS[b]
        Z0, Z1 = [], []
        for i in range(N):
            obs, aux = c.states(c.EVAL_N)
            r = c.reward(b, aux)
            z0, m2 = c.encode_reward(c.p0['backward'], obs, r)
            z1, _ = c.encode_reward(c.p1['backward'], obs, r)
            Z0.append(np.asarray(z0, np.float64)); Z1.append(np.asarray(z1, np.float64))
        Z0, Z1 = np.stack(Z0), np.stack(Z1)
        V = {k: Z1 @ M.T for k, M in maps.items()}
        zm0, zm1 = Z0.mean(0), Z1.mean(0)
        wA.writerow([label, '%.4f' % cos(zm0, zm1)] + ['%.4f' % cos(zm0, V[k].mean(0)) for k in ('free_b_plain', 'free_c_white', 'free_d_backmap')])
        rh = np.asarray(c.reward(b, aux_h), np.float64)
        rb = rh / math.sqrt(max(float(m2), 1e-12))                   # same scale as rbar of the latents
        zb = lambda Z: jnp.asarray(math.sqrt(d) * unit(Z), dtype=jnp.float32)
        # rows: (model, variant, zs, representation (B, Sigma^-1, mean latent))
        rows = [('th0', 'baseline_z0', Z0, (B0h, S0i)), ('th1', 'baseline_z1', Z1, (B1h, S1i)), ('th1', 'old_z0', Z0, (B1h, S1i))]
        rows += [('th0', k, V[k], (B0h, S0i)) for k in maps]
        for model, variant, Z, (Bh, Si) in rows:
            e_c, e_u = eps_pair(rb, Bh @ (Si @ Z.mean(0)))
            key = ('E', model, variant, label)
            meta[key] = dict(model=model, variant=variant, task=label, eps_c=e_c, eps_unc=e_u, cos_z0=cos(Z.mean(0), zm0))
            (items0 if model == 'th0' else items1).append((key, b, zb(Z)))
        for model, Z, ts in (('th0', Z0, c.ts0), ('th1', Z1, c.ts1)):
            for ang in ANG:
                key = ('S', model, ang, label)
                zs = jnp.asarray(math.sqrt(d) * rotate(unit(Z), ang, c.rng), dtype=jnp.float32)
                sens_meta[key] = (model, label, ang)
                (items0 if model == 'th0' else items1).append((key, b, zs))
    fA.close()
    # bootstrap of the labels for theta_1 on run
    boot = []
    if 'run' in c.LABELS:
        b = c.LABELS.index('run')
        obs, aux = c.states(c.EVAL_N)
        r = np.asarray(c.reward(b, aux), np.float64)
        rbar = r / math.sqrt(float(np.mean(r ** 2)) + 1e-12)
        B1l = c.feats(c.p1, obs)
        for j in range(10):
            ix = c.rng.integers(0, len(r), len(r))
            z = (B1l[ix] * rbar[ix, None]).mean(0)
            key = ('B', 'th1', j, 'run')
            boot.append(key)
            items1.append((key, b, jnp.tile(jnp.asarray(math.sqrt(d) * unit(z), dtype=jnp.float32)[None, :], (N, 1))))
    print('E: %d items for theta_0, %d for theta_1' % (len(items0), len(items1)), flush=True)
    ret = {}
    ret.update(c.rollout_returns(c.ts0.actor_params, items0))
    ret.update(c.rollout_returns(c.ts1.actor_params, items1))
    w, f = c.writer('E_returns.csv', ['model', 'variant', 'task', 'mean', 'std', 'eps_centered', 'eps_uncentered', 'cos_to_z0'])
    for key, m in meta.items():
        mu, sd = ret[key]
        w.writerow([m['model'], m['variant'], m['task'], '%.2f' % mu, '%.2f' % sd, '%.4f' % m['eps_c'], '%.4f' % m['eps_unc'], '%.4f' % m['cos_z0']])
        print('E %-4s %-16s %-11s return %7.1f +- %5.1f | eps_c %.3f eps_unc %.3f | cos to z0 %.3f' % (m['model'], m['variant'], m['task'], mu, sd, m['eps_c'], m['eps_unc'], m['cos_z0']), flush=True)
    f.close()
    w, f = c.writer('E_sensitivity.csv', ['model', 'task', 'angle_deg', 'mean', 'std'])
    for key, (model, label, ang) in sens_meta.items():
        w.writerow([model, label, ang, '%.2f' % ret[key][0], '%.2f' % ret[key][1]])
        print('E sens %-4s %-11s %2d deg return %7.1f +- %5.1f' % (model, label, ang, ret[key][0], ret[key][1]), flush=True)
    f.close()
    w, f = c.writer('E_bootstrap.csv', ['bootstrap', 'mean', 'std'])
    for key in boot:
        w.writerow([key[2], '%.2f' % ret[key][0], '%.2f' % ret[key][1]])
    if boot:
        ms = np.array([ret[k][0] for k in boot])
        print('E bootstrap (theta_1, run): mean of the 10 means %.1f, std across bootstrap latents %.1f, min %.1f max %.1f' % (ms.mean(), ms.std(), ms.min(), ms.max()), flush=True)
    f.close()


def run_F(c):
    import jax
    import jax.numpy as jnp
    d, agent, gamma = c.d, c.agent, c.DISCOUNT
    bs, n_batches = 1024, 40
    off = 1.0 - jnp.eye(bs)
    n_off = off.sum()

    @jax.jit
    def fb_terms(fwd, bwd, actor, obs, action, next_obs, z):
        F1, F2 = agent.forward_net.apply({'params': fwd}, obs, z, action)
        B = agent.backward_net.apply({'params': bwd}, next_obs)
        next_a = agent.actor_net.apply({'params': actor}, next_obs, z)
        t1, t2 = agent.forward_net.apply({'params': fwd}, next_obs, z, next_a)
        target = jnp.minimum(t1 @ B.T, t2 @ B.T)
        M1, M2 = F1 @ B.T, F2 @ B.T
        offd = 0.5 * sum((((M - gamma * target) ** 2) * off).sum() / n_off for M in (M1, M2))
        diag = -sum(jnp.diag(M).mean() for M in (M1, M2))
        return offd, diag

    P = {0: (c.ts0.fb_params, c.ts0.actor_params), 1: (c.ts1.fb_params, c.ts1.actor_params)}
    combos = [('F0pi0B0', 0, 0), ('F0pi0B1', 0, 1), ('F1pi1B1', 1, 1), ('F1pi1B0', 1, 0)]     # (name, model of F and pi, model of B)
    w, f = c.writer('F_fb_loss.csv', ['combo', 'latents', 'fb_loss', 'fb_offdiag', 'fb_diag'])
    for name, mf, mb in combos:
        for lat in ('uniform', 'goal'):
            tot = np.zeros(2)
            for i in range(n_batches):
                k = jax.random.PRNGKey(1000 + i)
                obs, action, next_obs, _ = c.replay_buffer.sample(k, bs)
                if lat == 'uniform':
                    z = agent.sample_z(jax.random.fold_in(k, 1), bs)
                else:
                    z = agent.backward_net.apply({'params': P[mf][0]['backward']}, obs[jax.random.permutation(jax.random.fold_in(k, 2), bs)])
                o, dg = fb_terms(P[mf][0]['forward'], P[mb][0]['backward'], P[mf][1], obs, action, next_obs, z)
                tot += np.array([float(o), float(dg)])
            tot /= n_batches
            w.writerow([name, lat, '%.4f' % tot.sum(), '%.4f' % tot[0], '%.4f' % tot[1]])
            print('F fb loss %-8s %-8s total %.4f (offdiag %.4f diag %.4f)' % (name, lat, tot.sum(), tot[0], tot[1]), flush=True)
    f.close()

    # latent of the new task (run) for each model, and the loss against the angle
    b_run = c.LABELS.index('run')
    zrun = {}
    obs, aux = c.states(100_000)
    r = c.reward(b_run, aux)
    for m in (0, 1):
        z, m2 = c.encode_reward(P[m][0]['backward'], obs, r)
        zrun[m] = unit(np.asarray(z, np.float64))
    w, f = c.writer('F_fb_loss_angle.csv', ['model', 'angle_deg', 'fb_loss', 'fb_offdiag', 'fb_diag'])
    for m in (0, 1):
        for ang in (0, 15, 30, 60, 90):
            tot = np.zeros(2)
            for i in range(n_batches):
                k = jax.random.PRNGKey(2000 + i)
                obs_b, action, next_obs, _ = c.replay_buffer.sample(k, bs)
                zu = np.tile(zrun[m][None, :], (bs, 1))
                z = jnp.asarray(math.sqrt(d) * rotate(zu, ang, c.rng), dtype=jnp.float32)
                o, dg = fb_terms(P[m][0]['forward'], P[m][0]['backward'], P[m][1], obs_b, action, next_obs, z)
                tot += np.array([float(o), float(dg)])
            tot /= n_batches
            w.writerow(['th%d' % m, ang, '%.4f' % tot.sum(), '%.4f' % tot[0], '%.4f' % tot[1]])
            print('F angle th%d %2d deg: fb loss %.4f (offdiag %.4f diag %.4f)' % (m, ang, tot.sum(), tot[0], tot[1]), flush=True)
    f.close()

    # value accuracy against Monte Carlo returns
    T, tgrid = c.EVAL_STEPS, list(range(0, 801, 100))

    @jax.jit
    def q_pred(fwd, obs, action, zbar, z):
        F1, F2 = agent.forward_net.apply({'params': fwd}, obs, zbar, action)
        return jnp.minimum((F1 * z).sum(-1), (F2 * z).sum(-1))

    def rollout_data(actor, zs):
        n = zs.shape[0]
        while len(c.eval_envs) < n:
            c.eval_envs.append(c.make_eval_env())
        envs = c.eval_envs[:n]
        states = [c.env_reset(e) for e in envs]
        O, A, X = [], [], []
        for _ in range(T):
            obs = np.stack(states).astype(np.float32)
            X.append(np.stack([c.env_aux(e, s) for e, s in zip(envs, states)]))
            act = np.asarray(agent.act(actor, jnp.asarray(obs), zs))
            O.append(obs); A.append(act)
            states = [c.env_step(e, a) for e, a in zip(envs, act)]
        return np.stack(O, 1), np.stack(A, 1), np.stack(X, 1)

    w, f = c.writer('F_value.csv', ['model', 'task', 'q_pred_mean', 'mc_mean', 'bias', 'rmse', 'corr', 'slope_mc_on_pred', 'n_points'])
    for m in (0, 1):
        for b in c.EVAL_BENCH:
            label = c.LABELS[b]
            obs_l, aux_l = c.states(100_000)
            r = c.reward(b, aux_l)
            z, m2 = c.encode_reward(P[m][0]['backward'], obs_l, r)
            z = jnp.asarray(z, jnp.float32)
            zbar = math.sqrt(d) * z / (jnp.linalg.norm(z) + 1e-12)
            zs = jnp.tile(zbar[None, :], (c.NUM_EVALS, 1))
            O, A, X = rollout_data(P[m][1], zs)
            rbar = np.asarray(c.reward(b, jnp.asarray(X, jnp.float32)), np.float64) / math.sqrt(float(m2) + 1e-12)      # [n, T]
            G = np.zeros_like(rbar)
            for t in range(T - 2, -1, -1):
                G[:, t] = rbar[:, t + 1] + gamma * G[:, t + 1]
            pred = []
            for t in tgrid:
                pred.append(np.asarray(q_pred(P[m][0]['forward'], jnp.asarray(O[:, t]), jnp.asarray(A[:, t]), zs, jnp.tile(z[None, :], (c.NUM_EVALS, 1)))))
            pred = np.stack(pred, 1).reshape(-1)
            mc = np.stack([G[:, t] for t in tgrid], 1).reshape(-1)
            slope = float((pred * mc).sum() / ((pred ** 2).sum() + 1e-12))
            row = [pred.mean(), mc.mean(), (pred - mc).mean(), np.sqrt(((pred - mc) ** 2).mean()), np.corrcoef(pred, mc)[0, 1], slope, len(pred)]
            w.writerow(['th%d' % m, label] + [('%.4f' % v if i < 6 else int(v)) for i, v in enumerate(row)])
            print('F value th%d %-11s pred %.3f mc %.3f bias %+.3f rmse %.3f corr %.3f slope %.2f' % ((m, label) + tuple(row[:6])), flush=True)
    f.close()


def run(which, c):
    os.makedirs(c['LOGS_FOLDER'], exist_ok=True)
    c = Ctx(c)
    print('diag %s: theta_0 = %s, theta_1 = %s, seed %d, free states %d' % (which, c.CHECKPOINT, c.EVAL_CKPT, c.diag_seed, len(c.free)), flush=True)
    if which == 'E':
        run_E(c)
    elif which == 'F':
        run_F(c)
    else:
        raise ValueError('diag= must be E or F')
