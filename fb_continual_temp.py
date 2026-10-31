"""
TEMPORARY COPY of fb_continual.py, to find which differences between L_rec (continual) and fit_backward_to_reward
(finetune=reward_critic, phase 1) matter for how well B regresses onto the reward. Everything is off by default, so with
no new flag this script behaves like fb_continual.py. New flags, all about the L_rec path:
    rec_fixed_w=0     1: L_rec uses the FIXED readout w_fit = Sigma_old^-1 z_run of the stage (computed once from B at the start
                         of the stage) instead of re-solving w = sg(Sigma_theta^-1 z_hat) at every step. Loss = mean((rbar - B w_fit)^2).
    rec_bs=0          >0: L_rec on a random minibatch of D_k of this size at every step (0 = the whole labeled set, as before).
    lr_coef=1.0       lr of B = lr_coef * lr (already in the config, now a flag). E3 phase 1 uses 3e-4 = 3 * lr.
    b_cosine=0        1: cosine decay of the lr of B over steps_per_stage (Adam is re-initialised at every stage). E3 phase 1 does this.
  Existing flags that also matter for this question: lam, n_est, n_labeled, cross_fit, and fb_objective=0 ortho_coef=0
  (then B receives gradient from L_rec only: the Q loss uses a stop-gradient B).
  New eval columns (continual mode): <task>_fit_{eps,corr} = held-out quality of B(s)^T w_fit with the FIXED readout of the
  stage, the quantity fit_backward_to_reward reports as [reward fit] eps/corr. <task>_fresh_eps is the quality with a re-solved readout.

Continual offline reward encoding in a forward-backward (FB) model at fixed rank.

The FB model is NOT pretrained here: it is loaded from a checkpoint written by the FB pretraining script
(fb_agent_*.pickle with 'fb_params' and 'actor_params'; 'fb_target_params' / 'g_params' are ignored).
Always classical FB (actor in the bootstrap) + DDPG actor (optional TD3+BC term via bc_coef), full dataset.

Usage:
    python fb_continual.py env_name=walker checkpoint=path/to/fb_agent_latest.pickle tasks=walk,run
    python fb_continual.py env_name=walker checkpoint=... tasks=[walk,run,vel8] lam=1 mu_b=1 mu_q=0.01 alpha=0.5

Flags:
    env_name=      cheetah | walker | quadruped | antmaze | pointmaze           (required)
    checkpoint=    path of the pretrained FB pickle                              (required)
    tasks=         ordered list of eval-task labels of the env (>= 1, unique): the continual stream.
                   labels: cheetah: vel10Back vel2Back vel2 vel10 flip | walker: stand walk vel8 run flip
                           quadruped: stand walk run jump | pointmaze: goal_0_20 goal_20_20 goal_20_0 goal_4_4 goal_12_12
                           antmaze: goal_bottom ... (see `benchmarks` below)
    seed=<int>               default: random (logged in info.txt)
    discount=0.95
    steps_per_stage=100000   number of updates per stage (multiple of scan_length)
    scan_length=1000         updates per jitted chunk (= logging granularity)
    n_labeled=10000          size of the labeled set D_k of each task (sampled once per stage, fixed during the stage)
    lam=1.0 mu_b=1.0 mu_q=0.01 alpha=0.5     lambda, mu_B, mu_Q, mixture weight of the atoms in nu
    z_task_frac=0.0          fraction of the z of the FB loss and of the actor loss replaced by the latent of the CURRENT task
                             (z_k = mean(rbar_k B) re-encoded at every step with the current B on the stage's labeled set)
    z_task_spread=0.5        angular noise around that latent (0 = exactly the latent; 1 = 45 degrees)
    actor_z_task_frac=-1     fraction of the ACTOR loss's z that is the task latent, independent of the FB loss (-1 = same as z_task_frac)
    freeze_fb=0              1 = diagnostic: F, B (and their Adam state) are not updated, only the actor trains
    freeze_b=0               1 = diagnostic: B is not updated; F and the actor are still trained
    n_est=4096               number of states of D used at every step to estimate Sigma_theta and A_theta
    n_big=50000              number of states used once per stage for Sigma_old, and at the end of the stage for T_theta
    ridge=1e-5               ridge added to every covariance before inversion
    cross_fit=0              1: L_rec with cross-fitting (z_hat on one half of D_k, loss on the other)
    bc_coef=0.0              TD3+BC weight of the DDPG actor (0 -> plain -Q.mean())
    FB hyper-parameters, they MUST be the ones of the pretraining run: lr=1e-4 fb_target_tau=0.01 ortho_coef=1.0
                             q_loss_coef=0.01 mix_ratio=0.5 discount=0.95 target_agg=min|mean (min = original FB)
    eval_chunk=              number of tasks rolled out in parallel (each needs NUM_EVALS env instances)
    eval_each_steps=0        if > 0: also evaluate (fresh zero-shot only) every this many chunks inside a stage
    eval_baseline=1          evaluate the pretrained checkpoint (stage 0) before the first stage
    eval_fresh=1             0 = skip the fresh zero-shot evaluation of every env task (much faster evals)
    fb_objective=1           1: critic loss = L_FB + q_loss_coef * Q loss (+ ortho_coef * ortho)
                             0: critic loss = Q loss only (+ ortho_coef * ortho). L_FB is not computed (branch skipped).
                             ortho_coef=0 skips the orthonormality loss; lam=0 / mu_b=0 / mu_q=0 skip L_rec / R^B / R^Q.

Single-task fine-tuning (finetune=, default none = the continual method above). In every fine-tuning mode, the z of the
critic, of the bootstrap and of the actor is the FIXED latent of the task, z_cond = sqrt(d) * normalize(z_k),
z_k = mean(rbar_k B) with B at the start of the stage. L_FB, ortho, L_rec, R^B, R^Q are NOT computed.
    finetune=actor          F and B frozen, only the actor is trained (maximize F(s, pi(s, z), z)^T z)
    finetune=critic         B frozen. F trained with the Q loss only, reward r(s') = B(s')^T w_q with the fixed readout
                            w_q = Sigma_old^-1 z_cond (Sigma_old on n_big states at the start of the stage). Actor trained.
    finetune=reward_critic  phase 1 (before the stage's RL steps): B alone is regressed on the labeled set with the readout
                            FIXED, min mean (B(s)^T w_fit - rbar(s))^2, w_fit = Sigma_old^-1 z_k (the fixed-z experiment,
                            warm-started). phase 2: same as finetune=critic with the new B (B frozen).
        rfit_steps=30000 rfit_lr=3e-4 rfit_bs=4096   (phase 1, cosine lr decay)
    reinit_critic=0         1 (finetune=critic|reward_critic only): F and the actor are re-initialised at random at the start of
                            the RL steps (after the reward fit, B is kept), instead of starting from the pretrained ones.
                            Fresh Adam states and target network. Diagnostic: warm start vs. TD3 from scratch.
    Evaluation: '<task>_ft_{mean,std,rew_corr,eps}' = rollouts with z_cond, and the reconstruction quality of the reward the
    critic is trained on (B(s)^T w_fit vs rbar, held-out states). An eval tagged 'ft_start' is run before the RL steps.
    Example:
        python fb_continual.py env_name=walker checkpoint=... tasks=walk n_labeled=100000 finetune=critic \
            steps_per_stage=1000000 eval_each_steps=10 eval_baseline=0 eval_fresh=0

Stage k (see the method): theta_old = theta_{k-1} (frozen), minimise
    L_k = L_FB + lam * L_rec(r_k) + mu_b * R^B + mu_q * R^Q.
  L_rec : detached-coefficient regression of the normalized reward on B_theta over the labeled set.
  R^B   : closed form E_{z~nu}[ z^T M z / z^T Sigma_old^-1 z ],  M = Sigma_old^-1 - T^T Sigma_theta^-1 T,  T = A Sigma_old^-1.
  R^Q   : value matching between old and new model on the re-encoded old family r_z (z' = sg(T z)).
          Both heads of F are matched head-to-head: (F_theta^i(s,a,zbar')^T z' - F_old^i(s,a,z)^T z)^2, averaged over i=1,2.
  nu    : (1 - alpha) * uniform sphere + alpha * uniform mixture of the stored atoms zbar_j (j < k).
End of stage: z_j <- T z_j for j < k (T between theta_k and theta_{k-1}, on a fresh batch of n_big states),
              z_k = mean_i rbar_k(s_i) B_{theta_k}(s_i).  Only d numbers per task are stored.

Logs (./logs/<run>/):
    training_log.csv : per chunk, FB metrics + new_reward_unexplained_fraction_training, old_rewards_lost_fraction,
                       old_rewards_value_change, old_latent_length_ratio (= mean ||T z|| / ||z||), total_loss
    eval_log.csv     : per evaluation, for every env task '<label>_fresh_{mean,std,rew_corr,eps}' (z inferred from fresh
                       10k labels, as in the pretraining script) and, for the tasks of the stream seen so far,
                       '<label>_tracked_{mean,std,rew_corr,eps}' (z = stored latent z_j carried through the T's).
                       eps = fraction of the reward energy NOT explained by B on held-out states (the quantity L_rec / R^B estimate).
                       The tracked metrics use the scalar m2_j (second moment of r_j on D_j) to put the reward on the scale of rbar_j:
                       it is only used for evaluation, never in training.
"""
import os
LOCAL = 'nazim' in os.getcwd()
DATASET_FOLDER = '../offline_qd/datasets/d4rl/'
if not LOCAL:
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    DATASET_FOLDER = '../fre_experiments/datasets/'

import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm

import csv
import sys, shutil
from datetime import datetime

import math
import time
import pickle

import dataclasses
from functools import partial
from typing import Any, Optional

import jax
import flax
from flax import serialization
import jax.numpy as jnp
import optax

from _utils.logs import parse_kwargs, resolve_seed, run_info


class CSVLogger:
    def __init__(self, filepath, fieldnames):
        self.filepath = filepath
        self.fieldnames = fieldnames
        # Write header only if file doesn't exist
        if not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def log(self, row: dict):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def save_self_copy(dst_dir="backups"):
    os.makedirs(dst_dir, exist_ok=True)
    src = os.path.abspath(__file__)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy(src, dst)
    return dst


def parse_task_list(value):
    """tasks=walk,run  |  tasks=[walk,run]  |  already a list/tuple -> list of str."""
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        items = str(value).strip().strip('[]()').split(',')
    items = [t.strip().strip('\'"') for t in items]
    return [t for t in items if t]


# ── Pure helpers of the method (jnp only) ────────────────────────────────────────────────────

def normalize(x, eps=1e-12):
    """torch F.normalize(x, dim=-1)."""
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def reconstruction_loss(B_lab, r_lab, Sigma_hat, cross_fit=False):
    """L_rec(theta; r) = mean_i (rbar_i - B(s_i)^T sg[Sigma_hat^-1 z_hat])^2, z_hat = mean_i rbar_i B(s_i).
    The stop-gradient covers the whole coefficient vector (detached coefficient); gradients flow only through the
    B(s_i) of the prediction. cross_fit: z_hat on one half of the labeled set, loss on the other half (and vice versa)."""
    def fit_eval(B_fit, r_fit, B_ev, r_ev):
        z_hat = jnp.mean(r_fit[:, None] * B_fit, axis=0)
        w = jax.lax.stop_gradient(jnp.linalg.solve(Sigma_hat, z_hat))
        return jnp.mean((r_ev - B_ev @ w) ** 2)

    if cross_fit:
        h = B_lab.shape[0] // 2
        return 0.5 * (fit_eval(B_lab[:h], r_lab[:h], B_lab[h:], r_lab[h:])
                      + fit_eval(B_lab[h:], r_lab[h:], B_lab[:h], r_lab[:h]))
    return fit_eval(B_lab, r_lab, B_lab, r_lab)


def old_family_matrices(B_new, B_old, sigma_old_inv, ridge):
    """From B_new, B_old evaluated on the same n states of D (rows = states):
       Sigma_theta = B_new^T B_new / n (+ ridge I),  A = B_new^T B_old / n,  T = A Sigma_old^-1,
       M = Sigma_old^-1 - T^T Sigma_theta^-1 T.   Returns (Sigma_theta, T, M)."""
    n, d = B_new.shape
    Sigma_th = B_new.T @ B_new / n + ridge * jnp.eye(d)
    A = B_new.T @ B_old / n
    T = A @ sigma_old_inv
    M = sigma_old_inv - T.T @ jnp.linalg.solve(Sigma_th, T)
    M = 0.5 * (M + M.T)
    return Sigma_th, T, M


def retention_basis_loss(z, M, sigma_old_inv):
    """R^B = mean_z [ z^T M z / z^T Sigma_old^-1 z ]  (z: [batch, d])."""
    num = jnp.einsum('bi,ij,bj->b', z, M, z)
    den = jnp.einsum('bi,ij,bj->b', z, sigma_old_inv, z)
    return jnp.mean(num / den)


if __name__ == '__main__':
    #######################################################################################

    kwargs = parse_kwargs(sys.argv[1:])
    print(kwargs)
    print(jax.devices())

    assert 'env_name' in kwargs, "env_name= is required"
    ENV_NAME = kwargs['env_name']
    assert ENV_NAME in ['cheetah', 'walker', 'quadruped', 'antmaze', 'pointmaze']

    assert 'checkpoint' in kwargs, "checkpoint= is required (pretrained FB pickle)"
    CHECKPOINT = str(kwargs['checkpoint'])

    assert 'tasks' in kwargs, "tasks= is required (ordered list of eval tasks of the env, at least one)"
    TASKS = parse_task_list(kwargs['tasks'])
    assert len(TASKS) >= 1, "tasks= must contain at least one task"
    assert len(set(TASKS)) == len(TASKS), f"tasks= must be unique, got {TASKS}"

    SEED = resolve_seed(kwargs)

    # ── Continual hyper-parameters ────────────────────────────────────────────────
    STEPS_PER_STAGE = int(kwargs['steps_per_stage']) if 'steps_per_stage' in kwargs else 100_000
    SCAN_LENGTH     = int(kwargs['scan_length']) if 'scan_length' in kwargs else 1_000
    assert STEPS_PER_STAGE % SCAN_LENGTH == 0, "steps_per_stage must be a multiple of scan_length"
    CHUNKS_PER_STAGE = STEPS_PER_STAGE // SCAN_LENGTH
    N_LABELED       = int(kwargs['n_labeled']) if 'n_labeled' in kwargs else 10_000
    N_BIG           = int(kwargs['n_big']) if 'n_big' in kwargs else 50_000
    EVAL_EACH_STEPS = int(kwargs['eval_each_steps']) if 'eval_each_steps' in kwargs else 0   # in chunks; 0 = only at the end of a stage
    EVAL_BASELINE   = bool(int(kwargs['eval_baseline'])) if 'eval_baseline' in kwargs else True
    EVAL_N          = 10_000      # number of states for the eval-time reward / reconstruction statistics
    EVAL_FRESH      = bool(int(kwargs['eval_fresh'])) if 'eval_fresh' in kwargs else True

    # ── Single-task fine-tuning ───────────────────────────────────────────────────
    FINETUNE   = str(kwargs['finetune']) if 'finetune' in kwargs else 'none'
    assert FINETUNE in ['none', 'actor', 'critic', 'reward_critic'], \
        "finetune= must be none, actor, critic or reward_critic"
    RFIT_STEPS = int(kwargs['rfit_steps']) if 'rfit_steps' in kwargs else 30_000
    RFIT_LR    = float(kwargs['rfit_lr']) if 'rfit_lr' in kwargs else 3e-4
    RFIT_BS    = int(kwargs['rfit_bs']) if 'rfit_bs' in kwargs else 4096
    RFIT_CHUNK = min(1000, RFIT_STEPS)
    REINIT_CRITIC = bool(int(kwargs['reinit_critic'])) if 'reinit_critic' in kwargs else False
    assert not REINIT_CRITIC or FINETUNE in ('critic', 'reward_critic'), "reinit_critic=1 needs finetune=critic or reward_critic"
    assert RFIT_STEPS > 0 and RFIT_STEPS % RFIT_CHUNK == 0, "rfit_steps must be a multiple of 1000 (or < 1000)"

    # ── Logging ───────────────────────────────────────────────────────────────────
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    if FINETUNE == 'none':
        exp_name = f'FBcontinual-{ENV_NAME}-' + '+'.join(TASKS)
    else:
        exp_name = f'FBfinetune-{FINETUNE}-{ENV_NAME}-' + '+'.join(TASKS)
        if 'n_labeled' in kwargs:
            exp_name += f'-nl{N_LABELED}'
    base_name = f"{date_time_str}_{exp_name}"

    suffix = 0
    while True:
        name = base_name if suffix == 0 else f"{base_name}_{suffix}"
        LOGS_FOLDER = f"./logs/{name}"
        try:
            os.makedirs(LOGS_FOLDER)      # atomic: fails if it already exists
            break
        except FileExistsError:
            suffix += 1

    MODEL_SAVE_FOLDER = f"./models/{name}"
    os.makedirs(MODEL_SAVE_FOLDER)
    print('LOGS_FOLDER      :', LOGS_FOLDER)
    print('MODEL_SAVE_FOLDER:', MODEL_SAVE_FOLDER)

    save_self_copy(dst_dir=LOGS_FOLDER)

    TRAIN_METRICS = ["fb_loss", "fb_diag", "fb_offdiag", "q_loss",
                     "orth_loss", "orth_loss_diag", "orth_loss_offdiag", "orth_linf", "orth_l2",
                     "target_M", "M1", "F1", "B", "B_norm", "z_norm", "next_Q", "q_reward",
                     "b_cov_eig_min", "b_cov_eig_max",
                     "actor_loss", "q", "bc_dist",
                     "total_loss", "new_reward_unexplained_fraction_training", "old_rewards_lost_fraction",
                     "old_rewards_value_change", "old_latent_length_ratio"]
    train_logger = CSVLogger(
        filepath=f"{LOGS_FOLDER}/training_log.csv",
        fieldnames=["stage", "step"] + TRAIN_METRICS + ["time"],
    )

    with open(f'{LOGS_FOLDER}/info.txt', 'a') as file:
        file.write(str(kwargs) + '\n')
        file.write(run_info())

    # ── Reward functions (jnp ports of new_utils/benchmark.py and utils/antmaze_benchmark.py) ──
    # Every reward function has the signature  fn(aux, param) -> array of shape aux.shape[:-1]

    INF = float('inf')

    def _sigmoid(x, value_at_1, sigmoid):
        if sigmoid == 'linear':
            scaled = x * (1.0 - value_at_1)
            return jnp.where(jnp.abs(scaled) < 1, 1.0 - scaled, 0.0)
        if sigmoid == 'gaussian':
            scale = math.sqrt(-2 * math.log(value_at_1))
            return jnp.exp(-0.5 * (x * scale) ** 2)
        raise NotImplementedError(sigmoid)

    def tolerance(x, lower, upper, margin, sigmoid='linear', value_at_margin=0.0):
        in_bounds = jnp.logical_and(lower <= x, x <= upper)
        d = jnp.where(x < lower, lower - x, x - upper) / margin
        return jnp.where(in_bounds, 1.0, _sigmoid(d, value_at_margin, sigmoid))

    # Cheetah: aux = [horizontal_velocity, angmomentum]
    def cheetah_reward(aux, param):
        if param == 'flip':
            _SPIN_SPEED = 5.0
            return tolerance(aux[..., 1], _SPIN_SPEED, INF, _SPIN_SPEED, 'linear', 0.0)
        velocity = aux[..., 0] * float(np.sign(param))
        bound    = float(abs(param))
        return tolerance(velocity, bound, INF, bound, 'linear', 0.0)

    # Walker: aux = [horizontal_velocity, torso_upright, torso_height, angmomentum]
    def walker_reward(aux, param):
        _STAND_HEIGHT = 1.2
        torso_upright, torso_height = aux[..., 1], aux[..., 2]
        standing    = tolerance(torso_height, _STAND_HEIGHT, INF, _STAND_HEIGHT / 2, 'gaussian', 0.1)
        upright     = (1 + torso_upright) / 2
        stand_reward = (3 * standing + upright) / 4
        if param == 'flip':
            _SPIN_SPEED = 5.0
            move_reward = tolerance(aux[..., 3], _SPIN_SPEED, INF, _SPIN_SPEED, 'linear', 0.5)
        elif param == 0:   # standing: ignore `move_reward`
            move_reward = jnp.ones_like(stand_reward)
        else:
            move_reward = tolerance(aux[..., 0], float(param), INF, param / 2, 'linear', 0.5)
        return stand_reward * (5 * move_reward + 1) / 6

    # Quadruped: aux = [torso_velocity(3), torso_upright, com_height]
    def _quadruped_upright(aux):
        deviation = math.cos(math.radians(0))
        return tolerance(aux[..., 3], deviation, INF, 1 + deviation, 'linear', 0.0)

    def _quadruped_move(aux, desired_speed):
        move = tolerance(aux[..., 0], desired_speed, INF, desired_speed, 'linear', 0.5)
        return _quadruped_upright(aux) * move

    def quadruped_stand(aux, _=None): return _quadruped_upright(aux)
    def quadruped_walk(aux, _=None):  return _quadruped_move(aux, 0.5)
    def quadruped_run(aux, _=None):   return _quadruped_move(aux, 5.0)
    def quadruped_jump(aux, _=None):
        _desired_height = 1.0
        jump_up = tolerance(aux[..., 4], _desired_height, INF, _desired_height, 'linear', 0.5)
        return _quadruped_upright(aux) * jump_up

    # Antmaze: aux = full observation (29 dims)
    XY_VEL_SCALE = 0.33820298

    def goal_reaching_reward(state, goal):
        dist = jnp.linalg.norm(state[..., :2] - jnp.asarray(goal, dtype=jnp.float32), axis=-1)
        return (dist < 2).astype(jnp.float32) * 2 - 1

    def velocity_reward(states, direction):
        xy_vels = states[..., 15:17] * XY_VEL_SCALE
        return jnp.sum(xy_vels * jnp.asarray(direction, dtype=jnp.float32), axis=-1)

    def _grid_indices(states, y_max):
        x = jnp.clip(states[..., 0].astype(jnp.int32), 0, 35)   # int cast truncates toward 0, like torch .long()
        y = jnp.clip(states[..., 1].astype(jnp.int32), 0, y_max)
        return x, y

    def make_simplex_reward(num_simplex):
        import opensimplex
        pos  = np.zeros((num_simplex, 36, 25))
        xvel = np.zeros((num_simplex, 36, 25))
        yvel = np.zeros((num_simplex, 36, 25))
        print("Generating simplex seeds")
        xi, yi = np.arange(36), np.arange(25)
        for r in tqdm(range(num_simplex)):
            opensimplex.seed(r)
            pos[r]  = opensimplex.noise2array(x=xi / 20.0, y=yi / 20.0).T
            opensimplex.seed(r + num_simplex)
            xvel[r] = opensimplex.noise2array(x=xi / 20.0, y=yi / 20.0).T
            opensimplex.seed(r + num_simplex * 2)
            yvel[r] = opensimplex.noise2array(x=xi / 20.0, y=yi / 20.0).T
        xvel[np.abs(xvel) < 0.5] = 0
        yvel[np.abs(yvel) < 0.5] = 0
        pos_mask = jnp.asarray(pos > 0.3, dtype=jnp.float32)   # thresholded in float64, as in torch
        xvel, yvel = jnp.asarray(xvel, dtype=jnp.float32), jnp.asarray(yvel, dtype=jnp.float32)

        def simplex_reward(states, simplex_id):
            x, y = _grid_indices(states, 24)
            rews = -1 + pos_mask[simplex_id, x, y] * 0.5
            xy_vels = states[..., 15:17] * XY_VEL_SCALE
            return rews + xy_vels[..., 0] * xvel[simplex_id, x, y] + xy_vels[..., 1] * yvel[simplex_id, x, y]
        return simplex_reward

    def make_matrix_reward(pos, xvel, yvel, edges=False):
        pos_mask = jnp.asarray(pos > 0.3, dtype=jnp.float32)
        xvel, yvel = jnp.asarray(xvel, dtype=jnp.float32), jnp.asarray(yvel, dtype=jnp.float32)

        def matrix_reward(states, _=None):
            x, y = _grid_indices(states, 23)
            if edges:
                return pos_mask[x, y] * 2 - 1
            xy_vels = states[..., 15:17] * XY_VEL_SCALE
            return pos_mask[x, y] * 0.5 + xy_vels[..., 0] * xvel[x, y] + xy_vels[..., 1] * yvel[x, y]
        return matrix_reward

    def matrices():
        return [np.zeros((36, 25)) for _ in range(3)]

    def path_matrices():   # TestRewPath
        pos, xvel, yvel = matrices()
        pos[3:21, 7:10] = 1;  xvel[3:21, 7:10] = -1
        pos[0:3, 3:10] = 1;   yvel[0:3, 3:10] = -1
        pos[0:18, 0:3] = 1;   xvel[0:18, 0:3] = 1
        return pos, xvel, yvel

    def loop_matrices():   # TestRewLoop
        pos, xvel, yvel = matrices()
        pos[22:33, 14:18] = 1; xvel[22:33, 14:18] = -1
        pos[21:, 0:3] = 1;     xvel[21:, 0:3] = 1
        pos[33:, 3:18] = 1;    yvel[33:, 3:18] = 1
        pos[18:21, 0:7] = 1;   yvel[18:21, 0:7] = -1
        return pos, xvel, yvel

    def edges_matrices():  # TestRewMatrixEdges
        pos, xvel, yvel = matrices()
        pos[:3, :] = 1; pos[-3:, :] = 1; pos[:, :3] = 1; pos[:, -3:] = 1
        return pos, xvel, yvel

    # Pointmaze (ogbench): aux = full observation (x, y, vx, vy). Same 0/1 goal reward as the old ogbench antmaze code.
    POINTMAZE_GOAL_RADIUS = 1.0

    def pointmaze_goal_reward(state, goal):
        dist = jnp.linalg.norm(state[..., :2] - jnp.asarray(goal, dtype=jnp.float32), axis=-1)
        return (dist < POINTMAZE_GOAL_RADIUS).astype(jnp.float32)

    # ── Environment & dataset setup ───────────────────────────────────────────────

    if ENV_NAME in ['cheetah', 'walker', 'quadruped']:
        from dm_control import suite

        if ENV_NAME == 'cheetah':
            STATE_DIM, ACTION_DIM = 17, 6
            NUM_TRAJECTORIES, TRAJECTORY_LEN = 10000, 1000
            dataset = np.load(f'{DATASET_FOLDER}/cheetah_rnd.npy', allow_pickle=True).item()
            aux = np.concatenate((
                np.load(f'{DATASET_FOLDER}/aux_cheetah.npy').reshape(10000, 1000, 1),
                np.load(f'{DATASET_FOLDER}/angmomentum_aux_cheetah.npy').reshape(10000, 1000, 1),
            ), axis=-1)
            benchmarks = [
                (cheetah_reward, 'vel10Back', -10),
                (cheetah_reward, 'vel2Back',   -2),
                (cheetah_reward, 'vel2',        2),
                (cheetah_reward, 'vel10',      10),
                (cheetah_reward, 'flip',   'flip'),
            ]

        elif ENV_NAME == 'walker':
            STATE_DIM, ACTION_DIM = 24, 6
            NUM_TRAJECTORIES, TRAJECTORY_LEN = 10000, 1000
            dataset = np.load(f'{DATASET_FOLDER}/walker_rnd.npy', allow_pickle=True).item()
            aux = np.concatenate((
                np.load(f'{DATASET_FOLDER}/aux_walker.npy').reshape(10000, 1000, 3),
                np.load(f'{DATASET_FOLDER}/angmomentum_aux_walker.npy').reshape(10000, 1000, 1),
            ), axis=-1)
            benchmarks = [
                (walker_reward, 'stand',  0),
                (walker_reward, 'walk',   2),
                (walker_reward, 'vel8',   8),
                (walker_reward, 'run',   10),
                (walker_reward, 'flip', 'flip'),
            ]

        elif ENV_NAME == 'quadruped':
            STATE_DIM, ACTION_DIM = 78, 12
            NUM_TRAJECTORIES, TRAJECTORY_LEN = 10000, 1001
            dataset = np.load(f'{DATASET_FOLDER}/quadruped_rnd_without_physics.npy', allow_pickle=True).item()
            aux = np.load(f'{DATASET_FOLDER}/aux_quadruped.npy', allow_pickle=True)
            benchmarks = [
                (quadruped_stand, 'stand', None),
                (quadruped_walk,  'walk',  None),
                (quadruped_run,   'run',   None),
                (quadruped_jump,  'jump',  None),
            ]

        dataset_obs     = np.asarray(dataset['observations']).reshape(-1, STATE_DIM)
        dataset_actions = np.asarray(dataset['actions']).reshape(-1, ACTION_DIM)
        dataset_aux     = np.asarray(aux).reshape(dataset_obs.shape[0], -1)
        del dataset, aux

        EVAL_STEPS = 1000

        def make_eval_env():
            return suite.load(domain_name=ENV_NAME, task_name='run',
                              environment_kwargs=dict(flat_observation=True))

        def timestep2obs(timestep):
            return np.concatenate([v if len(v.shape) != 0 else v.reshape(-1)
                                   for k, v in timestep.observation.items()])

        def env_reset(e):
            return timestep2obs(e.reset())

        def env_step(e, action):
            return timestep2obs(e.step(action))

        def env_aux(e, state):
            physics = e.physics
            if ENV_NAME == 'walker':
                return np.array([physics.horizontal_velocity(), physics.torso_upright(), physics.torso_height(),
                                 physics.named.data.subtree_angmom['torso'][1]])
            if ENV_NAME == 'cheetah':
                return np.array([physics.speed(), physics.named.data.subtree_angmom['torso'][1]])
            com_height = physics.named.data.sensordata['center_of_mass'].copy()[2]   # quadruped
            return np.concatenate([physics.torso_velocity(), np.array([physics.torso_upright()]), np.array([com_height])])

    elif ENV_NAME == 'antmaze':
        import gym
        import d4rl   # registers the antmaze environments

        STATE_DIM, ACTION_DIM = 29, 8
        NUM_TRAJECTORIES, TRAJECTORY_LEN = 999, 1001
        env = gym.make('antmaze-large-diverse-v2')
        dataset = env.get_dataset()

        N = NUM_TRAJECTORIES * TRAJECTORY_LEN
        dataset_obs     = np.asarray(dataset['observations'])[:N].reshape(-1, STATE_DIM)
        dataset_actions = np.asarray(dataset['actions'])[:N].reshape(-1, ACTION_DIM)
        dataset_aux     = dataset_obs       # rewards are computed from the full observation
        del dataset

        simplex_reward = make_simplex_reward(num_simplex=10)
        benchmarks = [
            (goal_reaching_reward, 'goal_bottom', np.array([28, 0])),
            (goal_reaching_reward, 'goal_left',   np.array([0, 15])),
            (goal_reaching_reward, 'goal_top',    np.array([35, 24])),
            (goal_reaching_reward, 'goal_center', np.array([12, 24])),
            (goal_reaching_reward, 'goal_right',  np.array([33, 16])),
            (velocity_reward, 'vel_left',  [-1, 0]),
            (velocity_reward, 'vel_up',    [0, 1]),
            (velocity_reward, 'vel_down',  [0, -1]),
            (velocity_reward, 'vel_right', [1, 0]),
            (simplex_reward, 'simplex_1', 1),
            (simplex_reward, 'simplex_2', 2),
            (simplex_reward, 'simplex_3', 3),
            (simplex_reward, 'simplex_4', 4),
            (simplex_reward, 'simplex_5', 5),
            (make_matrix_reward(*path_matrices()),          'path_center', None),
            (make_matrix_reward(*loop_matrices()),          'path_loop',   None),
            (make_matrix_reward(*edges_matrices(), edges=True), 'path_edges',  None),
        ]

        EVAL_STEPS = 2000

        def make_eval_env():
            return gym.make('antmaze-large-diverse-v2')

        def env_reset(e):
            e.reset()
            location = (20, 15)
            # reset_to_location_antmaze
            e.sim.reset()
            qpos = e.init_qpos + e.np_random.uniform(low=-.1, high=.1, size=e.model.nq)
            qpos[:2] = np.array(location).astype(e.observation_space.dtype)
            qvel = e.init_qvel + e.np_random.randn(e.model.nv) * .1
            e.set_state(qpos, qvel)
            return e.unwrapped._get_obs()

        def env_step(e, action):
            new_state, _, _, _ = e.step(action)
            return new_state

        def env_aux(e, state):
            return state

    else:  # pointmaze (ogbench pointmaze-medium-navigate-v0)
        import ogbench

        POINTMAZE_DATASET = 'pointmaze-medium-navigate-v0'
        _, ogb_dataset, _ = ogbench.make_env_and_datasets(POINTMAZE_DATASET)

        dataset_obs     = np.asarray(ogb_dataset['observations'], dtype=np.float32)
        dataset_actions = np.asarray(ogb_dataset['actions'],      dtype=np.float32)
        STATE_DIM, ACTION_DIM = dataset_obs.shape[-1], dataset_actions.shape[-1]   # read from the data (obs starts with (x, y))
        print('pointmaze STATE_DIM:', STATE_DIM, 'ACTION_DIM:', ACTION_DIM)
        dataset_aux     = dataset_obs       # rewards are computed from the observation
        TRAJECTORY_LEN  = 1000
        assert len(dataset_obs) % TRAJECTORY_LEN == 0, \
            f"dataset size {len(dataset_obs)} is not a multiple of TRAJECTORY_LEN={TRAJECTORY_LEN}, adjust TRAJECTORY_LEN"
        NUM_TRAJECTORIES = len(dataset_obs) // TRAJECTORY_LEN
        ends = np.where(np.asarray(ogb_dataset['terminals']) > 0)[0]
        if len(ends):   # sanity check: episode boundaries must match TRAJECTORY_LEN, else (s, s') pairs get corrupted
            assert np.all(ends % TRAJECTORY_LEN == TRAJECTORY_LEN - 1), \
                "episode boundaries in the dataset do not match TRAJECTORY_LEN, adjust TRAJECTORY_LEN"
        del ogb_dataset

        # same free cells as antmaze-medium (cell (i, j) <-> xy = (4 j - 4, 4 i - 4))
        targets = [[0, 20], [20, 20], [20, 0], [4, 4], [12, 12]]
        benchmarks = [(pointmaze_goal_reward, f'goal_{x}_{y}', np.array([x, y])) for x, y in targets]

        EVAL_STEPS = 1000

        def make_eval_env():
            return ogbench.make_env_and_datasets(POINTMAZE_DATASET, env_only=True)

        def env_reset(e):
            e.reset()
            e.unwrapped.set_xy((0, 0))           # fixed start, as in the old ogbench antmaze eval
            ob = np.asarray(e.unwrapped.get_ob())
            assert ob.shape[-1] == STATE_DIM, f"env obs dim {ob.shape[-1]} != dataset obs dim {STATE_DIM}"
            return ob

        def env_step(e, action):
            ob, _, _, _, _ = e.step(action)       # gymnasium API: 5-tuple
            return ob

        def env_aux(e, state):
            return state

    N_BENCH   = len(benchmarks)
    NUM_EVALS = int(kwargs['num_evals']) if 'num_evals' in kwargs else 5
    print(NUM_TRAJECTORIES, TRAJECTORY_LEN, NUM_TRAJECTORIES * TRAJECTORY_LEN)

    # ── The stream of tasks ───────────────────────────────────────────────────────
    LABELS = [label for _, label, _ in benchmarks]
    for t in TASKS:
        assert t in LABELS, f"unknown task '{t}' for env {ENV_NAME}; available: {LABELS}"
    TASK_IDX = [LABELS.index(t) for t in TASKS]
    print('continual stream:', TASKS)

    BATCH_SIZE = 1024
    # Number of benchmarks rolled out in parallel during eval (each one needs NUM_EVALS env instances)
    EVAL_CHUNK = int(kwargs['eval_chunk']) if 'eval_chunk' in kwargs else N_BENCH

    EVAL_FIELDS = (["stage", "step", "tag", "mean_fresh_all", "mean_fresh_tasks", "mean_tracked_tasks"]
                   + [f"{label}_fresh_{s}" for label in LABELS for s in ("mean", "std", "rew_corr", "eps")]
                   + [f"{label}_tracked_{s}" for label in TASKS for s in ("mean", "std", "rew_corr", "eps")]
                   + [f"{label}_ft_{s}" for label in TASKS for s in ("mean", "std", "rew_corr", "eps")]
                   + [f"{label}_fit_{s}" for label in TASKS for s in ("rew_corr", "eps")]
                   + ["time"])
    eval_logger = CSVLogger(filepath=f"{LOGS_FOLDER}/eval_log.csv", fieldnames=EVAL_FIELDS)

    # ── Replay buffer ─────────────────────────────────────────────────────────────

    class OfflineReplayBuffer(flax.struct.PyTreeNode):
        obs:       jnp.ndarray
        actions:   jnp.ndarray
        aux:       Optional[jnp.ndarray]
        valid_idx: jnp.ndarray
        size:      int = flax.struct.field(pytree_node=False)

        @classmethod
        def init(cls, observations, actions, trajectory_len, aux=None) -> "OfflineReplayBuffer":
            n_total = (len(observations) // trajectory_len) * trajectory_len
            idx     = np.arange(n_total)
            # a transition (i, i+1) is valid iff i is not the last state of its trajectory
            valid_idx = idx[(idx % trajectory_len) != trajectory_len - 1]
            return cls(
                obs=jnp.asarray(observations[:n_total], dtype=jnp.float32),
                actions=jnp.asarray(actions[:n_total], dtype=jnp.float32),
                aux=None if aux is None else jnp.asarray(aux[:n_total], dtype=jnp.float32),
                valid_idx=jnp.asarray(valid_idx, dtype=jnp.int32),
                size=len(valid_idx),
            )

        def sample_indices(self, random_key, sample_size: int):
            idx = jax.random.randint(random_key, shape=(sample_size,), minval=0, maxval=self.size)
            return self.valid_idx[idx]

        def sample(self, random_key, sample_size: int):
            """Returns (obs, actions, next_obs, aux)."""
            i   = self.sample_indices(random_key, sample_size)
            aux = None if self.aux is None else self.aux[i]
            return self.obs[i], self.actions[i], self.obs[i + 1], aux

    replay_buffer = OfflineReplayBuffer.init(dataset_obs, dataset_actions, TRAJECTORY_LEN, aux=dataset_aux)
    print('transitions available for training:', replay_buffer.size)
    del dataset_obs, dataset_actions, dataset_aux

    # ── Neural networks (identical to the pretraining script) ─────────────────────

    import flax.linen as nn

    class TorchDense(nn.Module):
        """nn.Dense with torch's default nn.Linear init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for W and b."""
        features: int

        @nn.compact
        def __call__(self, x):
            fan_in = x.shape[-1]
            bound  = 1.0 / math.sqrt(fan_in)
            init   = lambda key, shape, dtype=jnp.float32: jax.random.uniform(key, shape, dtype, -bound, bound)
            kernel = self.param('kernel', init, (fan_in, self.features))
            bias   = self.param('bias', init, (self.features,))
            return x @ kernel + bias

    class NTanh(nn.Module):
        """torch 'ntanh': LayerNorm followed by Tanh."""
        @nn.compact
        def __call__(self, x):
            return jnp.tanh(nn.LayerNorm(epsilon=1e-5)(x))

    class ForwardMap(nn.Module):
        """torch ForwardMap (preprocess=True). Returns (F1, F2)."""
        hidden_dim:  int
        feature_dim: int
        z_dim:       int
        add_trunk:   bool = False

        @nn.compact
        def __call__(self, obs, z, action):
            # obs_action_net: Linear -> ntanh -> Linear -> relu
            oa = TorchDense(self.hidden_dim)(jnp.concatenate([obs, action], axis=-1))
            oa = NTanh()(oa)
            oa = nn.relu(TorchDense(self.feature_dim)(oa))
            # obs_z_net: Linear -> ntanh -> Linear -> relu
            oz = TorchDense(self.hidden_dim)(jnp.concatenate([obs, z], axis=-1))
            oz = NTanh()(oz)
            oz = nn.relu(TorchDense(self.feature_dim)(oz))

            h = jnp.concatenate([oa, oz], axis=-1)
            if self.add_trunk:
                h = nn.relu(TorchDense(self.hidden_dim)(h))

            def head(h):
                x = nn.relu(TorchDense(self.hidden_dim)(h))
                return TorchDense(self.z_dim)(x)

            return head(h), head(h)

    class BackwardMap(nn.Module):
        """torch BackwardMap: mlp(obs, hidden, 'ntanh', hidden, 'relu', z_dim), then sqrt(z_dim) * normalize."""
        hidden_dim: int
        z_dim:      int
        norm_z:     bool = True

        @nn.compact
        def __call__(self, obs):
            x = NTanh()(TorchDense(self.hidden_dim)(obs))
            x = nn.relu(TorchDense(self.hidden_dim)(x))
            x = TorchDense(self.z_dim)(x)
            if self.norm_z:
                x = math.sqrt(self.z_dim) * normalize(x)
            return x

    class Actor(nn.Module):
        """torch Actor (preprocess=True). Returns the mean action mu = tanh(policy(h))."""
        hidden_dim:  int
        feature_dim: int
        action_dim:  int
        add_trunk:   bool = False

        @nn.compact
        def __call__(self, obs, z):
            o = NTanh()(TorchDense(self.hidden_dim)(obs))
            o = nn.relu(TorchDense(self.feature_dim)(o))
            oz = NTanh()(TorchDense(self.hidden_dim)(jnp.concatenate([obs, z], axis=-1)))
            oz = nn.relu(TorchDense(self.feature_dim)(oz))

            h = jnp.concatenate([o, oz], axis=-1)
            if self.add_trunk:
                h = nn.relu(TorchDense(self.hidden_dim)(h))

            h = nn.relu(TorchDense(self.hidden_dim)(h))
            return jnp.tanh(TorchDense(self.action_dim)(h))

    def sample_truncated_normal(mu, random_key, std, clip, low=-1.0, high=1.0, eps=1e-6):
        """torch TruncatedNormal.sample(clip): clipped Gaussian noise, output clamped with a straight-through gradient."""
        noise   = jax.random.normal(random_key, mu.shape) * std
        noise   = jnp.clip(noise, -clip, clip)
        x       = mu + noise
        clamped = jnp.clip(x, low + eps, high - eps)
        return x + jax.lax.stop_gradient(clamped - x)

    # ── FB Agent ──────────────────────────────────────────────────────────────────

    @dataclasses.dataclass(frozen=True)
    class FBConfig:
        # --- FB (same defaults as the pretraining script: they must match the checkpoint) ---
        bc_coef:             float = 0.0      # weight of the behavior-cloning term of the DDPG actor (0 -> plain -Q.mean())
        lr:                  float = 1e-4
        lr_coef:             float = 1.0     # backward net lr = lr_coef * lr
        fb_target_tau:       float = 0.01
        hidden_dim:          int   = 1024
        backward_hidden_dim: int   = 526     # (sic) same value as in the torch script
        feature_dim:         int   = 512
        z_dim:               int   = 50
        stddev:              float = 0.2
        stddev_clip:         float = 0.3
        batch_size:          int   = 1024
        ortho_coef:          float = 1.0
        mix_ratio:           float = 0.5
        norm_z:              bool  = True
        q_loss:              bool  = True
        q_loss_coef:         float = 0.01
        add_trunk:           bool  = False
        discount:            float = 0.95
        target_agg:          str   = 'min'    # bootstrap target of M over the two heads: 'min' (original FB / pretraining script) | 'mean'
        # --- continual terms ---
        lam:                 float = 1.0      # weight of L_rec
        mu_b:                float = 1.0      # weight of R^B
        mu_q:                float = 0.01     # weight of R^Q (not dimensionless: it is a squared error on values)
        alpha:               float = 0.5      # weight of the atoms in nu
        n_est:               int   = 4096     # states of D per step to estimate Sigma_theta, A_theta
        ridge:               float = 1e-5     # ridge on every covariance before inversion
        cross_fit:           bool  = False    # cross-fitted L_rec
        rec_fixed_w:         bool  = False    # L_rec with the fixed readout w_fit of the stage instead of the re-solved one
        rec_bs:              int   = 0        # L_rec minibatch size on D_k (0 = the whole labeled set)
        b_decay_steps:       int   = 0        # >0: cosine decay of the lr of B over this many updates (set by b_cosine=1)
        # --- latent of the current task inside the FB / actor loss (0 = off = original behaviour) ---
        z_task_frac:         float = 0.0      # fraction of the z of L_FB and of the actor loss replaced by the latent of the current task
        z_task_spread:       float = 0.5      # angular noise around that latent (0 = exactly the latent; 1 = 45 degrees)
        actor_z_task_frac:   float = -1.0     # fraction of the ACTOR's z that is the task latent (-1 = same as z_task_frac)
        freeze_fb:           bool  = False    # diagnostic: do not update F and B, only the actor
        freeze_b:            bool  = False    # diagnostic: do not update B (F and the actor are still trained)
        # --- critic loss / fine-tuning ---
        fb_objective:        bool  = True     # True: L_FB + q_loss_coef * Q loss. False: Q loss only (L_FB not computed)
        finetune:            str   = 'none'   # none | actor | critic | reward_critic (single-task fine-tuning, see docstring)

    class FBTrainingState(flax.struct.PyTreeNode):
        fb_params:             Any          # {'forward': ..., 'backward': ...}
        fb_target_params:      Any
        actor_params:          Any
        fb_optimizer_state:    optax.OptState
        actor_optimizer_state: optax.OptState
        random_key:            jnp.ndarray
        steps:                 jnp.ndarray

    class StageData(flax.struct.PyTreeNode):
        """Everything frozen during a stage."""
        old_params:    Any             # theta_old = {'forward': ..., 'backward': ...}
        sigma_old_inv: jnp.ndarray     # [d, d]
        lab_obs:       jnp.ndarray     # [n_labeled, obs_dim]   labeled states of D_k
        lab_r:         jnp.ndarray     # [n_labeled]            normalized rewards rbar_k (unit second moment)
        atoms:         jnp.ndarray     # [K, d]  stored latents z_j (rows >= n_atoms are zeros), normalized when sampling
        n_atoms:       jnp.ndarray     # int32 scalar = k - 1
        z_run:         jnp.ndarray     # [d] E[rbar_k B] with B at the start of the stage (used instead of re-encoding when freeze_b=1)
        z_cond:        jnp.ndarray     # [d] sqrt(d) * normalize(z_run): the fixed task latent of the fine-tuning modes
        w_fit:         jnp.ndarray     # [d] Sigma_old^-1 z_run : fixed readout, B(s)^T w_fit ~ rbar_k(s)
        w_q:           jnp.ndarray     # [d] Sigma_old^-1 z_cond: fixed readout of the reward in the fine-tuning Q loss

    class FBAgent:

        def __init__(self, config: FBConfig, obs_dim: int, action_dim: int):
            self.cfg        = config
            self.obs_dim    = obs_dim
            self.action_dim = action_dim

            self.forward_net  = ForwardMap(config.hidden_dim, config.feature_dim, config.z_dim, config.add_trunk)
            self.backward_net = BackwardMap(config.backward_hidden_dim, config.z_dim, config.norm_z)
            self.actor_net    = Actor(config.hidden_dim, config.feature_dim, action_dim, config.add_trunk)

            # Forward net at lr, backward net at lr_coef * lr (same as the two torch param groups)
            labels = {'forward': 'forward', 'backward': 'backward'}
            b_lr = config.lr * config.lr_coef
            if config.b_decay_steps > 0:
                b_lr = optax.cosine_decay_schedule(b_lr, config.b_decay_steps)
            self.fb_optimizer = optax.multi_transform(
                {'forward': optax.adam(config.lr), 'backward': optax.adam(b_lr)},
                lambda params: {k: jax.tree_util.tree_map(lambda _: labels[k], v) for k, v in params.items()},
            )
            self.actor_optimizer = optax.adam(config.lr)

        # ── init ──────────────────────────────────────────────────────────────

        def init(self, random_key) -> FBTrainingState:
            cfg = self.cfg
            random_key, kf, kb, ka = jax.random.split(random_key, 4)
            obs    = jnp.zeros((1, self.obs_dim))
            action = jnp.zeros((1, self.action_dim))
            z      = jnp.zeros((1, cfg.z_dim))

            fb_params = {
                'forward':  self.forward_net.init(kf, obs, z, action)['params'],
                'backward': self.backward_net.init(kb, obs)['params'],
            }
            actor_params = self.actor_net.init(ka, obs, z)['params']

            return FBTrainingState(
                fb_params=fb_params,
                fb_target_params=jax.tree_util.tree_map(jnp.copy, fb_params),
                actor_params=actor_params,
                fb_optimizer_state=self.fb_optimizer.init(fb_params),
                actor_optimizer_state=self.actor_optimizer.init(actor_params),
                random_key=random_key,
                steps=jnp.zeros((), dtype=jnp.int32),
            )

        # ── helpers ───────────────────────────────────────────────────────────

        def sample_z(self, random_key, size: int):
            cfg = self.cfg
            k1, k2 = jax.random.split(random_key)
            gaussian = normalize(jax.random.normal(k1, (size, cfg.z_dim)))
            if cfg.norm_z:
                return math.sqrt(cfg.z_dim) * gaussian
            uniform = jax.random.uniform(k2, (size, cfg.z_dim))
            return math.sqrt(cfg.z_dim) * uniform * gaussian

        @partial(jax.jit, static_argnames=("self",))
        def act(self, actor_params, obs, z):
            """Deterministic action (eval_mode=True in the torch agent). Works on batches of (obs, z)."""
            return self.actor_net.apply({'params': actor_params}, obs, z)

        # ── single-task fine-tuning step (called from update when finetune != 'none') ──

        def _finetune_step(self, ts, sd, obs, action, next_obs, k_next, k_act, random_key, metrics):
            """One update on the task of the stage. Every z (critic, bootstrap, actor) is the fixed latent z_cond.
            L_FB, ortho, L_rec, R^B, R^Q are NOT computed.
              actor         : F and B frozen, only the actor maximizes F(s, pi(s, z), z)^T z
              critic        : B frozen; F trained with the Q loss only, reward r(s') = B(s')^T w_q (fixed readout)
              reward_critic : same as critic (B was fit to the reward before the stage, see fit_backward_to_reward)"""
            cfg = self.cfg
            bs, d = cfg.batch_size, cfg.z_dim
            discount = cfg.discount
            z = jnp.broadcast_to(sd.z_cond[None, :], (bs, d))
            fb_params, fb_optimizer_state = ts.fb_params, ts.fb_optimizer_state

            if cfg.finetune in ('critic', 'reward_critic'):
                # Q(s, a) = F(s, a, z)^T z regressed on r(s') + discount * min_heads F_bar(s', a', z)^T z,  a' = pi(s', z) + noise
                reward = jax.lax.stop_gradient(
                    self.backward_net.apply({'params': ts.fb_params['backward']}, next_obs) @ sd.w_q)
                next_mu     = self.actor_net.apply({'params': ts.actor_params}, next_obs, z)
                next_action = jax.lax.stop_gradient(
                    sample_truncated_normal(next_mu, k_next, cfg.stddev, cfg.stddev_clip))
                tF1, tF2 = self.forward_net.apply({'params': ts.fb_target_params['forward']}, next_obs, z, next_action)
                next_Q   = jax.lax.stop_gradient(jnp.minimum((tF1 * z).sum(-1), (tF2 * z).sum(-1)))
                target_Q = jax.lax.stop_gradient(reward + discount * next_Q)

                def q_loss_fn(f_params):
                    F1, F2 = self.forward_net.apply({'params': f_params}, obs, z, action)
                    return (jnp.mean(((F1 * z).sum(-1) - target_Q) ** 2)
                            + jnp.mean(((F2 * z).sum(-1) - target_Q) ** 2))

                # gradient on F only; B gets zero gradients and is left untouched
                q_loss, f_grads = jax.value_and_grad(q_loss_fn)(ts.fb_params['forward'])
                zeros_b = jax.tree_util.tree_map(jnp.zeros_like, ts.fb_params['backward'])
                fb_updates, fb_optimizer_state = self.fb_optimizer.update(
                    {'forward': f_grads, 'backward': zeros_b}, ts.fb_optimizer_state, ts.fb_params)
                fb_params = {'forward':  optax.apply_updates(ts.fb_params['forward'], fb_updates['forward']),
                             'backward': ts.fb_params['backward']}
                metrics.update({'q_loss': q_loss, 'next_Q': next_Q.mean(), 'q_reward': reward.mean(),
                                'total_loss': q_loss})

            # ── actor: maximize F(s, pi(s, z), z)^T z (TD3+BC if bc_coef > 0), with the freshly updated F
            def actor_loss_fn(actor_params):
                mu = self.actor_net.apply({'params': actor_params}, obs, z)
                a  = sample_truncated_normal(mu, k_act, cfg.stddev, cfg.stddev_clip)
                F1, F2 = self.forward_net.apply({'params': fb_params['forward']}, obs, z, a)
                Q  = jnp.minimum((F1 * z).sum(-1), (F2 * z).sum(-1))
                sq = jnp.mean((mu - action) ** 2, axis=-1)
                if cfg.bc_coef > 0:
                    actor_loss = -Q.mean() / jax.lax.stop_gradient(jnp.abs(Q).mean()) + cfg.bc_coef * sq.mean()
                else:
                    actor_loss = -Q.mean()
                return actor_loss, {'actor_loss': actor_loss, 'q': Q.mean(), 'bc_dist': sq.mean()}

            (_, actor_metrics), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(ts.actor_params)
            actor_updates, actor_optimizer_state = self.actor_optimizer.update(
                actor_grads, ts.actor_optimizer_state, ts.actor_params)
            actor_params = optax.apply_updates(ts.actor_params, actor_updates)
            metrics.update(actor_metrics)

            tau_t = cfg.fb_target_tau
            fb_target_params = jax.tree_util.tree_map(
                lambda p, t: tau_t * p + (1 - tau_t) * t, fb_params, ts.fb_target_params)
            ts = ts.replace(
                fb_params=fb_params,
                fb_target_params=fb_target_params,
                actor_params=actor_params,
                fb_optimizer_state=fb_optimizer_state,
                actor_optimizer_state=actor_optimizer_state,
                random_key=random_key,
                steps=ts.steps + 1,
            )
            return ts, metrics

        # ── update────────────────────────────────────────────────────────────

        @partial(jax.jit, static_argnames=("self",))
        def update(self, ts: FBTrainingState, replay_buffer: OfflineReplayBuffer, sd: StageData):
            cfg = self.cfg
            bs  = cfg.batch_size
            d   = cfg.z_dim
            discount = cfg.discount

            (random_key, k_batch, k_z, k_perm, k_mix, k_next, k_act,
             k_est, k_nu0, k_nu1, k_nu2, k_tz1, k_tz2) = jax.random.split(ts.random_key, 13)
            obs, action, next_obs, _ = replay_buffer.sample(k_batch, bs)

            # every training metric is always returned (NaN when its term is not computed), so that the pytree of
            # metrics does not depend on the branches taken
            metrics = {m: jnp.full((), jnp.nan, dtype=jnp.float32) for m in TRAIN_METRICS}

            if cfg.finetune != 'none':
                return self._finetune_step(ts, sd, obs, action, next_obs, k_next, k_act, random_key, metrics)

            # states of D used to estimate Sigma_theta and A_theta (n >> d)
            obs_est = replay_buffer.obs[replay_buffer.sample_indices(k_est, cfg.n_est)]

            # ── z sampling for L_FB: random z, with a fraction replaced by B(obs) of shuffled states
            z = self.sample_z(k_z, bs)
            if cfg.mix_ratio > 0:
                perm  = jax.random.permutation(k_perm, bs)
                mix_z = jax.lax.stop_gradient(
                    self.backward_net.apply({'params': ts.fb_params['backward']}, obs[perm]))
                mix_mask = jax.random.uniform(k_mix, (bs, 1)) < cfg.mix_ratio
                z = jnp.where(mix_mask, mix_z, z)

            # ── task latent of the CURRENT stage in the z distribution of L_FB and of the actor:
            #    z_k = mean(rbar_k B_current) on the stage's labeled set D_k (fixed for the stage), plus a bit of angular noise
            z_act = z      # z used by the actor loss (= the z of the FB loss unless actor_z_task_frac is set)
            if cfg.z_task_frac > 0 or cfg.actor_z_task_frac > 0:
                if cfg.freeze_b:
                    z_cur = sd.z_run                                                         # computed once at the start of the stage
                else:
                    B_lab_sg = jax.lax.stop_gradient(
                        self.backward_net.apply({'params': ts.fb_params['backward']}, sd.lab_obs))
                    z_cur    = (sd.lab_r[:, None] * B_lab_sg).mean(axis=0)                   # [d] re-encoded with the current B
                noise    = normalize(jax.random.normal(k_tz1, (bs, d)))
                z_task   = math.sqrt(d) * normalize(normalize(z_cur)[None, :] + cfg.z_task_spread * noise)
                u_task   = jax.random.uniform(k_tz2, (bs, 1))
                z_base   = z
                z        = jax.lax.stop_gradient(jnp.where(u_task < cfg.z_task_frac, z_task, z_base))
                if cfg.actor_z_task_frac >= 0:     # the actor gets its own fraction of task latents
                    z_act = jax.lax.stop_gradient(jnp.where(u_task < cfg.actor_z_task_frac, z_task, z_base))
                else:
                    z_act = z

            use_ret        = cfg.mu_b > 0 or cfg.mu_q > 0     # the retention terms need latents z ~ nu
            use_old_family = cfg.lam > 0 or use_ret           # Sigma_theta / T / M are needed

            # ── latents of the retention losses: z ~ nu = (1 - alpha) * uniform sphere + alpha * atoms ──
            if use_ret:
                z_sphere = self.sample_z(k_nu0, bs)
                n_at     = jnp.maximum(sd.n_atoms, 1)
                at_idx   = jnp.minimum((jax.random.uniform(k_nu1, (bs,)) * n_at).astype(jnp.int32), n_at - 1)
                z_atom   = math.sqrt(d) * normalize(sd.atoms[at_idx])
                use_atom = (jax.random.uniform(k_nu2, (bs, 1)) < cfg.alpha) & (sd.n_atoms > 0)
                z_ret    = jax.lax.stop_gradient(jnp.where(use_atom, z_atom, z_sphere))

            off_diag = 1.0 - jnp.eye(bs)
            n_off    = off_diag.sum()

            if not cfg.freeze_fb:
                # ── bootstrap of the critic losses (classical: through the actor) ──
                next_mu     = self.actor_net.apply({'params': ts.actor_params}, next_obs, z)
                next_action = jax.lax.stop_gradient(
                    sample_truncated_normal(next_mu, k_next, cfg.stddev, cfg.stddev_clip))
                target_F1, target_F2 = self.forward_net.apply(
                    {'params': ts.fb_target_params['forward']}, next_obs, z, next_action)
                if cfg.fb_objective:
                    target_B = self.backward_net.apply({'params': ts.fb_target_params['backward']}, next_obs)
                    tM1, tM2 = target_F1 @ target_B.T, target_F2 @ target_B.T
                    if cfg.target_agg == 'min':
                        target_M = jax.lax.stop_gradient(jnp.minimum(tM1, tM2))
                    else:
                        target_M = jax.lax.stop_gradient(0.5 * (tM1 + tM2))
                if cfg.q_loss:
                    next_Q = jax.lax.stop_gradient(jnp.minimum((target_F1 * z).sum(-1), (target_F2 * z).sum(-1)))

            # ── FB update: [L_FB] + [Q loss] + [ortho] + lam * L_rec + mu_b * R^B + mu_q * R^Q
            #    every term is computed only if it is used (static branches: fb_objective, q_loss, ortho_coef, lam, mu_b, mu_q)
            def fb_loss_fn(fb_params):
                F1, F2 = self.forward_net.apply({'params': fb_params['forward']}, obs, z, action)
                B      = self.backward_net.apply({'params': fb_params['backward']}, next_obs)
                total  = jnp.zeros(())
                m      = {}

                if cfg.fb_objective:
                    # ---- L_FB (identical to the pretraining script)
                    M1, M2 = F1 @ B.T, F2 @ B.T
                    fb_offdiag = 0.5 * sum(
                        (((Mh - discount * target_M) ** 2) * off_diag).sum() / n_off for Mh in (M1, M2))
                    fb_diag = -sum(jnp.diag(Mh).mean() for Mh in (M1, M2))
                    fb_loss = fb_offdiag + fb_diag
                    total   = total + fb_loss
                    m.update({'fb_loss': fb_loss, 'fb_diag': fb_diag, 'fb_offdiag': fb_offdiag,
                              'target_M': target_M.mean(), 'M1': M1.mean()})

                if cfg.q_loss:
                    # ---- Q loss with the implicit reward B(s')^T Sigma^-1 z (Sigma on the batch)
                    B_sg    = jax.lax.stop_gradient(B)
                    cov     = B_sg.T @ B_sg / B_sg.shape[0]
                    inv_cov = jnp.linalg.inv(cov)
                    implicit_reward = ((B_sg @ inv_cov) * z).sum(axis=-1)
                    target_Q = jax.lax.stop_gradient(implicit_reward + discount * next_Q)
                    Q1, Q2   = (F1 * z).sum(-1), (F2 * z).sum(-1)
                    q_loss   = jnp.mean((Q1 - target_Q) ** 2) + jnp.mean((Q2 - target_Q) ** 2)
                    # fb_objective=1: L_FB + q_loss_coef * Q loss ; fb_objective=0: the Q loss alone
                    total    = total + (cfg.q_loss_coef if cfg.fb_objective else 1.0) * q_loss
                    m.update({'q_loss': q_loss, 'next_Q': next_Q.mean(), 'q_reward': implicit_reward.mean()})

                if cfg.ortho_coef > 0:
                    Cov               = B @ B.T
                    orth_loss_diag    = -2 * jnp.diag(Cov).mean()
                    orth_loss_offdiag = ((Cov ** 2) * off_diag).sum() / n_off
                    orth_loss         = orth_loss_offdiag + orth_loss_diag
                    total             = total + cfg.ortho_coef * orth_loss
                    m.update({'orth_loss': orth_loss, 'orth_loss_diag': orth_loss_diag,
                              'orth_loss_offdiag': orth_loss_offdiag})

                if use_old_family:
                    # ---- batch estimates of Sigma_theta, A_theta, T_theta, M_theta (n_est states of D)
                    B_new_est = self.backward_net.apply({'params': fb_params['backward']}, obs_est)
                    B_old_est = jax.lax.stop_gradient(
                        self.backward_net.apply({'params': sd.old_params['backward']}, obs_est))
                    Sigma_th, T, M = old_family_matrices(B_new_est, B_old_est, sd.sigma_old_inv, cfg.ridge)

                if cfg.lam > 0:
                    # ---- L_rec on the labeled set (whole set at every step, detached coefficient)
                    if cfg.rec_bs > 0:        # random minibatch of D_k (key derived from the step key: the other streams are unchanged)
                        idx_rec = jax.random.randint(jax.random.fold_in(random_key, 7), (cfg.rec_bs,), 0, sd.lab_obs.shape[0])
                        lab_obs_rec, lab_r_rec = sd.lab_obs[idx_rec], sd.lab_r[idx_rec]
                    else:
                        lab_obs_rec, lab_r_rec = sd.lab_obs, sd.lab_r
                    B_lab = self.backward_net.apply({'params': fb_params['backward']}, lab_obs_rec)
                    if cfg.rec_fixed_w:       # fixed readout of the stage, as in fit_backward_to_reward
                        l_rec = jnp.mean((lab_r_rec - B_lab @ sd.w_fit) ** 2)
                    else:
                        l_rec = reconstruction_loss(B_lab, lab_r_rec, Sigma_th, cfg.cross_fit)
                    total = total + cfg.lam * l_rec
                    m['new_reward_unexplained_fraction_training'] = l_rec

                if cfg.mu_b > 0:
                    # ---- R^B (closed form, differentiated through A_theta and Sigma_theta)
                    r_b   = retention_basis_loss(z_ret, M, sd.sigma_old_inv)
                    total = total + cfg.mu_b * r_b
                    m['old_rewards_lost_fraction'] = r_b

                if cfg.mu_q > 0:
                    # ---- R^Q: values of the re-encoded old family r_z, matched head by head. z' = sg(T z).
                    z_new     = jax.lax.stop_gradient(z_ret @ T.T)               # rows: (T z)^T
                    z_new_bar = math.sqrt(d) * normalize(z_new)
                    nF1, nF2  = self.forward_net.apply({'params': fb_params['forward']}, obs, z_new_bar, action)
                    oF1, oF2  = jax.lax.stop_gradient(self.forward_net.apply(
                        {'params': sd.old_params['forward']}, obs, z_ret, action))
                    r_q = 0.5 * (jnp.mean(((nF1 * z_new).sum(-1) - (oF1 * z_ret).sum(-1)) ** 2)
                                 + jnp.mean(((nF2 * z_new).sum(-1) - (oF2 * z_ret).sum(-1)) ** 2))
                    total = total + cfg.mu_q * r_q
                    m.update({'old_rewards_value_change': r_q,
                              'old_latent_length_ratio': jnp.mean(jnp.linalg.norm(z_new, axis=-1)
                                                                  / jnp.linalg.norm(z_ret, axis=-1))})

                B_d      = jax.lax.stop_gradient(B)
                eye_diff = B_d.T @ B_d / B_d.shape[0] - jnp.eye(B_d.shape[1])
                b_eigs   = jnp.linalg.eigvalsh(B_d.T @ B_d / B_d.shape[0])
                m.update({
                    'F1':            F1.mean(),
                    'B':             B_d.mean(),
                    'B_norm':        jnp.linalg.norm(B_d, axis=-1).mean(),
                    'z_norm':        jnp.linalg.norm(z, axis=-1).mean(),
                    'orth_linf':     jnp.max(jnp.abs(eye_diff)),
                    'orth_l2':       jnp.linalg.norm(eye_diff) / math.sqrt(B_d.shape[1]),
                    'b_cov_eig_min': b_eigs[0],
                    'b_cov_eig_max': b_eigs[-1],
                    'total_loss':    total,
                })
                return total, m

            if cfg.freeze_fb:      # diagnostic: F and B (and their Adam state) are not updated, FB losses not computed
                fb_params, fb_optimizer_state = ts.fb_params, ts.fb_optimizer_state
            else:
                (_, fb_metrics), fb_grads = jax.value_and_grad(fb_loss_fn, has_aux=True)(ts.fb_params)
                metrics.update(fb_metrics)
                fb_updates, fb_optimizer_state = self.fb_optimizer.update(
                    fb_grads, ts.fb_optimizer_state, ts.fb_params)
                if cfg.freeze_b:       # B is not updated (F and the actor still are)
                    fb_updates = {'forward': fb_updates['forward'],
                                  'backward': jax.tree_util.tree_map(jnp.zeros_like, fb_updates['backward'])}
                fb_params = optax.apply_updates(ts.fb_params, fb_updates)

            # ── Actor update (DDPG / TD3-like, uses the freshly updated forward net, as in torch) ──
            # maximize F(s, pi(s,z), z)^T z. With bc_coef > 0: TD3+BC (scale-free Q term, the normalizer is not differentiated).
            def actor_loss_fn(actor_params):
                mu = self.actor_net.apply({'params': actor_params}, obs, z_act)
                a  = sample_truncated_normal(mu, k_act, cfg.stddev, cfg.stddev_clip)
                F1, F2 = self.forward_net.apply({'params': fb_params['forward']}, obs, z_act, a)
                Q = jnp.minimum((F1 * z_act).sum(-1), (F2 * z_act).sum(-1))
                sq = jnp.mean((mu - action) ** 2, axis=-1)
                if cfg.bc_coef > 0:
                    actor_loss = -Q.mean() / jax.lax.stop_gradient(jnp.abs(Q).mean()) + cfg.bc_coef * sq.mean()
                else:
                    actor_loss = -Q.mean()
                return actor_loss, {'actor_loss': actor_loss, 'q': Q.mean(), 'bc_dist': sq.mean()}

            (_, actor_metrics), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(ts.actor_params)
            actor_updates, actor_optimizer_state = self.actor_optimizer.update(
                actor_grads, ts.actor_optimizer_state, ts.actor_params)
            actor_params = optax.apply_updates(ts.actor_params, actor_updates)

            metrics.update({
                'actor_loss': actor_metrics['actor_loss'],
                'q':          actor_metrics['q'],
                'bc_dist':    actor_metrics['bc_dist'],
            })

            # ── Target update ─────────────────────────────────────────────────
            tau_t = cfg.fb_target_tau
            fb_target_params = jax.tree_util.tree_map(
                lambda p, t: tau_t * p + (1 - tau_t) * t, fb_params, ts.fb_target_params)

            ts = ts.replace(
                fb_params=fb_params,
                fb_target_params=fb_target_params,
                actor_params=actor_params,
                fb_optimizer_state=fb_optimizer_state,
                actor_optimizer_state=actor_optimizer_state,
                random_key=random_key,
                steps=ts.steps + 1,
            )
            return ts, metrics

    fb_config = FBConfig(batch_size=BATCH_SIZE)
    for _k, _typ in (('discount', float), ('bc_coef', float), ('ortho_coef', float), ('q_loss_coef', float),
                     ('lr', float), ('lr_coef', float), ('fb_target_tau', float), ('mix_ratio', float),
                     ('lam', float), ('mu_b', float), ('mu_q', float),
                     ('alpha', float), ('n_est', int), ('ridge', float),
                     ('z_task_frac', float), ('z_task_spread', float), ('actor_z_task_frac', float)):
        if _k in kwargs:
            fb_config = dataclasses.replace(fb_config, **{_k: _typ(kwargs[_k])})
    if 'target_agg' in kwargs:
        assert kwargs['target_agg'] in ['min', 'mean'], 'target_agg= must be min or mean'
        fb_config = dataclasses.replace(fb_config, target_agg=kwargs['target_agg'])
    if 'freeze_b' in kwargs:
        fb_config = dataclasses.replace(fb_config, freeze_b=bool(int(kwargs['freeze_b'])))
    if 'freeze_fb' in kwargs:
        fb_config = dataclasses.replace(fb_config, freeze_fb=bool(int(kwargs['freeze_fb'])))
    if 'cross_fit' in kwargs:
        fb_config = dataclasses.replace(fb_config, cross_fit=bool(int(kwargs['cross_fit'])))
    if 'rec_fixed_w' in kwargs:
        fb_config = dataclasses.replace(fb_config, rec_fixed_w=bool(int(kwargs['rec_fixed_w'])))
    if 'rec_bs' in kwargs:
        fb_config = dataclasses.replace(fb_config, rec_bs=int(kwargs['rec_bs']))
    if 'b_cosine' in kwargs and bool(int(kwargs['b_cosine'])):
        fb_config = dataclasses.replace(fb_config, b_decay_steps=STEPS_PER_STAGE)
    if 'fb_objective' in kwargs:
        fb_config = dataclasses.replace(fb_config, fb_objective=bool(int(kwargs['fb_objective'])))
    fb_config = dataclasses.replace(fb_config, finetune=FINETUNE)
    assert fb_config.fb_objective or fb_config.q_loss, 'fb_objective=0 needs the Q loss (q_loss=True)'
    if FINETUNE != 'none':
        print(f'finetune={FINETUNE}: z = fixed task latent; fb_objective, ortho_coef, lam, mu_b, mu_q, alpha, '
              f'z_task_*, freeze_b, freeze_fb and mix_ratio are ignored')
    assert 0.0 <= fb_config.alpha <= 1.0, 'alpha= must be in [0, 1]'
    assert fb_config.norm_z, 'the method assumes norm_z=True (z on the sphere of radius sqrt(d))'
    Z_DIM = fb_config.z_dim
    RIDGE = fb_config.ridge
    print(fb_config)
    with open(f'{LOGS_FOLDER}/info.txt', 'a') as file:
        file.write(str(fb_config) + '\n')
        file.write(f'tasks={TASKS} steps_per_stage={STEPS_PER_STAGE} n_labeled={N_LABELED} n_big={N_BIG}\n')

    agent = FBAgent(fb_config, obs_dim=STATE_DIM, action_dim=ACTION_DIM)

    random_key = jax.random.PRNGKey(SEED)
    random_key, init_key = jax.random.split(random_key)
    train_state = agent.init(init_key)

    eval_random_state  = {'key': jax.random.PRNGKey(SEED + 10_000)}
    stage_random_state = {'key': jax.random.PRNGKey(SEED + 20_000)}
    def next_eval_key():
        eval_random_state['key'], sub = jax.random.split(eval_random_state['key'])
        return sub
    def next_stage_key():
        stage_random_state['key'], sub = jax.random.split(stage_random_state['key'])
        return sub

    # ── Checkpoint loading (pretrained FB) ────────────────────────────────────────

    def load_checkpoint(ts, path):
        with open(path, 'rb') as f:
            ckpt = pickle.load(f)

        def restore(template, state, what):
            restored = jax.tree_util.tree_map(jnp.asarray, serialization.from_state_dict(template, state))
            def _check(a, b):
                assert a.shape == b.shape, (f"checkpoint mismatch in {what}: expected shape {a.shape}, got {b.shape} "
                                            f"(env_name / network hyper-parameters differ from the pretraining run?)")
            jax.tree_util.tree_map(_check, template, restored)
            return restored

        fb_params    = restore(ts.fb_params, ckpt['fb_params'], 'fb_params')
        actor_params = restore(ts.actor_params, ckpt['actor_params'], 'actor_params')
        return ts.replace(
            fb_params=fb_params,
            fb_target_params=jax.tree_util.tree_map(jnp.copy, fb_params),
            actor_params=actor_params,
            fb_optimizer_state=agent.fb_optimizer.init(fb_params),
            actor_optimizer_state=agent.actor_optimizer.init(actor_params),
        )

    train_state = load_checkpoint(train_state, CHECKPOINT)
    print('loaded pretrained FB from', CHECKPOINT)

    @jax.jit
    def train_chunk(train_state, replay_buffer, stage_data):
        """SCAN_LENGTH consecutive (FB + continual terms) + actor updates."""
        def body(ts, _):
            return agent.update(ts, replay_buffer, stage_data)
        return jax.lax.scan(body, train_state, None, length=SCAN_LENGTH)

    # ── Stage machinery ───────────────────────────────────────────────────────────

    @jax.jit
    def backward_apply(backward_params, obs):
        return agent.backward_net.apply({'params': backward_params}, obs)

    def big_backward(backward_params, obs, chunk=25_000):
        return jnp.concatenate([backward_apply(backward_params, obs[i:i + chunk]) for i in range(0, obs.shape[0], chunk)])

    @jax.jit
    def ft_recon_stats(backward_params, hold_obs, r_scaled, w):
        """Reconstruction with the FIXED readout w of the stage, r_hat = B(s)^T w (scale of rbar), on held-out states.
        Returns (fraction of the energy of r_scaled not explained, correlation(r_scaled, r_hat))."""
        rec  = agent.backward_net.apply({'params': backward_params}, hold_obs) @ w
        eps  = jnp.mean((r_scaled - rec) ** 2) / (jnp.mean(r_scaled ** 2) + 1e-12)
        rc, rr = rec - rec.mean(), r_scaled - r_scaled.mean()
        corr = (rc * rr).sum() / (jnp.sqrt((rc ** 2).sum() * (rr ** 2).sum()) + 1e-12)
        return eps, corr

    def ft_reward_quality(backward_params, task_idx, w_fit, m2):
        """Quality of the reward the fine-tuned critic is trained on (B^T w_fit vs rbar), on EVAL_N fresh dataset states."""
        reward_fn, _, param = benchmarks[task_idx]
        hold_obs, _, _, hold_aux = replay_buffer.sample(next_eval_key(), EVAL_N)
        r_hold = reward_fn(hold_aux, param) / math.sqrt(m2 + 1e-12)
        eps, corr = ft_recon_stats(backward_params, hold_obs, r_hold, w_fit)
        return float(eps), float(corr)

    def fit_backward_to_reward(ts, sd, task_idx, m2):
        """finetune=reward_critic, phase 1: B alone is regressed on the labeled set of the stage with the readout FIXED,
            min mean_i (B(s_i)^T w_fit - rbar(s_i))^2    (the fixed-z experiment, warm-started from the current B).
        F, the actor and w_fit / w_q / z_cond are not touched. Returns ts with the new B (target = copy, fresh Adam)."""
        n   = int(sd.lab_obs.shape[0])
        bs  = min(RFIT_BS, n)
        opt = optax.adam(optax.cosine_decay_schedule(RFIT_LR, RFIT_STEPS))
        params    = ts.fb_params['backward']
        opt_state = opt.init(params)

        @jax.jit
        def run_chunk(params, opt_state, key, lab_obs, lab_r, w_fit):
            def body(carry, k):
                p, s = carry
                idx  = jax.random.randint(k, (bs,), 0, n)
                def loss_fn(p):
                    B = agent.backward_net.apply({'params': p}, lab_obs[idx])
                    return jnp.mean((B @ w_fit - lab_r[idx]) ** 2)
                loss, g = jax.value_and_grad(loss_fn)(p)
                upd, s  = opt.update(g, s, p)
                return (optax.apply_updates(p, upd), s), loss
            (params, opt_state), losses = jax.lax.scan(body, (params, opt_state), jax.random.split(key, RFIT_CHUNK))
            return params, opt_state, losses.mean()

        eps0, corr0 = ft_reward_quality(params, task_idx, sd.w_fit, m2)
        print(f'[reward fit] before: held-out eps={eps0:.4e} corr={corr0:.4f}')
        for c in tqdm(range(RFIT_STEPS // RFIT_CHUNK), desc='reward fit (B only)'):
            params, opt_state, loss = run_chunk(params, opt_state, next_stage_key(), sd.lab_obs, sd.lab_r, sd.w_fit)
            if (c + 1) % 5 == 0:
                tqdm.write(f'  reward fit step {(c + 1) * RFIT_CHUNK}: train loss {float(loss):.4e}')
        eps1, corr1 = ft_reward_quality(params, task_idx, sd.w_fit, m2)
        msg = (f'[reward fit] {RFIT_STEPS} steps, lr={RFIT_LR}, bs={bs}, n_labeled={n}: held-out eps {eps0:.4e} -> {eps1:.4e}, '
               f'corr {corr0:.4f} -> {corr1:.4f}')
        print(msg)
        with open(f'{LOGS_FOLDER}/info.txt', 'a') as file:
            file.write(msg + '\n')

        fb_params = {'forward': ts.fb_params['forward'], 'backward': params}
        return ts.replace(fb_params=fb_params,
                          fb_target_params=jax.tree_util.tree_map(jnp.copy, fb_params),
                          fb_optimizer_state=agent.fb_optimizer.init(fb_params))

    def begin_stage(ts, k, task_idx, latents):
        """theta_old <- theta; Sigma_old; labeled set D_k with normalized rewards; atoms; reset of Adam / target nets."""
        reward_fn, label, param = benchmarks[task_idx]
        old_params = jax.tree_util.tree_map(jnp.copy, ts.fb_params)

        # Sigma_old on a large batch of D (once per stage)
        idx_big   = replay_buffer.sample_indices(next_stage_key(), N_BIG)
        B_old_big = big_backward(old_params['backward'], replay_buffer.obs[idx_big])
        sigma_old = B_old_big.T @ B_old_big / B_old_big.shape[0] + RIDGE * jnp.eye(Z_DIM)
        sigma_old_inv = jnp.linalg.inv(sigma_old)

        # labeled set D_k: n_labeled states ~ rho, fixed for the whole stage; rewards rescaled to unit second moment (no centering)
        rng = np.random.default_rng(SEED * 1000 + k)
        sel = rng.choice(replay_buffer.size, size=N_LABELED, replace=False)
        idx_lab = replay_buffer.valid_idx[jnp.asarray(sel)]
        lab_obs = replay_buffer.obs[idx_lab]
        r_lab   = reward_fn(replay_buffer.aux[idx_lab], param)
        m2      = float(jnp.mean(r_lab ** 2))
        assert m2 > 0, f"task {label}: the reward is identically zero on the {N_LABELED} labeled states"
        lab_r   = r_lab / math.sqrt(m2)
        print(f"[stage {k}] task {label}: m2={m2:.4g}, fraction of non-zero rewards={float((r_lab != 0).mean()):.4f}")

        atoms = jnp.zeros((len(TASKS), Z_DIM))
        if len(latents) > 0:
            atoms = atoms.at[:len(latents)].set(jnp.stack(latents))

        z_run  = jnp.mean(lab_r[:, None] * backward_apply(ts.fb_params['backward'], lab_obs), axis=0)
        z_cond = math.sqrt(Z_DIM) * normalize(z_run)
        sd = StageData(old_params=old_params, sigma_old_inv=sigma_old_inv, lab_obs=lab_obs, lab_r=lab_r,
                       atoms=atoms, n_atoms=jnp.asarray(len(latents), dtype=jnp.int32),
                       z_run=z_run, z_cond=z_cond, w_fit=sigma_old_inv @ z_run, w_q=sigma_old_inv @ z_cond)

        # fresh Adam states; target network = copy of theta_old
        ts = ts.replace(
            fb_target_params=jax.tree_util.tree_map(jnp.copy, ts.fb_params),
            fb_optimizer_state=agent.fb_optimizer.init(ts.fb_params),
            actor_optimizer_state=agent.actor_optimizer.init(ts.actor_params),
        )
        return ts, sd, m2

    def end_stage(ts, sd, latents):
        """z_j <- T z_j for the previous tasks (T between theta_k and theta_{k-1}, fresh large batch); z_k from D_k."""
        new_backward = ts.fb_params['backward']
        if len(latents) > 0:
            idx_big = replay_buffer.sample_indices(next_stage_key(), N_BIG)
            obs_big = replay_buffer.obs[idx_big]
            B_new = big_backward(new_backward, obs_big)
            B_old = big_backward(sd.old_params['backward'], obs_big)
            T = (B_new.T @ B_old / B_new.shape[0]) @ sd.sigma_old_inv
            latents = [T @ z_j for z_j in latents]
        z_k = jnp.mean(sd.lab_r[:, None] * backward_apply(new_backward, sd.lab_obs), axis=0)
        return latents + [z_k]

    # ── Evaluation ────────────────────────────────────────────────────────────────

    eval_envs = []   # pool of env instances, grown on demand and reused across evals

    def rollout_batch(actor_params, zs):
        """Rolls out len(zs) envs in lock-step (one batched, jitted policy call per step).
        Env i is conditioned on zs[i]. Returns aux (n, T, aux_dim)."""
        n = zs.shape[0]
        while len(eval_envs) < n:
            eval_envs.append(make_eval_env())
        envs = eval_envs[:n]

        states = [env_reset(e) for e in envs]
        aux_l = []
        for _ in tqdm(range(EVAL_STEPS), leave=False):
            obs = np.stack(states).astype(np.float32)
            aux_l.append(np.stack([env_aux(e, s) for e, s in zip(envs, states)]))
            actions = np.asarray(agent.act(actor_params, jnp.asarray(obs), zs))
            states  = [env_step(e, a) for e, a in zip(envs, actions)]
        return np.stack(aux_l).transpose(1, 0, 2)

    def rollout_returns(actor_params, items):
        """items: list of (key, task_idx, zs[NUM_EVALS, d]). Returns {key: (mean, std)} of the trajectory returns."""
        out = {}
        for start in range(0, len(items), EVAL_CHUNK):
            chunk = items[start:start + EVAL_CHUNK]
            zs = jnp.concatenate([z for _, _, z in chunk], axis=0)
            aux_traj = rollout_batch(actor_params, zs)
            for j, (key, b, _) in enumerate(chunk):
                reward_fn, label, param = benchmarks[b]
                sl = slice(j * NUM_EVALS, (j + 1) * NUM_EVALS)
                # rewards of the produced trajectories: (NUM_EVALS, EVAL_STEPS) -> (NUM_EVALS,)
                trajectory_rewards = np.asarray(reward_fn(jnp.asarray(aux_traj[sl], dtype=jnp.float32), param)).sum(axis=-1)
                if ENV_NAME == 'antmaze' and 'goal' in label:
                    # goal tasks: 1 if the goal was reached at least once, else 0
                    trajectory_rewards = np.where(trajectory_rewards != -EVAL_STEPS, 1.0, 0.0)
                print(f"{key[0]} {label}: mean {trajectory_rewards.mean():.3f}  std {trajectory_rewards.std(ddof=1):.3f}  "
                      f"{trajectory_rewards.tolist()}")
                out[key] = (float(trajectory_rewards.mean()), float(trajectory_rewards.std(ddof=1)))
        return out

    @jax.jit
    def encode_reward(backward_params, obs, r):
        """Unnormalized encoding of the reward normalized to unit second moment: z = mean(rbar B). Returns (z, m2)."""
        B = agent.backward_net.apply({'params': backward_params}, obs)
        m2 = jnp.mean(r ** 2)
        r_bar = r / jnp.sqrt(m2 + 1e-12)
        return jnp.mean(r_bar[:, None] * B, axis=0), m2

    @jax.jit
    def recon_stats(backward_params, exp_obs, hold_obs, r_scaled, z):
        """Reconstruct r_hat = B^T Sigma^-1 z (Sigma on exp_obs) on held-out states.
        Returns (fraction of the energy of r_scaled not explained, correlation(r_scaled, r_hat))."""
        B_exp  = agent.backward_net.apply({'params': backward_params}, exp_obs)
        cov    = B_exp.T @ B_exp / B_exp.shape[0] + RIDGE * jnp.eye(Z_DIM)
        B_hold = agent.backward_net.apply({'params': backward_params}, hold_obs)
        rec    = B_hold @ jnp.linalg.solve(cov, z)
        eps    = jnp.mean((r_scaled - rec) ** 2) / (jnp.mean(r_scaled ** 2) + 1e-12)
        rc, rr = rec - rec.mean(), r_scaled - r_scaled.mean()
        corr   = (rc * rr).sum() / (jnp.sqrt((rc ** 2).sum() * (rr ** 2).sum()) + 1e-12)
        return eps, corr

    eval_history = []   # for rewards.png

    def run_eval(ts, stage, step, tag, latents, m2s, ft=None, fit=None):
        """fresh: z inferred from fresh 10k labels for every env task (as in the pretraining script). Skipped if eval_fresh=0.
        tracked: z = stored latent z_j (carried through the T's) for the tasks of the stream already learned.
        ft = (task_idx, label, z_cond, w_fit, m2) in the fine-tuning modes: rollouts with the fixed task latent z_cond."""
        start_time = time.time()
        backward_params = ts.fb_params['backward']
        items, stats = [], {}

        for b in (range(N_BENCH) if EVAL_FRESH else []):
            reward_fn, label, param = benchmarks[b]
            zs = []
            for _ in range(NUM_EVALS):
                fit_obs, _, _, fit_aux = replay_buffer.sample(next_eval_key(), EVAL_N)
                z_raw, m2_fit = encode_reward(backward_params, fit_obs, reward_fn(fit_aux, param))
                zs.append(math.sqrt(Z_DIM) * normalize(z_raw))
            hold_obs, _, _, hold_aux = replay_buffer.sample(next_eval_key(), EVAL_N)
            exp_obs = replay_buffer.sample(next_eval_key(), EVAL_N)[0]
            r_hold  = reward_fn(hold_aux, param) / jnp.sqrt(m2_fit + 1e-12)
            eps, corr = recon_stats(backward_params, exp_obs, hold_obs, r_hold, z_raw)   # last fit batch
            stats[('fresh', b)] = (float(eps), float(corr))
            items.append((('fresh', b), b, jnp.stack(zs)))

        for j, z_j in enumerate(latents):
            b = TASK_IDX[j]
            reward_fn, label, param = benchmarks[b]
            hold_obs, _, _, hold_aux = replay_buffer.sample(next_eval_key(), EVAL_N)
            exp_obs = replay_buffer.sample(next_eval_key(), EVAL_N)[0]
            r_hold  = reward_fn(hold_aux, param) / math.sqrt(m2s[j] + 1e-12)   # same scale as rbar_j
            eps, corr = recon_stats(backward_params, exp_obs, hold_obs, r_hold, z_j)
            stats[('tracked', j)] = (float(eps), float(corr))
            z_bar = math.sqrt(Z_DIM) * normalize(z_j)
            items.append((('tracked', j), b, jnp.tile(z_bar[None, :], (NUM_EVALS, 1))))

        if ft is not None:
            ft_b, ft_label, ft_z, ft_w, ft_m2 = ft
            stats[('ft', 0)] = ft_reward_quality(backward_params, ft_b, ft_w, ft_m2)
            items.append((('ft', 0), ft_b, jnp.tile(ft_z[None, :], (NUM_EVALS, 1))))

        returns = rollout_returns(ts.actor_params, items)

        row = {'stage': stage, 'step': step, 'tag': tag}
        if fit is not None:     # continual mode: quality of B^T w_fit (fixed readout of the stage), same quantity as [reward fit]
            fit_b, fit_label, fit_w, fit_m2 = fit
            fit_eps, fit_corr = ft_reward_quality(backward_params, fit_b, fit_w, fit_m2)
            row.update({f'{fit_label}_fit_eps': fit_eps, f'{fit_label}_fit_rew_corr': fit_corr})
        fresh_means = np.full(N_BENCH, np.nan)
        for b, label in (enumerate(LABELS) if EVAL_FRESH else []):
            m, s = returns[('fresh', b)]
            eps, corr = stats[('fresh', b)]
            row.update({f'{label}_fresh_mean': m, f'{label}_fresh_std': s, f'{label}_fresh_rew_corr': corr, f'{label}_fresh_eps': eps})
            fresh_means[b] = m
        tracked_means = []
        for j in range(len(latents)):
            label = TASKS[j]
            m, s = returns[('tracked', j)]
            eps, corr = stats[('tracked', j)]
            row.update({f'{label}_tracked_mean': m, f'{label}_tracked_std': s, f'{label}_tracked_rew_corr': corr, f'{label}_tracked_eps': eps})
            tracked_means.append(m)
        if ft is not None:
            m, s = returns[('ft', 0)]
            eps, corr = stats[('ft', 0)]
            row.update({f'{ft_label}_ft_mean': m, f'{ft_label}_ft_std': s, f'{ft_label}_ft_rew_corr': corr,
                        f'{ft_label}_ft_eps': eps})
            print(f'[{tag}] {ft_label} fine-tuned: return {m:.2f} +/- {s:.2f}   reward eps={eps:.4e} corr={corr:.4f}')
        row['mean_fresh_all']     = float(fresh_means.mean())
        row['mean_fresh_tasks']   = float(fresh_means[TASK_IDX].mean())
        row['mean_tracked_tasks'] = float(np.mean(tracked_means)) if tracked_means else float('nan')
        row['time'] = time.time() - start_time
        eval_logger.log(row)
        print({k: v for k, v in row.items() if k.startswith('mean_')})

        eval_history.append(row)
        plot_rewards()

    def plot_rewards():
        """One column per env task. Row 1: return, mean +/- std over the episodes (fresh z, and stored latent for stream tasks).
        Row 2: reward-reconstruction correlation. Vertical lines = stage ends."""
        n = len(LABELS)
        fig, axs = plt.subplots(2, n, figsize=(n * 4.5, 8), squeeze=False)
        xs = np.arange(len(eval_history))
        end_xs = [x for x, r in zip(xs, eval_history) if r['tag'] == 'end']

        def series(label, kind, suffix):
            return np.array([r.get(f'{label}_{kind}_{suffix}', np.nan) for r in eval_history], dtype=float)

        for i, label in enumerate(LABELS):
            in_stream = label in TASKS
            kinds = [('fresh', 'fresh latent (zero-shot)')]
            if in_stream:
                kinds.append(('tracked', 'stored latent (carried through T)'))
            if in_stream and FINETUNE != 'none':
                kinds.append(('ft', f'fine-tuned ({FINETUNE}, fixed task z)'))

            # row 1: return with a +/- 1 std band
            ax = axs[0, i]
            for kind, name in kinds:
                m, s = series(label, kind, 'mean'), series(label, kind, 'std')
                line, = ax.plot(xs, m, label=name)
                ax.fill_between(xs, m - s, m + s, color=line.get_color(), alpha=0.2)
            # row 2: correlation
            ax2 = axs[1, i]
            for kind, name in kinds:
                ax2.plot(xs, series(label, kind, 'rew_corr'), label=name)
            ax2.set_ylim(-0.1, 1.05)

            for a, ylabel in ((ax, 'return (mean ± std over episodes)'), (ax2, 'reward reconstruction correlation')):
                for x in end_xs:
                    a.axvline(x, color='gray', alpha=0.3)
                a.set_title(label + (' (learned task)' if in_stream else ''))
                a.set_xlabel('evaluation index')
                a.set_ylabel(ylabel)
                a.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(f"{LOGS_FOLDER}/rewards_and_correlations.png")
        plt.close()

    def save_agent(ts, path, latents, m2s, stage):
        state = {
            'fb_params':        ts.fb_params,
            'fb_target_params': ts.fb_target_params,
            'actor_params':     ts.actor_params,
        }
        state_dict = serialization.to_state_dict(jax.device_get(state))
        state_dict['latents'] = np.asarray(jnp.stack(latents)) if latents else np.zeros((0, Z_DIM))   # z_j of the tasks seen
        state_dict['m2s']     = np.asarray(m2s, dtype=np.float64)
        state_dict['tasks']   = TASKS[:stage]
        with open(path, "wb") as params_file:
            pickle.dump(state_dict, params_file)

    # ── Continual training ────────────────────────────────────────────────────────

    latents, m2s = [], []      # stored z_j (d numbers per task) and m2_j (evaluation only)

    if EVAL_BASELINE:
        run_eval(train_state, 0, 0, 'baseline', [], [])

    for k, (task, task_idx) in enumerate(zip(TASKS, TASK_IDX), start=1):
        print(f"\n===== Stage {k}/{len(TASKS)}: task '{task}' ({len(latents)} previous task(s)) =====")
        train_state, stage_data, m2_k = begin_stage(train_state, k, task_idx, latents)

        ft_info = None
        fit_info = (task_idx, task, stage_data.w_fit, m2_k) if FINETUNE == 'none' else None   # fixed-readout quality of B on this task
        if FINETUNE != 'none':
            if FINETUNE == 'reward_critic':
                train_state = fit_backward_to_reward(train_state, stage_data, task_idx, m2_k)
            ft_info = (task_idx, task, stage_data.z_cond, stage_data.w_fit, m2_k)
            # step 0 of the fine-tuning: zero-shot policy with z_cond (+ quality of the reward the critic will use)
            run_eval(train_state, k, 0, 'ft_start', [], [], ft_info)
            if REINIT_CRITIC:
                # F and the actor from scratch (B kept); target = copy of the new F; fresh Adam states
                fresh = agent.init(next_stage_key())
                fb_params = {'forward': fresh.fb_params['forward'], 'backward': train_state.fb_params['backward']}
                train_state = train_state.replace(
                    fb_params=fb_params,
                    fb_target_params=jax.tree_util.tree_map(jnp.copy, fb_params),
                    actor_params=fresh.actor_params,
                    fb_optimizer_state=agent.fb_optimizer.init(fb_params),
                    actor_optimizer_state=agent.actor_optimizer.init(fresh.actor_params))
                print('reinit_critic=1: F and the actor re-initialised (B kept)')

        for chunk in tqdm(range(CHUNKS_PER_STAGE), desc=f'stage {k}'):
            start_time = time.time()
            train_state, current_metrics = train_chunk(train_state, replay_buffer, stage_data)
            current_metrics = jax.device_get(current_metrics)
            timelapse = time.time() - start_time

            step_in_stage = (chunk + 1) * SCAN_LENGTH
            row = {'stage': k, 'step': step_in_stage, 'time': float(timelapse)}
            row.update({m: float(np.mean(current_metrics[m])) for m in TRAIN_METRICS})
            train_logger.log(row)

            is_last = (chunk + 1) == CHUNKS_PER_STAGE
            if EVAL_EACH_STEPS > 0 and (chunk + 1) % EVAL_EACH_STEPS == 0 and not is_last:
                # mid-stage: only fresh zero-shot (the stored latents are still expressed in the coordinates of theta_old)
                run_eval(train_state, k, step_in_stage, 'mid', [], [], ft_info, fit_info)

        latents = end_stage(train_state, stage_data, latents)
        m2s.append(m2_k)

        save_agent(train_state, f"{MODEL_SAVE_FOLDER}/fb_stage_{k}_{task}.pickle", latents, m2s, k)
        run_eval(train_state, k, STEPS_PER_STAGE, 'end', latents, m2s, ft_info, fit_info)