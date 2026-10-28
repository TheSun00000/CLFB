"""
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
    seed=0 discount=0.95
    steps_per_stage=100000   number of updates per stage (multiple of scan_length)
    scan_length=1000         updates per jitted chunk (= logging granularity)
    n_labeled=10000          size of the labeled set D_k of each task (sampled once per stage, fixed during the stage)
    lam=1.0 mu_b=1.0 mu_q=0.01 alpha=0.5     lambda, mu_B, mu_Q, mixture weight of the atoms in nu
    n_est=4096               number of states of D used at every step to estimate Sigma_theta and A_theta
    n_big=50000              number of states used once per stage for Sigma_old, and at the end of the stage for T_theta
    ridge=1e-5               ridge added to every covariance before inversion
    cross_fit=0              1: L_rec with cross-fitting (z_hat on one half of D_k, loss on the other)
    bc_coef=0.0              TD3+BC weight of the DDPG actor (0 -> plain -Q.mean())
    eval_chunk=              number of tasks rolled out in parallel (each needs NUM_EVALS env instances)
    eval_each_steps=0        if > 0: also evaluate (fresh zero-shot only) every this many chunks inside a stage
    eval_baseline=1          evaluate the pretrained checkpoint (stage 0) before the first stage

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
    training_log.csv : per chunk, FB metrics + l_rec, r_b, r_q, z_shrink (= mean ||T z|| / ||z||), total_loss
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

from _utils.logs import parse_kwargs


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

    SEED = int(kwargs['seed']) if 'seed' in kwargs else 0

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

    # ── Logging ───────────────────────────────────────────────────────────────────
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    exp_name = f'FBcontinual-{ENV_NAME}-' + '+'.join(TASKS)
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
                     "target_M", "M1", "F1", "B", "B_norm", "z_norm", "next_Q",
                     "b_cov_eig_min", "b_cov_eig_max",
                     "actor_loss", "q", "bc_dist",
                     "total_loss", "l_rec", "r_b", "r_q", "z_shrink"]
    train_logger = CSVLogger(
        filepath=f"{LOGS_FOLDER}/training_log.csv",
        fieldnames=["stage", "step"] + TRAIN_METRICS + ["time"],
    )

    with open(f'{LOGS_FOLDER}/info.txt', 'a') as file:
        file.write(str(kwargs) + '\n')

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
    NUM_EVALS = 5
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
        lr:                  float = 3e-4
        lr_coef:             float = 1.0     # backward net lr = lr_coef * lr
        fb_target_tau:       float = 0.005
        hidden_dim:          int   = 1024
        backward_hidden_dim: int   = 526     # (sic) same value as in the torch script
        feature_dim:         int   = 512
        z_dim:               int   = 50
        stddev:              float = 0.2
        stddev_clip:         float = 0.3
        batch_size:          int   = 1024
        ortho_coef:          float = 10.0
        mix_ratio:           float = 0.5
        norm_z:              bool  = True
        q_loss:              bool  = True
        q_loss_coef:         float = 0.01
        add_trunk:           bool  = False
        discount:            float = 0.95
        # --- continual terms ---
        lam:                 float = 1.0      # weight of L_rec
        mu_b:                float = 1.0      # weight of R^B
        mu_q:                float = 0.01     # weight of R^Q (not dimensionless: it is a squared error on values)
        alpha:               float = 0.5      # weight of the atoms in nu
        n_est:               int   = 4096     # states of D per step to estimate Sigma_theta, A_theta
        ridge:               float = 1e-5     # ridge on every covariance before inversion
        cross_fit:           bool  = False    # cross-fitted L_rec

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
            self.fb_optimizer = optax.multi_transform(
                {'forward': optax.adam(config.lr), 'backward': optax.adam(config.lr * config.lr_coef)},
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

        # ── update ────────────────────────────────────────────────────────────

        @partial(jax.jit, static_argnames=("self",))
        def update(self, ts: FBTrainingState, replay_buffer: OfflineReplayBuffer, sd: StageData):
            cfg = self.cfg
            bs  = cfg.batch_size
            d   = cfg.z_dim
            discount = cfg.discount

            (random_key, k_batch, k_z, k_perm, k_mix, k_next, k_act,
             k_est, k_nu0, k_nu1, k_nu2) = jax.random.split(ts.random_key, 11)
            obs, action, next_obs, _ = replay_buffer.sample(k_batch, bs)

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

            # ── latents of the retention losses: z ~ nu = (1 - alpha) * uniform sphere + alpha * atoms ──
            z_sphere = self.sample_z(k_nu0, bs)
            n_at     = jnp.maximum(sd.n_atoms, 1)
            at_idx   = jnp.minimum((jax.random.uniform(k_nu1, (bs,)) * n_at).astype(jnp.int32), n_at - 1)
            z_atom   = math.sqrt(d) * normalize(sd.atoms[at_idx])
            use_atom = (jax.random.uniform(k_nu2, (bs, 1)) < cfg.alpha) & (sd.n_atoms > 0)
            z_ret    = jax.lax.stop_gradient(jnp.where(use_atom, z_atom, z_sphere))

            off_diag = 1.0 - jnp.eye(bs)
            n_off    = off_diag.sum()

            # ── bootstrap term of the FB loss (classical: through the actor) ──
            target_B = self.backward_net.apply({'params': ts.fb_target_params['backward']}, next_obs)
            next_mu     = self.actor_net.apply({'params': ts.actor_params}, next_obs, z)
            next_action = jax.lax.stop_gradient(
                sample_truncated_normal(next_mu, k_next, cfg.stddev, cfg.stddev_clip))
            target_F1, target_F2 = self.forward_net.apply(
                {'params': ts.fb_target_params['forward']}, next_obs, z, next_action)
            # target_M = jax.lax.stop_gradient(jnp.minimum(target_F1 @ target_B.T, target_F2 @ target_B.T))
            target_M = jax.lax.stop_gradient(0.5 * (target_F1 @ target_B.T + target_F2 @ target_B.T))
            next_Q   = jax.lax.stop_gradient(jnp.minimum((target_F1 * z).sum(-1), (target_F2 * z).sum(-1)))

            # ── FB update: L_FB + lam * L_rec + mu_b * R^B + mu_q * R^Q ───────
            def fb_loss_fn(fb_params):
                # ---- L_FB (identical to the pretraining script)
                F1, F2 = self.forward_net.apply({'params': fb_params['forward']}, obs, z, action)
                B      = self.backward_net.apply({'params': fb_params['backward']}, next_obs)
                M1, M2 = F1 @ B.T, F2 @ B.T

                fb_offdiag = 0.5 * sum(
                    (((M - discount * target_M) ** 2) * off_diag).sum() / n_off for M in (M1, M2))
                fb_diag = -sum(jnp.diag(M).mean() for M in (M1, M2))
                fb_loss = fb_offdiag + fb_diag

                q_loss = jnp.zeros(())
                if cfg.q_loss:
                    B_sg    = jax.lax.stop_gradient(B)
                    cov     = B_sg.T @ B_sg / B_sg.shape[0]
                    inv_cov = jnp.linalg.inv(cov)
                    implicit_reward = ((B_sg @ inv_cov) * z).sum(axis=-1)
                    target_Q = jax.lax.stop_gradient(implicit_reward + discount * next_Q)
                    Q1, Q2   = (F1 * z).sum(-1), (F2 * z).sum(-1)
                    q_loss   = jnp.mean((Q1 - target_Q) ** 2) + jnp.mean((Q2 - target_Q) ** 2)
                    fb_loss  = fb_loss + cfg.q_loss_coef * q_loss

                Cov               = B @ B.T
                orth_loss_diag    = -2 * jnp.diag(Cov).mean()
                orth_loss_offdiag = ((Cov ** 2) * off_diag).sum() / n_off
                orth_loss         = orth_loss_offdiag + orth_loss_diag
                fb_loss           = fb_loss + cfg.ortho_coef * orth_loss

                # ---- batch estimates of Sigma_theta, A_theta, T_theta, M_theta (n_est states of D)
                B_new_est = self.backward_net.apply({'params': fb_params['backward']}, obs_est)
                B_old_est = jax.lax.stop_gradient(
                    self.backward_net.apply({'params': sd.old_params['backward']}, obs_est))
                Sigma_th, T, M = old_family_matrices(B_new_est, B_old_est, sd.sigma_old_inv, cfg.ridge)

                # ---- L_rec on the labeled set (whole set at every step, detached coefficient)
                B_lab = self.backward_net.apply({'params': fb_params['backward']}, sd.lab_obs)
                l_rec = reconstruction_loss(B_lab, sd.lab_r, Sigma_th, cfg.cross_fit)

                # ---- R^B (closed form, differentiated through A_theta and Sigma_theta)
                r_b = retention_basis_loss(z_ret, M, sd.sigma_old_inv)

                # ---- R^Q: values of the re-encoded old family r_z, matched head by head. z' = sg(T z).
                z_new     = jax.lax.stop_gradient(z_ret @ T.T)               # rows: (T z)^T
                z_new_bar = math.sqrt(d) * normalize(z_new)
                nF1, nF2  = self.forward_net.apply({'params': fb_params['forward']}, obs, z_new_bar, action)
                oF1, oF2  = jax.lax.stop_gradient(self.forward_net.apply(
                    {'params': sd.old_params['forward']}, obs, z_ret, action))
                r_q = 0.5 * (jnp.mean(((nF1 * z_new).sum(-1) - (oF1 * z_ret).sum(-1)) ** 2)
                             + jnp.mean(((nF2 * z_new).sum(-1) - (oF2 * z_ret).sum(-1)) ** 2))
                z_shrink = jnp.mean(jnp.linalg.norm(z_new, axis=-1) / jnp.linalg.norm(z_ret, axis=-1))

                total = fb_loss + cfg.lam * l_rec + cfg.mu_b * r_b + cfg.mu_q * r_q

                eye_diff = B.T @ B / B.shape[0] - jnp.eye(B.shape[1])
                b_eigs = jnp.linalg.eigvalsh(jax.lax.stop_gradient(B).T @ jax.lax.stop_gradient(B) / B.shape[0])
                metrics = {
                    'target_M':          target_M.mean(),
                    'M1':                M1.mean(),
                    'F1':                F1.mean(),
                    'B':                 B.mean(),
                    'B_norm':            jnp.linalg.norm(B, axis=-1).mean(),
                    'z_norm':            jnp.linalg.norm(z, axis=-1).mean(),
                    'fb_loss':           fb_loss,
                    'fb_diag':           fb_diag,
                    'fb_offdiag':        fb_offdiag,
                    'q_loss':            q_loss,
                    'orth_loss':         orth_loss,
                    'orth_loss_diag':    orth_loss_diag,
                    'orth_loss_offdiag': orth_loss_offdiag,
                    'orth_linf':         jnp.max(jnp.abs(eye_diff)),
                    'orth_l2':           jnp.linalg.norm(eye_diff) / math.sqrt(B.shape[1]),
                    'next_Q':            next_Q.mean(),
                    'b_cov_eig_min':     b_eigs[0],
                    'b_cov_eig_max':     b_eigs[-1],
                    'total_loss':        total,
                    'l_rec':             l_rec,
                    'r_b':               r_b,
                    'r_q':               r_q,
                    'z_shrink':          z_shrink,
                }
                return total, metrics

            (_, metrics), fb_grads = jax.value_and_grad(fb_loss_fn, has_aux=True)(ts.fb_params)
            fb_updates, fb_optimizer_state = self.fb_optimizer.update(
                fb_grads, ts.fb_optimizer_state, ts.fb_params)
            fb_params = optax.apply_updates(ts.fb_params, fb_updates)

            # ── Actor update (DDPG / TD3-like, uses the freshly updated forward net, as in torch) ──
            # maximize F(s, pi(s,z), z)^T z. With bc_coef > 0: TD3+BC (scale-free Q term, the normalizer is not differentiated).
            def actor_loss_fn(actor_params):
                mu = self.actor_net.apply({'params': actor_params}, obs, z)
                a  = sample_truncated_normal(mu, k_act, cfg.stddev, cfg.stddev_clip)
                F1, F2 = self.forward_net.apply({'params': fb_params['forward']}, obs, z, a)
                Q = jnp.minimum((F1 * z).sum(-1), (F2 * z).sum(-1))
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
    for _k, _typ in (('discount', float), ('bc_coef', float), ('lam', float), ('mu_b', float), ('mu_q', float),
                     ('alpha', float), ('n_est', int), ('ridge', float)):
        if _k in kwargs:
            fb_config = dataclasses.replace(fb_config, **{_k: _typ(kwargs[_k])})
    if 'cross_fit' in kwargs:
        fb_config = dataclasses.replace(fb_config, cross_fit=bool(int(kwargs['cross_fit'])))
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

        sd = StageData(old_params=old_params, sigma_old_inv=sigma_old_inv, lab_obs=lab_obs, lab_r=lab_r,
                       atoms=atoms, n_atoms=jnp.asarray(len(latents), dtype=jnp.int32))

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

    def run_eval(ts, stage, step, tag, latents, m2s):
        """fresh: z inferred from fresh 10k labels for every env task (as in the pretraining script).
        tracked: z = stored latent z_j (carried through the T's) for the tasks of the stream already learned."""
        start_time = time.time()
        backward_params = ts.fb_params['backward']
        items, stats = [], {}

        for b in range(N_BENCH):
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

        returns = rollout_returns(ts.actor_params, items)

        row = {'stage': stage, 'step': step, 'tag': tag}
        fresh_means = np.zeros(N_BENCH)
        for b, label in enumerate(LABELS):
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
        row['mean_fresh_all']     = float(fresh_means.mean())
        row['mean_fresh_tasks']   = float(fresh_means[TASK_IDX].mean())
        row['mean_tracked_tasks'] = float(np.mean(tracked_means)) if tracked_means else float('nan')
        row['time'] = time.time() - start_time
        eval_logger.log(row)
        print({k: v for k, v in row.items() if k.startswith('mean_')})

        eval_history.append(row)
        plot_rewards()

    def plot_rewards():
        """One panel per task of the stream: fresh and tracked returns at every evaluation; vertical lines = stage ends."""
        n = len(TASKS)
        num_cols = min(4, n)
        num_rows = n // num_cols + int(n % num_cols != 0)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 4.5))
        axs = np.atleast_1d(axs).flatten()
        xs = np.arange(len(eval_history))
        for i, label in enumerate(TASKS):
            axs[i].plot(xs, [r.get(f'{label}_fresh_mean', np.nan) for r in eval_history], label='fresh z')
            axs[i].plot(xs, [r.get(f'{label}_tracked_mean', np.nan) for r in eval_history], label='tracked z_j')
            for x, r in zip(xs, eval_history):
                if r['tag'] == 'end':
                    axs[i].axvline(x, color='gray', alpha=0.3)
            axs[i].set_title(label)
            axs[i].legend()
        plt.tight_layout()
        plt.savefig(f"{LOGS_FOLDER}/rewards.png")
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
                run_eval(train_state, k, step_in_stage, 'mid', [], [])

        latents = end_stage(train_state, stage_data, latents)
        m2s.append(m2_k)

        save_agent(train_state, f"{MODEL_SAVE_FOLDER}/fb_stage_{k}_{task}.pickle", latents, m2s, k)
        run_eval(train_state, k, STEPS_PER_STAGE, 'end', latents, m2s)
