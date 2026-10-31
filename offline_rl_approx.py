"""
Single-reward offline TD3 baseline, derived from fb.py, without anything FB-specific.

Same envs / datasets / eval as fb.py, but ONE reward function per env instead of the zero-shot family:
    F(s, a, z) -> Q(s, a)   (scalar, two heads)
    no B(s), no z (no z sampling, no orthogonality loss, no task inference)

Usage:
    python offline_rl_approx.py env_name=walker task=walk
    optional: task=<label or index in the benchmark list of the env>   (default: 0, the available labels are printed)
              seed=<int> (default: random, logged in info.txt) discount=0.95
              bc_coef=0.0     (0 -> plain TD3 actor loss -Q.mean(); > 0 -> TD3+BC)
              n_transitions=100000 data_seed=0      (uniform subsample of the dataset, default: full dataset)
              eval_each_steps=10

The Q bootstrap goes through the actor (TD3-like):
    L_Q = sum_heads (Q_h(s, a) - (r(s') + discount * min_heads Q_bar(s', a'))^2,  a' = pi(s') + clipped noise
The reward of the transition (s, a, s') is r(s'), as in fb.py where the reward is read on B(next_obs).
No terminal handling (same as fb.py).

Actor: maximize Q(s, pi(s)). With bc_coef > 0 this becomes TD3+BC:
    -Q.mean() / stopgrad(|Q|.mean()) + bc_coef * mean((pi(s) - a_data)^2)

Learned reward (offline_rl_approx.py)
    The policy can be trained on an APPROXIMATION of the reward instead of the true reward. The approximator is first fit
    offline on `approx_n` states of the dataset (supervised by the true reward), then its prediction replaces the reward of
    every dataset state. Evaluation (rollouts) is ALWAYS on the true reward, so the learning curves are comparable with
    reward_approx=true (the reference behaviour).
        reward_approx=true | lrec_detach | lrec_full | fixed_z | mlp          (default: true)
            lrec_detach : B(s) = sqrt(d) normalize(MLP(s)), r_hat = B^T Sigma^-1 z (z = mean(rbar B)), L_rec with detached coefficient
            lrec_full   : same, gradient through Sigma^-1 z
            fixed_z     : r_hat = B(s)^T z_fixed, z_fixed = sqrt(d) * normalize(random vector)
            mlp         : MLP(s) regressed directly on the reward
            fb_pretrained : B of the pretrained FB (checkpoint=), r_hat = B^T w, w = Sigma^-1 z (z = mean(rbar B), on the
                            approximator states). No training: reference for finetune=critic of fb_continual.py
            fb_finetuned  : same B and w, then B regressed with w FIXED (warm-started fixed-z experiment):
                            reference for finetune=reward_critic of fb_continual.py (same reward, critic from scratch)
        checkpoint=          pretrained FB pickle (required for fb_pretrained / fb_finetuned; approx_hidden / approx_zdim
                             must match it: 526 / 50 by default)
        approx_n=            number of dataset states used to fit the approximator (default: all the states of the dataset)
        approx_steps=30000 approx_lr=1e-3 approx_bs=4096 approx_est=4096 approx_decay=1 (cosine lr decay)
        approx_hidden=526 approx_zdim=50 approx_ridge=1e-5 approx_ortho=0.0 (optional orthonormality term on B, lrec only)
        approx_seed=0
    rbar = r / sqrt(mean r^2) is what the approximators regress; r_hat is multiplied back by that scale.
    Logged: the held-out quality of the approximation (corr, R2, eps) in info.txt, and reward_approx_scatter.png.
    Usage:
        python offline_rl_approx.py env_name=walker task=walk reward_approx=true
        python offline_rl_approx.py env_name=walker task=walk reward_approx=lrec_full approx_n=100000
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
import matplotlib.patches as patches

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


if __name__ == '__main__':
    #######################################################################################

    kwargs = parse_kwargs(sys.argv[1:])
    print(kwargs)
    print(jax.devices())

    assert 'env_name' in kwargs, "env_name= is required"
    ENV_NAME = kwargs['env_name']
    assert ENV_NAME in ['cheetah', 'walker', 'quadruped', 'antmaze', 'pointmaze']

    # ── Config ────────────────────────────────────────────────────────────────────
    SEED = resolve_seed(kwargs)
    N_TRANSITIONS = int(kwargs['n_transitions']) if 'n_transitions' in kwargs else None
    DATA_SEED     = int(kwargs['data_seed']) if 'data_seed' in kwargs else 0

    # ── Reward approximator flags ─────────────────────────────────────────────────
    REWARD_APPROX = str(kwargs['reward_approx']) if 'reward_approx' in kwargs else 'true'
    assert REWARD_APPROX in ['true', 'lrec_detach', 'lrec_full', 'fixed_z', 'mlp', 'fb_pretrained', 'fb_finetuned'], \
        "reward_approx= must be true, lrec_detach, lrec_full, fixed_z, mlp, fb_pretrained or fb_finetuned"
    CHECKPOINT = str(kwargs['checkpoint']) if 'checkpoint' in kwargs else None
    if REWARD_APPROX in ['fb_pretrained', 'fb_finetuned']:
        assert CHECKPOINT is not None, f"reward_approx={REWARD_APPROX} needs checkpoint=<pretrained FB pickle>"
    APPROX_N      = int(kwargs['approx_n']) if 'approx_n' in kwargs else None      # None = all the states of the dataset
    APPROX_STEPS  = int(kwargs['approx_steps']) if 'approx_steps' in kwargs else 30_000
    APPROX_LR     = float(kwargs['approx_lr']) if 'approx_lr' in kwargs else 1e-3
    APPROX_BS     = int(kwargs['approx_bs']) if 'approx_bs' in kwargs else 4096
    APPROX_EST    = int(kwargs['approx_est']) if 'approx_est' in kwargs else 4096  # states per step to estimate Sigma
    APPROX_DECAY  = bool(int(kwargs['approx_decay'])) if 'approx_decay' in kwargs else True
    APPROX_HIDDEN = int(kwargs['approx_hidden']) if 'approx_hidden' in kwargs else 526
    APPROX_ZDIM   = int(kwargs['approx_zdim']) if 'approx_zdim' in kwargs else 50
    APPROX_RIDGE  = float(kwargs['approx_ridge']) if 'approx_ridge' in kwargs else 1e-5
    APPROX_ORTHO  = float(kwargs['approx_ortho']) if 'approx_ortho' in kwargs else 0.0
    APPROX_SEED   = int(kwargs['approx_seed']) if 'approx_seed' in kwargs else 0

    # ── Logging ───────────────────────────────────────────────────────────────────
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    exp_name = f'TD3-{ENV_NAME}'
    if 'task' in kwargs:
        exp_name += f'-{kwargs["task"]}'
    if N_TRANSITIONS is not None:
        exp_name += f'-n{N_TRANSITIONS}'
    if REWARD_APPROX != 'true':
        exp_name += f'-rew_{REWARD_APPROX}'
        if APPROX_N is not None:
            exp_name += f'-an{APPROX_N}'

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

    TRAIN_METRICS = ["q_loss", "actor_loss",
                     "q_data", "q_pi", "q_gap", "bc_dist",
                     "next_Q", "target_Q", "reward"]
    train_logger = CSVLogger(
        filepath=f"{LOGS_FOLDER}/training_log.csv",
        fieldnames=["step"] + TRAIN_METRICS + ["time"],
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

    POINTMAZE_MEDIUM_MAP = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]])

    def add_pointmaze_walls(ax, maze_unit=4.0):
        # ogbench: cell (i, j) is centered at (maze_unit * j - 4, maze_unit * i - 4)
        for i, j in zip(*np.where(POINTMAZE_MEDIUM_MAP == 1)):
            cx, cy = maze_unit * j - 4, maze_unit * i - 4
            ax.add_patch(patches.Rectangle((cx - maze_unit / 2, cy - maze_unit / 2), maze_unit, maze_unit,
                                           linewidth=0, facecolor='gray'))

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

        def env_physics(e):
            return e.physics.get_state()

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

        def env_physics(e):
            return None

        def add_largest_maze_walls(ax):
            maze_optim = [
                (1, 1, 1, 2), (0, 4, 2, 1), (3, 1, 1, 4), (5, 0, 1, 1), (4, 2, 3, 1), (1, 6, 3, 1),
                (4, 4, 2, 1), (5, 6, 2, 1), (1, 8, 1, 1), (3, 7, 1, 2), (5, 8, 1, 2),
            ]
            block_size = 0.025 * 80
            height, width = 7, 10
            torso_x, torso_y = (width - 1) * block_size, (height - 1) * block_size
            for (y, x, w, h) in maze_optim:
                x = x * block_size * 2 - torso_x + (h - 1) * block_size - h * block_size + 18
                y = y * block_size * 2 - torso_y + (w - 1) * block_size - w * block_size + 12
                h, w = h * block_size * 2, w * block_size * 2
                ax.add_patch(patches.Rectangle((x, y), h, w, linewidth=2, edgecolor='gray', facecolor='gray'))

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

        def env_physics(e):
            return None

    # ── Choice of the single reward function ──────────────────────────────────────
    TASK_LABELS = [label for _, label, _ in benchmarks]
    print('available tasks:', {i: l for i, l in enumerate(TASK_LABELS)})
    TASK = str(kwargs['task']) if 'task' in kwargs else '0'
    if TASK in TASK_LABELS:
        TASK_ID = TASK_LABELS.index(TASK)
    else:
        assert TASK.lstrip('-').isdigit(), f"task={TASK} is neither a label nor an index, available: {TASK_LABELS}"
        TASK_ID = int(TASK)
    assert 0 <= TASK_ID < len(benchmarks), f"task index {TASK_ID} out of range, available: {TASK_LABELS}"
    reward_fn, TASK_LABEL, TASK_PARAM = benchmarks[TASK_ID]
    print('TASK:', TASK_ID, TASK_LABEL)

    # Reward of every dataset state (computed once, in chunks). The reward of the transition (s_i, a_i, s_{i+1}) is
    # rewards[i + 1], i.e. r(s'), consistent with fb.py where the reward is read through B(next_obs).
    _CH = 1_000_000
    dataset_rewards = np.concatenate([
        np.asarray(reward_fn(jnp.asarray(dataset_aux[s:s + _CH], dtype=jnp.float32), TASK_PARAM))
        for s in range(0, len(dataset_aux), _CH)
    ]).astype(np.float32)
    print(f'dataset rewards: mean={dataset_rewards.mean():.4f} std={dataset_rewards.std():.4f} '
          f'min={dataset_rewards.min():.4f} max={dataset_rewards.max():.4f} '
          f'frac_nonzero={(dataset_rewards != 0).mean():.4f}')

    NUM_EVALS = 5
    print(NUM_TRAJECTORIES, TRAJECTORY_LEN, NUM_TRAJECTORIES * TRAJECTORY_LEN)

    BATCH_SIZE      = 1024
    EVAL_EACH_STEPS = 10      # in outer iterations (1 outer iteration = SCAN_LENGTH updates)
    SCAN_LENGTH     = 1_000
    NUM_OUTER_STEPS = 1_000   # 1M updates in total
    PLOT_EACH_EVALS = 5
    SAVE_EACH_STEPS = 100
    if 'eval_each_steps' in kwargs:
        EVAL_EACH_STEPS = int(kwargs['eval_each_steps'])

    eval_logger = CSVLogger(
        filepath=f"{LOGS_FOLDER}/eval_log.csv",
        fieldnames=["step", f"{TASK_LABEL}_mean", f"{TASK_LABEL}_std", "time"],
    )

    # ── Replay buffer ─────────────────────────────────────────────────────────────

    class OfflineReplayBuffer(flax.struct.PyTreeNode):
        obs:       jnp.ndarray
        actions:   jnp.ndarray
        aux:       Optional[jnp.ndarray]
        rewards:   jnp.ndarray            # r(s_i) for every dataset state
        valid_idx: jnp.ndarray
        size:      int = flax.struct.field(pytree_node=False)

        @classmethod
        def init(cls, observations, actions, trajectory_len, rewards, aux=None,
                 n_transitions=None, subsample_seed=0) -> "OfflineReplayBuffer":
            n_total = (len(observations) // trajectory_len) * trajectory_len
            idx     = np.arange(n_total)
            # a transition (i, i+1) is valid iff i is not the last state of its trajectory
            valid_idx = idx[(idx % trajectory_len) != trajectory_len - 1]
            # Optional uniform subsample of transitions. The obs / actions arrays are kept whole (the next state is
            # obs[i + 1]); only the set of indices that can be sampled is restricted.
            if n_transitions is not None and n_transitions < len(valid_idx):
                rng = np.random.default_rng(subsample_seed)
                valid_idx = np.sort(rng.choice(valid_idx, size=n_transitions, replace=False))

            return cls(
                obs=jnp.asarray(observations[:n_total], dtype=jnp.float32),
                actions=jnp.asarray(actions[:n_total], dtype=jnp.float32),
                aux=None if aux is None else jnp.asarray(aux[:n_total], dtype=jnp.float32),
                rewards=jnp.asarray(rewards[:n_total], dtype=jnp.float32),
                valid_idx=jnp.asarray(valid_idx, dtype=jnp.int32),
                size=len(valid_idx),
            )

        def sample_indices(self, random_key, sample_size: int):
            idx = jax.random.randint(random_key, shape=(sample_size,), minval=0, maxval=self.size)
            return self.valid_idx[idx]

        def sample(self, random_key, sample_size: int):
            """Returns (obs, actions, next_obs, reward) with reward = r(next_obs)."""
            i = self.sample_indices(random_key, sample_size)
            return self.obs[i], self.actions[i], self.obs[i + 1], self.rewards[i + 1]

        def sample_states(self, random_key, sample_size: int):
            """Returns (obs, aux) of random dataset states (for the plots)."""
            i = self.sample_indices(random_key, sample_size)
            return self.obs[i], self.aux[i]

    replay_buffer = OfflineReplayBuffer.init(dataset_obs, dataset_actions, TRAJECTORY_LEN, rewards=dataset_rewards,
                                             aux=dataset_aux, n_transitions=N_TRANSITIONS, subsample_seed=DATA_SEED)
    print('transitions available for training:', replay_buffer.size)
    del dataset_obs, dataset_actions, dataset_aux, dataset_rewards

    # ── Neural networks ───────────────────────────────────────────────────────────

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

    class QMap(nn.Module):
        """Q(s, a): same as ForwardMap without z. Two heads (Q1, Q2), each a scalar.
        The obs_z branch of ForwardMap becomes an obs-only branch, so that the width of the concatenation (and thus of
        the heads) is the same as in ForwardMap."""
        hidden_dim:  int
        feature_dim: int
        add_trunk:   bool = False

        @nn.compact
        def __call__(self, obs, action):
            # obs_action_net: Linear -> ntanh -> Linear -> relu
            oa = TorchDense(self.hidden_dim)(jnp.concatenate([obs, action], axis=-1))
            oa = NTanh()(oa)
            oa = nn.relu(TorchDense(self.feature_dim)(oa))
            # obs_net (replaces obs_z_net): Linear -> ntanh -> Linear -> relu
            o = NTanh()(TorchDense(self.hidden_dim)(obs))
            o = nn.relu(TorchDense(self.feature_dim)(o))

            h = jnp.concatenate([oa, o], axis=-1)
            if self.add_trunk:
                h = nn.relu(TorchDense(self.hidden_dim)(h))

            def head(h):
                x = nn.relu(TorchDense(self.hidden_dim)(h))
                return TorchDense(1)(x)[..., 0]

            return head(h), head(h)

    class Actor(nn.Module):
        """Same as the Actor of fb.py without z (single obs branch). Returns the mean action mu = tanh(policy(h))."""
        hidden_dim:  int
        feature_dim: int
        action_dim:  int
        add_trunk:   bool = False

        @nn.compact
        def __call__(self, obs):
            o = NTanh()(TorchDense(self.hidden_dim)(obs))
            h = nn.relu(TorchDense(self.feature_dim)(o))

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

    # ── Reward approximator ───────────────────────────────────────────────────────
    # reward_approx=true : the policy is trained on the TRUE reward of the dataset (reference behaviour).
    # otherwise          : a reward approximator is first fit offline on `approx_n` states of the dataset (supervised by
    #                      the true reward), then r_hat(s) replaces the reward of EVERY dataset state in the replay
    #                      buffer. The policy is still evaluated on the TRUE reward (rollouts + reward_fn), as always.
    #   lrec_detach : B(s) = sqrt(d) * normalize(MLP(s)); r_hat = B^T Sigma^-1 z, z = mean(rbar B); L_rec with the
    #                 coefficient detached (as in fb_continual.py)
    #   lrec_full   : same, but the gradient goes through Sigma^-1 z
    #   fixed_z     : same B, r_hat = B(s)^T z_fixed with z_fixed = sqrt(d) * normalize(random vector)
    #   mlp         : MLP(s) regressed directly on the reward
    # rbar = r / sqrt(mean(r^2)) (unit second moment, no centering) is what the approximators regress; r_hat is
    # multiplied back by that scale so the policy sees a reward on the same scale as the true one.

    def l2_normalize(x, eps=1e-12):
        return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)

    class BackwardMap(nn.Module):
        """Same B as in fb_continual.py: sqrt(z_dim) * normalize(MLP(s))."""
        hidden_dim: int
        z_dim:      int

        @nn.compact
        def __call__(self, obs):
            x = NTanh()(TorchDense(self.hidden_dim)(obs))
            x = nn.relu(TorchDense(self.hidden_dim)(x))
            x = TorchDense(self.z_dim)(x)
            return math.sqrt(self.z_dim) * l2_normalize(x)

    class RewardNet(nn.Module):
        """Direct reward regressor s -> r (same body as B, scalar output, no normalization)."""
        hidden_dim: int

        @nn.compact
        def __call__(self, obs):
            x = NTanh()(TorchDense(self.hidden_dim)(obs))
            x = nn.relu(TorchDense(self.hidden_dim)(x))
            return TorchDense(1)(x)[..., 0]

    def fit_reward_approximator(mode, obs_all, r_true, pool_idx, seed):
        """Fits the approximator on the states obs_all[pool_idx] (targets r_true[pool_idx]).
        Returns (predict, info): predict(obs[n, obs_dim]) -> reward estimates [n] on the scale of the TRUE reward."""
        n         = int(pool_idx.shape[0])
        pool_obs  = obs_all[pool_idx]
        pool_r    = r_true[pool_idx]
        scale     = float(jnp.sqrt(jnp.mean(pool_r ** 2)))
        assert scale > 0, 'the true reward is identically zero on the approximator states'
        pool_rbar = pool_r / scale
        d, bs, n_est = APPROX_ZDIM, min(APPROX_BS, n), min(APPROX_EST, n)

        net = RewardNet(APPROX_HIDDEN) if mode == 'mlp' else BackwardMap(APPROX_HIDDEN, d)
        key = jax.random.PRNGKey(seed)
        key, k_init, k_z = jax.random.split(key, 3)
        params  = net.init(k_init, jnp.zeros((1, STATE_DIM)))['params']
        z_fixed = math.sqrt(d) * l2_normalize(jax.random.normal(k_z, (d,)))
        n_steps = APPROX_STEPS
        if mode in ('fb_pretrained', 'fb_finetuned'):
            # B of the pretrained FB; its readout w = Sigma^-1 z (z = mean(rbar B), Sigma = mean(B B^T) + ridge, on the
            # approximator states) is then kept FIXED: fb_pretrained = no training, fb_finetuned = B regressed with w fixed
            with open(CHECKPOINT, 'rb') as f:
                ckpt = pickle.load(f)
            params = jax.tree_util.tree_map(
                jnp.asarray, serialization.from_state_dict(params, ckpt['fb_params']['backward']))
            B0 = jax.jit(lambda x: net.apply({'params': params}, x))
            S, zs = np.zeros((d, d)), np.zeros(d)
            for i in range(0, n, 200_000):
                Bc  = B0(pool_obs[i:i + 200_000])
                S  += np.asarray(Bc.T @ Bc, dtype=np.float64)
                zs += np.asarray(pool_rbar[i:i + 200_000] @ Bc, dtype=np.float64)
            z_fixed = jnp.asarray(np.linalg.solve(S / n + APPROX_RIDGE * np.eye(d), zs / n), dtype=jnp.float32)
            print(f'  loaded pretrained B from {CHECKPOINT}; fixed readout ||w||={float(jnp.linalg.norm(z_fixed)):.2f}')
            if mode == 'fb_pretrained':
                n_steps = 0
        lr  = optax.cosine_decay_schedule(APPROX_LR, n_steps) if (APPROX_DECAY and n_steps > 0) else APPROX_LR
        opt = optax.adam(lr)
        opt_state = opt.init(params)

        def loss_fn(p, pool_obs, pool_rbar, idx, est_idx):
            x, r = pool_obs[idx], pool_rbar[idx]
            out  = net.apply({'params': p}, x)
            extra = 0.0
            if mode == 'mlp':
                pred = out
            elif mode in ('fixed_z', 'fb_pretrained', 'fb_finetuned'):
                pred = out @ z_fixed
            else:
                B_e   = net.apply({'params': p}, pool_obs[est_idx])
                S_e   = B_e.T @ B_e / n_est
                Sigma = S_e + APPROX_RIDGE * jnp.eye(d)
                w     = jnp.linalg.solve(Sigma, r @ out / bs)             # w = Sigma^-1 z_hat
                if mode == 'lrec_detach':
                    w = jax.lax.stop_gradient(w)
                pred = out @ w
                if APPROX_ORTHO > 0:                                      # optional: keep Sigma close to I
                    extra = APPROX_ORTHO * (S_e ** 2).sum() / d
            return jnp.mean((r - pred) ** 2) + extra

        @jax.jit
        def step(p, s, key, pool_obs, pool_rbar):
            k1, k2  = jax.random.split(key)
            idx     = jax.random.randint(k1, (bs,), 0, n)
            est_idx = jax.random.randint(k2, (n_est,), 0, n)
            loss, g = jax.value_and_grad(loss_fn)(p, pool_obs, pool_rbar, idx, est_idx)
            upd, s  = opt.update(g, s, p)
            return optax.apply_updates(p, upd), s, loss

        print(f'[reward approx] mode={mode} n_states={n} steps={n_steps} lr={APPROX_LR} bs={bs} '
              f'decay={APPROX_DECAY} ortho={APPROX_ORTHO}')
        for i in tqdm(range(n_steps), desc=f'fit {mode}'):
            key, k = jax.random.split(key)
            params, opt_state, loss = step(params, opt_state, k, pool_obs, pool_rbar)
            if i == 0 or (i + 1) % 2000 == 0:
                tqdm.write(f'  approx step {i + 1}: minibatch loss {float(loss):.4f}')

        # final readout for the L_rec modes: z and Sigma on ALL the approximator states (no minibatch noise)
        apply_fn = jax.jit(lambda x: net.apply({'params': params}, x))
        w, info = None, {'scale': scale, 'n_states': n}
        if mode in ('lrec_detach', 'lrec_full'):
            S, zs = np.zeros((d, d)), np.zeros(d)
            for i in range(0, n, 200_000):
                Bc  = apply_fn(pool_obs[i:i + 200_000])
                S  += np.asarray(Bc.T @ Bc, dtype=np.float64)
                zs += np.asarray(pool_rbar[i:i + 200_000] @ Bc, dtype=np.float64)
            S = S / n + APPROX_RIDGE * np.eye(d)
            w = jnp.asarray(np.linalg.solve(S, zs / n), dtype=jnp.float32)
            eig = np.linalg.eigvalsh(S)
            info.update({'sigma_eig_min': float(eig[0]), 'sigma_eig_max': float(eig[-1]), 'w_norm': float(jnp.linalg.norm(w))})
            print(f'  Sigma eig min={eig[0]:.2e} max={eig[-1]:.2e}  ||w||={float(jnp.linalg.norm(w)):.2f}')

        if mode == 'mlp':
            read = jax.jit(lambda x: net.apply({'params': params}, x))
        elif mode in ('fixed_z', 'fb_pretrained', 'fb_finetuned'):
            read = jax.jit(lambda x: net.apply({'params': params}, x) @ z_fixed)
        else:
            read = jax.jit(lambda x: net.apply({'params': params}, x) @ w)

        def predict(X, chunk=200_000):
            return jnp.concatenate([read(X[i:i + chunk]) for i in range(0, X.shape[0], chunk)]) * scale

        return predict, info

    if REWARD_APPROX != 'true':
        n_states_total = int(replay_buffer.obs.shape[0])
        # the approximator sees the states s_i of the transitions of the (possibly n_transitions-subsampled) dataset
        cand    = np.asarray(replay_buffer.valid_idx)
        rng_a   = np.random.default_rng(APPROX_SEED)
        n_approx = len(cand) if APPROX_N is None else min(APPROX_N, len(cand))
        pool_np = cand if n_approx == len(cand) else rng_a.choice(cand, size=n_approx, replace=False)

        predict_reward, approx_info = fit_reward_approximator(
            REWARD_APPROX, replay_buffer.obs, replay_buffer.rewards, jnp.asarray(pool_np, dtype=jnp.int32), APPROX_SEED)
        r_hat = predict_reward(replay_buffer.obs)                       # [n_total]: reward estimate of EVERY state

        # quality of the approximation on states NOT used to fit it (in-sample if the pool is the whole dataset)
        not_pool = np.ones(n_states_total, dtype=bool)
        not_pool[pool_np] = False
        held_np  = np.where(not_pool)[0]
        in_sample = len(held_np) == 0
        eval_pool = pool_np if in_sample else held_np
        eval_idx  = rng_a.choice(eval_pool, size=min(200_000, len(eval_pool)), replace=False)
        r_t = np.asarray(replay_buffer.rewards[eval_idx])
        r_h = np.asarray(r_hat[eval_idx])
        approx_corr = float(np.corrcoef(r_t, r_h)[0, 1])
        approx_eps  = float(((r_t - r_h) ** 2).mean() / (r_t ** 2).mean())
        approx_r2   = float(1 - ((r_t - r_h) ** 2).sum() / ((r_t - r_t.mean()) ** 2).sum())
        msg = (f'[reward approx] {REWARD_APPROX}, fit on {len(pool_np)} states, '
               f'{"IN-SAMPLE" if in_sample else "held-out"} quality: corr={approx_corr:.4f} R2={approx_r2:.4f} eps={approx_eps:.4e}')
        print(msg)
        with open(f'{LOGS_FOLDER}/info.txt', 'a') as file:
            file.write(msg + '\n' + str(approx_info) + '\n')

        sel = rng_a.choice(len(eval_idx), size=min(5000, len(eval_idx)), replace=False)
        plt.figure(figsize=(5, 5))
        plt.scatter(r_t[sel], r_h[sel], s=3, alpha=0.4)
        lo, hi = min(r_t[sel].min(), r_h[sel].min()), max(r_t[sel].max(), r_h[sel].max())
        plt.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
        plt.xlabel('true reward'); plt.ylabel('approximated reward')
        plt.title(f'{TASK_LABEL} / {REWARD_APPROX}: corr={approx_corr:.3f}, R2={approx_r2:.3f}')
        plt.tight_layout()
        plt.savefig(f'{LOGS_FOLDER}/reward_approx_scatter.png')
        plt.close()

        # from here on, the policy is trained on the approximated reward
        replay_buffer = replay_buffer.replace(rewards=jnp.asarray(r_hat, dtype=jnp.float32))
        del r_hat

    # ── Agent ─────────────────────────────────────────────────────────────────────

    @dataclasses.dataclass(frozen=True)
    class TD3Config:
        bc_coef:             float = 0.0      # weight of the behavior-cloning term (0 -> plain TD3, -Q.mean(); > 0 -> TD3+BC)
        lr:                  float = 3e-4
        target_tau:          float = 0.005    # same as fb_target_tau
        hidden_dim:          int   = 1024
        feature_dim:         int   = 512
        stddev:              float = 0.2
        stddev_clip:         float = 0.3
        batch_size:          int   = 1024
        add_trunk:           bool  = False
        discount:            float = 0.95

    class TD3TrainingState(flax.struct.PyTreeNode):
        q_params:              Any
        q_target_params:       Any
        actor_params:          Any
        q_optimizer_state:     optax.OptState
        actor_optimizer_state: optax.OptState
        random_key:            jnp.ndarray
        steps:                 jnp.ndarray

    class TD3Agent:

        def __init__(self, config: TD3Config, obs_dim: int, action_dim: int):
            self.cfg        = config
            self.obs_dim    = obs_dim
            self.action_dim = action_dim

            self.q_net     = QMap(config.hidden_dim, config.feature_dim, config.add_trunk)
            self.actor_net = Actor(config.hidden_dim, config.feature_dim, action_dim, config.add_trunk)

            self.q_optimizer     = optax.adam(config.lr)
            self.actor_optimizer = optax.adam(config.lr)

        def init(self, random_key) -> TD3TrainingState:
            # 4-way split kept (the 3rd key was used by the removed V network) so that Q / actor inits match the reference script
            random_key, kq, _, ka = jax.random.split(random_key, 4)
            obs    = jnp.zeros((1, self.obs_dim))
            action = jnp.zeros((1, self.action_dim))

            q_params     = self.q_net.init(kq, obs, action)['params']
            actor_params = self.actor_net.init(ka, obs)['params']

            return TD3TrainingState(
                q_params=q_params,
                q_target_params=jax.tree_util.tree_map(jnp.copy, q_params),
                actor_params=actor_params,
                q_optimizer_state=self.q_optimizer.init(q_params),
                actor_optimizer_state=self.actor_optimizer.init(actor_params),
                random_key=random_key,
                steps=jnp.zeros((), dtype=jnp.int32),
            )

        @partial(jax.jit, static_argnames=("self",))
        def act(self, actor_params, obs):
            """Deterministic action (eval_mode=True in the torch agent). Works on batches of obs."""
            return self.actor_net.apply({'params': actor_params}, obs)

        @partial(jax.jit, static_argnames=("self",))
        def update(self, ts: TD3TrainingState, replay_buffer: OfflineReplayBuffer):
            cfg = self.cfg
            bs  = cfg.batch_size
            discount = cfg.discount

            random_key, k_batch, k_next, k_act = jax.random.split(ts.random_key, 4)
            obs, action, next_obs, reward = replay_buffer.sample(k_batch, bs)

            # ── bootstrap: through the actor (TD3-like), a' = pi(s') + clipped noise
            next_mu     = self.actor_net.apply({'params': ts.actor_params}, next_obs)
            next_action = jax.lax.stop_gradient(
                sample_truncated_normal(next_mu, k_next, cfg.stddev, cfg.stddev_clip))
            tqn1, tqn2 = self.q_net.apply({'params': ts.q_target_params}, next_obs, next_action)
            next_Q   = jax.lax.stop_gradient(jnp.minimum(tqn1, tqn2))
            target_Q = jax.lax.stop_gradient(reward + discount * next_Q)

            # ── Q update: TD toward r(s') + discount * min_heads Q_bar(s', a')
            def q_loss_fn(q_params):
                Q1, Q2 = self.q_net.apply({'params': q_params}, obs, action)
                q_loss = jnp.mean((Q1 - target_Q) ** 2) + jnp.mean((Q2 - target_Q) ** 2)
                return q_loss, jax.lax.stop_gradient(jnp.minimum(Q1, Q2).mean())

            (q_loss, q_data), q_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(ts.q_params)
            q_updates, q_optimizer_state = self.q_optimizer.update(q_grads, ts.q_optimizer_state, ts.q_params)
            q_params = optax.apply_updates(ts.q_params, q_updates)

            # ── Actor update (uses the freshly updated Q, as in fb.py): maximize Q(s, pi(s)).
            # With bc_coef > 0: TD3+BC (scale-free Q term, the normalizer is not differentiated).
            def actor_loss_fn(actor_params):
                mu = self.actor_net.apply({'params': actor_params}, obs)
                a  = sample_truncated_normal(mu, k_act, cfg.stddev, cfg.stddev_clip)
                Q1, Q2 = self.q_net.apply({'params': q_params}, obs, a)
                Q  = jnp.minimum(Q1, Q2)
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

            metrics = {
                'q_loss':     q_loss,
                'actor_loss': actor_metrics['actor_loss'],
                'q_data':     q_data,                                   # Q at the dataset action
                'q_pi':       actor_metrics['q'],                       # Q at the actor action
                'q_gap':      actor_metrics['q'] - q_data,
                'bc_dist':    actor_metrics['bc_dist'],
                'next_Q':     next_Q.mean(),                            # bootstrap value
                'target_Q':   target_Q.mean(),
                'reward':     reward.mean(),
            }

            # ── Target update ─────────────────────────────────────────────────
            tau_t = cfg.target_tau
            q_target_params = jax.tree_util.tree_map(
                lambda p, t: tau_t * p + (1 - tau_t) * t, q_params, ts.q_target_params)

            ts = ts.replace(
                q_params=q_params,
                q_target_params=q_target_params,
                actor_params=actor_params,
                q_optimizer_state=q_optimizer_state,
                actor_optimizer_state=actor_optimizer_state,
                random_key=random_key,
                steps=ts.steps + 1,
            )
            return ts, metrics

    td3_config = TD3Config(batch_size=BATCH_SIZE)
    if 'discount' in kwargs:
        td3_config = dataclasses.replace(td3_config, discount=float(kwargs['discount']))
    if 'bc_coef' in kwargs:
        td3_config = dataclasses.replace(td3_config, bc_coef=float(kwargs['bc_coef']))
    print(td3_config)
    with open(f'{LOGS_FOLDER}/info.txt', 'a') as file:
        file.write(f'task={TASK_ID} ({TASK_LABEL})\n')
        file.write(str(td3_config) + '\n')

    agent = TD3Agent(td3_config, obs_dim=STATE_DIM, action_dim=ACTION_DIM)

    random_key = jax.random.PRNGKey(SEED)
    random_key, init_key = jax.random.split(random_key)
    train_state = agent.init(init_key)

    eval_random_state = {'key': jax.random.PRNGKey(SEED + 10_000)}
    def next_eval_key():
        eval_random_state['key'], sub = jax.random.split(eval_random_state['key'])
        return sub

    @jax.jit
    def train_chunk(train_state, replay_buffer):
        """SCAN_LENGTH consecutive Q + actor updates."""
        def body(ts, _):
            return agent.update(ts, replay_buffer)
        return jax.lax.scan(body, train_state, None, length=SCAN_LENGTH)

    # ── Evaluation ────────────────────────────────────────────────────────────────

    eval_envs = []   # pool of env instances, grown on demand and reused across evals

    def rollout_batch(actor_params, n):
        """Rolls out n envs in lock-step (one batched, jitted policy call per step).
        Returns obs (n, T, obs_dim), aux (n, T, aux_dim), physics (n, T, P) or None."""
        while len(eval_envs) < n:
            eval_envs.append(make_eval_env())
        envs = eval_envs[:n]

        states = [env_reset(e) for e in envs]
        obs_l, aux_l, phys_l = [], [], []
        for _ in tqdm(range(EVAL_STEPS), leave=False):
            obs = np.stack(states).astype(np.float32)
            obs_l.append(obs)
            aux_l.append(np.stack([env_aux(e, s) for e, s in zip(envs, states)]))
            if env_physics(envs[0]) is not None:
                phys_l.append(np.stack([env_physics(e) for e in envs]))

            actions = np.asarray(agent.act(actor_params, jnp.asarray(obs)))
            states  = [env_step(e, a) for e, a in zip(envs, actions)]

        obs_traj  = np.stack(obs_l).transpose(1, 0, 2)
        aux_traj  = np.stack(aux_l).transpose(1, 0, 2)
        phys_traj = np.stack(phys_l).transpose(1, 0, 2) if phys_l else None
        return obs_traj, aux_traj, phys_traj

    def plot_task(axs, obs_traj, aux_traj, phys_traj):
        """2 panels: real reward on dataset states, agent trajectories."""
        eval_obs, eval_aux = replay_buffer.sample_states(next_eval_key(), 10_000)
        real_rewards = np.asarray(reward_fn(eval_aux, TASK_PARAM))
        eval_obs, eval_aux = np.asarray(eval_obs), np.asarray(eval_aux)
        time_axis = np.tile(np.arange(EVAL_STEPS), (obs_traj.shape[0], 1))

        if ENV_NAME == 'cheetah':
            axs[0].scatter(eval_obs[..., 8], eval_aux[..., 0], c=real_rewards)
            axs[1].scatter(phys_traj[..., 0], time_axis, c='red', s=1)
            axs[1].set_xlim([-25, 25])
        elif ENV_NAME == 'walker':
            axs[0].scatter(eval_obs[..., 16], eval_aux[..., 0], c=real_rewards)
            axs[1].scatter(phys_traj[..., 1], time_axis, c='red', s=1)
        elif ENV_NAME == 'quadruped':
            axs[0].scatter(eval_aux[..., 0], eval_aux[..., 3], c=real_rewards)
            for j in range(aux_traj.shape[0]):
                axs[1].plot(aux_traj[j, :, 0], aux_traj[j, :, 3], c='red')
        elif ENV_NAME == 'antmaze':
            axs[0].scatter(eval_obs[..., 0], eval_obs[..., 1], c=real_rewards)
            axs[1].scatter(obs_traj[..., 0], obs_traj[..., 1], c='red', s=5)
            for k in range(2):
                add_largest_maze_walls(axs[k])
        elif ENV_NAME == 'pointmaze':
            axs[0].scatter(eval_obs[..., 0], eval_obs[..., 1], c=real_rewards, s=3)
            axs[1].scatter(obs_traj[..., 0], obs_traj[..., 1], c='red', s=2)
            for k in range(2):
                add_pointmaze_walls(axs[k])
                axs[k].set_aspect('equal')

        axs[0].set_title(f'{TASK_LABEL}')
        axs[1].set_title('Agent Trajectory')

    def run_eval(train_state, steps, make_plot):
        obs_traj, aux_traj, phys_traj = rollout_batch(train_state.actor_params, NUM_EVALS)

        # rewards of the produced trajectories: (NUM_EVALS, EVAL_STEPS) -> (NUM_EVALS,)
        trajectory_rewards = np.asarray(reward_fn(jnp.asarray(aux_traj, dtype=jnp.float32), TASK_PARAM)).sum(axis=-1)
        if ENV_NAME == 'antmaze' and 'goal' in TASK_LABEL:
            # goal tasks: 1 if the goal was reached at least once, else 0
            trajectory_rewards = np.where(trajectory_rewards != -EVAL_STEPS, 1.0, 0.0)

        print(TASK_LABEL, ':')
        print('\tRewards:', trajectory_rewards.tolist())
        print('\tmean:', trajectory_rewards.mean())
        print('\tstd:',  trajectory_rewards.std(ddof=1))

        if make_plot:
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            plot_task(axs, obs_traj, aux_traj, phys_traj)
            plt.tight_layout()
            plt.savefig(f"{LOGS_FOLDER}/eval-steps:{steps}.png")
            plt.close()

        return trajectory_rewards.mean(), trajectory_rewards.std(ddof=1)

    def save_agent(train_state, path):
        state = {
            'q_params':        train_state.q_params,
            'q_target_params': train_state.q_target_params,
            'actor_params':    train_state.actor_params,
        }
        state_dict = serialization.to_state_dict(jax.device_get(state))
        with open(path, "wb") as params_file:
            pickle.dump(state_dict, params_file)

    # ── Training ──────────────────────────────────────────────────────────────────

    rewards_logs = []
    n_evals = 0

    for steps in tqdm(range(NUM_OUTER_STEPS)):
        num_updates = (steps + 1) * SCAN_LENGTH

        start_time = time.time()
        train_state, current_metrics = train_chunk(train_state, replay_buffer)
        current_metrics = jax.device_get(current_metrics)
        timelapse = time.time() - start_time

        row = {'step': num_updates, 'time': float(timelapse)}
        row.update({k: float(np.mean(current_metrics[k])) for k in TRAIN_METRICS})
        train_logger.log(row)

        if (steps + 1) % EVAL_EACH_STEPS == 0:
            make_plot = (n_evals % PLOT_EACH_EVALS == 0)
            start_time = time.time()
            mean, std = run_eval(train_state, num_updates, make_plot)
            n_evals += 1

            eval_logger.log({'step': num_updates, f'{TASK_LABEL}_mean': float(mean), f'{TASK_LABEL}_std': float(std),
                             'time': time.time() - start_time})

            rewards_logs.append(mean)
            plt.figure(figsize=(6, 5))
            plt.plot(rewards_logs)
            plt.title(TASK_LABEL)
            plt.tight_layout()
            plt.savefig(f"{LOGS_FOLDER}/rewards.png")
            plt.close()

            save_agent(train_state, f"{MODEL_SAVE_FOLDER}/td3_agent_latest.pickle")

        if (steps + 1) % SAVE_EACH_STEPS == 0:
            save_agent(train_state, f"{MODEL_SAVE_FOLDER}/td3_agent_step:{num_updates}.pickle")