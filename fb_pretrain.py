"""
Forward-backward (FB) pretraining: classical FB (actor in the bootstrap) + DDPG actor, full dataset. Nothing else.
The checkpoints are meant to be loaded by fb_continual.py (checkpoint=...).

Usage:
    python fb_pretrain.py env_name=walker
    python fb_pretrain.py env_name=walker seed=0 num_steps=1000000 ortho_coef=10

Flags:
    env_name=       cheetah | walker | quadruped | antmaze | pointmaze           (required)
    seed=<int>               default: random (logged in info.txt)
    num_steps=1000000        number of updates of THIS run (multiple of scan_length)
    checkpoint=              optional fb_agent_*.pickle to start from (fine-tune / continue). Logged 'step' then counts the
                             checkpoint's updates too (e.g. 800000 + ...), and a baseline evaluation is done at the start.
    load_target=1            with checkpoint: use the saved target networks (0 -> targets = copy of the online networks)
    load_optimizer=1         with checkpoint: use the saved Adam states (0 -> fresh Adam). Pickles written by this script contain them.
    eval_baseline=           evaluate before training (default: 1 with a checkpoint, 0 otherwise)
    scan_length=1000         updates per jitted chunk (= logging granularity)
    eval_each_steps=10       evaluate every this many chunks (10 -> every 10k updates) and at the end
    save_every=50            save fb_agent_<step>.pickle every this many chunks (50 -> every 50k updates); fb_agent_latest.pickle
                             is rewritten at every evaluation and at the end
    num_evals=5              episodes per task and evaluation
    eval_chunk=              number of tasks rolled out in parallel (each needs num_evals env instances)
    FB hyper-parameters (defaults = the script that produced your previous checkpoint; they must be the same in fb_continual.py):
    discount=0.95 ortho_coef=1.0 q_loss_coef=0.01 lr=1e-4 fb_target_tau=0.01 mix_ratio=0.5 bc_coef=0.0
    target_agg=min   bootstrap target of M over the two heads: min (original FB, previous checkpoint) | mean

Outputs (./logs/<run>/ and ./models/<run>/):
    info.txt           kwargs + the full FBConfig of the run
    training_log.csv   per chunk: FB metrics (mean over the chunk)
    eval_log.csv       per evaluation: '<task>_{mean,std,rew_corr,eps}' for every task of the env (z inferred from fresh 10k labels)
    rewards.png        returns of every task along training
    fb_agent_<step>.pickle, fb_agent_latest.pickle:
        'fb_params', 'fb_target_params', 'actor_params'   (what fb_continual.py loads)
        'steps', 'config'                                 (number of updates, FBConfig as a dict)
        'optimizer'                                       (Adam states of F/B and actor; best effort, absent if serialization fails)
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


def normalize(x, eps=1e-12):
    """torch F.normalize(x, dim=-1)."""
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


if __name__ == '__main__':
    #######################################################################################

    kwargs = parse_kwargs(sys.argv[1:])
    print(kwargs)
    print(jax.devices())

    assert 'env_name' in kwargs, "env_name= is required"
    ENV_NAME = kwargs['env_name']
    assert ENV_NAME in ['cheetah', 'walker', 'quadruped', 'antmaze', 'pointmaze']

    SEED = resolve_seed(kwargs)

    NUM_STEPS       = int(kwargs['num_steps']) if 'num_steps' in kwargs else 1_000_000
    SCAN_LENGTH     = int(kwargs['scan_length']) if 'scan_length' in kwargs else 1_000
    assert NUM_STEPS % SCAN_LENGTH == 0, "num_steps must be a multiple of scan_length"
    NUM_CHUNKS      = NUM_STEPS // SCAN_LENGTH
    EVAL_EACH_STEPS = int(kwargs['eval_each_steps']) if 'eval_each_steps' in kwargs else 10   # in chunks
    SAVE_EVERY      = int(kwargs['save_every']) if 'save_every' in kwargs else 50             # in chunks
    EVAL_N          = 10_000      # number of states for the eval-time reward / reconstruction statistics

    # ── Logging ───────────────────────────────────────────────────────────────────
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    exp_name = f'FBpretrain-{ENV_NAME}'
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
                     "actor_loss", "q", "bc_dist"]
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
    LABELS    = [label for _, label, _ in benchmarks]
    print(NUM_TRAJECTORIES, TRAJECTORY_LEN, NUM_TRAJECTORIES * TRAJECTORY_LEN)

    BATCH_SIZE = 1024
    # Number of benchmarks rolled out in parallel during eval (each one needs NUM_EVALS env instances)
    EVAL_CHUNK = int(kwargs['eval_chunk']) if 'eval_chunk' in kwargs else N_BENCH

    EVAL_FIELDS = (["step", "mean_all"]
                   + [f"{label}_{s}" for label in LABELS for s in ("mean", "std", "rew_corr", "eps")]
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

    # ── Neural networks (identical to fb_continual.py) ────────────────────────────

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
        bc_coef:             float = 0.0      # weight of the behavior-cloning term of the DDPG actor (0 -> plain -Q.mean())
        lr:                  float = 1e-4
        lr_coef:             float = 1.0      # backward net lr = lr_coef * lr
        fb_target_tau:       float = 0.01
        hidden_dim:          int   = 1024
        backward_hidden_dim: int   = 526      # (sic) same value as in the torch script
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
        target_agg:          str   = 'min'    # bootstrap target of M over the two heads: 'min' (original FB / previous pretraining) | 'mean'

    class FBTrainingState(flax.struct.PyTreeNode):
        fb_params:             Any          # {'forward': ..., 'backward': ...}
        fb_target_params:      Any
        actor_params:          Any
        fb_optimizer_state:    optax.OptState
        actor_optimizer_state: optax.OptState
        random_key:            jnp.ndarray
        steps:                 jnp.ndarray

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
        def update(self, ts: FBTrainingState, replay_buffer: OfflineReplayBuffer):
            cfg = self.cfg
            bs  = cfg.batch_size
            discount = cfg.discount

            random_key, k_batch, k_z, k_perm, k_mix, k_next, k_act = jax.random.split(ts.random_key, 7)
            obs, action, next_obs, _ = replay_buffer.sample(k_batch, bs)

            # ── z sampling: random z, with a fraction replaced by B(obs) of shuffled states
            z = self.sample_z(k_z, bs)
            if cfg.mix_ratio > 0:
                perm  = jax.random.permutation(k_perm, bs)
                mix_z = jax.lax.stop_gradient(
                    self.backward_net.apply({'params': ts.fb_params['backward']}, obs[perm]))
                mix_mask = jax.random.uniform(k_mix, (bs, 1)) < cfg.mix_ratio
                z = jnp.where(mix_mask, mix_z, z)

            off_diag = 1.0 - jnp.eye(bs)
            n_off    = off_diag.sum()

            # ── bootstrap term of the FB loss (classical: through the actor) ──
            target_B = self.backward_net.apply({'params': ts.fb_target_params['backward']}, next_obs)
            next_mu     = self.actor_net.apply({'params': ts.actor_params}, next_obs, z)
            next_action = jax.lax.stop_gradient(
                sample_truncated_normal(next_mu, k_next, cfg.stddev, cfg.stddev_clip))
            target_F1, target_F2 = self.forward_net.apply(
                {'params': ts.fb_target_params['forward']}, next_obs, z, next_action)
            tM1, tM2 = target_F1 @ target_B.T, target_F2 @ target_B.T
            if cfg.target_agg == 'min':
                target_M = jax.lax.stop_gradient(jnp.minimum(tM1, tM2))
            else:
                target_M = jax.lax.stop_gradient(0.5 * (tM1 + tM2))
            next_Q   = jax.lax.stop_gradient(jnp.minimum((target_F1 * z).sum(-1), (target_F2 * z).sum(-1)))

            # ── FB update ─────────────────────────────────────────────────────
            def fb_loss_fn(fb_params):
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
                }
                return fb_loss, metrics

            (_, metrics), fb_grads = jax.value_and_grad(fb_loss_fn, has_aux=True)(ts.fb_params)
            fb_updates, fb_optimizer_state = self.fb_optimizer.update(
                fb_grads, ts.fb_optimizer_state, ts.fb_params)
            fb_params = optax.apply_updates(ts.fb_params, fb_updates)

            # ── Actor update (DDPG / TD3-like, uses the freshly updated forward net) ──
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
    for _k, _typ in (('discount', float), ('bc_coef', float), ('ortho_coef', float), ('q_loss_coef', float),
                     ('lr', float), ('fb_target_tau', float), ('mix_ratio', float)):
        if _k in kwargs:
            fb_config = dataclasses.replace(fb_config, **{_k: _typ(kwargs[_k])})
    if 'target_agg' in kwargs:
        assert kwargs['target_agg'] in ['min', 'mean'], 'target_agg= must be min or mean'
        fb_config = dataclasses.replace(fb_config, target_agg=kwargs['target_agg'])
    assert fb_config.norm_z, 'the method assumes norm_z=True (z on the sphere of radius sqrt(d))'
    Z_DIM = fb_config.z_dim
    RIDGE = 1e-5     # eval-time ridge on the covariance used to reconstruct rewards
    print(fb_config)
    with open(f'{LOGS_FOLDER}/info.txt', 'a') as file:
        file.write(str(fb_config) + '\n')
        file.write(f'num_steps={NUM_STEPS} scan_length={SCAN_LENGTH} eval_each_steps={EVAL_EACH_STEPS} save_every={SAVE_EVERY}\n')

    agent = FBAgent(fb_config, obs_dim=STATE_DIM, action_dim=ACTION_DIM)

    random_key = jax.random.PRNGKey(SEED)
    random_key, init_key = jax.random.split(random_key)
    train_state = agent.init(init_key)

    # ── Optional: start from a checkpoint (fine-tuning / continued pretraining) ───
    CHECKPOINT      = str(kwargs['checkpoint']) if 'checkpoint' in kwargs else None
    LOAD_TARGET     = bool(int(kwargs['load_target']))    if 'load_target'    in kwargs else True    # use the saved target networks
    LOAD_OPTIMIZER  = bool(int(kwargs['load_optimizer'])) if 'load_optimizer' in kwargs else True    # use the saved Adam states
    EVAL_BASELINE   = bool(int(kwargs['eval_baseline']))  if 'eval_baseline'  in kwargs else (CHECKPOINT is not None)
    START_STEP      = 0

    if CHECKPOINT is not None:
        with open(CHECKPOINT, 'rb') as f:
            ckpt = pickle.load(f)

        def restore(template, state, what):
            restored = jax.tree_util.tree_map(jnp.asarray, serialization.from_state_dict(template, state))
            def _check(a, b):
                assert a.shape == b.shape, (f"checkpoint mismatch in {what}: expected shape {a.shape}, got {b.shape} "
                                            f"(env_name / network hyper-parameters differ from the pretraining run?)")
            jax.tree_util.tree_map(_check, template, restored)
            return restored

        fb_params    = restore(train_state.fb_params, ckpt['fb_params'], 'fb_params')
        actor_params = restore(train_state.actor_params, ckpt['actor_params'], 'actor_params')
        if LOAD_TARGET and 'fb_target_params' in ckpt:
            fb_target_params = restore(train_state.fb_target_params, ckpt['fb_target_params'], 'fb_target_params')
        else:
            fb_target_params = jax.tree_util.tree_map(jnp.copy, fb_params)
        fb_opt_state, actor_opt_state = agent.fb_optimizer.init(fb_params), agent.actor_optimizer.init(actor_params)
        opt_loaded = False
        if LOAD_OPTIMIZER and 'optimizer' in ckpt:
            try:
                fb_opt_state    = restore(fb_opt_state, ckpt['optimizer']['fb_optimizer_state'], 'fb_optimizer_state')
                actor_opt_state = restore(actor_opt_state, ckpt['optimizer']['actor_optimizer_state'], 'actor_optimizer_state')
                opt_loaded = True
            except Exception as e:
                print(f'WARNING: could not restore the Adam states ({type(e).__name__}: {e}); using fresh Adam')
                fb_opt_state, actor_opt_state = agent.fb_optimizer.init(fb_params), agent.actor_optimizer.init(actor_params)
        train_state = train_state.replace(
            fb_params=fb_params, fb_target_params=fb_target_params, actor_params=actor_params,
            fb_optimizer_state=fb_opt_state, actor_optimizer_state=actor_opt_state)
        START_STEP = int(ckpt.get('steps', 0))
        msg = (f'loaded {CHECKPOINT} (steps={START_STEP}, target networks from checkpoint={LOAD_TARGET and "fb_target_params" in ckpt}, '
               f'Adam states from checkpoint={opt_loaded})')
        print(msg)
        with open(f'{LOGS_FOLDER}/info.txt', 'a') as file:
            file.write(msg + '\n')

    eval_random_state = {'key': jax.random.PRNGKey(SEED + 10_000)}
    def next_eval_key():
        eval_random_state['key'], sub = jax.random.split(eval_random_state['key'])
        return sub

    @jax.jit
    def train_chunk(train_state, replay_buffer):
        """SCAN_LENGTH consecutive FB + actor updates."""
        def body(ts, _):
            return agent.update(ts, replay_buffer)
        return jax.lax.scan(body, train_state, None, length=SCAN_LENGTH)

    # ── Evaluation (fresh zero-shot inference from EVAL_N labeled states, as in the original script) ──

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
                print(f"{label}: mean {trajectory_rewards.mean():.3f}  std {trajectory_rewards.std(ddof=1):.3f}  "
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

    def run_eval(ts, step):
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
            stats[b] = (float(eps), float(corr))
            items.append((b, b, jnp.stack(zs)))

        returns = rollout_returns(ts.actor_params, items)

        row = {'step': step}
        means = np.zeros(N_BENCH)
        for b, label in enumerate(LABELS):
            m, s = returns[b]
            eps, corr = stats[b]
            row.update({f'{label}_mean': m, f'{label}_std': s, f'{label}_rew_corr': corr, f'{label}_eps': eps})
            means[b] = m
        row['mean_all'] = float(means.mean())
        row['time'] = time.time() - start_time
        eval_logger.log(row)
        print(f"[eval @ {step}] mean over tasks: {row['mean_all']:.2f}   " +
              "  ".join(f"{l}: {row[f'{l}_mean']:.1f}" for l in LABELS))

        eval_history.append(row)
        plot_rewards()

    def plot_rewards():
        """One panel per task: return along training."""
        n = N_BENCH
        num_cols = min(4, n)
        num_rows = n // num_cols + int(n % num_cols != 0)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 4.5))
        axs = np.atleast_1d(axs).flatten()
        xs = [r['step'] for r in eval_history]
        for i, label in enumerate(LABELS):
            m = np.array([r[f'{label}_mean'] for r in eval_history])
            s = np.array([r[f'{label}_std'] for r in eval_history])
            axs[i].plot(xs, m)
            axs[i].fill_between(xs, m - s, m + s, alpha=0.2)
            axs[i].set_title(label)
        plt.tight_layout()
        plt.savefig(f"{LOGS_FOLDER}/rewards.png")
        plt.close()

    # ── Checkpointing ─────────────────────────────────────────────────────────────

    def save_agent(ts, path, step):
        """Atomic write (tmp file + rename), so that a killed job never leaves a half-written fb_agent_latest.pickle."""
        state = {
            'fb_params':        ts.fb_params,
            'fb_target_params': ts.fb_target_params,
            'actor_params':     ts.actor_params,
        }
        state_dict = serialization.to_state_dict(jax.device_get(state))
        state_dict['steps']  = int(step)
        state_dict['config'] = dataclasses.asdict(fb_config)
        try:   # best effort: Adam states (lets a later script restart without the "fresh Adam" transient)
            opt = {'fb_optimizer_state': ts.fb_optimizer_state, 'actor_optimizer_state': ts.actor_optimizer_state}
            state_dict['optimizer'] = serialization.to_state_dict(jax.device_get(opt))
        except Exception as e:
            print(f'WARNING: optimizer state not saved ({type(e).__name__}: {e})')
        tmp = path + '.tmp'
        with open(tmp, "wb") as params_file:
            pickle.dump(state_dict, params_file)
        os.replace(tmp, path)

    # ── Training ──────────────────────────────────────────────────────────────────

    if EVAL_BASELINE:
        run_eval(train_state, START_STEP)

    for chunk in tqdm(range(NUM_CHUNKS), desc='pretraining'):
        start_time = time.time()
        train_state, current_metrics = train_chunk(train_state, replay_buffer)
        current_metrics = jax.device_get(current_metrics)
        timelapse = time.time() - start_time

        step = START_STEP + (chunk + 1) * SCAN_LENGTH      # total number of updates, including those of the checkpoint
        row = {'step': step, 'time': float(timelapse)}
        row.update({m: float(np.mean(current_metrics[m])) for m in TRAIN_METRICS})
        train_logger.log(row)

        is_last = (chunk + 1) == NUM_CHUNKS
        if (chunk + 1) % EVAL_EACH_STEPS == 0 or is_last:
            run_eval(train_state, step)
            save_agent(train_state, f"{MODEL_SAVE_FOLDER}/fb_agent_latest.pickle", step)
        if (chunk + 1) % SAVE_EVERY == 0 or is_last:
            save_agent(train_state, f"{MODEL_SAVE_FOLDER}/fb_agent_{step}.pickle", step)