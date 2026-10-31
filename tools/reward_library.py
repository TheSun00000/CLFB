"""A library of candidate walker rewards defined on the 24-d OBSERVATION (what B sees, and what rollouts can compute).

Observation layout (dm_control walker, flat): [orientations(14), height(1), velocity(9)]
    orientations = (cos, sin) of the absolute angle of 7 bodies: torso, r.thigh, r.leg, r.foot, l.thigh, l.leg, l.foot
    height       = torso height;  velocity = qvel = [z, x, torso angular, r.hip, r.knee, r.ankle, l.hip, l.knee, l.ankle]
Forward kinematics from the orientations (lengths read from the walker model, checked on the data: lowest ankle height ~ 0.03, min -0.04):
    torso 0.6 (hip joint 0.3 below the torso centre), thigh 0.45, leg 0.5, foot 0.2.
Every reward is bounded in [0, 1] (dm_control tolerance functions). `LIBRARY` = {name: (description, fn(obs) -> r)}.
"""
import numpy as np

INF = float('inf')
D = np.pi / 180


def tol(x, lo, hi, margin):
    """1 inside [lo, hi], linear decay to 0 at `margin` outside."""
    inb = (lo <= x) & (x <= hi)
    d = np.where(x < lo, lo - x, x - hi) / margin
    return np.where(inb, 1.0, np.clip(1.0 - d, 0.0, 1.0))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def feats(o):
    f = {}
    c = lambda i: o[..., 2 * i]
    s = lambda i: o[..., 2 * i + 1]
    ang = lambda i: np.arctan2(s(i), c(i))
    f['h'] = o[..., 14]
    f['c0'], f['s0'] = c(0), s(0)
    f['vz'], f['vx'], f['w'] = o[..., 15], o[..., 16], o[..., 17]
    f['hv_r'], f['kv_r'], f['av_r'], f['hv_l'], f['kv_l'], f['av_l'] = (o[..., 18], o[..., 19], o[..., 20], o[..., 21], o[..., 22], o[..., 23])
    f['hip_r'], f['knee_r'], f['ank_r'] = wrap(ang(1) - ang(0)), wrap(ang(2) - ang(1)), wrap(ang(3) - ang(2))
    f['hip_l'], f['knee_l'], f['ank_l'] = wrap(ang(4) - ang(0)), wrap(ang(5) - ang(4)), wrap(ang(6) - ang(5))
    f['th_r'], f['th_l'] = ang(1), ang(4)
    # heights (y) and horizontal offsets (x, relative to the torso centre) of joints
    f['hip_y'] = f['h'] - 0.3 * c(0)
    f['top_y'] = f['h'] + 0.3 * c(0)
    f['knee_r_y'], f['knee_l_y'] = f['hip_y'] - 0.45 * c(1), f['hip_y'] - 0.45 * c(4)
    f['ank_r_y'], f['ank_l_y'] = f['knee_r_y'] - 0.5 * c(2), f['knee_l_y'] - 0.5 * c(5)
    f['hip_x'] = -0.3 * s(0)
    f['knee_r_x'], f['knee_l_x'] = f['hip_x'] - 0.45 * s(1), f['hip_x'] - 0.45 * s(4)
    f['ank_r_x'], f['ank_l_x'] = f['knee_r_x'] - 0.5 * s(2), f['knee_l_x'] - 0.5 * s(5)
    f['jv_mean'] = np.mean(np.abs(o[..., 18:24]), axis=-1)
    f['jv_max'] = np.max(np.abs(o[..., 18:24]), axis=-1)
    f['jv_r'] = np.mean(np.abs(o[..., 18:21]), axis=-1)
    f['jv_l'] = np.mean(np.abs(o[..., 21:24]), axis=-1)
    f['foot_r_c'], f['foot_l_c'] = c(3), c(6)
    return f


L = {}


def add(name, desc):
    def deco(fn):
        L[name] = (desc, lambda o, fn=fn: fn(feats(o)))
        return fn
    return deco


# ------------------------------------------------------------------------------------------------ static postures of the legs
@add('split', 'thighs >= 60 deg apart')
def _(f): return tol(np.abs(wrap(f['th_r'] - f['th_l'])), 60 * D, INF, 40 * D)
@add('knees_bent', 'both knees bent >= 60 deg')
def _(f): return tol(np.abs(f['knee_r']), 60 * D, INF, 40 * D) * tol(np.abs(f['knee_l']), 60 * D, INF, 40 * D)
@add('r_knee_only', 'right knee bent >= 80 deg, left leg straight')
def _(f): return tol(np.abs(f['knee_r']), 80 * D, INF, 50 * D) * tol(np.abs(f['knee_l']), 0, 25 * D, 30 * D)
@add('l_knee_only', 'left knee bent >= 80 deg, right leg straight')
def _(f): return tol(np.abs(f['knee_l']), 80 * D, INF, 50 * D) * tol(np.abs(f['knee_r']), 0, 25 * D, 30 * D)
@add('legs_straight', 'both knees straight (<= 15 deg)')
def _(f): return tol(np.abs(f['knee_r']), 0, 15 * D, 30 * D) * tol(np.abs(f['knee_l']), 0, 15 * D, 30 * D)
@add('hips_flexed', 'both thighs rotated forward by >= 70 deg relative to the torso (sitting-like)')
def _(f): return tol(-f['hip_r'], 70 * D, INF, 40 * D) * tol(-f['hip_l'], 70 * D, INF, 40 * D)
@add('hips_extended', 'both thighs behind the torso axis (hip angle >= +5 deg)')
def _(f): return tol(f['hip_r'], 5 * D, INF, 30 * D) * tol(f['hip_l'], 5 * D, INF, 30 * D)
@add('r_leg_fwd', 'right thigh ahead of the left by >= 45 deg (signed)')
def _(f): return tol(wrap(f['th_r'] - f['th_l']), 45 * D, INF, 40 * D)
@add('l_leg_fwd', 'left thigh ahead of the right by >= 45 deg (signed)')
def _(f): return tol(wrap(f['th_l'] - f['th_r']), 45 * D, INF, 40 * D)
@add('toes_up', 'both feet dorsiflexed >= 30 deg')
def _(f): return tol(f['ank_r'], 30 * D, INF, 30 * D) * tol(f['ank_l'], 30 * D, INF, 30 * D)
@add('toes_down', 'both feet plantarflexed >= 30 deg (tiptoe)')
def _(f): return tol(-f['ank_r'], 30 * D, INF, 30 * D) * tol(-f['ank_l'], 30 * D, INF, 30 * D)
@add('tuck', 'knees bent >= 90 deg and thighs forward >= 70 deg (curled legs)')
def _(f): return (tol(np.abs(f['knee_r']), 90 * D, INF, 40 * D) * tol(np.abs(f['knee_l']), 90 * D, INF, 40 * D)
                  * tol(-f['hip_r'], 70 * D, INF, 40 * D) * tol(-f['hip_l'], 70 * D, INF, 40 * D))

# ------------------------------------------------------------------------------------------------ feet geometry (forward kinematics)
@add('flamingo_r', 'right foot >= 0.35 above the ground, left foot on the ground')
def _(f): return tol(f['ank_r_y'], 0.35, INF, 0.25) * tol(f['ank_l_y'], -INF, 0.12, 0.12)
@add('flamingo_l', 'left foot >= 0.35 above the ground, right foot on the ground')
def _(f): return tol(f['ank_l_y'], 0.35, INF, 0.25) * tol(f['ank_r_y'], -INF, 0.12, 0.12)
@add('airborne', 'both feet >= 0.8 above the ground')
def _(f): return tol(np.minimum(f['ank_r_y'], f['ank_l_y']), 0.8, INF, 0.4)
@add('wide_stride', 'feet >= 0.7 apart horizontally')
def _(f): return tol(np.abs(f['ank_r_x'] - f['ank_l_x']), 0.7, INF, 0.4)
@add('r_foot_ahead', 'right foot >= 0.5 ahead of the left (signed)')
def _(f): return tol(f['ank_r_x'] - f['ank_l_x'], 0.5, INF, 0.4)
@add('l_foot_ahead', 'left foot >= 0.5 ahead of the right (signed)')
def _(f): return tol(f['ank_l_x'] - f['ank_r_x'], 0.5, INF, 0.4)
@add('feet_together', 'feet within 0.1 of each other in x and y')
def _(f): return tol(np.abs(f['ank_r_x'] - f['ank_l_x']), 0, 0.1, 0.3) * tol(np.abs(f['ank_r_y'] - f['ank_l_y']), 0, 0.1, 0.2)
@add('knee_high_r', 'right knee at or above the hip')
def _(f): return tol(f['knee_r_y'] - f['hip_y'], -0.05, INF, 0.3)
@add('knee_high_l', 'left knee at or above the hip')
def _(f): return tol(f['knee_l_y'] - f['hip_y'], -0.05, INF, 0.3)
@add('foot_above_hip', 'a foot >= 0.4 above the hip (leg kicked up)')
def _(f): return tol(np.maximum(f['ank_r_y'], f['ank_l_y']) - f['hip_y'], 0.4, INF, 0.3)

# ------------------------------------------------------------------------------------------------ torso, height
@add('hip_low', 'hip joint within 0.35 of the ground (sitting or lying)')
def _(f): return tol(f['hip_y'], -INF, 0.35, 0.3)
@add('head_low', 'top of the torso within 0.4 of the ground')
def _(f): return tol(f['top_y'], -INF, 0.4, 0.4)
@add('inverted', 'torso pointing down (upside down)')
def _(f): return tol(-f['c0'], 0.5, INF, 0.5)
@add('prone', 'torso horizontal, one side (sin > 0.8)')
def _(f): return tol(f['s0'], 0.8, INF, 0.4)
@add('supine', 'torso horizontal, other side (sin < -0.8)')
def _(f): return tol(-f['s0'], 0.8, INF, 0.4)
@add('lean_a', 'torso tilted 25..50 deg to one side')
def _(f): return tol(-np.arctan2(f['s0'], f['c0']), 25 * D, 50 * D, 25 * D)
@add('lean_b', 'torso tilted 25..50 deg to the other side')
def _(f): return tol(np.arctan2(f['s0'], f['c0']), 25 * D, 50 * D, 25 * D)
@add('squat_deep', 'hip low (0.3..0.55) with an upright torso')
def _(f): return tol(f['hip_y'], 0.3, 0.55, 0.25) * tol(f['c0'], 0.7, INF, 0.4)
@add('tall', 'torso centre >= 1.35 high')
def _(f): return tol(f['h'], 1.35, INF, 0.3)

# ------------------------------------------------------------------------------------------------ instantaneous motion
@add('jump_any', 'upward velocity >= 3')
def _(f): return tol(f['vz'], 3.0, INF, 3.0)
@add('fall', 'downward velocity >= 3')
def _(f): return tol(-f['vz'], 3.0, INF, 3.0)
@add('spin_fwd', 'torso angular velocity >= +8 rad/s')
def _(f): return tol(f['w'], 8.0, INF, 8.0)
@add('spin_back', 'torso angular velocity <= -8 rad/s')
def _(f): return tol(-f['w'], 8.0, INF, 8.0)
@add('kick_r', 'right hip velocity >= +20 rad/s')
def _(f): return tol(f['hv_r'], 20.0, INF, 20.0)
@add('kick_l', 'left hip velocity >= +20 rad/s')
def _(f): return tol(f['hv_l'], 20.0, INF, 20.0)
@add('kick_back_r', 'right hip velocity <= -20 rad/s')
def _(f): return tol(-f['hv_r'], 20.0, INF, 20.0)
@add('hips_antiphase', 'hips swinging in opposite directions (product of velocities <= -150)')
def _(f): return tol(-f['hv_r'] * f['hv_l'], 150.0, INF, 150.0)
@add('hips_inphase', 'hips swinging the same way (product of velocities >= 150)')
def _(f): return tol(f['hv_r'] * f['hv_l'], 150.0, INF, 150.0)
@add('knee_flap', 'both knees moving fast (mean |velocity| >= 28 rad/s)')
def _(f): return tol(0.5 * (np.abs(f['kv_r']) + np.abs(f['kv_l'])), 28.0, INF, 14.0)
@add('ankle_snap', 'both ankles moving fast (mean |velocity| >= 32 rad/s)')
def _(f): return tol(0.5 * (np.abs(f['av_r']) + np.abs(f['av_l'])), 32.0, INF, 16.0)
@add('flail', 'all joints moving fast (mean |velocity| >= 24 rad/s)')
def _(f): return tol(f['jv_mean'], 24.0, INF, 12.0)
@add('calm', 'joints nearly still (max |velocity| <= 15 rad/s) and the torso not rotating')
def _(f): return tol(f['jv_max'], 0, 15.0, 10.0) * tol(np.abs(f['w']), 0, 2.0, 4.0)
@add('right_only', 'right leg moving (>= 15), left leg still (<= 5)')
def _(f): return tol(f['jv_r'], 15.0, INF, 15.0) * tol(f['jv_l'], 0, 5.0, 10.0)
@add('left_only', 'left leg moving (>= 15), right leg still (<= 5)')
def _(f): return tol(f['jv_l'], 15.0, INF, 15.0) * tol(f['jv_r'], 0, 5.0, 10.0)
@add('slide_fast', 'horizontal speed >= 3 in either direction, any posture')
def _(f): return tol(np.abs(f['vx']), 3.0, INF, 3.0)

# ------------------------------------------------------------------------------------------------ compound
@add('tuck_jump', 'rising (vz >= 2) with knees bent')
def _(f): return tol(f['vz'], 2.0, INF, 2.0) * tol(np.abs(f['knee_r']), 50 * D, INF, 40 * D) * tol(np.abs(f['knee_l']), 50 * D, INF, 40 * D)
@add('airborne_spin', 'both feet >= 0.6 above the ground and rotating (|w| >= 8)')
def _(f): return tol(np.minimum(f['ank_r_y'], f['ank_l_y']), 0.6, INF, 0.3) * tol(np.abs(f['w']), 8.0, INF, 6.0)
@add('upright_spin', 'rotating (|w| >= 5) while the torso is roughly upright')
def _(f): return tol(np.abs(f['w']), 5.0, INF, 5.0) * tol(f['c0'], 0.7, INF, 0.4)
@add('standing_flail', 'torso high (>= 1.0) with the legs flailing')
def _(f): return tol(f['h'], 1.0, INF, 0.3) * tol(f['jv_mean'], 20.0, INF, 12.0)
@add('crouch_still', 'hip low (<= 0.5) and the joints nearly still (<= 20 rad/s)')
def _(f): return tol(f['hip_y'], -INF, 0.5, 0.3) * tol(f['jv_max'], 0, 20.0, 12.0)
@add('kicked_up', 'a foot above the hip while the other leg is straight')
def _(f): return (tol(np.maximum(f['ank_r_y'], f['ank_l_y']) - f['hip_y'], 0.0, INF, 0.3)
                  * tol(np.minimum(np.abs(f['knee_r']), np.abs(f['knee_l'])), 0, 20 * D, 30 * D))
@add('hop_pose', 'one foot up, rising')
def _(f): return tol(np.abs(f['ank_r_y'] - f['ank_l_y']), 0.25, INF, 0.25) * tol(f['vz'], 1.0, INF, 1.0)

LIBRARY = L
