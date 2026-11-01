#!/usr/bin/env python3
"""Fast retention study, eps only (2026-11-01): fine-tune the pretrained walker FB model on ONE task for a few thousand updates, no rollouts, and look at the
reward error eps of the 8 tasks (tools/eps_report.py). Goal: eps of the trained task drops, eps of the other tasks stays at the pretrained level.
Each queue entry is ONE job that runs many python commands one after the other (one job keeps its priority; many small jobs do not).
Usage: python tools/queue_eps.py <wave>      (waves defined below; see docs/eps_plan.md for the motivation of each)
Common setup: the best setup of the return study (100k labels, lam 100, fixed readout, minibatch 4096, B lr x3 with cosine decay, focus 0.5 / noise 0.2),
est_same=1 ridge=0 (user's decision), 30k updates, eval_rollouts=0 (eps only; evaluated at 0, 10k, 20k, 30k).
"""
import subprocess, sys, os

Q = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queue.py')
BASE = ('n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 z_task_frac=0.5 '
        'actor_z_task_frac=0.5 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=10 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0')
SEEDS = 2
TASKS8 = ['stand', 'run', 'flip', 'supine', 'jump_any', 'kick_l', 'split', 'flamingo_r']


def mu(mb, mq):
    return 'mu_b=%s mu_q=%s' % (mb, mq)


# wave -> job name -> (title, [(config name, task, extra flags, repeats)])
WAVES = {
    'e1': {
        'e1_mub': ('mu_b alone (+ controls without retention terms)',
                   [('e1_ctrl', 'run', mu(0, 0), 3)] + [('e1_mb%s' % v, 'run', mu(v, 0), SEEDS) for v in (10, 100, 1000, 10000)]),
        'e1_muq': ('mu_q alone', [('e1_mq%s' % v, 'run', mu(0, v), SEEDS) for v in ('0.01', '0.1', '1', '10')]),
        'e1_both': ('mu_b and mu_q together', [('e1_mb%s_mq%s' % (b, q), 'run', mu(b, q), SEEDS) for b, q in ((100, '0.1'), (1000, '0.1'), (1000, '0.01'), (1000, '1'))]),
    },
    # e1 result (run, 30k updates): trained eps 0.036 without retention terms and the unaligned tasks rise (jump_any +0.12, supine/kick_l +0.03); mu_b alone helps
    # little until it blocks the fit (10000: trained eps 0.19); mu_q alone at 1 gives trained eps 0.062 and LOWERS the eps of every other task by 0.06-0.14; mu_q 10 blocks the fit.
    # e2: refine mu_q around 1 on run, and test the key settings when the fine-tuned task is not run (4 tasks x control / mu_q 1 / mu_b 100 + mu_q 1, 2 seeds).
    'e2': {
        'e2_run': ('refine mu_q around 1 on run', [('e2_mq%s' % v, 'run', mu(0, v), SEEDS) for v in ('0.3', '0.5', '2', '3')] + [('e2_mb100_mq1', 'run', mu(100, 1), SEEDS)]),
        'e2_ta': ('other fine-tuned tasks: supine, split',
                  [('e2_%s_%s' % (n, t), t, f, SEEDS) for t in ('supine', 'split') for n, f in (('ctrl', mu(0, 0)), ('mq1', mu(0, 1)), ('mb100_mq1', mu(100, 1)))]),
        'e2_tb': ('other fine-tuned tasks: jump_any, flamingo_r',
                  [('e2_%s_%s' % (n, t), t, f, SEEDS) for t in ('jump_any', 'flamingo_r') for n, f in (('ctrl', mu(0, 0)), ('mq1', mu(0, 1)), ('mb100_mq1', mu(100, 1)))]),
    },
    # e2 result: on run, mu_q 0.5-1 keeps every other task flat or lower (trained eps .055-.062 against .036), mu_q >= 2 blocks the fit (.10-.14); mb100+mq1 about the same as mq1.
    # Other fine-tuned tasks: supine and split have little forgetting without retention terms (<= +0.016); jump_any and flamingo_r do (+0.08) and mb100+mq1 removes it (others <= -0.03,
    # trained .047 / .065 against .030 / .043); split gets trained .095 against .057 with mq1 or mb100+mq1. e3: one setting across the 8 tasks: mq 0.5, mb100+mq0.5, mb100+mq1
    # on all tasks, plus the controls of the tasks that have none (stand, flip, kick_l). 14 runs per job.
    'e3': {
        'e3_a': ('mu_q 0.5 on 7 tasks', [('e3_mq0.5_%s' % t, t, mu(0, '0.5'), SEEDS) for t in ('supine', 'split', 'jump_any', 'flamingo_r', 'stand', 'flip', 'kick_l')]),
        'e3_b': ('mu_b 100 + mu_q 0.5 on 4 tasks, controls of stand, flip, kick_l',
                 [('e3_mb100_mq0.5_%s' % t, t, mu(100, '0.5'), SEEDS) for t in ('run', 'supine', 'split', 'jump_any')] + [('e3_ctrl_%s' % t, t, mu(0, 0), SEEDS) for t in ('stand', 'flip', 'kick_l')]),
        'e3_c': ('mu_b 100 + mu_q 0.5 on 4 tasks, mu_b 100 + mu_q 1 on stand, flip, kick_l',
                 [('e3_mb100_mq0.5_%s' % t, t, mu(100, '0.5'), SEEDS) for t in ('flamingo_r', 'stand', 'flip', 'kick_l')] + [('e3_mb100_mq1_%s' % t, t, mu(100, 1), SEEDS) for t in ('stand', 'flip', 'kick_l')]),
    },
    # e3b (launched in parallel with e3 to use free slots; at most 10 jobs at a time are fine): two more settings around the candidates, on the 8 tasks, 8 runs per job:
    # mb100 + mq0.3 (a smaller trained-eps price than mq 0.5-1) and mb300 + mq0.5 (a stronger span term).
    'e3b': {
        'e3b_%s_%d' % (n, k): ('mu_b %s + mu_q %s on tasks %d-%d' % (b, q, 4 * k + 1, 4 * k + 4),
                              [('e3b_mb%s_mq%s_%s' % (b, q, t), t, mu(b, q), SEEDS) for t in TASKS8[4 * k:4 * k + 4]])
        for (n, b, q) in (('a', 100, '0.3'), ('b', 300, '0.5')) for k in (0, 1)
    },
}


def add(job, title, items, wave):
    cmds = []
    for cfg, task, extra, reps in items:
        for k in range(1, reps + 1):
            cmds.append('python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=%s %s %s model_dir=models/eps/%s_r%d' % (task, BASE, extra, cfg, k))
    args = [sys.executable, Q, 'add', job, '--time', '1:55:00', '--title', 'Eps study %s: %s (%d runs of 30k updates, one job)' % (wave, title, len(cmds))]
    for c in cmds:
        args += ['--cmd', c]
    subprocess.run(args, check=True)


# e5 (after the streams of e4): the single-task settings (mu_b 100/300, mu_q 0.5) do not keep a task learned earlier in a stream (run eps .04 after its stage, .35 after stage 3, as in the control)
# and the stored latents carried through the stages blow up with mu_b (eps 1e3-1e10, vs 0.5 for the control): suspected ill-conditioned Sigma_old^-1 at ridge=0. One job per configuration, stream s1, 2 seeds.
E5 = [('r1e5_mb100', 'ridge=1e-5 ' + mu(100, '0.5')), ('r1e5_mb300', 'ridge=1e-5 ' + mu(300, '0.5')), ('mb1000', mu(1000, '0.5')), ('mb3000', mu(3000, '0.5')),
      ('a09_mb100', 'alpha=0.9 ' + mu(100, '0.5')), ('a09_mb1000', 'alpha=0.9 ' + mu(1000, '0.5'))]
WAVES['e5'] = {'e5_%s' % n: ('stream s1 (run > jump_any > flamingo_r > supine), %s, 2 seeds' % f, [('e5_stream_s1_%s' % n, 'run,jump_any,flamingo_r,supine', f, 2)]) for n, f in E5}

STREAMS = {'s1': 'run,jump_any,flamingo_r,supine', 's2': 'flamingo_r,split,jump_any,run'}


def e4_wave(settings):
    """Confirmation round (settings = [(mu_b, mu_q), ...] that passed in e3): a third seed of every setting and of the control on the 8 tasks (4 runs per job),
    and two streams of 4 tasks, one stage each (30k updates), with every setting and the control, 2 seeds (2 runs per job, ~30 min each)."""
    jobs = {}
    n = 0
    for sk, tl in STREAMS.items():
        for cn, f in [('ctrl', mu(0, 0))] + [('mb%s_mq%s' % st, mu(*st)) for st in settings]:
            jobs['e4_stream_%d' % n] = ('stream %s (%s), %s, 2 seeds' % (sk, tl.replace(',', ' > '), cn), [('e4_stream_%s_%s' % (sk, cn), tl, f, 2)])
            n += 1
    singles = [('e4_ctrl', mu(0, 0))] + [('e4_mb%s_mq%s' % st, mu(*st)) for st in settings]
    items = [('%s_%s' % (nm, t), t, f, 1) for nm, f in singles for t in TASKS8]
    for k in range(0, len(items), 4):
        jobs['e4_single_%d' % (k // 4)] = ('third seed on single tasks (%d-%d)' % (k + 1, min(k + 4, len(items))), items[k:k + 4])
    return jobs


def main():
    wave = sys.argv[1]
    if wave == 'e4':
        WAVES['e4'] = e4_wave([tuple(x.split(',')) for x in sys.argv[2].split(';')])      # e.g. "100,0.5;300,0.5"
    for job, (title, items) in WAVES[wave].items():
        add(job, title, items, wave)


if __name__ == '__main__':
    main()
