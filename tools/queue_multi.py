#!/usr/bin/env python3
"""Phase 2 of the eps study (2026-11-01): B frozen (checkpoints of the best retention setups), F trained with the Q loss and the actor with TD3 on ALL 8 task latents
(multi_task_frac of the z, the rest uniform), no FB objective, 500k updates, returns evaluated. Script: fb_continual_temp.py finetune=multi.
Usage: python tools/queue_multi.py smoke | <wave>
"""
import subprocess, sys, os
Q = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queue.py')
T8 = 'stand,run,flip,supine,jump_any,kick_l,split,flamingo_r'
BASE = ('env_name=walker tasks=%s n_labeled=100000 num_evals=5 lr_coef=1 z_task_spread=0.2 multi_task_frac=0.5 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 finetune=multi' % T8)
S8 = 'models/eps/e7_stream_s8_a099_mb300_r1/fb_stage_8_flip.pickle'


def cmdline(job, ckpt, steps, every, extra=''):
    return 'python fb_continual_temp.py checkpoint=%s %s steps_per_stage=%d eval_each_steps=%d %s model_dir=models/multi/%s' % (ckpt, BASE, steps, every, extra, job)


def add(job, title, runs, tlimit='1:55:00', steps=500000, every=50, extra=''):
    # one job = several runs [(run name, checkpoint)] one after the other
    cmds = [cmdline(n, c, steps, every, extra if isinstance(extra, str) else extra[n]) for n, c in runs]
    args = [sys.executable, Q, 'add', job, '--time', tlimit, '--title', title]
    for c in cmds:
        args += ['--cmd', c]
    subprocess.run(args, check=True)


if __name__ == '__main__':
    if sys.argv[1] == 'smoke':
        add('m0_smoke', 'Multi-task F + actor training, smoke test (10k updates, speed check)', [('m0_smoke', S8)], '0:40:00', 10000, 5)
    if sys.argv[1] == 'm1':
        E = 'models/eps/'
        PRE = 'models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle'
        st = lambda n, r: ('m1_s8_%s_r%d' % (n, r), E + 'e7_stream_s8_%s_r%d/fb_stage_8_flip.pickle' % ({'mb300': 'a099_mb300', 'ctrl': 'ctrl'}[n], r))
        sing = lambda kind, t: ('m1_%s_%s' % (kind, t), E + ('e2_mq0.5_r1' if (kind, t) == ('mq05', 'run') else 'e3_%s_%s_r1' % ({'mq05': 'mq0.5', 'mb100': 'mb100_mq0.5'}[kind], t)) + '/fb_stage_1_%s.pickle' % t)
        T = ['run', 'stand', 'flip', 'supine', 'jump_any', 'kick_l', 'split', 'flamingo_r']
        runs = [st('mb300', 1), st('mb300', 2), st('ctrl', 1), ('m1_pretrained', PRE)] + [sing(k, t) for k in ('mq05', 'mb100') for t in T]
        for i in range(0, len(runs), 3):
            add('m1_%d' % (i // 3), 'Multi-task F + actor, 500k updates, B frozen: ' + ', '.join(n[3:] for n, _ in runs[i:i + 3]), runs[i:i + 3])

    if sys.argv[1] == 'm2':
        # sanity: 100% task-driven z, only the latent of run (spread 0.2, and 0 = a fixed latent), returns on run only
        X = 'tasks=run multi_task_frac=1 eval_stream_only=1 num_evals=10'
        PRE = 'models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle'
        runs = [('m2_pre_run_sp02', PRE), ('m2_pre_run_sp0', PRE), ('m2_mq05_run_sp02', 'models/eps/e2_mq0.5_r1/fb_stage_1_run.pickle')]
        add('m2_0', 'Sanity: F + actor on 100% run latent (pretrained spread 0.2 and 0, mu_q 0.5 run checkpoint), 500k updates', runs, extra={n: X + (' z_task_spread=0' if n.endswith('sp0') else '') for n, _ in runs})
    if sys.argv[1] == 'm2b':
        # same sanity on STREAM checkpoints (final stage of the 8-task stream): retention setting mu_b 300 + mu_q 0.5, and the control stream
        X = 'tasks=run multi_task_frac=1 eval_stream_only=1 num_evals=10'
        E = 'models/eps/'
        runs = [('m2_s8mb300_run_sp02', E + 'e7_stream_s8_a099_mb300_r1/fb_stage_8_flip.pickle'), ('m2_s8ctrl_run_sp02', E + 'e7_stream_s8_ctrl_r1/fb_stage_8_flip.pickle')]
        add('m2_1', 'Sanity: F + actor on 100% run latent from the 8-task stream checkpoints (mu_b 300 + mu_q 0.5, and control), 500k updates', runs, extra=X)
    if sys.argv[1] == 'm3':
        # one task at a time, 100% of its latent (spread 0.2), from the 8-task stream checkpoints (final stage): mu_b 300 + mu_q 0.5 and control; run is done in m2
        E = 'models/eps/'
        CK = {'mb300': E + 'e7_stream_s8_a099_mb300_r1/fb_stage_8_flip.pickle', 'ctrl': E + 'e7_stream_s8_ctrl_r1/fb_stage_8_flip.pickle'}
        T = ['stand', 'flip', 'supine', 'jump_any', 'kick_l', 'split', 'flamingo_r']
        runs = [('m3_%s_%s' % (c, t), CK[c], t) for t in T for c in CK]
        for i in range(0, len(runs), 3):
            chunk = runs[i:i + 3]
            add('m3_%d' % (i // 3), 'One task at a time, 100%% of its latent, stream checkpoints: ' + ', '.join(n[3:] for n, _, _ in chunk), [(n, c) for n, c, _ in chunk],
                extra={n: 'tasks=%s multi_task_frac=1 eval_stream_only=1 num_evals=10' % t for n, _, t in chunk})
