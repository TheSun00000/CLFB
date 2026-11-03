#!/usr/bin/env python3
"""Cheetah, fine-tuning on vel10 from the NEW pretrained model (2026-11-03: saved_models/fb_agent_cheetah_pre2.pickle, FB pretraining of this repo, 1M updates; the old
fb_agent_classical_cheetah.pickle is no longer used). Round 1 (B frozen, no stage 1): stage 2 only (finetune=multi, 300k updates, tasks= the 6 cheetah tasks evaluated, vel10 trained),
1 seed, varying p (multi_task_frac), lr (F and actor), z_task_spread, bc_coef, and F frozen or not. Jobs ft1_*: two runs each. Read with tools/multi_report.py ft1 .
Usage: python tools/queue_cheetah_ft.py r1
"""
import subprocess, sys, os
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import queue_rows_cheetah as R
Q = os.path.join(D, 'queue.py')
PRE2 = 'saved_models/fb_agent_cheetah_pre2.pickle'


def run(name, extra):
    return R.stage2(PRE2, name, extra + ' ').replace('models/cheetah_multi/', 'models/cheetah_ft/')


def add(job, title, cmds, **kw):
    subprocess.run([sys.executable, Q, 'add', job, '--time', '1:55:00', '--title', title] + sum([['--cmd', c] for c in cmds], []), check=True)


if __name__ == '__main__' and sys.argv[1] == 'r1':
    cfgs = [('ft1_fa_p25', ''), ('ft1_fa_p50', 'multi_task_frac=0.5'), ('ft1_fa_p100', 'multi_task_frac=1'),
            ('ft1_fa_p25_lr3', 'lr=3e-5'), ('ft1_fa_p50_lr3', 'multi_task_frac=0.5 lr=3e-5'), ('ft1_fa_p100_lr3', 'multi_task_frac=1 lr=3e-5'),
            ('ft1_act_p25', 'multi_freeze_f=1'), ('ft1_act_p100', 'multi_freeze_f=1 multi_task_frac=1'),
            ('ft1_fa_p50_sp3', 'multi_task_frac=0.5 z_task_spread=0.3'), ('ft1_fa_p50_bc', 'multi_task_frac=0.5 bc_coef=0.1')]
    for i in range(0, len(cfgs), 2):
        pair = cfgs[i:i + 2]
        add('ft1_%d' % (i // 2), 'Cheetah finetune vel10 (new pretrained model, B frozen): ' + ', '.join(n for n, _ in pair),
            [run(n, e) for n, e in pair])


def pipe(name, mu_b, mu_q, extra=''):
    s1 = ('python fb_continual_temp.py env_name=cheetah checkpoint=%s tasks=vel10 %s ortho_coef=100 mu_b=%s mu_q=%s fb_objective=0 z_task_frac=0 actor_z_task_frac=0 '
          'alpha=0 sigma_report=1 model_dir=models/cheetah_eps/%s' % (PRE2, R.B0, mu_b, mu_q, name))
    return [s1, R.stage2('models/cheetah_eps/%s/fb_stage_1_vel10.pickle' % name, name, extra + ' ').replace('models/cheetah_multi/', 'models/cheetah_ft/')]


if __name__ == '__main__' and sys.argv[1] == 'r2':
    P = {'a': [('ft2_fa_p100_bc1', 'multi_task_frac=1 bc_coef=0.1'), ('ft2_fa_p50_bc3', 'multi_task_frac=0.5 bc_coef=0.3'),
               ('ft2_fa_p50_lr3_bc1', 'multi_task_frac=0.5 lr=3e-5 bc_coef=0.1'), ('ft2_fa_p100_lr1', 'multi_task_frac=1 lr=1e-5')],
         'b': [('ft2_fa_p75', 'multi_task_frac=0.75'), ('ft2_act_p50', 'multi_freeze_f=1 multi_task_frac=0.5'),
               ('ft2_fa_p50_tau5', 'multi_task_frac=0.5 fb_target_tau=0.005'), ('ft2_fa_p50_tau20', 'multi_task_frac=0.5 fb_target_tau=0.02')]}
    for k, lst in P.items():
        add('ft2_' + k, 'Cheetah finetune vel10 (new pretrained model, B frozen) round 2: ' + ', '.join(n for n, _ in lst), [run(n, e) for n, e in lst])
    Q3 = {'c': [('ft2_r4_p25', 300, 0.5, ''), ('ft2_r4_p50', 300, 0.5, 'multi_task_frac=0.5'), ('ft2_r4_p100', 300, 0.5, 'multi_task_frac=1')],
          'd': [('ft2_r3_p25', 0, 0, ''), ('ft2_r3_p100', 0, 0, 'multi_task_frac=1')],
          'e': [('ft2_r4_p50_lr3', 300, 0.5, 'multi_task_frac=0.5 lr=3e-5'), ('ft2_r4_p100_bc1', 300, 0.5, 'multi_task_frac=1 bc_coef=0.1')]}
    for k, lst in Q3.items():
        add('ft2_' + k, 'Cheetah finetune vel10 (new pretrained model) round 2, B stage + F/actor: ' + ', '.join(n for n, *_ in lst),
            sum([pipe(n, mb, mq, e) for n, mb, mq, e in lst], []))


def pipe2(name, s1extra, extra):
    c = pipe(name, 300, 0.5, extra)
    c[0] = c[0].replace(' sigma_report=1', ' ' + s1extra + ' sigma_report=1') if s1extra else c[0]
    return c


if __name__ == '__main__' and sys.argv[1] == 'r3':
    # retention on (user 2026-11-03): gentler / different B stages, bc_coef 0 always, stage 2 p = 0.5 or 0.75, 1 seed
    S = {'lr1': 'lr_coef=1', 'lr03': 'lr_coef=0.3', 'st10k': 'steps_per_stage=10000', 'st3k': 'steps_per_stage=3000',
         'mb1000q2': 'mu_b=1000 mu_q=2', 'lam30': 'lam=30', 'lam300': 'lam=300', 'weak': 'mu_b=100 mu_q=0.1'}
    for k, (tag, s1) in enumerate(S.items()):
        cmds = []
        for p in ('0.5', '0.75'):
            n = 'ft3_%s_p%s' % (tag, p.replace('0.', ''))
            cmds += pipe2(n, s1, 'multi_task_frac=%s' % p)
        add('ft3_%d' % k, 'Cheetah finetune vel10 with retention, B stage variant %s (mu_b 300, mu_q 0.5 unless changed), F/actor p 0.5 and 0.75, bc 0' % tag, cmds)


if __name__ == '__main__' and sys.argv[1] == 'r4':
    # round 4: 3 seeds of the two best retention configs of round 3 + new combinations (1 seed); bc_coef 0
    for tag, s1, p in (('lr03', 'lr_coef=0.3', '0.75'), ('weak', 'mu_b=100 mu_q=0.1', '0.75')):
        for s in (1, 2, 3):
            n = 'ft4_%s_p%s_s%d' % (tag, p.replace('0.', ''), s)
            add(n, 'Cheetah finetune vel10 with retention, 3 seeds: B stage %s, F/actor p %s, seed %d' % (tag, p, s), pipe2(n, s1, 'multi_task_frac=' + p))
    for tag, s1, p in (('lr01', 'lr_coef=0.1', '0.75'), ('lr03weak', 'lr_coef=0.3 mu_b=100 mu_q=0.1', '0.75'),
                       ('lr03st10k', 'lr_coef=0.3 steps_per_stage=10000', '0.75'), ('lr03', 'lr_coef=0.3', '1')):
        n = 'ft4_%s_p%s_s1' % (tag, p.replace('0.', ''))
        add(n, 'Cheetah finetune vel10 with retention: B stage %s, F/actor p %s' % (tag, p), pipe2(n, s1, 'multi_task_frac=' + p))


if __name__ == '__main__' and sys.argv[1] == 'r5':
    # clean p sweep with retention (standard row 4 B stage), same seed => same B for every p; bc_coef 0
    for s in (1, 2, 3):
        for part, ps in (('a', ('0.1', '0.25', '0.5')), ('b', ('0.75', '0.9', '1'))):
            cmds = []
            for p in ps:
                n = 'ft5_s%d_p%s' % (s, p.replace('0.', ''))
                c = pipe(n, 300, 0.5, 'multi_task_frac=%s' % p)
                cmds += [x.replace(' model_dir=', ' seed=%d model_dir=' % (100 + s)) for x in c]
            add('ft5_s%d%s' % (s, part), 'Cheetah finetune vel10 p sweep with retention (row 4 B stage, seed %d): p = %s' % (100 + s, ', '.join(ps)), cmds)
