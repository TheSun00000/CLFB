#!/usr/bin/env python3
"""Retention experiments (wave given as argument): the best setup of the return study fine-tuned on run WITH the retention terms mu_b, mu_q,
then a zero-shot evaluation on the 8 tasks in the same job. Reference without retention: the run model of array 1788307 (mu_b = mu_q = 0).
Usage: python tools/queue_retention.py w1      (adds the entries of wave 1 to the queue; the order of the list is the order of submission)
Waves are defined below with their motivation; add a new wave only from the results of the previous one (see docs/retention_plan.md).
"""
import subprocess, sys, os

Q = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queue.py')
BASE = ('n_labeled=100000 num_evals=20 steps_per_stage=500000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 '
        'z_task_frac=0.5 actor_z_task_frac=1.0 z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=10 eval_baseline=1 eval_stream_only=1')

# name -> (flags that differ from the best setup, why)
WAVES = {
    'w1': {
        'ret_mb3_mq001':   ('mu_b=3 mu_q=0.01',   'weak retention of the span of B'),
        'ret_mb30_mq001':  ('mu_b=30 mu_q=0.01',  'medium retention of the span of B'),
        'ret_mb300_mq001': ('mu_b=300 mu_q=0.01', 'strong retention of the span of B'),
        'ret_mb30_mq1':    ('mu_b=30 mu_q=1',     'medium span retention + strong value matching (does the Q term add anything?)'),
    },
    # w1 result: B is kept (supine eps 0.34 -> 0.37 with mu_b=300, FB_run 0.48) but supine return stays ~13: the ACTOR (trained on z_run only,
    # actor_z_task_frac=1.0) forgets the other latents. So w2 gives the actor some uniform latents. Flags appended after BASE override it.
    'w2': {
        'ret_af05_mb300':  ('actor_z_task_frac=0.5 mu_b=300 mu_q=0.01',  'half of the actor updates on uniform latents + strong span retention'),
        'ret_af025_mb300': ('actor_z_task_frac=0.25 mu_b=300 mu_q=0.01', 'only a quarter of actor updates on z_run (is run still learned?)'),
        'ret_af05_mb1000': ('actor_z_task_frac=0.5 mu_b=1000 mu_q=0.01', 'does a stronger R^B add to the actor change? (R_B still 0.1 at 300)'),
        'ret_af05_nor':    ('actor_z_task_frac=0.5 mu_b=0 mu_q=0',      'control: actor change alone, no retention term'),
    },
    # w2 result: actor frac 0.5 restores supine (policy default behaviour); mu_b=1000 restores eps of all old tasks but split stays ~620 and flamingo_r ~690;
    # actor frac 0.25 gives split 681 / flamingo_r 584 with run 394. R_Q still ~1500 at mu_q=0.01 => w3 raises mu_q now that the actor is trained on uniform latents.
    'w3': {
        'ret_af05_mb1000_mq01':   ('actor_z_task_frac=0.5 mu_b=1000 mu_q=0.1',   'value matching 10x stronger on the best w2 config (R_Q ~1500)'),
        'ret_af05_mb1000_mq003':  ('actor_z_task_frac=0.5 mu_b=1000 mu_q=0.03',  'intermediate mu_q in case 0.1 destabilises'),
        'ret_af025_mb1000_mq01':  ('actor_z_task_frac=0.25 mu_b=1000 mu_q=0.1',  'less actor focus (best split in w2) + strong value matching'),
        'ret_af025_mb1000_mq001': ('actor_z_task_frac=0.25 mu_b=1000 mu_q=0.01', 'actor 0.25 with mu_b 1000 (untested pair) as the mu_q=0.01 reference'),
    },
    # w4 (user request 2026-11-01): ridge=0 so that T = I exactly when B_new = B_old (R^B is then exactly 0 at the start of a stage; with ridge 1e-5 it
    # has a floor of ~0.007 on a random B), and a much smaller lam than 100 (0.5, 0.25). Rest = best w2 config (af 0.5, mu_b 1000, mu_q 0.01).
    'w4': {
        'ret_r0_lam05':  ('ridge=0 lam=0.5 actor_z_task_frac=0.5 mu_b=1000 mu_q=0.01',  'ridge 0 and lam 0.5 on the best w2 config'),
        'ret_r0_lam025': ('ridge=0 lam=0.25 actor_z_task_frac=0.5 mu_b=1000 mu_q=0.01', 'ridge 0 and lam 0.25 on the best w2 config'),
    },
}


def main():
    wave = sys.argv[1]
    for name, (flags, why) in WAVES[wave].items():
        train = 'python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run %s %s model_dir=models/ret/%s' % (BASE, flags, name)
        ev = 'python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run bench=final num_evals=20 eval_ckpt=models/ret/%s/fb_stage_1_run.pickle' % name
        subprocess.run([sys.executable, Q, 'add', name, '--time', '1:50:00', '--title', 'Retention %s: best setup on run + %s (%s)' % (wave, flags, why), '--cmd', train, '--cmd', ev], check=True)


if __name__ == '__main__':
    main()
