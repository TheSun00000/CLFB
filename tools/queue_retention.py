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
}


def main():
    wave = sys.argv[1]
    for name, (flags, why) in WAVES[wave].items():
        train = 'python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run %s %s model_dir=models/ret/%s' % (BASE, flags, name)
        ev = 'python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run bench=final num_evals=20 eval_ckpt=models/ret/%s/fb_stage_1_run.pickle' % name
        subprocess.run([sys.executable, Q, 'add', name, '--time', '1:50:00', '--title', 'Retention %s: best setup on run + %s (%s)' % (wave, flags, why), '--cmd', train, '--cmd', ev], check=True)


if __name__ == '__main__':
    main()
