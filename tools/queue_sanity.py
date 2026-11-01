#!/usr/bin/env python3
"""Sanity check of the retention terms (2026-11-01): lam=0, uniform latents everywhere (z_task_frac=0, actor_z_task_frac=0), ridge=0, so that
theta_new = theta_old at the start of the stage and nothing asks the model to learn run. The model should not move away from the pretrained one
(zero-shot returns on the 8 tasks close to the pretrained model's). Groups:
  A: pretraining loss on (L_FB + Q loss + ortho), mu_b / mu_q varied.   B: fb_objective=0 (Q loss + ortho, no L_FB).
  C: fb_objective=0 q_loss=0 ortho_coef=0: the only signal on F and B is mu_b * R^B + mu_q * R^Q.
Usage: python tools/queue_sanity.py [s1|s2]     (s1 default: the 12 entries above; s2: the estimator fix, see docs/sanity_plan.md)
"""
import subprocess, sys, os

Q = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queue.py')
BASE = ('n_labeled=100000 num_evals=20 steps_per_stage=200000 lam=0 ridge=0 z_task_frac=0 actor_z_task_frac=0 ztask_bs=4096 '
        'bench=final eval_each_steps=10 eval_baseline=1 eval_stream_only=1')
B_ON, C_ON = 'fb_objective=0', 'fb_objective=0 q_loss=0 ortho_coef=0'
RUNS = {
    'san_A0': ('mu_b=0 mu_q=0',           'control: pretraining loss alone'),
    'san_A1': ('mu_b=100 mu_q=0',         'only R^B'),
    'san_A2': ('mu_b=0 mu_q=0.1',         'only R^Q'),
    'san_A3': ('mu_b=100 mu_q=0.1',       'R^B and R^Q'),
    'san_A4': ('mu_b=1000 mu_q=0.01',     'values of the retention waves'),
    'san_B0': (B_ON + ' mu_b=0 mu_q=0',       'control: Q loss + ortho only (L_FB off)'),
    'san_B1': (B_ON + ' mu_b=100 mu_q=0',     'L_FB off, only R^B'),
    'san_B2': (B_ON + ' mu_b=0 mu_q=0.1',     'L_FB off, only R^Q'),
    'san_B3': (B_ON + ' mu_b=100 mu_q=0.1',   'L_FB off, R^B and R^Q'),
    'san_C1': (C_ON + ' mu_b=100 mu_q=0',     'only signal: R^B'),
    'san_C2': (C_ON + ' mu_b=0 mu_q=0.1',     'only signal: R^Q'),
    'san_C3': (C_ON + ' mu_b=100 mu_q=0.1',   'only signal: R^B and R^Q'),
}

# s2: est_same=1 (Sigma_old on the n_est states of the step: exactly 0 at theta_old) on the runs where the terms moved the model (A2, A3) or the
# ones that stood out (A1, A4, C2, C3, B3), and a repeat of the control A0 for the run-to-run noise.
S2 = {
    'san2_A0b':  ('mu_b=0 mu_q=0',                              'repeat of the control A0 (run-to-run noise)'),
    'san2_A1f':  ('est_same=1 mu_b=100 mu_q=0',                 'A1 with Sigma_old on the same states'),
    'san2_A2f':  ('est_same=1 mu_b=0 mu_q=0.1',                 'A2 with Sigma_old on the same states'),
    'san2_A3f':  ('est_same=1 mu_b=100 mu_q=0.1',               'A3 with Sigma_old on the same states'),
    'san2_A4f':  ('est_same=1 mu_b=1000 mu_q=0.01',             'A4 with Sigma_old on the same states'),
    'san2_B3f':  (B_ON + ' est_same=1 mu_b=100 mu_q=0.1',       'B3 with Sigma_old on the same states'),
    'san2_C2f':  (C_ON + ' est_same=1 mu_b=0 mu_q=0.1',         'C2 with Sigma_old on the same states'),
    'san2_C3f':  (C_ON + ' est_same=1 mu_b=100 mu_q=0.1',       'C3 with Sigma_old on the same states'),
}


def main():
    runs = S2 if len(sys.argv) > 1 and sys.argv[1] == 's2' else RUNS
    for name, (flags, why) in runs.items():
        train = 'python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run %s %s model_dir=models/ret/%s' % (BASE, flags, name)
        ev = 'python fb_continual_temp.py env_name=walker checkpoint=$CKPT tasks=run bench=final num_evals=20 eval_ckpt=models/ret/%s/fb_stage_1_run.pickle' % name
        subprocess.run([sys.executable, Q, 'add', name, '--time', '1:20:00', '--title', 'Sanity (lam=0, ridge=0, uniform z): %s (%s)' % (flags, why),
                        '--cmd', train, '--cmd', ev], check=True)


if __name__ == '__main__':
    main()
