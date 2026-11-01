# Sanity checks of mu_b and mu_q (autonomous work, started 2026-11-01 ~07:00)

Instruction from the user (going to sleep): the goal is to sanity-check the mu_b and mu_q losses. When the experiments finish, launch new ones to
investigate more. Free to push, pull, sbatch. (CLAUDE.md still applies: never cancel or delete on the cluster; no B(s) locally except what the
user explicitly asked.)

## Question
With lam=0, uniform latents and theta_new = theta_old at the start, does anything in the retention terms move the model, and can mu_b / mu_q alone
(fb_objective=0, no Q loss, no ortho) keep it where it is? Expected: zero-shot returns and eps equal to the pretrained model's, R^B and R^Q small.
Known before the runs: the estimator has a noise floor at theta_old (R^B ~0.07, R^Q ~2000 at n_est=4096, Sigma_old on 50000 other states).

## Runs (wave s1, jobs 1807646-55 + san_C2, san_C3)
`tools/queue_sanity.py`: A0-A4 (pretraining loss on), B0-B3 (fb_objective=0), C1-C3 (only mu terms). Report: `tools/sync_from_hpc.sh`, `python tools/retention_report.py`
(columns: returns and eps on 8 tasks vs pretrained, R^B, R^Q at the end), plus training_log.csv (orth_loss, fb_loss, old_latent_length_ratio).

## Decision rules for the next wave
- If R^B / R^Q at the end sit at the noise floor (~0.07 / ~2000) and the models do not move: the terms are dominated by estimator noise -> test the fix
  (Sigma_old estimated on the same n_est batch as A, or n_est=50000) with the same runs C3/A3 before any other sweep.
- If a model without the retention terms (A0, B0) drifts and the one with them (A1-A3, B1-B3) does not: the terms work; report which of mu_b/mu_q does it.
- If C runs (only mu terms) collapse or drift (orth_loss up, returns down): the terms alone are not a valid regularizer; look at which of R^B / R^Q rises.
- At most 2 follow-up waves of <= 8 runs; each must be motivated here before it is submitted.

## Log
