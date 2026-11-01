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

### 2026-11-01 ~08:00 - wave s1 result (12 runs, 200k updates, folders 2026-11-01_07-04 ... in logs/, models/ret/san_*; `python tools/retention_report.py`)
Pretrained eps (stand run flip supine jump_any kick_l split flamingo_r): .298 .327 .343 .337 .420 .674 .480 .793. Returns pretrained: stand 711, run 352, flip 515, supine 981, split 962, flamingo_r 813.
- C (only mu terms, L_FB/Q/ortho off): the model stays where it was. eps within +-0.03 of pretrained on all 8 tasks (C1 .290 .321 .330 .343 .433 .673 .476 .765; C2 and C3 alike), orth_l2 0.43-0.47. => R^B and R^Q alone are valid
  regularisers at theta_old (the 'sanity' passes for B and the span). Returns: C1 stand +129, flip +88, split -161; C2 split -130, flamingo +17; C3 split -130, flamingo +72, stand -109. F and B hardly move, so the return changes come from the ACTOR
  (trained 200k updates on uniform z against nearly fixed F): the pretrained actor is not a fixed point of its own update (it was the actor that forgot split in the retention waves too).
- A0 (pretraining loss alone, 200k updates): eps rises .3 -> .48-.5 on stand/run/flip/supine (B drifts away from the pretrained one; hyper-parameters equal to the pretraining run's, fresh Adam and target nets at the start of the stage), returns stand +132, split -58, flamingo -52.
  So part of what the earlier waves called forgetting is drift caused by restarting training on the same data. A repeat is in s2 (san2_A0b) for the noise.
- mu_b: A1 (100) eps .35-.40 (half of the A0 drift), A4 (1000, mu_q 0.01) eps = pretrained (.311 .348 .349 .346 .432 .680 .473 .798): R^B protects the span of B.
- L_FB off: B0 and B2 collapse B (eps .91 on every task, orth 0.22, run 93 / 55); mu_b=100 restores eps to the pretrained level (B1, B3). mu_q does not protect B (B2).
- mu_q alone with the pretraining loss (A2): worse than the control (stand 566, flip 340, supine 841, split 771 vs A0 843/602/989/904); R^Q stays at ~1700 (the noise floor measured before, ~2000); with mu_b it is better (A3 eps .21-.26 on aligned tasks) but split -170, flamingo -287.
  Reading: with Sigma_old on other states, T != I at theta_old (noise 0.1), so the stop-gradient target z' = sg(T z) is a randomly rotated latent and R^Q pulls F off its pretrained values.
- Final R^B/R^Q: A1 .067, A3 .082/1420, A4 -.002/1386, C1 .015, C3 .029/2079, C2 1228: all at the noise floor (R^B ~0.07, R^Q ~2000), none near 0.
Decision (rule 1 of the plan): test the estimator fix. New flag `est_same=1`: Sigma_old re-estimated at every step on the n_est states used for Sigma_theta and A (T = I, M = 0 exactly at theta_old).
Wave s2 (8 runs, jobs 1807955-62): A0 repeat (noise), est_same=1 on A1, A2, A3, A4, B3, C2, C3. Expect R^B ~0 and R^Q ~0 at the first chunks and a smaller effect of mu_q alone on the returns.
