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

### 2026-11-01 ~09:30 - wave s2 result (8 runs; est_same=1 and the control repeat)
- Noise: control A0 vs repeat A0b: stand +132/+57, split -58/-90, flamingo -52/-83, eps .48/.43 on stand. So single-run return deltas of +-50-100 are noise; B drifts in both (eps .3 -> .43-.49).
- Fix works on the estimator: final R^Q 1228-2079 -> 159 (C2f, C3f), 633 (A2f), 487 (A3f), 544 (A4f), 1052 (B3f); R^B 0.000-0.012 in C3f/A4f/B3f. Still R^B ~0.07 in A1f/A3f: with the pretraining loss on and mu_b=100 there is a real pull on B (A4f with mu_b=1000: 0.007).
  The remaining R^Q ~159 in C (B unchanged, T = I) is probably Adam amplifying float32 noise on a loss that is at its minimum (hypothesis, not tested).
- mu_q alone with the pretraining loss no longer hurts the aligned tasks: A2f stand +64, flip -19, supine -42 (A2: -144, -175, -140), eps back to pretrained (.295 .340 .358 vs A2 .248 .315 .306). Supports the reading that the old harm came from T != I.
- B: eps stays at pretrained level in the runs with mu_b=1000 (A4f .312 .356 .347 ..) and in C/B3f; with mu_b=100 and L_FB on (A1f, A3f) it drifts (.43-.45 / .25-.30).
- Returns of split: -42..-305 in all runs with the pretraining loss and -130..-161 in the five C runs (F and B at the pretrained values), flamingo_r -347..+119. Not removed by the fix.
Conclusion s2: the estimator fix removes the noise floor and the spurious harm of mu_q; what remains (split, flamingo returns, B drift with L_FB) does not depend on mu_b / mu_q.
Wave s3 (last allowed by the plan, 6 runs): `freeze_fb=1` (only the actor trained) 200k x2 and 50k; control A0 at 20k and 50k updates; A0 with lr/3.
Question answered by s3: is the split loss the actor's own drift (freeze_fb), does it grow with steps, and is the B drift an immediate effect of restarting (Adam / target reset) or accumulates.

### 2026-11-01 ~10:30 - wave s3 result (6 runs) and end of the sanity study
- Actor only (freeze_fb=1, F and B exactly pretrained, eps = pretrained +-0.01): returns still move: split -188 (50k), -160 and -48 (200k, two seeds), flip +42 / -159 / +43, stand +61 / +20 / +107, flamingo_r +25 / -54 / +22.
  => the actor's own update, with nothing else changing, moves the returns by up to ~190 per task and by a different amount in each repeat. Single-run differences below ~200 on split are within this actor noise.
- B drift grows with the number of updates (control A0, eps of stand/run/flip): 20k .27 .33 .30 (not worse than pretrained .30 .33 .34), 50k .33 .39 .39, 200k .43-.48 .46-.49 .44-.50; with lr/3 at 200k .34 .38 .41. So it accumulates (scales with lr x steps), it is not a jump at the restart.
- Conclusions of the sanity study (waves s1-s3, 26 runs):
  1. R^B and R^Q alone (group C) keep the model at its pretrained values (eps +-0.03, orth 0.43-0.47) - they pass the check. mu_b=100-1000 also keeps B from collapsing when L_FB is off (B1, B3: eps pretrained vs 0.91 without).
  2. With Sigma_old on other states than A, the terms have a noise floor (R^B ~0.07, R^Q ~2000 at n_est=4096) and mu_q alone then harms the model (A2). `est_same=1` removes the floor (R^Q 160-1050, R^B ~0) and the harm.
  3. Returns are noisy: control repeats and actor-only repeats differ by 50-200 per task. The earlier retention waves (single runs) cannot resolve differences of that size: the split gap (-285 for FB_run, 570-680 in the waves) is larger than this noise, but the differences between waves are not.
  4. Drift not related to mu_b / mu_q: B drifts with the pretraining loss alone (continuing at lr 1e-4 for 200k updates raises eps from .3 to .45), the actor drifts alone.
Next, if continued: re-run the best retention configuration with est_same=1 and 3 seeds (no repeat exists), and cap the drift of B and the actor (fewer steps, lower lr). Nothing else queued.
