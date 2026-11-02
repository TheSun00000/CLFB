# Decisions

Newest first. One entry per decision: what, why, what else was considered.

## 2026-11-01: retention setting for the continual runs (eps study)
- Decision (recommended, not a default in the script): alpha 0.99, mu_b 300, mu_q 0.5, with ridge=0 and est_same=1, 30k updates per stage. The script defaults are unchanged (alpha 0.5, mu_b 1, mu_q 0.01, ridge 1e-5, est_same 0).
- Why: single task (3 seeds): mu_q 0.5 stops the rise of the other tasks' eps; streams of 4 and 8 tasks: only alpha 0.99 keeps the earlier tasks (alpha 0.5 and 0.9 do not, ridge 1e-5 does not help), mu_b 300 keeps more than 100 over 8 stages, stored latents stay valid; without mu_b the learned tasks are lost, without mu_q the tasks not yet learned stop improving. Evidence: docs/eps_plan.md, runs models/eps/*, tracker family 'eps-study'.
- Limits: eps only (returns not checked), one dataset/environment (walker), 2 seeds per stream configuration, mu_b 1000 and longer streams untested; mu_b 300 + mu_q 0.5 failed the single-task criterion on stand in one of three seeds (+0.032).

## 2026-11-01: `est_same` flag (Sigma_old on the step's own n_est states), default off
- Decision: add `est_same` to `fb_continual_temp.py` (default 0, so earlier runs are unchanged). With 1, Sigma_old^-1 used in T, M and the R^B denominator is re-estimated at every step on the same n_est states as Sigma_theta and A, so that R^B = R^Q = 0 exactly at theta = theta_old.
- Why: with Sigma_old on n_big other states, R^B/R^Q have a noise floor (R^B ~0.07, R^Q ~2000 at n_est=4096) and mu_q alone pushes F off its pretrained values (sanity A2 vs A2f). Not made the default: the retention waves and docs/experiments.md were run without it; the stage-end T (n_big fresh states, Sigma_old on others) still has a ~3% mismatch.
- Evidence: waves s1/s2 in docs/sanity_plan.md. Also new flag `q_loss` (default 1) to allow runs where only mu_b R^B and mu_q R^Q train F and B.

## 2026-11-01: sanity runs with lam=0 before more retention sweeps
- Decision: 12 runs with lam=0, ridge=0, uniform latents (z_task_frac=actor_z_task_frac=0), to see what mu_b and mu_q do alone when theta_new = theta_old (no returns should drop). Also fb_objective=0, and a mode where mu_b R^B + mu_q R^Q is the only loss on F and B (new flag `q_loss=0`, default unchanged).
- Why: R^B / R^Q are not 0 at theta = theta_old with the current estimator (Sigma_old and A on different state batches: R^B -0.10, R^Q ~2400 at n_est=4096), so the earlier waves cannot separate real drift from noise. The ridge=0 decision below was used for wave 4 only, which the user cancelled; the sanity runs use ridge=0 too.
- Not done: fixing the estimator (same batch for Sigma_old and A, or larger n_est); asked to the user.

## 2026-11-01: ridge=0 for the retention runs
With B_new = B_old, R^B should be 0. It is only if the ridge is 0: `old_family_matrices` adds the ridge to Sigma_theta and Sigma_old but not to A, so T != I (R^B floor 0.007 at ridge 1e-5 on a random B). The user chose ridge=0. Applied by passing `ridge=0` explicitly in wave 4 (ret_r0_lam05, ret_r0_lam025, lam 0.5 / 0.25); the default in the script is NOT changed (it is recorded in info.txt and docs/experiments.md of earlier runs). Change the default only on request.

## 2026-10-31: new walker tasks defined on the observation, to test retention
- Decision: add six walker tasks to the benchmark list of `fb_continual_temp.py`: `spin_back`, `kick_r`, `lean_a`, `jump`, `knees_bent`, `split`. Their rewards are functions of the 24-d observation (angles of the 7 bodies, joint and torso velocities), carried in `aux[..., 4:28]` (dataset and rollouts). `tools/task_design.py` has the study and the definitions.
- Reason: the five built-in tasks are all functions of the same 4 physics quantities; B fine-tuned on run encodes all of them (eps 0.29-0.38 -> 0.04-0.14, latent cosine 0.91-1.0 with run), and fitting B on one after run costs run nothing (eps_run 0.037 -> 0.035-0.06). The new tasks have reward correlation ~0 with run and, fitted after run, raise eps_run to 0.11-0.26; fitted before run, their own eps rises from 0.01-0.06 to 0.08-0.18 once run is learned (stand, walk, flip: +0.004-0.025). So they can show what mu_b and mu_q protect.
- Alternatives considered: rewards on the same 4 quantities (backward, crouch, lying): still aligned in B space; `feet_up` (not learnable by B, eps 0.44 after the fit), `lean_b` (redundant with `lean_a`), the left-knee/right-knee variants (redundant with `knees_bent`).
- Evidence: B-only fits on 10k labels, 30k steps (`tools/task_design.py --start/--xfirst`), pretrained and round-4 checkpoints; not yet confirmed on zero-shot returns or on the full method (L_FB, ortho).
- Update 2 (random-latent control): of the six, only split is a genuine latent-driven zero-shot ability of the pretrained model (977 vs 386 with a random latent) that is lost after fine-tuning on run (594 vs 602). The others gain little from the latent. A broader set was designed afterwards (tools/reward_library.py, 54 rewards, matrix in the tracker).
- Update: the zero-shot returns of the round-4 checkpoints on these tasks (eval-only job 1793300) are dominated by the default posture of the policy (no run was trained on the tasks), so they neither confirm nor refute the choice. lean_a is not usable (return 29-978 at the same eps); a task needs to be learned first for its return to say anything.

## 2026-10-31: task list = stand, run, flip + supine, jump_any, kick_l, split, flamingo_r
- Decision: the study uses three built-in walker tasks (stand, run, flip) and five new tasks defined on the observation: supine (torso lying horizontal), jump_any (upward velocity >= 3), kick_l (left hip velocity >= 20 rad/s), split (thighs >= 60 deg apart), flamingo_r (right foot >= 0.35 above the ground, left foot on it). `bench=final` in `fb_continual_temp.py` selects exactly this list.
- Reason: priority on low correlation among the chosen tasks (tolerated 0.3), then distinct skills for the story. Largest |corr| among the new tasks 0.08, between a new task and stand, run or flip 0.19 (supine). Walk and vel8 dropped (0.92 and 1.0 with run).
- Alternatives considered: slide_fast (run's speed without the upright term, 0.25 with run), spin_back (flip reversed, 0.45 with kick_l), kicked_up (0.2 with all three kept tasks, 0.79 with knees_bent), the greedy set of the matrix (r_foot_ahead, toes_down, hips_antiphase, ...: gait-like), lean_a, hop_pose (0.73 with jump_any).
- Evidence: tools/reward_library.py and tools/reward_corr.py (|corr| on 100k dataset states, matrix on the tracker). The three kept tasks are correlated with each other (stand-run 0.88, run-flip 0.77, stand-flip 0.77). Not yet checked for supine, jump_any, kick_l, flamingo_r: B-encodability, reachability by a policy, the random-latent control; split is validated.

## Template
### YYYY-MM-DD: short title
- Decision:
- Reason:
- Alternatives considered:
- Evidence (run folder, notebook, discussion):

## Recorded from the code (date of the decision not known)
### Classical FB with a DDPG actor only
- Decision: the continual script always uses classical FB (actor in the bootstrap) with a DDPG actor, on the full dataset.
- Reason: to be filled from the discussions.

### Fixed rank, no growth of the model
- Decision: rank `d` stays fixed across stages; old latents are carried through `T` instead of storing old models.
- Reason: to be filled.

### Store only d numbers per task
- Decision: keep one latent `z_j` per task, updated by `T` at each stage end.
- Reason: to be filled.

## 2026-11-01 - Retention study (waves 1-3, 12 runs): stop, report
Decision: stop the mu_b / mu_q / actor-fraction sweep after 3 waves without meeting the 'others' target. Keep mu_q = 0.01 (larger values hurt flamingo_r), mu_b 300-1000 and actor_z_task_frac 0.25-0.5 as the best partial setting.
Why: B retention restores eps and the actor fix restores supine, but split stays at 59-70% of pretrained and flamingo_r at 61-85%; the loss is in F/actor for those latents. Details and numbers: docs/retention_plan.md (Log).

## 2026-11-02: retention test for F and the actor, and what may be used
- The retention test is: B fine-tuned on one task (with mu_b / mu_q), then F (Q loss only) and the actor (TD3) trained with the labels of THAT task only plus generic latents, then ALL tasks evaluated (they are test tasks: no labels, latents or rewards of them in training). Joint training on all 8 latents (docs/joint_plan.md part A) is an upper bound, not a retention test.
- Rejected: kd_actor / kd_q (an anchor of pi / Q to the starting model assumes that z keeps its meaning while B changes; kd_q duplicates mu_q). No new loss without a principled derivation.
- Result: training on the task latent alone erases the other tasks (supine 989 -> 24); mixing random latents into the training (p = 0.25-0.5 of the task latent) keeps them when starting from the 8-task stream model (run 286 -> 435, supine / split / flamingo_r / stand kept); from single-task B starts, the random latents must be whitened (Sigma of the single-task B is ill-conditioned) and split / flamingo_r are still partly lost. Recommended for now: stream model start, whitened random latents (`multi_white=1`), p = 0.25. Not a script default (flags). jump_any still drops. Evidence and numbers: docs/joint_plan.md.
- Rejected (2026-11-02, user): `multi_randtask` (random latents E[r B] of threshold rewards on one observation coordinate). It gave the best retention (run 433, jump_any 190 -> 232, kick_l 153 -> 287, nothing dropped) but it assumes the test rewards are thresholds of observation coordinates: a prior on the test functions, hard-coded, which would not exist for visual inputs. The flag stays in `fb_continual_temp.py` but must not be used in reported results. Results kept in docs/joint_plan.md (m20) for the record.

## 2026-11-02: Math tab in the tracker
- The tracker has a Math tab (`tracker/index.html`, section `math`) with the full derivation of the method in LaTeX: notation, reward encoding, FB losses, stage objective (L_rec, T, R^B, R^Q, nu), end of stage, metrics, analysis (suboptimality, ceiling, retention chain, capacity). Source of the analysis part: the LaTeX write-up the user pasted on 2026-11-02.
- Typesetting: MathJax 3.2.2 (SVG output) from cdnjs, because the page may load stylesheets only from Google Fonts (KaTeX needs a CSS file). The text is static HTML in the page, not in the artifact database, so changing it needs a republish.
- Not yet in the tab: the fine-tuning modes (`finetune=...`) and a derivation that R^Q is the exact condition for value preservation.

## 2026-11-02 (night 2): orthonormality of B in the stage, ortho_coef = 100
- Finding: with retention terms (mu_b 300, mu_q 0.5) and ortho_coef 1, Sigma = E[B B^T] collapses in a few directions after a single-task stage (smallest eigenvalue 1e-4 to 3e-3; trace Sigma^-1 600-10000 vs 81 for the pretrained B). Cause: mu_q through L_FB (mu_b alone and L_FB off keep Sigma healthy). Consequence: uniform random latents have a reward energy of 3000+ vs 21-27 for tasks.
- Decision (user): the pipeline is redone with ortho_coef = 100 (Sigma = identity, trace 50, from stage 2 on); the B stages and F + actor tests are in docs/joint_plan.md part D. Cost: slightly less retention of the sparse tasks in B (flamingo_r eps .23 vs .15). Not a script default (flag). The earlier checkpoints (ortho_coef 1) stay in models/eps/ for comparison.
- Random latents for the F + actor phase: whitened (multi_white=1) or uniform both keep the other tasks from the new stream model; use whitened when Sigma may be ill conditioned.
