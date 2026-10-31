# Decisions

Newest first. One entry per decision: what, why, what else was considered.

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
