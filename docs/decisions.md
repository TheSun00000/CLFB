# Decisions

Newest first. One entry per decision: what, why, what else was considered.

## 2026-10-31: new walker tasks defined on the observation, to test retention
- Decision: add six walker tasks to the benchmark list of `fb_continual_temp.py`: `spin_back`, `kick_r`, `lean_a`, `jump`, `knees_bent`, `split`. Their rewards are functions of the 24-d observation (angles of the 7 bodies, joint and torso velocities), carried in `aux[..., 4:28]` (dataset and rollouts). `tools/task_design.py` has the study and the definitions.
- Reason: the five built-in tasks are all functions of the same 4 physics quantities; B fine-tuned on run encodes all of them (eps 0.29-0.38 -> 0.04-0.14, latent cosine 0.91-1.0 with run), and fitting B on one after run costs run nothing (eps_run 0.037 -> 0.035-0.06). The new tasks have reward correlation ~0 with run and, fitted after run, raise eps_run to 0.11-0.26; fitted before run, their own eps rises from 0.01-0.06 to 0.08-0.18 once run is learned (stand, walk, flip: +0.004-0.025). So they can show what mu_b and mu_q protect.
- Alternatives considered: rewards on the same 4 quantities (backward, crouch, lying): still aligned in B space; `feet_up` (not learnable by B, eps 0.44 after the fit), `lean_b` (redundant with `lean_a`), the left-knee/right-knee variants (redundant with `knees_bent`).
- Evidence: B-only fits on 10k labels, 30k steps (`tools/task_design.py --start/--xfirst`), pretrained and round-4 checkpoints; not yet confirmed on zero-shot returns or on the full method (L_FB, ortho).

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
