# Decisions

Newest first. One entry per decision: what, why, what else was considered.

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
