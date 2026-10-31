# Journal

Newest first. One entry per work session: what was done, what was learned, what comes next.

## 2026-10-31
- Set up key based `ssh jz`, the one-way sync into `hpc/` and `tools/hpc_status.py` (Slurm status of jobs launched from `scripts/`, timer installed by the user, tracker pushed on demand).
- The tracker now lists the 43 jobs submitted since 2026-10-30 (older jobs removed from the tracker only; nothing was touched on the cluster). Titles and descriptions were derived from the commands and the script docstrings, not from results.
- Oct 30 jobs, in order: frozen-B runs with the task latent (spread 0.2, then 0); critic-oracle upper bounds (first two launches failed on an empty `seed=$s`, relaunched as 1725616); single-reward TD3 with learned reward approximations (cancelled, then 15 minute time limit); fine-tuning comparison of FB (actor, critic, reward_critic) against TD3 references R1 and R2 on walker run with 100k and 10k labeled states (a first submission on task=walk was cancelled, relaunched as 1744592).
- No results have been read yet.

## 2026-10-30
- Looked through the repo. Method and flags are documented in the docstring of `fb_continual.py`.
- Created the tracker page, `CLAUDE.md` and this `docs/` skeleton.
- No experiment results recorded yet. Runs are on the HPC; log access to be set up.
- Next: move the key conclusions from the chat project into `docs/method.md` and `docs/decisions.md`.
