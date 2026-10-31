# CLFB: continual learning with Behavioral Foundation Models

Research code. A pretrained forward-backward (FB) model encodes a stream of reward tasks, one stage per task, at fixed rank, while keeping the latents of earlier tasks valid. Full method: `docs/method.md`.

## Shared context (read first)
- `docs/method.md`: method, notation, losses.
- `docs/decisions.md`: decisions and why. Add an entry whenever one is made.
- `docs/journal.md`: dated session notes. Add one at the end of each work session.
- `docs/experiments.md`: how runs are named, logged and reported.
- Tracker web page (progress, experiments, updates, slides): https://claude.ai/artifact/Wjzxw6x8Jk4igUg1MCNSeK . Source: `tracker/index.html`. Its data lives in the artifact database; update it with the ArtifactData tool, not by republishing the page.
- A claude.ai chat project, "Continual Learning with Behavioral Foundation Models", holds earlier discussions. Claude Code cannot read it. Conclusions from it are copied into `docs/`.

## Layout
- `fb_continual.py`: single script for training and evaluation. The flags and the log format are documented in its module docstring.
- `_utils/`: `logs.py` (`parse_kwargs`), `wrappers.py` (environments).
- `saved_models/`: pretrained FB pickles (`fb_agent_classical_{cheetah,walker}.pickle`), keys `fb_params` and `actor_params`.
- `logs/<timestamp>_FBcontinual-<env>-<tasks>/`: `training_log.csv`, `eval_log.csv`, `info.txt`, a copy of the script.
- `models/<run>/`: checkpoints. `csv_files/`: exported CSVs for plotting.
- `b_reward_test.ipynb`, `plot_csv.ipynb`: analysis notebooks.
- `tracker/index.html`: source of the tracker page.
- `logs/`, `models/`, `saved_models/`, `csv_files/` and `.slurm_logs/` are gitignored.

## Running
```
python fb_continual.py env_name=walker checkpoint=saved_models/fb_agent_classical_walker.pickle tasks=walk,run
```
Flags are `key=value` (no dashes). Required: `env_name`, `checkpoint`, `tasks`. See the docstring for the rest (`lam`, `mu_b`, `mu_q`, `alpha`, `steps_per_stage`, `n_labeled`, `cross_fit`, `bc_coef`, ...).

## Environments
- Experiments run on the HPC cluster. There is a clone of this repo there; the local machine is for editing, notebooks and light checks.
- The script detects local vs cluster by whether the working directory contains `nazim`. This sets the dataset path (`../offline_qd/datasets/d4rl/` locally, `../fre_experiments/datasets/` on the cluster) and, on the cluster, `MUJOCO_GL=egl` and single-threaded BLAS.
- Stack: JAX, Flax, Optax.
- Results from the cluster do not exist locally unless copied. Do not assume the local `logs/` reflect the latest runs.

## Conventions
- Keep the script a single file with flat `key=value` flags. Match the existing style (pure helpers of the method near the top, `__main__` below).
- Do not change defaults silently: they are recorded in `info.txt` for each run and in `docs/experiments.md`.
- Log new metrics as extra CSV columns. Do not rename existing columns, because the plotting notebooks and the tracker depend on them.
- Do not commit logs, models or datasets.
- Keep notes factual: no result goes into `docs/` or the tracker without the run folder it came from.

## HPC (Jean Zay)
- Access: `ssh jz` (key based, set up in `~/.ssh/config`). Over a plain `ssh jz 'cmd'` the variable `$WORK` is empty, so run `ssh jz 'bash -l -s'` with the commands on stdin, or `ssh jz 'bash -lc "..."'`. Repo clone: `/lustre/fswork/projects/rech/vcj/uyo96iu/CLFB`.
- Read-only by default. Never cancel, delete or overwrite anything on the cluster without asking.
- One-way sync of logs and code into `hpc/` (gitignored): `tools/sync_from_hpc.sh`. Nothing is ever pushed to the cluster by it.
- All sbatch files are in `scripts/` on the cluster (`a100_dev_jax_array.sh` is the main one: it runs `COMMANDS[$SLURM_ARRAY_TASK_ID]`, one array task per command; each job's `.slurm_logs/<script>_<jobid>.out` starts with `Running: <command>` and prints `LOGS_FOLDER`).
- Launching an experiment: when the user asks to add one, write a new `.sh` in `scripts/` on the cluster (copy the header of `a100_dev_jax_array.sh`, set `--array` to the number of commands, put the commands in `COMMANDS`), submit it with `sbatch`, then tell the user the job id and add the experiment to the tracker with a title and a description. The user has authorized this submission step; cancelling jobs is not covered.

## Job status and tracker
- `tools/hpc_status.py --diff` (read-only, one ssh call) reads Slurm for jobs launched from `scripts/` and writes `hpc/status/jobs.json` and `hpc/status/snapshot.json`. A systemd user timer (`tools/systemd/clfb-hpc-status.{service,timer}`, every 2 minutes) runs it once installed by the user.
- The scripts cannot write to the tracker; only Claude can, through the ArtifactData tool. To refresh it ("update the tracker"), in this order:
  1. `tools/sync_from_hpc.sh` (the headline numbers are read from the local copy of the logs, so a stale copy gives stale or missing results);
  2. `tools/hpc_status.py --diff`; if it prints "changed since last push":
  3. `ArtifactData get` of `snapshot/jobs` (with `out_dir`) for its `version`, then `set` it from `hpc/status/snapshot.json` with `if_version`;
  4. `tools/hpc_status.py --mark-pushed`.
- Tracker data: `snapshot/jobs` holds every job since 2026-10-30 (machine-written, includes `results` for finished jobs, see `tools/job_results.py`). Collection `tracking`, one doc per job id, holds what a person or Claude decides: `tracked`, `title`, `description`, `family`, `variant`, `order`. Collection `families` (title, order, description) defines the families of the Results tab. The Experiments tab lists tracked jobs for monitoring; the Results tab groups finished tracked jobs by family and variant and reports mean ± std across runs of the same variant.
- New jobs are untracked by default. When the user says a job is an experiment, add its `tracking/<jobId>` doc (and for an array, every task of it): `tools/describe_jobs.py` writes title, description, family and variant from the command (add a branch there for a new kind of experiment), then push the files it writes under `hpc/status/tracking/` with an ArtifactData batch (50 writes at most per batch; `if_version` is required for documents that already exist).

## Updating the shared context
At the end of a session: append to `docs/journal.md`, record any decision in `docs/decisions.md`, and add an Updates entry to the tracker. When runs finish, add Experiments rows to the tracker with the run folder name.
