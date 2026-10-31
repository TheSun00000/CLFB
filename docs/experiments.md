# Experiments

## Where runs happen
On the HPC cluster. Each run writes `logs/<timestamp>_FBcontinual-<env>-<tasks>/` with `training_log.csv`, `eval_log.csv`, `info.txt` (flags and config) and a copy of the script.

## Reading the logs
- `training_log.csv`: one row per chunk of `scan_length` updates: FB metrics, `l_rec`, `r_b`, `r_q`, `z_shrink`, `total_loss`.
- `eval_log.csv`: one row per evaluation (columns `stage`, `step`, `tag`, `mean_fresh_all`, `mean_fresh_tasks`, `mean_tracked_tasks`, then per task `<task>_{fresh,tracked}_{mean,std,rew_corr,eps}`, `time`). `tag=baseline` is the pretrained checkpoint before stage 1.

## Reporting a run on the tracker
Add one Experiments row with: name, date, run folder, status, config (only the flags that differ from the defaults), the headline numbers (fresh and tracked mean return per task, final `eps`), a verdict, and notes. Always include the run folder name.

## Runs
No results recorded yet. Local runs from 2026-10-28 (`walker`, `walk+run`, five runs) are debugging runs and are not reported.

| Date | Run folder | Env / tasks | Differs from defaults | Headline | Verdict |
|------|------------|-------------|-----------------------|----------|---------|
