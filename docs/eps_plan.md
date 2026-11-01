# Fast retention study, eps only (autonomous work, started 2026-11-01)

## Instruction from the user
Do not care about returns for now: if a reward is not represented by B, the actor will not solve it. What is needed: when the model is fine-tuned on one task,
the eps of that task drops and the eps of the other tasks stays at the pretrained level. eps drops in a few thousand updates (~30k), so runs are very short.
Settings: est_same=1 and ridge=0 always. One queue entry = one job with many python commands (keeps the priority). Free to push, pull, sbatch, read logs,
think of new experiments, repeat; do NOT push to the tracker; stop when the results are satisfying. CLAUDE.md still applies (never cancel/delete on the cluster).

## Protocol
`tools/queue_eps.py <wave>`: pretrained walker FB, fine-tune on one task (run first), 30k updates, 100k labels, best setup (lam 100, fixed readout, minibatch 4096,
B lr x3 cosine, focus 0.5 noise 0.2, actor focus 0.5), eval_rollouts=0 (eps only; at 0, 10k, 20k, 30k; all 8 tasks), folders `models/eps/<config>_r<k>`.
Report: `tools/sync_from_hpc.sh; python tools/eps_report.py [prefix]`.
Success: eps of the trained task within 0.03 of the no-retention control, and the largest increase of eps on the unaligned tasks (supine, jump_any, kick_l, split, flamingo_r)
<= +0.03 (evaluation noise ~0.01-0.02; stand and flip are aligned with run and are allowed to drop). Then check on a second fine-tuned task (not run) and with 2 repeats.
Unknown before the first wave: how much the other tasks' eps rises at 30k without retention (500k gave +0.08..0.12 on the unaligned tasks).

## Log
