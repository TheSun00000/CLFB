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

### 2026-11-01 - wave e1 result (jobs 1810531-33; 27 runs of 30k updates on run, 2-3 seeds per configuration)
Pretrained eps: stand .295 run .351 flip .335 supine .345 jump_any .417 kick_l .673 split .473 flamingo_r .780.
- trained eps is at its floor after 10k updates (0.034-0.037), so 30k is more than enough (10k would do for run).
- No retention terms (ctrl, 3 seeds): trained 0.036; others: stand/run/flip -0.24/-0.30/-0.20; unaligned supine +0.03, jump_any +0.12, kick_l +0.03, split -0.01, flamingo_r +0.01. So the forgetting is mostly jump_any (+0.12-0.14) at 30k.
- mu_b alone: 10 -> same as ctrl; 100 -> trained .046, jump_any +0.06; 1000 -> .050, +0.08; 10000 -> trained .188 (blocks the fit), jump_any +0.01. mu_b alone does not protect jump_any until it blocks the fit.
- mu_q alone: 0.01, 0.1 -> as ctrl (jump_any +0.09, +0.11); 1 -> trained .062, and EVERY other task's eps falls (supine -0.13, jump_any -0.14, kick_l -0.06, split -0.08, flamingo_r -0.10, in both seeds); 10 -> trained .199, others -0.03..-0.16.
- both: (100, 0.1) jump_any +0.07, (1000, 0.01/0.1) +0.07/+0.08, (1000, 1) trained .078 and others -0.04..+0.03.
Reading: mu_q is the term that holds the others (it freezes F on the old family and F is coupled to B through L_FB); at 1 it costs ~0.026 of trained eps. mu_b does nothing useful on top.
The drop of the others' eps below the pretrained level is unexpected (the retention terms were meant to keep them, not improve them); to understand, not to rely on.
Next (wave e2, 3 jobs, 2 seeds): mu_q 0.3, 0.5, 2, 3 and mb100+mq1 on run; control / mq1 / mb100+mq1 when fine-tuning supine, split, jump_any, flamingo_r (report checks every other task).
