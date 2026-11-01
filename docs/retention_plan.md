# Retention plan (autonomous work, started 2026-11-01 ~00:40)

## Instruction from the user (before going to sleep)
Goal: when the model is fine-tuned on a task, the other tasks must not lose performance. Focus on FB_run (the best setup fine-tuned on run) and
measure the other tasks. If they did not drop, fine. If they dropped, make retention work with mu_b and mu_q; code changes allowed; free to
launch experiments, none that is not motivated; no questions needed. Never cancel or delete anything on the cluster. Never run B(s) locally
(cluster only). The queue (tools/queue.py, systemd timer every 3 min) submits whatever fits in the 10 job slots; use `queue.py prio` to order.

## Definitions
- Best setup: lam 100, focus 0.5, noise 0.2, actor focus 1.0, rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1, 100k labels, 500k updates, ztask_bs=4096, bench=final.
- FB_run: the model of job 1788356.0 (models/2026-10-31_18-57-15_FBcontinual-walker-run_5), same setup, mu_b = mu_q = 0.
- Evaluation: zero-shot return and held-out eps on the 8 tasks (stand run flip supine jump_any kick_l split flamingo_r), latent from fresh labels, 20 episodes.
- Report: `python tools/retention_report.py` (after tools/sync_from_hpc.sh). Success of a retention run: run return >= 0.9 x FB_run's;
  tasks the pretrained model does well (>= 300) keep >= 85% of their return; eps drift on unaligned tasks <= +0.03.

## What is known (before wave 1)
Old evaluation, pretrained -> FB_run: stand 689 -> 960, walk 700 -> 875, vel8 409 -> 517, flip 513 -> 911 (aligned with run: better, eps 0.3 -> 0.04-0.14);
unaligned tasks: eps up by 0.08-0.12 (split 0.47 -> 0.59, knees_bent 0.57 -> 0.67, kick_r 0.66 -> 0.78, jump 0.43 -> 0.53) and the returns the pretrained
policy had fall: split 977 -> 672, knees_bent 460 -> 253. So B loses the span of unaligned rewards (-> mu_b) and the policy changes (-> mu_q, actor focus).
supine/jump_any/kick_l/flamingo_r on FB_run: not known yet (entry final_eval_base).

## Wave 1 (motivated by the eps rise of the unaligned tasks: B-level forgetting, so mu_b first; the scale of R^B is unknown, hence a log-spaced sweep)
ret_mb3_mq001, ret_mb30_mq001, ret_mb300_mq001, ret_mb30_mq1 (tools/queue_retention.py w1). Read R_B, R_Q, run return/eps, other tasks.

## Decision rules after a wave
- If a run meets all three targets: stop the sweep, verify it with a second seed, write it up (journal, decisions, tracker Updates + family 'retention').
- If mu_b keeps the eps/returns of the other tasks but the run return drops below 0.9 x FB_run: the trade-off is the lam / mu_b ratio -> next wave: raise lam (300) with the best mu_b, or relax the actor focus; 3-4 runs.
- If eps of the other tasks is preserved but their returns still drop: the policy changed, not B -> next wave on mu_q (1, 10) and actor_z_task_frac 0.5 (the actor then sees uniform latents too).
- If neither helps: look at R_B / R_Q values (are they small? then the terms are too weak: raise by x10) and the training curves before adding experiments.
- At most 4 runs per wave, at most 3 waves without a success; then stop and report what was learned.

## Log
(append dated entries: wave submitted, results, decision)

### 2026-11-01 00:50 - measured (final_eval_base, TD3 upper bounds, FT supine)
Zero-shot return, pretrained -> FB_run (TD3 upper bound): stand 711 -> 965 (971), run 352 -> 441 (461), flip 515 -> 906 (939),
supine 981 -> 19 (991), jump_any 104 -> 126 (396), kick_l 84 -> 184 (484), split 962 -> 676 (999), flamingo_r 813 -> 395 (990).
=> FB_run is near the upper bound on stand/run/flip but LOSES a lot on the unaligned tasks the pretrained policy does well: supine -962, flamingo_r -418, split -285.
FT_supine (best setup on supine): supine 991 (= upper bound) but stand 108, run 26, flip 91: the forgetting is symmetric.
(Caveat: part of supine/split/flamingo_r at the pretrained model may be default behaviour of the policy, not the latent: only split validated by the random-latent control.)
Wave 1 submitted at 00:45: jobs 1802637 (mb30), 1802638 (mb3), 1802639 (mb300), 1802640 (mb30 mq1); each ~60-70 min + eval. Entries final_ft_* (kick_l, split, flamingo_r, stand, flip) also running.

### 2026-11-01 01:13 - wave 1 half way (step ~240k of 500k, no errors, 5.6 s/1k updates -> ~47 min + eval)
Diagnostics (mean of the last chunks): R_B = share of old-family reward energy lost: mb3 0.265, mb30 0.247, mb300 0.104, mb30_mq1 0.520 (first chunk 0.002-0.04).
=> mu_b = 3 and 30 barely change R_B (L_rec lam 100 dominates); only mu_b = 300 moves it (x2.5 lower), still 0.10. R_Q (squared value error) is huge (2000-2700, 5800 with mu_q 1),
mu_q=1 destabilises the run (run return 75 at 240k, orth 1.7) -> mq1 is a dead end at this scale; mu_q 0.01 is already worth ~25 in the loss.
Run return at 240k: mb3 434, mb30 411, mb300 426 (eps 0.03): the retention terms do not hurt the run so far.
Wait for the final evaluation of the other tasks (expected ~01:40) before wave 2; likely wave 2: mu_b 1000 and 3000 (R_B still >0.1 at 300), mu_q 0.01 and 0.
