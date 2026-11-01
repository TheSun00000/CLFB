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

### 2026-11-01 01:50 - wave 1 result: FAILED (run return fine, others lost)
Run return 440-450 (FB_run 441) for mb3/mb30/mb300 at mu_q 0.01; mb30_mq1 collapses (run 103, everything lower). Other tasks (mb300 / FB_run / pretrained):
supine 13 / 19 / 981, split 581 / 676 / 962, flamingo_r 342 / 395 / 813, kick_l 157 / 184 / 84. No run meets the 'others' target (supine alone fails it).
Diagnostics: R_B 0.27 (mb3), 0.26 (mb30), 0.11 (mb300); R_Q 1850-2270, 5660 with mu_q 1.
Key observation: with mu_b=300 the eps of supine is 0.37 (pretrained 0.34, FB_run 0.48), split 0.51 (0.48, 0.60), flamingo_r 0.75 (0.79, 0.83), jump_any 0.52 (0.42, 0.63):
B keeps (part of) the span, yet the returns do not come back. => the policy is what forgets: the actor is trained on z_run only (actor_z_task_frac=1.0).
Decision rule 'eps preserved, returns drop' applies: wave 2 = actor_z_task_frac 0.5/0.25 with mu_b 300 (+ mb1000, + a control without retention terms to separate the actor effect from R^B).
Risk: the earlier return study found actor focus 1.0 best for run return; the run target is >= 397.

### 2026-11-01 02:45 - wave 2 (3 of 4 finished; af025 still running) - actor fix works for supine, NOT for split / flamingo_r
Zero-shot return (pretrained / FB_run in brackets: supine 981/19, split 962/676, flamingo_r 813/395, run 352/441):
- ret_af05_nor   (actor 0.5, no retention term): run 443, supine 980, split 637, flamingo_r 499, stand 938, flip 847, kick_l 165.
- ret_af05_mb300 (R_B 0.014, R_Q 1675):          run 401, supine 984, split 574, flamingo_r 665.
- ret_af05_mb1000 (R_B ~0, R_Q 1455):            run 401, supine 868, split 621, flamingo_r 694.
Run target (>= 397) met by all three. 'Others' target (>= 85% of pretrained on supine/split/flamingo_r) NOT met: split 60-66%, flamingo_r 61-85%.
eps: mb1000 brings the others back to the pretrained level (supine 0.36, split 0.47, flamingo 0.77, jump_any 0.50) but the control does not (0.47/0.56/0.82).
Reading: (1) giving the actor uniform latents alone restores supine (980 even though eps 0.47): supine is mostly policy default behaviour (the caveat noted before), so it is a weak retention probe;
(2) R_B restores the eps of every old task but only helps flamingo_r's return (+170-200); split stays at ~600 even with eps back to 0.47 => the policy / F values for the old latents changed:
R_Q is still 1450-1700 (squared value error), mu_q = 0.01 is too small and mu_q=1 (at actor frac 1.0) was unstable.
Next (wave 3, after af025 finishes): mu_q 0.1 and 0.03 on top of af05 + mb1000, since the actor is now stable on uniform latents; the earlier collapse of mq1 happened with actor frac 1.0.

### 2026-11-01 03:15 - wave 2 complete; wave 3 queued
ret_af025_mb300 (R_B 0.056, R_Q 1643): run 394 (just below 397), supine 989, split 681 (71%), flamingo_r 584 (72%), stand 926, flip 744. Fails 'others' (and run by 3).
Wave 2 = no success (waves without success: 2 of 3). Best so far on the old tasks: af025 for split, mb1000 for flamingo_r. Remaining gap is in the policy/F values (R_Q ~1500) -> wave 3 raises mu_q
(0.1, 0.03 at af 0.5 + mb1000; 0.1 and 0.01 at af 0.25 + mb1000). If this wave fails, stop the sweep and report (rule: at most 3 waves).
