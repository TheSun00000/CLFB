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
