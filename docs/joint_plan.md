# Phase 2: F and actor on top of a frozen B (2026-11-02)

Goal (user): after B has been fine-tuned with the retention terms (eps study, docs/eps_plan.md), train F (Q loss only) and the actor (TD3) for many steps,
no FB objective, and reach the single-task upper bounds on all tasks while keeping the tasks that are not trained on.
Code: `fb_continual_temp.py finetune=multi` (B frozen; one stage; z = task latent with prob `multi_task_frac`, otherwise a random latent; reward B(s')^T Sigma^-1 z, Sigma on n_big states).
Queue: `tools/queue_multi.py <wave>`, report: `tools/multi_report.py <name>` (run on the cluster). Runs in `models/multi/<run>`, logs in `logs/*FBfinetune-multi*`.
Offline TD3 upper bounds (1M updates, true reward): stand 971, run 461, flip 939, supine 991, jump_any 396, kick_l 484, split 999, flamingo_r 990.

## Part A: joint training on the 8 task latents (an upper bound, NOT a retention test: the learner sees the latents of the evaluated tasks)
Checkpoint: final stage of the 8-task stream, alpha 0.99, mu_b 300, mu_q 0.5, seed 1 (stage 8 = flip). Mean of the last 3 evaluations, 5 episodes.
- m1 (50% task latents, 50% uniform): stream mu_b 300 stable (stand ~900, flip ~760, jump_any 206-281, kick_l 92-240), single-task checkpoints unstable and worse.
- m2/m3 (one task at a time, 100% of its latent, only that task evaluated): stand 976, run 454, flip 908, supine 993, jump_any 349, kick_l 402, split 998, flamingo_r 990. Control stream: jump_any 147, kick_l 140.
- m4/m5/m6/m7 (8 tasks together, 100% task latents): spread 0.2: stand 941 run 419 flip 872 jump_any 301 kick_l 192; spread 0: stand 959 run 432 flip 875 jump_any 357 kick_l 332 split 998 flamingo_r 987.
  1M updates no better (jump_any 351, kick_l 244); lr 3e-4 worse; TD3+BC 0.1 / 0.5 worse (flip 630 / 330); hard tasks (run, flip, jump_any, kick_l) sampled x3: stand 964 run 436 flip 899 jump_any 371 kick_l 355 (best so far); batch 4096 similar (kick_l ~300).
- Random agent (dm_control, 20 episodes): split 227, flamingo_r 98, supine 640 +- 470, jump_any 77, kick_l 127. Random latent in a trained policy (eval_z=random): split 520-674, flamingo_r 82-227, supine 218-280.

## Part B: the retention test (user's definition)
Fine-tune B on ONE task (with mu_b / mu_q), then train F and the actor on that task only (+ generic latents), and evaluate ALL tasks (not seen in training). Only the labels of the trained task may be used.
- m8 (p = 1, 500k updates): from pretrained, run: run 322->361, supine 989->24, flamingo_r 979->288, split 934->705, stand 697->842, flip 515->730. From B(kick_l): kick_l 84->365, supine 979->10, split 960->103, flamingo_r 964->61.
  From B(run) mu_b 300: run 186->404 but supine 972->24, split 974->671, flamingo_r 713->290. From the 8-stream model on run: run 305->408, supine 992->15, split 997->682. So training the actor/F on one latent erases the others.
- m10/m11 (run latent with prob p, uniform sphere otherwise): p = 0.9, 0.5, 0.25, 0.1: supine 24-89, split 498-647, flamingo_r 253-363 at the end: mixing uniform latents does not help. kick_l with p = 0.5 diverged (Q ~ 1500, q_loss 1e5).
  Lower lr (3e-5) the same. Longer (1M) at p = 0.5: supine 270, split 574 (slightly better).
- Rejected ideas: kd_actor / kd_q (anchoring pi and Q to the starting model on generic latents assume that z keeps its meaning while B changes, and kd_q is mu_q); they were queued once (m10_1) and cancelled before running.
- Hypothesis: the reward B(s')^T Sigma^-1 z of a uniform random z has far more energy than a task reward (Sigma^-1 amplifies low-variance directions), so the random-latent part of the loss is noise. Tests queued: random latents drawn as in the pretraining (B(s) of random states, `multi_bmix`, m12) and isotropic in the whitened B space (`multi_white`, m13).
