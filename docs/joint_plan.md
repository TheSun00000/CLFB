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
- Hypothesis: the reward B(s')^T Sigma^-1 z of a uniform random z has far more energy than a task reward (Sigma^-1 amplifies low-variance directions), so the random-latent part of the loss is noise.
  Spectrum of Sigma (`sigma_report=1`, job 1838255): pretrained B and 8-task stream B are well conditioned (smallest eigenvalue 0.055, trace Sigma^-1 81-84; reward energy of a uniform latent 81-85 vs 50 whitened vs 25-55 for task latents);
  the B fine-tuned on run only (mu_b 300 + mu_q 0.5) is not (smallest eigenvalue 0.00035, trace Sigma^-1 3399; uniform-latent energy 3394, q90 8272 vs 51 whitened vs 21-27 for tasks). So the energy explanation holds for the single-task B starts only.

## Part C: results of the random-latent variants (retention test, B frozen, 300k updates unless said, 5 episodes, mean of the last 3 evaluations; start = B(run) mu_b 300 unless said)
- m12 (random latents = B(s) of random states, `multi_bmix`): p = 0.5, bmix 0.5: supine kept (989) but split 617, flamingo_r 300; p = 0.9: supine 913, split 616; bmix 1: supine 671.
- m13 (random latents whitened, `multi_white=1`): p = 0.5: run 333->438, supine 946->994, stand 388->945, flip 673->830, but split 990->657, flamingo_r 746->332, jump_any 160->124. p = 0.9: run 402, flip 917, supine 966, split 714.
  Whitened + B(s) mix: run 435, supine 981, split 714, flamingo_r 274.
- m14: whitened p = 0.25: run 445, supine 957, split 512, flamingo_r 521; p = 0.1: run 371, split 628; repeat of p = 0.5: run 425, supine 994, split 564, flamingo_r 303 (so differences < 50 are noise);
  1M updates: p = 0.5 run 439, split 536; p = 0.25 run 422, split 681, flamingo_r 832. From the kick_l B (p = 0.5 / 0.25): kick_l 39->258 / 79->189, supine kept, split 887->595 / 973->399. From the pretrained model, p = 0.5: run 330->364, supine 988->983, split 955->646.
- m15/m16/m17/m18 (START = 8-task stream model, whitened random latents, learning ONE task): the stream model is much more robust than the single-task B starts.
  learning run, p = 0.25: run 286->435 (m15), 400 (repeat), 424 (stream seed 2), 440 (batch 4096), 415 (1M); p = 0.5: 432; p = 0.1: 427 (388 mean of last 3). supine, split, flamingo_r, stand kept (supine 974->991, split 993->993, flamingo_r 976->975, stand 870->962), flip rises (707->829); jump_any drops 212->138 at every p.
  learning kick_l, p = 0.25: kick_l 168->333 (bound 484), all others kept (jump_any 217->249). learning jump_any: 207->304, flip drops 703->627. learning flip: 731->795, others kept. learning stand: 878->904, jump_any 210->131.
  learning supine: kick_l 69->106, flip 586->666. learning split: split 989->997 but flip 683->503, stand 861->783, kick_l 120->38. learning flamingo_r: run 282->271, flip 683->649, jump_any 210->158.
- m19 (stream model, uniform / B(s) random latents instead of whitened, learning run): uniform p = 0.25: run 372, others kept; uniform p = 0.5: run 401; bmix 0.5 p = 0.25: run 352. So from a well-conditioned B the whitening is not what keeps the other tasks; random latents at p <= 0.5 are.
- Still open: jump_any (and some pairs: flip after split / jump_any) drops when another task is learned.

## Running / queued when the laptop was shut down (2026-11-02, night)
- m20_a (1840703), m20_b (1840704): random latents that look like tasks (`multi_randtask`: E[r B] for smooth threshold rewards on one observation coordinate, random sign and quantile; prior-based, flagged as such), stream model learning run (3 variants) and kick_l / split / jump_any.
- m21_a (1840712), m21_b (1840713): the main candidate (stream model, whitened p = 0.25, learning run) 3 times with 20 episodes per task, and run-only baseline (p = 1), uniform p = 0.25, whitened p = 0.5, also 20 episodes.
- To read: `ssh jz` then `python3 tools/multi_report.py m20_` / `m21_` in the repo (git pull first).
