"""Command builders of the fast-adaptation experiments (see tools/overnight.py). Cluster paths are relative to the repo root on the cluster."""

PRE = '$PRE'
# stage-1 checkpoints (B tuned with mu_q=0; user decision 2026-11-04: the old row-4 stage-1 checkpoints, mu_q=0.5, are not used any more)
SEED1 = {('run', 's1'): 101, ('run', 's2'): 102, ('run', 's3'): 103, ('jump_any', 's1'): 201, ('jump_any', 's2'): 202, ('jump_any', 's3'): 203}   # explicit seeds of stage 1 (b_diag rebuilds D_k from them)
CHOSEN = {'run': 'fa2v', 'jump_any': 'fa2v'}     # which copy of the stage-1 checkpoints stage 2 uses: 'fa2' (A100 job) or 'fa2v' (V100 duplicate); set to the copy that finished first


def s1dir(task, seed, prefix=None):
    return 'models/eps/%s_%s_%s' % (prefix or CHOSEN[task], task, seed)


class _S1:
    def __getitem__(self, task):
        return {s: s1dir(task, s) for s in ('s1', 's2', 's3')}


S1 = _S1()


def stage1(task, seed, prefix='fa2'):
    """B tuned on `task` as in row 4 (L_rec lam 100, retention mu_b 300, mu_q 0, ortho 100, 30k updates); F and the actor of that stage are discarded by b_adapter."""
    return ('python $SCRIPT env_name=walker checkpoint=%s tasks=%s n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 '
            'z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=0 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 ortho_coef=100 mu_b=300 mu_q=0 fb_objective=0 '
            'z_task_frac=0 actor_z_task_frac=0 seed=%d model_dir=%s' % (PRE, task, SEED1[(task, seed)], s1dir(task, seed, prefix)))


def stage2(task, seed, tag, lr=1e-4, frac=0.25, adapter=True, steps=200000, every=5, dense_until=50, late_every=20, extra=''):
    """Stage 2 (finetune=multi, B frozen). every = evaluation period in chunks of 1000 updates up to dense_until, then late_every (5k up to 50k, then every 20k). Run folder is found through model_dir=models/multi/fa_<tag>."""
    c = ('python $SCRIPT checkpoint=%s/fb_stage_1_%s.pickle env_name=walker n_labeled=100000 num_evals=5 lr_coef=1 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 '
         'finetune=multi steps_per_stage=%d eval_each_steps=%d eval_dense_until=%d eval_late_every=%d z_task_spread=0 eval_stream_only=0 tasks=%s multi_task_frac=%g lr=%g'
         % (S1[task][seed], task, steps, every, dense_until, late_every, task, frac, lr))
    if adapter:
        c += ' b_adapter=1 adapter_ckpt=%s' % PRE
    if extra:
        c += ' ' + extra
    return c + ' model_dir=models/multi/fa_%s' % tag


def bdiag(task, prefix='fa2'):
    """Quality check of the 3 new stage-1 B's of a task (R^B, principal angles, reward fit eps, latent drift), after the stage-1 commands of the same job."""
    runs = ' '.join('--run %s_%s_%s:%s/fb_stage_1_%s.pickle:%d' % (prefix, task, s, s1dir(task, s, prefix), task, SEED1[(task, s)]) for s in ('s1', 's2', 's3'))
    return 'python tools/b_diag.py --pre %s %s' % (PRE, runs)
