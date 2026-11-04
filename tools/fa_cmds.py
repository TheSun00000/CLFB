"""Command builders of the fast-adaptation experiments (see tools/overnight.py). Cluster paths are relative to the repo root on the cluster."""

PRE = '$PRE'
S1 = {'run': {'s1': 'models/eps/r1_nofb_uni_run', 's2': 'models/eps/r2_row4_s2', 's3': 'models/eps/r2_row4_s3'},
      'jump_any': {'s1': 'models/eps/fa_jump_any_s1', 's2': 'models/eps/fa_jump_any_s2', 's3': 'models/eps/fa_jump_any_s3'}}


def stage1(task, seed):
    """B tuned on `task` as in row 4 (L_rec lam 100, retention mu_b 300 mu_q 0.5, ortho 100, 30k updates); F and the actor of that stage are discarded by b_adapter."""
    return ('python $SCRIPT env_name=walker checkpoint=%s tasks=%s n_labeled=100000 num_evals=5 steps_per_stage=30000 lam=100 rec_fixed_w=1 rec_bs=4096 lr_coef=3 b_cosine=1 '
            'z_task_spread=0.2 ztask_bs=4096 bench=final eval_each_steps=0 eval_baseline=1 eval_rollouts=0 est_same=1 ridge=0 ortho_coef=100 mu_b=300 mu_q=0.5 fb_objective=0 '
            'z_task_frac=0 actor_z_task_frac=0 model_dir=%s' % (PRE, task, S1[task][seed]))


def stage2(task, seed, tag, lr=1e-4, frac=0.25, adapter=True, steps=30000, every=2, extra=''):
    """Stage 2 (finetune=multi, B frozen). every = evaluation period in chunks of 1000 updates. Run folder is found through model_dir=models/multi/fa_<tag>."""
    c = ('python $SCRIPT checkpoint=%s/fb_stage_1_%s.pickle env_name=walker n_labeled=100000 num_evals=5 lr_coef=1 bench=final eval_baseline=1 eval_rollouts=1 ridge=0 '
         'finetune=multi steps_per_stage=%d eval_each_steps=%d z_task_spread=0 eval_stream_only=0 tasks=%s multi_task_frac=%g lr=%g'
         % (S1[task][seed], task, steps, every, task, frac, lr))
    if adapter:
        c += ' b_adapter=1 adapter_ckpt=%s' % PRE
    if extra:
        c += ' ' + extra
    return c + ' model_dir=models/multi/fa_%s' % tag
