"""Titles, descriptions, families and variants for the tracked jobs (what the logs cannot tell).

Reads hpc/status/snapshot.json and writes, for every job, hpc/status/tracking/<jobId>.json (full tracking document)
and hpc/status/tracking_family/<jobId>.json (only family, variant, order: for updating documents that already exist).
Also writes hpc/status/families.json. A new kind of experiment needs a new branch in describe() and classify().
Run from the repo root: python3 tools/describe_jobs.py
"""
import json, os, re
jobs = json.load(open('hpc/status/snapshot.json'))['jobs']
CK = "models/2026-10-29_14-44-51_FBpretrain-walker (pretrained FB, walker)"

def flag(cmd, k, default=None):
    m = re.search(r'(?:^|\s)%s=(\S+)' % re.escape(k), cmd)
    return m.group(1) if m else default

def abl(cmd):
    """Variant name of an E3 ablation (reinit_critic / lr / fb_target_tau overrides), '' if the command is a plain run."""
    parts = []
    if flag(cmd, 'reinit_critic') == '1':
        parts.append('F and actor re-initialised')
    if flag(cmd, 'lr') or flag(cmd, 'fb_target_tau'):
        parts.append('lr %s, tau %s' % (flag(cmd, 'lr', 'default'), flag(cmd, 'fb_target_tau', 'default')))
    return ' + '.join(parts)

def abl_text(cmd):
    return abl(cmd) + ' (the TD3 reference uses lr 3e-4, tau 0.005, a critic from scratch)'

def describe(j):
    cmd, st = j['command'] or '', j['status']
    arr = j['arrayJob'] or j['jobId']
    tail = ''
    if 'fb_critic_oracle.py' in cmd:
        real = flag(cmd, 'q_true_reward')
        what = 'true task reward' if real == '1' else 'implicit reward B Sigma^-1 z'
        seed = [t for t in cmd.split() if t.startswith('seed=')][-1].split('=')[1]
        title = 'Critic oracle, %s' % what + ('' if seed == '$s' else ', seed %s' % seed)
        desc = ('Diagnostic upper bound on walker run. B is pretrained and frozen; F and the actor are fine-tuned on one task '
                'with the task latent as the only z and the Q loss only (no FB objective, no orthonormality, no continual terms). '
                'The TD target uses the %s. 200k steps.' % ('TRUE reward of every dataset state (oracle)' if real == '1' else 'implicit reward of the original auxiliary loss (version 1)'))
        if seed == '$s':
            tail = ' Failed at start: the launch loop left seed empty (seed=$s), so the script stopped while parsing arguments; relaunched as job 1725616.'
    elif 'fb_continual.py' in cmd and flag(cmd, 'freeze_b') == '1':
        spread = flag(cmd, 'z_task_spread')
        title = 'Frozen B, task latent, spread %s' % spread
        desc = ('Diagnostic on walker run (500k steps, ortho_coef=0, lam=mu_b=mu_q=0, so the continual terms are off). B is frozen; F and the actor '
                'train with z equal to the current task latent (z_task_frac=1) plus angular noise of %s (0 = exactly the latent). '
                'Question: how well does the policy do when B and the task latent are fixed and only F and the actor adapt?' % spread)
    elif 'offline_rl_approx.py' in cmd and flag(cmd, 'reward_approx') in ('true', 'lrec_detach', 'lrec_full', 'fixed_z', 'mlp'):
        ra, an, nt = flag(cmd, 'reward_approx'), flag(cmd, 'approx_n'), flag(cmd, 'n_transitions')
        title = 'Single-reward TD3, reward = %s%s' % (ra, ', approx_n %s' % an if an else '')
        desc = ('Single-reward offline TD3 baseline on walker walk (no FB), trained on a learned approximation of the reward instead of the true one: '
                'reward_approx=%s%s. Evaluation is always on the true reward, so the curves compare with reward_approx=true.' % (ra, '' if not an else ' fit on %s dataset states' % an))
        if nt:
            desc += ' Dataset subsampled to %s transitions.' % nt
        if arr in ('1741653', '1741821'):
            tail = ' Cancelled after about 6 minutes; resubmitted with n_transitions=100000 and approx_n=10000 (jobs 1742199 and following).'
        if arr in ('1742199',) or j['jobId'] in ('1742345', '1742346', '1742368', '1742199'):
            tail = ' Stopped by the 15 minute time limit.'
    elif 'offline_rl_approx.py' in cmd:
        ra, an, task = flag(cmd, 'reward_approx'), flag(cmd, 'approx_n'), flag(cmd, 'task')
        ref = 'R1' if ra == 'fb_pretrained' else 'R2'
        title = 'TD3 reference %s (%s), %s labeled states, %s' % (ref, ra, an, task)
        desc = ('Single-reward TD3 from scratch on walker %s with the reward of the fine-tuning options. '
                '%s: reward = B^T w from the pretrained FB (w = Sigma^-1 z); this is the reward that finetune=critic trains on. '
                % (task, 'R1' if ref == 'R1' else 'R2')
                ) if ref == 'R1' else (
                'Single-reward TD3 from scratch on walker %s. R2: same B and w, then B regressed with w fixed (warm-started fixed-z experiment); '
                'this is the reward that finetune=reward_critic trains on.' % task)
        desc += ' Label budget: %s states.' % an
    else:
        ft, nl, task = flag(cmd, 'finetune'), flag(cmd, 'n_labeled'), flag(cmd, 'tasks')
        names = {'actor': ('E1', 'F and B frozen, only the actor is trained'),
                 'critic': ('E2', 'B frozen; F trained with the Q loss only on reward B^T w_q (w_q = Sigma_old^-1 z); actor trained'),
                 'reward_critic': ('E3', 'first B is regressed alone on the labeled set with the readout fixed, then same as E2 with the new B')}
        e, how = names[ft]
        title = 'Fine-tune FB, %s (%s), %s labeled states, %s' % (ft, e, nl, task)
        desc = ('Single-task adaptation of a pretrained FB on walker %s, 1M steps: %s.' % (task, how))
        if abl(cmd):
            title = 'E3 ablation, %s, %s labeled states' % (abl(cmd), nl)
            desc = ('Why is R2 (TD3 from scratch on the fine-tuned-B reward) better than E3? Same as E3 (finetune=reward_critic, walker %s, 1M steps) '
                    'but with: %s. Control: the E3 runs of the finetune family.' % (task, abl_text(cmd)))
        if arr == '1744529':
            tail = ' Cancelled after about 2 minutes; this submission used task=walk and was resubmitted on task=run as job 1744592.'
    if arr == '1744529' and 'offline_rl_approx.py' in cmd:
        tail = ' Cancelled after about 2 minutes; this submission used task=walk and was resubmitted on task=run as job 1744592.'
    desc += ' Checkpoint: ' + CK + '.' if 'checkpoint=' in cmd else ''
    return title, (desc + tail).strip()

FAMILIES = {
    'frozen-b': ('Frozen B with the task latent', 1,
                 'Walker run, 500k steps. B frozen, continual terms off. F and the actor train with z equal to the task latent plus angular noise (spread).'),
    'oracle': ('Critic oracle (upper bound)', 2,
               'Walker run, 200k steps. B frozen; F and the actor fine-tuned with the Q loss only. The TD target uses either the true task reward (oracle) or the implicit reward B Sigma^-1 z.'),
    'finetune': ('Fine-tuning FB vs TD3 references', 3,
                 'Single-task adaptation of a pretrained FB (E1 actor only, E2 critic, E3 reward regression on B then critic) against TD3 from scratch on the two rewards the critics train on (R1 pretrained B, R2 fine-tuned B), for two label budgets. 1M steps.'),
    'finetune-ablation': ('E3 ablation: warm start, lr, tau', 5,
                          'Walker run, 10k labels, 1M steps. E3 (finetune=reward_critic) with F and the actor re-initialised and/or the TD3 learning rate (3e-4) and target tau (0.005), to explain the gap to the TD3 reference R2. Two runs per variant.'),
    'td3-learned': ('Single-reward TD3 with learned rewards', 4,
                    'Walker walk. Offline TD3 on an approximation of the reward (lrec_detach, lrec_full, fixed_z, mlp) instead of the true one; evaluation on the true reward.'),
}


def kilo(n):
    n = int(n)
    return '%dk' % (n // 1000) if n >= 1000 else str(n)


def classify(j):
    """(family, variant, order, panel, series): jobs with the same family and variant are repeats of one configuration.
    On the Results tab a panel is one chart and a series is one line of it (same series name = same color across panels)."""
    cmd = j['command'] or ''
    if 'fb_critic_oracle.py' in cmd:
        real = flag(cmd, 'q_true_reward')
        v = 'true reward' if real == '1' else 'implicit reward'
        return 'oracle', v, 1 if real == '1' else 2, '', v
    if 'fb_continual.py' in cmd and flag(cmd, 'freeze_b') == '1':
        sp = flag(cmd, 'z_task_spread')
        v = 'spread %s' % sp
        return 'frozen-b', v, 1 if sp == '0.2' else 2, '', v
    task = flag(cmd, 'tasks') or flag(cmd, 'task') or ''
    suffix = '' if task == 'run' else ' (%s)' % task
    if 'offline_rl_approx.py' in cmd and flag(cmd, 'reward_approx') in ('true', 'lrec_detach', 'lrec_full', 'fixed_z', 'mlp'):
        ra, an = flag(cmd, 'reward_approx'), flag(cmd, 'approx_n')
        order = ['true', 'lrec_detach', 'lrec_full', 'fixed_z', 'mlp'].index(ra) * 10 + (0 if an is None or an == '100000' else 1)
        panel = ('approx_n %s' % kilo(an)) if an else 'true reward'
        return 'td3-learned', 'reward %s%s' % (ra, ', approx_n %s' % kilo(an) if an else ''), order, panel, 'reward ' + ra
    if 'offline_rl_approx.py' in cmd:
        ra, an = flag(cmd, 'reward_approx'), flag(cmd, 'approx_n')
        ref = 'R1' if ra == 'fb_pretrained' else 'R2'
        name = 'R1 TD3 on pretrained-B reward' if ref == 'R1' else 'R2 TD3 on fine-tuned-B reward'
        return ('finetune', '%s, %s labels%s' % (name, kilo(an), suffix), (4 if ref == 'R1' else 5) * 10 + (0 if an == '100000' else 1),
                '%s labels%s' % (kilo(an), suffix), name)
    ft, nl = flag(cmd, 'finetune'), flag(cmd, 'n_labeled')
    if abl(cmd):
        return 'finetune-ablation', abl(cmd), 10 + len(abl(cmd)), '', abl(cmd)
    idx = {'actor': 1, 'critic': 2, 'reward_critic': 3}[ft]
    name = {'actor': 'E1 actor only', 'critic': 'E2 critic (B frozen)', 'reward_critic': 'E3 reward regression + critic'}[ft]
    return ('finetune', '%s, %s labels%s' % (name, kilo(nl), suffix), idx * 10 + (0 if nl == '100000' else 1),
            '%s labels%s' % (kilo(nl), suffix), name)


n = 0
os.makedirs('hpc/status/tracking_family', exist_ok=True)
for raw, j in jobs.items():
    if not j.get('command'):          # pending array range (or a job whose log has no 'Running:' line yet): no command to read
        if '[' not in raw:
            continue
        t, d = 'Queued array %s (%s)' % (j['arrayJob'], j['script']), \
            'Tasks %s of this array have not started yet; they appear as separate jobs when they do. Waiting reason is shown in the status column.' % j['arrayIndex']
        fam, var, order, panel, series = '', '', 0, '', ''
    else:
        t, d = describe(j)
        fam, var, order, panel, series = classify(j)
    json.dump({'tracked': True, 'title': t, 'description': d, 'family': fam, 'variant': var, 'order': order, 'panel': panel, 'series': series},
              open('hpc/status/tracking/%s.json' % raw.replace('[', '').replace(']', ''), 'w'))
    json.dump({'family': fam, 'variant': var, 'order': order, 'panel': panel, 'series': series}, open('hpc/status/tracking_family/%s.json' % raw.replace('[', '').replace(']', ''), 'w'))
    n += 1
    print(raw, j['status'], '|', fam, '|', var)
print(n)
json.dump({k: {'title': v[0], 'order': v[1], 'description': v[2]} for k, v in FAMILIES.items()}, open('hpc/status/families.json', 'w'), indent=1)
