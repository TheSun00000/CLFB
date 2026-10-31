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
    m = re.findall(r'(?:^|\s)%s=(\S+)' % re.escape(k), cmd)      # the last one wins, as in parse_kwargs
    return m[-1] if m else default

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

def lrec_round1(cmd):
    """Round 1 (array 1782771, returns on, L_FB on or off): variants named by the flags that differ from the baseline."""
    if flag(cmd, 'finetune') == 'reward_critic':
        return 'reference: E3 phase 1, 10k labels'
    parts = []
    if flag(cmd, 'rec_fixed_w') == '1': parts.append('fixed readout')
    if flag(cmd, 'rec_bs'): parts.append('minibatch %s' % flag(cmd, 'rec_bs'))
    if flag(cmd, 'lr_coef'): parts.append('B lr x%s' % flag(cmd, 'lr_coef'))
    if flag(cmd, 'b_cosine') == '1': parts.append('cosine')
    if flag(cmd, 'fb_objective') == '0': parts.append('B driven by L_rec only')
    if flag(cmd, 'lam') != '10': parts.append('lam %s' % flag(cmd, 'lam'))
    return 'round 1: ' + (' + '.join(parts) or 'baseline L_rec (lam 10)')


def is_free(cmd):
    """Free round: B driven by L_rec alone (fb_objective=0), no rollouts."""
    return 'fb_continual_temp.py' in cmd and flag(cmd, 'eval_rollouts') == '0'


def free_parts(cmd):
    """Differences of a free-round L_rec run from 'E3-like' (fixed readout, batch 4096, B lr x3 with cosine decay, ortho off)."""
    parts = []
    if flag(cmd, 'rec_fixed_w') != '1':
        parts.append('resolved readout')
        if flag(cmd, 'cross_fit') == '1': parts.append('cross-fit')
        if flag(cmd, 'n_est'): parts.append('n_est %s' % flag(cmd, 'n_est'))
        if flag(cmd, 'ridge'): parts.append('ridge %s' % flag(cmd, 'ridge'))
    bs = flag(cmd, 'rec_bs', '0')
    if bs != '4096': parts.append('whole batch' if bs == '0' else 'batch %s' % bs)
    lr, cos = flag(cmd, 'lr_coef', '1'), flag(cmd, 'b_cosine') == '1'
    if lr != '3' or not cos: parts.append('lr x%s %s' % (lr, 'cosine' if cos else 'constant'))
    if flag(cmd, 'ortho_coef') not in (None, '0'): parts.append('ortho %s' % flag(cmd, 'ortho_coef'))
    return parts


def free_variant(cmd):
    ref = flag(cmd, 'finetune') == 'reward_critic'
    name = 'E3 phase 1' if ref else (', '.join(free_parts(cmd)) or 'E3-like')
    extra = []
    nl = int(flag(cmd, 'n_labeled', '10000'))
    if nl != 10000: extra.append('%sk labels' % (nl // 1000))
    if flag(cmd, 'tasks') != 'run': extra.append(flag(cmd, 'tasks'))
    return name + (' (%s)' % ', '.join(extra) if extra else '')


FREE_PANELS = {      # key = free_parts joined by ', ' (run, 10k labels); value = panels whose line it is
    'resolved readout': ['readout (run, 10k labels)', 'ridge of the resolved readout', 'labels, resolved readout'],
    'resolved readout, cross-fit': ['readout (run, 10k labels)'],
    'resolved readout, n_est 16384': ['readout (run, 10k labels)'],
    'resolved readout, ridge 1e-2': ['ridge of the resolved readout'],
    'resolved readout, ridge 1e-1': ['ridge of the resolved readout'],
    'batch 1024': ['batch size'], 'whole batch': ['batch size'],
    'lr x1 cosine': ['learning rate'], 'lr x1 constant': ['learning rate'], 'lr x3 constant': ['learning rate'], 'lr x10 cosine': ['learning rate'],
    'ortho 0.01': ['ortho term, no L_FB'], 'ortho 0.1': ['ortho term, no L_FB'], 'ortho 1': ['ortho term, no L_FB'],
}
FREE_ORDER = list(FREE_PANELS)
FREE_TASKS = ['run', 'walk', 'stand', 'flip', 'vel8']
HOME = ['readout (run, 10k labels)', 'ridge of the resolved readout', 'batch size', 'learning rate', 'ortho term, no L_FB']   # panels with the E3-like line


def free_class(cmd):
    """(variant, order, panels, series, is_reference) of a free-round run. Panels are separated by '|'; series is the line name in each panel.
    A reference is named '<name of its partner line> (E3 phase 1)', which gives it the color of the partner."""
    ref = flag(cmd, 'finetune') == 'reward_critic'
    nl, task = int(flag(cmd, 'n_labeled', '10000')), flag(cmd, 'tasks')
    parts = [] if ref else free_parts(cmd)
    key, tag = ', '.join(parts), ' (E3 phase 1)' if ref else ''
    memb = []
    if task == 'run' and nl == 10000:
        if ref or not parts:
            memb = [(pn, 'E3 phase 1' if ref else 'E3-like') for pn in HOME]
            memb += [('number of labels', '10k labels' + tag), ('task', 'run' + tag)]
            if ref:
                memb += [(pn, 'E3 phase 1 (reference)' if pn == 'headline' else 'E3 phase 1') for pn in FB_PANELS]
        else:
            memb = [(pn, key) for pn in FREE_PANELS.get(key, [])]
            if key == 'resolved readout':
                memb = [(pn, '10k labels' if pn.startswith('labels') else key) for pn, _ in memb]
    elif not parts or parts == ['resolved readout']:
        base = '%dk labels' % (nl // 1000) if task == 'run' else task
        if parts:
            memb = [('labels, resolved readout', base)]
        else:
            memb = [('number of labels' if task == 'run' else 'task', base + tag)]
    if not memb:
        memb = [('', free_variant(cmd))]
    if task == 'run' and nl == 10000 and not parts:
        order = 0
    elif key in FREE_ORDER:
        order = 1 + FREE_ORDER.index(key)
    else:
        order = 200 + FREE_TASKS.index(task) * 10 if task in FREE_TASKS and task != 'run' else 100 + nl // 1000
    return free_variant(cmd), order, '|'.join(m[0] for m in memb), '|'.join(m[1] for m in memb), ref


def is_fb(cmd):
    """L_FB-on round (array of scripts/a100_dev_lrec_fb.sh): 500k updates, L_FB and ortho active, rollouts on."""
    return 'fb_continual_temp.py' in cmd and flag(cmd, 'steps_per_stage') == '500000' and flag(cmd, 'fb_objective') != '0'


FB_PANELS = ['headline', 'lam sweep', 'ortho_coef at lam 100', 'ortho_coef at lam 10', 'B-only prefit', 'readout and B lr (lam 100)', 'labels and task (lam 100)']


def fb_parts(cmd):
    """(lam, set of modifiers) of an L_FB-on run, relative to 'E3-like L_rec, ortho 1, 10k labels, run'."""
    lam, mods = flag(cmd, 'lam', '1'), []
    if flag(cmd, 'rec_prefit') == '1': mods.append('prefit')
    if flag(cmd, 'ortho_coef') not in (None, '1'): mods.append('ortho ' + flag(cmd, 'ortho_coef'))
    if flag(cmd, 'rec_fixed_w') != '1' and lam != '0': mods.append('resolved readout')
    if flag(cmd, 'lr_coef') == '1' and lam != '0': mods.append('lr x1 cosine')
    if flag(cmd, 'b_cosine') != '1' and flag(cmd, 'rec_fixed_w') == '1': mods.append('lr x3 constant')
    if flag(cmd, 'n_labeled', '10000') != '10000': mods.append('%sk labels' % (int(flag(cmd, 'n_labeled')) // 1000))
    if flag(cmd, 'tasks') != 'run': mods.append(flag(cmd, 'tasks'))
    return lam, mods


def fb_variant(cmd):
    lam, mods = fb_parts(cmd)
    return 'L_FB on: ' + ('lam 0 (control, no L_rec)' if lam == '0' else ', '.join(['lam %s' % lam] + mods))


def fb_class(cmd):
    """(variant, order, panels, series, False). Panels hold at most 5 lines; the default run (lam N, E3-like, ortho 1) is in several."""
    lam, mods = fb_parts(cmd)
    L = int(float(lam))
    m = []                                               # (panel, series)
    if not mods:
        if L > 0: m.append(('lam sweep', 'lam %d' % L))
        if L == 0: m.append(('headline', 'no L_rec (lam 0)'))
        if L == 100:
            m += [('headline', 'lam 100, 10k labels'), ('ortho_coef at lam 100', 'ortho 1'), ('B-only prefit', 'lam 100'),
                  ('readout and B lr (lam 100)', 'fixed readout, lr x3 cosine'), ('labels and task (lam 100)', 'run, 10k labels')]
        if L == 10:
            m += [('ortho_coef at lam 10', 'ortho 1'), ('B-only prefit', 'lam 10')]
    elif mods == ['prefit']:
        m.append(('B-only prefit', 'lam %d + prefit' % L))
    elif len(mods) == 1 and mods[0].startswith('ortho'):
        m.append(('ortho_coef at lam %d' % L, mods[0]))
    elif mods == ['resolved readout']:
        m.append(('readout and B lr (lam 100)', 'resolved readout, lam %d' % L))
        if L == 100: m.append(('headline', 'lam 100, 10k labels, resolved readout'))
    elif mods in (['lr x1 cosine'], ['lr x3 constant']):
        m.append(('readout and B lr (lam 100)', mods[0]))
    elif len(mods) == 1:
        m.append(('labels and task (lam 100)', mods[0]))
        if mods[0] == '100k labels' and L == 100: m.append(('headline', 'lam 100, 100k labels'))
    if not m:
        m = [('', fb_variant(cmd))]
    order = 1 + L // 10 + (0 if not mods else 40 + len(mods[0]))
    return fb_variant(cmd), order, '|'.join(x[0] for x in m), '|'.join(x[1] for x in m), False


def is_ret(cmd):
    """Round 4 (scripts/a100_dev_return.sh): the policy is pointed at the task (z_task_frac), return from fresh labels, stream-only evaluation."""
    return 'fb_continual_temp.py' in cmd and flag(cmd, 'eval_stream_only') == '1'


def ret_parts(cmd):
    lam, ztf, atf = flag(cmd, 'lam', '1'), flag(cmd, 'z_task_frac', '0'), flag(cmd, 'actor_z_task_frac')
    sp = flag(cmd, 'z_task_spread', '0.5')
    ex = []
    if atf not in (None, '-1', ztf): ex.append('actor focus %s' % atf)
    if flag(cmd, 'mix_ratio') == '0': ex.append('no B(s) latents')
    if flag(cmd, 'q_loss_coef'): ex.append('Q loss %s' % flag(cmd, 'q_loss_coef'))
    if flag(cmd, 'fb_objective') == '0': ex.append('Q loss only, no L_FB')
    if flag(cmd, 'rec_prefit') == '1' and flag(cmd, 'freeze_b') == '1': ex.append('prefit, B frozen')
    return lam, ztf, sp, ex


def ret_variant(cmd):
    lam, ztf, sp, ex = ret_parts(cmd)
    return ', '.join(['lam %s' % lam, 'focus %s' % ztf] + (['noise %s' % sp] if ztf != '0' or ex else []) + ex)


def ret_class(cmd):
    """(variant, order, panels, series, False). A panel holds at most 5 lines; the base run (lam 100, focus 1, noise 0.2) is in all of them."""
    lam, ztf, sp, ex = ret_parts(cmd)
    L, base = float(lam), (float(lam) == 100 and ztf == '1.0' and sp == '0.2' and not ex)
    m = []
    if not ex and L == 100 and (sp == '0.2' or ztf == '0'):
        name = {'0': 'uniform z (no focus)', '0.5': 'focus 0.5', '0.8': 'focus 0.8', '1.0': 'focus 1.0'}.get(ztf)
        if name: m.append(('task focus', name))
    if not ex and L == 0 and ztf == '0.5': m.append(('task focus', 'focus 0.5, no L_rec'))
    if not ex and L == 100 and ztf == '1.0': m.append(('noise around the latent', 'noise %s' % sp))
    if base:
        m += [('who sees the task latent', 'F and actor, focus 1.0'), ('is L_rec needed', 'lam 100'), ('critic objective', 'default (L_FB + Q loss 0.01)')]
    if not ex and L == 100 and ztf == '0.5' and sp == '0.2': m.append(('who sees the task latent', 'F and actor, focus 0.5'))
    if ex == ['actor focus 1.0'] and L == 100:
        m.append(('who sees the task latent', 'actor only, F half-focused' if ztf == '0.5' else 'actor only, F uniform'))
    if ex == ['no B(s) latents']: m.append(('who sees the task latent', 'F without B(s) latents'))
    if not ex and ztf == '1.0' and sp == '0.2' and L in (0, 30, 300): m.append(('is L_rec needed', 'lam %d%s' % (L, ' (no L_rec)' if L == 0 else '')))
    if ex and ex[0] in ('Q loss 0.1', 'Q loss 1.0'): m.append(('critic objective', ex[0]))
    if ex == ['Q loss only, no L_FB']: m.append(('critic objective', 'Q loss only (no L_FB)'))
    if ex == ['prefit, B frozen']: m.append(('critic objective', 'B-only prefit, then B frozen'))
    if not m: m = [('', ret_variant(cmd))]
    order = 1 + int(L // 10) + int(float(ztf) * 10) + 40 * len(ex)
    return ret_variant(cmd), order, '|'.join(x[0] for x in m), '|'.join(x[1] for x in m), False


def lrec_class(cmd):
    """(variant, order, panel, series, is_reference) for fb_continual_temp.py runs."""
    if is_ret(cmd):
        return ret_class(cmd)
    if is_fb(cmd):
        return fb_class(cmd)
    if is_free(cmd):
        return free_class(cmd)
    v = lrec_round1(cmd)
    R1A, R1B = 'round 1: single changes', 'round 1: combined changes'
    if v.startswith('reference'):
        return v, 0, R1A + '|' + R1B, 'E3 phase 1|E3 phase 1', True
    single = ('baseline', 'lam 100', 'fixed readout', 'minibatch 4096', 'B lr x3 + cosine')
    name = v[len('round 1: '):]
    if name.startswith('baseline'):
        return v, 1, R1A + '|' + R1B, 'baseline|baseline', False
    return v, 2 + len(name), R1A if name in single else R1B, name, False

def describe(j):
    cmd, st = j['command'] or '', j['status']
    arr = j['arrayJob'] or j['jobId']
    tail = ''
    if 'fb_continual_temp.py' in cmd:
        v = ret_variant(cmd) if is_ret(cmd) else (fb_variant(cmd) if is_fb(cmd) else (free_variant(cmd) if is_free(cmd) else lrec_round1(cmd)))
        title = ('Return with task focus: %s' if is_ret(cmd) else 'L_rec vs E3 phase 1: %s') % v
        desc = ('Which differences between L_rec (continual) and fit_backward_to_reward (E3 phase 1) make B regress onto the reward as well? '
                'Run of fb_continual_temp.py on walker %s, %s labeled states, variant: %s. ' % (flag(cmd, 'tasks'), flag(cmd, 'n_labeled'), v))
        if is_ret(cmd):
            desc += ('The policy is pointed at the task: %s of the z of L_FB and of the actor loss are the current task latent (angular noise %s), the rest uniform. '
                     'Return = zero-shot with the latent inferred from fresh labels, every 10k updates, 20 episodes. L_FB and ortho on, lam %s, E3-like L_rec, %s updates.'
                     % (flag(cmd, 'z_task_frac', '0'), flag(cmd, 'z_task_spread', '0.5'), flag(cmd, 'lam'), flag(cmd, 'steps_per_stage')))
        elif is_fb(cmd):
            desc += ('L_FB and the ortho term are active, %s updates, F and the actor on uniform z, returns evaluated every 25k updates. Question: how low can the held-out '
                     'eps go while L_FB and ortho stay active, and at what price in orth_l2, fb_loss and return? E3-like L_rec = fixed readout, batch 4096, B lr x3 with cosine decay.'
                     % flag(cmd, 'steps_per_stage'))
        elif is_free(cmd):
            desc += ('B is driven by L_rec alone (fb_objective=0, no L_FB), %s updates, no rollouts. E3-like = fixed readout, batch 4096, B lr x3 with cosine '
                     'decay, ortho_coef=0; the reference is fit_backward_to_reward itself (30k B-only steps). Metric: held-out eps (eval_log.csv, %s_fresh_eps).'
                     % (flag(cmd, 'steps_per_stage'), flag(cmd, 'tasks')))
        else:
            desc += 'Single stage of %s updates, mu_b=mu_q=0. Metric: held-out eps and correlation of the reward reconstruction.' % flag(cmd, 'steps_per_stage')
    elif 'fb_critic_oracle.py' in cmd:
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
                 'Does freezing B and training F and the actor on the task latent improve the return? Walker run, 500k updates.'),
    'oracle': ('Critic oracle (upper bound)', 2,
               'How much can the critic gain from knowing the true reward? B frozen, Q loss only, walker run, 200k updates.'),
    'finetune': ('Fine-tuning FB vs TD3 references', 3,
                 'How should a pretrained FB be adapted to one reward with few labels? E1 trains the actor only, E2 the critic with B frozen, E3 first fits B to the reward; R1 and R2 are TD3 from scratch on the same two rewards. Walker run, 1M updates.'),
    'finetune-ablation': ('E3 ablation: warm start, lr, tau', 5,
                          'Why does E3 end below its TD3 reference R2 although both train on the same fitted reward? Walker run, 10k labels, 1M updates.'),
    'return-focus': ('Return with the policy pointed at the task', 0,
                     "Can the return reach the E3 level (419-438 after 1M updates; TD3 on the same reward 475-485) when F and the actor train partly on the task latent, with L_rec, L_FB and the orthonormality term all active? Walker run, 100k labels, 500k updates."),
    'lrec-vs-fit': ('L_rec vs fit_backward_to_reward', 0.5,
                    "Can L_rec, the reward term of the continual method, fit the reward in B as well as the E3 reference (fit_backward_to_reward) while L_FB and the orthonormality term stay active? Walker run, one stage."),
    'task-design': ('Reward design: tasks that are not aligned with the built-in ones', -1,
                    'Which new walker rewards are uncorrelated with stand, walk, vel8, run and flip, and with each other? 54 rewards (48 new, 6 from the first study) defined on the observation, evaluated on 100k dataset states, no training of anything.'),
    'td3-learned': ('Single-reward TD3 with learned rewards', 4,
                    'Walker walk. Offline TD3 on an approximation of the reward (lrec_detach, lrec_full, fixed_z, mlp) instead of the true one; evaluation on the true reward.'),
}

# what each family concludes and the figures that show it (a figure = one metric of one panel, with the reason to look at it)
EXTRA = {
    'task-design': {
        'standalone': True,
        'summary': 'Almost all of the new rewards are nearly uncorrelated with the five built-in tasks (|corr| <= 0.1 for 34 of the 54 rewards, <= 0.25 for 41), because the built-ins are all functions of the same few quantities (velocity, uprightness, height). The ones that are correlated (0.4-0.9: head_low, standing_flail, tall, inverted, knee_high_*, hip_low, upright_spin, lean_b) are really about being upright or high and should be left out. The rest form small clusters (left/right mirror images; kicks and spins; jump, fall and hop; flamingo and airborne; the leg-posture family), so a set should take one reward per cluster.',
        'figures': [{'type': 'matrix', 'title': 'How correlated are the rewards?',
                     'why': 'Absolute Pearson correlation of the rewards over 100k dataset states (darker = more alike; hover for the value). Rows and columns are ordered by similarity, so blocks on the diagonal are groups of rewards that mean almost the same thing. A good set has a light background between its members and against the built-ins (red).'}]},
    'return-focus': {
        'summary': 'Pointing the policy at the task works: with L_rec on (lam 100) the return reaches 431 ± 18 (4 runs, mean of the last 50k updates), against 340 at the start and 329 without focus, which is the range E3 needed 1M updates for (419-438), while L_FB and the orthonormality term stay active (eps 0.05, orth_l2 0.60 against 0.43). L_rec is necessary: without it the return falls to 250-300, and with lam 30 it is 290. The focus level (0.5-1.0), the noise (0-0.5) and whether F or only the actor sees the latent all give 395-434, within the noise; a larger Q-loss weight breaks the reward fit (213 at 1.0). The other tasks do not suffer, they improve; a control without L_rec and without focus is missing to say how much of that comes from the extra L_FB training alone.',
        'figures': [
            {'plot': 'return', 'panel': 'task focus', 'title': 'Does training on the task latent raise the return?',
             'why': 'Zero-shot return on run, with the latent inferred from fresh labels (higher is better). Focus is the share of the training z replaced by the task latent; 0 is the unfocused run of the previous round. For scale: the pretrained policy is at 340 and E3 reaches 419-438 after 1M updates.'},
            {'type': 'latentcontrol', 'title': 'Does the task latent really drive the policy? Control with a random latent',
             'why': 'Same checkpoints and tasks, but the policy gets a random latent instead of the one inferred from labelled states; the gain is what the task latent contributes. The built-in tasks lose 150-500 points without it. Among the new tasks only split has a clear latent-driven ability in the pretrained model (977 against 386), and it is gone after fine-tuning on run (594 against 602): that one is real forgetting. spin_back, kick_r and jump gain little from the latent at either point, and for knees_bent it even hurts after fine-tuning (287 with the latent, 601 with a random one).'},
            {'type': 'alltasks', 'title': 'What happens to the other tasks?',
             'why': 'Zero-shot return on every walker task at the end of training (latent inferred from fresh labels, 20 episodes), with the change from the pretrained model in colour. The five built-in tasks (left) do not suffer from the focus on run. The six columns on the right are new tasks defined for the two-task test; no run was trained on them (the latent says little about them: eps 0.4-0.8), so their returns mostly reflect the default posture of the policy and vary a lot between runs (lean_a: 29 to 978 at the same eps). They are not evidence of forgetting by themselves; see the control with a random latent below.'},
            {'plot': 'eps', 'panel': 'task focus', 'title': 'Does the task focus hurt the reward fit?',
             'why': 'Held-out error of the reward reconstruction (lower is better; the E3 reference is 0.032). With the task focus it stays at 0.03-0.06, only a little above the unfocused run (0.036).'},
            {'plot': 'orth', 'panel': 'task focus', 'title': 'Is B still orthonormal?',
             'why': 'Orthonormality error of B (the pretrained model is at 0.43). The task focus pushes it up to 0.52-0.60, against 0.43-0.44 without focus: the price of pointing the policy at the task.'},
            {'plot': 'fb', 'panel': 'task focus', 'title': 'Is the FB objective still being optimised?',
             'why': 'FB loss (lower is better). It falls further the more focus there is, but the training z differs between the curves, so the levels are not comparable; what matters is that it keeps improving.'},
            {'plot': 'return', 'panel': 'noise around the latent', 'title': 'How much noise around the task latent?',
             'why': 'Angular noise added to the task latent during training (0 = exactly the latent). Noise spreads the policy over neighbouring latents, which may help it on the latent inferred from fresh labels.'},
            {'plot': 'return', 'panel': 'who sees the task latent', 'title': 'Does F need the focus, or only the actor?',
             'why': 'Share of the task latent given to F (through L_FB and the Q loss) and to the actor, and whether F also sees the latents of B(s) of shuffled states.'},
            {'plot': 'return', 'panel': 'is L_rec needed', 'title': 'Is the reward fit needed for the return?',
             'why': 'Without L_rec (lam 0) B no longer encodes the reward, so the task latent carries no information about it. Any return gain that disappears here comes from the encoded reward.'},
            {'plot': 'return', 'panel': 'critic objective', 'title': 'What trains the critic best?',
             'why': 'The weight of the Q loss next to L_FB, the E3-like critic (Q loss only), and a B fitted first and then frozen. E3 trains the critic on the Q loss alone.'}]},
    'frozen-b': {
        'summary': 'Training F and the actor with B and the task latent frozen gives only a small gain: the return goes from about 340 to 360-370.',
        'figures': [{'plot': 'return', 'panel': '', 'title': 'Return during training',
                     'why': 'Does adding noise around the task latent (spread 0.2, against exactly the latent) change anything? Both end near 360-370 and the three runs of each overlap.'}]},
    'oracle': {
        'summary': 'With B frozen, a critic trained on the true reward ends around 390 on average, against about 350 with the implicit reward. Two runs each and a large spread between runs: the gain over the pretrained 345 is modest either way.',
        'figures': [{'plot': 'return', 'panel': '', 'title': 'Return during training',
                     'why': 'Does knowing the true reward help the critic more than the implicit reward B Sigma^-1 z? The true reward ends at 379 and 403, the implicit one at 308 and 398 (two runs each): a hint, not a difference we can rely on.'}]},
    'finetune': {
        'summary': 'Only the variants that fit B to the reward improve clearly: E3 reaches 420-440 and TD3 on the fine-tuned reward (R2) 475-485, while the actor-only and frozen-B critic variants stay around or below the pretrained 345. The ranking is the same with 10k and 100k labels.',
        'figures': [{'plot': 'return', 'panel': '10k labels', 'title': 'Return, 10k labeled states',
                     'why': 'Which way of adapting works best with few labels? The two curves that rise (E3 and R2) are the ones where B was fitted to the reward.'},
                    {'plot': 'return', 'panel': '100k labels', 'title': 'Return, 100k labeled states',
                     'why': 'Do ten times more labels change the ranking? No: the order is the same and the final returns are within about 20.'}]},
    'finetune-ablation': {
        'summary': 'Re-initialising F and the actor helps (about 450 against 438 for E3), the TD3 learning rate and target tau alone do not, and none of the variants reaches R2 (485): the gap is not explained by one factor.',
        'figures': [{'plot': 'return', 'panel': '', 'title': 'Return during training',
                     'why': 'Which difference between E3 and TD3 (warm start, learning rate and tau, or both) explains the gap? Two runs per variant; the spread between runs is up to 30.'}]},
    'lrec-vs-fit': {
        'summary': 'With L_FB switched off, L_rec alone already matches the reference (epsilon 0.030 against 0.032, over 60 runs). With L_FB and the orthonormality term on, it does so with 100k labeled states (epsilon 0.026, B still orthonormal, FB loss intact). With 10k labels it gets to 0.04-0.05 within 25k updates and then drifts up to 0.09-0.14, because the fixed labeled set is overfit; only the resolved readout stays flat (0.06).',
        'figures': [
            {'plot': 'eps', 'panel': 'headline', 'title': 'Does L_rec fit the reward as well as the reference?',
             'why': 'Error of the reward reconstruction on held-out states (lower is better); the dashed line is the E3 reference. With 100k labels L_rec reaches it and stays; with 10k labels it gets close early and then drifts up; without L_rec the reward is lost.'},
            {'plot': 'orth', 'panel': 'headline', 'title': 'Is B still orthonormal?',
             'why': 'The orthonormality error of B (the pretrained model is at 0.43). It stays at 0.43-0.48 in every run, so fitting the reward does not break the FB structure.'},
            {'plot': 'fb', 'panel': 'headline', 'title': 'Is the FB objective still being optimised?',
             'why': 'The FB loss keeps improving with L_rec: it ends at -300 to -308, against -312 without L_rec, so the reward term costs the FB objective only a few points.'},
            {'plot': 'return', 'panel': 'headline', 'title': 'What happens to the return?',
             'why': 'F and the actor train on uniform z, not on this task, so the return is not expected to improve. It stays near the pretrained level (about 340), a little lower on average (300-330) with L_rec. Each point is 10 episodes, so the curves are noisy.'},
            {'plot': 'eps', 'panel': 'lam sweep', 'title': 'Does the weight lam of L_rec matter?',
             'why': 'lam 10 is too weak to hold the fit; from lam 30 to 1000 the curves are alike, so a larger weight does not stop the late rise.'},
            {'plot': 'eps', 'panel': 'ortho_coef at lam 100', 'title': 'Does a weaker orthonormality term help?',
             'why': 'A smaller ortho_coef (1 is the default) lowers epsilon (0.069 at 0.01 against 0.139 at 1, at 500k updates), but it costs orthonormality, shown next.'},
            {'plot': 'orth', 'panel': 'ortho_coef at lam 100', 'title': 'What a weaker orthonormality term costs',
             'why': 'The orthonormality error of B for the same runs: 1.0 at ortho_coef 0.1, 2.2 at 0.01 and 3.1 at 0, against 0.44 at the default.'}]},
}


def kilo(n):
    n = int(n)
    return '%dk' % (n // 1000) if n >= 1000 else str(n)


def classify(j):
    """(family, variant, order, panel, series): jobs with the same family and variant are repeats of one configuration.
    On the Results tab a panel is one chart and a series is one line of it (same series name = same color across panels)."""
    cmd = j['command'] or ''
    if 'fb_continual_temp.py' in cmd:
        v, order, panel, series, ref = lrec_class(cmd)
        return ('return-focus' if is_ret(cmd) else 'lrec-vs-fit'), v, order, panel, series, ref
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
    if j.get('children'):             # a job that ran several commands: its children are tracked, not the job
        continue
    if not j.get('command'):          # pending array range (or a job whose log has no 'Running:' line yet): no command to read
        if '[' not in raw:
            continue
        t, d = 'Queued array %s (%s)' % (j['arrayJob'], j['script']), \
            'Tasks %s of this array have not started yet; they appear as separate jobs when they do. Waiting reason is shown in the status column.' % j['arrayIndex']
        fam, var, order, panel, series, ref = '', '', 0, '', '', False
    else:
        t, d = describe(j)
        c = classify(j)
        fam, var, order, panel, series = c[:5]
        ref = len(c) > 5 and c[5]
    json.dump({'tracked': True, 'title': t, 'description': d, 'family': fam, 'variant': var, 'order': order, 'panel': panel, 'series': series, 'ref': ref},
              open('hpc/status/tracking/%s.json' % raw.replace('[', '').replace(']', ''), 'w'))
    json.dump({'family': fam, 'variant': var, 'order': order, 'panel': panel, 'series': series, 'ref': ref}, open('hpc/status/tracking_family/%s.json' % raw.replace('[', '').replace(']', ''), 'w'))
    n += 1
    print(raw, j['status'], '|', fam, '|', var)
print(n)
fam_docs = {k: {'title': v[0], 'order': v[1], 'description': v[2]} for k, v in FAMILIES.items()}
for k, extra in EXTRA.items():
    fam_docs[k].update(extra)
fam_docs['lrec-vs-fit']['table'] = 'eps'
fam_docs['return-focus']['table'] = 'eps'
json.dump(fam_docs, open('hpc/status/families.json', 'w'), indent=1)
