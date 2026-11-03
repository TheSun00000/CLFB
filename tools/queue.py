#!/usr/bin/env python3
"""Local queue of experiments to sbatch on Jean Zay, submitted whenever job slots are free.

Why: the dev QoS allows at most 10 submitted jobs per user (running + pending, every array task counts as one), so more experiments
than that wait here and are sent as slots free up. Nothing is run locally: `submit` only calls `ssh jz` (git pull of the repo, squeue, sbatch).

The list lives in queue/experiments.json (local, git-ignored). Each entry has an sbatch script scripts/queue/<name>.sh (committed and pushed,
because the cluster pulls it), written by `add` from a header and a list of shell commands (run one after the other in the same job).

    tools/queue.py add NAME --time 1:00:00 --cmd "python a ..." [--cmd "python b ..."] [--title "..."] [--slots 1]
    tools/queue.py status            # the list, the jobs on the cluster, the free slots (read-only)
    tools/queue.py submit [--dry-run] [--notify]  # submit, in list order, every queued entry that fits in the free slots (--notify: desktop notification)
    tools/queue.py sync              # update done / failed from Slurm
    tools/queue.py requeue NAME      # put a failed/cancelled entry back in the queue
    tools/queue.py prio NAME [NAME..]  # move queued entries to the front of the queue

`submit` refuses to run if the local scripts are not pushed (the cluster would not have them). It pulls the repo on the cluster (fast-forward only).
"""
import argparse, fcntl, json, os, re, subprocess, sys
from datetime import datetime

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
QFILE = os.path.join(ROOT, 'queue', 'experiments.json')
HOST, REMOTE = 'jz', '/lustre/fswork/projects/rech/vcj/uyo96iu/CLFB'
LIMIT = 10
HEADER = """#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time={time}
#SBATCH --exclude=jean-zay-iam26
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module unload cudnn
module load cudnn/9.8.0.87-cuda
conda activate jax_env

CKPT=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
CMD="{joined}"
echo "Running: $CMD"
mkdir -p models logs
"""

# Eval-only jobs (no training) go to the V100 dev partition, which is easy to get (user, 2026-11-03): --gpu v100, or automatic when every command has eval_ckpt=.
HEADER_V100 = (HEADER.replace('--cpus-per-task=8', '--cpus-per-task=16').replace('--account=vcj@a100', '--account=vcj@v100')
               .replace('#SBATCH -C a100\n', '').replace('--qos=qos_gpu_a100-dev', '--qos=qos_gpu-dev'))


def lock():
    """One queue.py that changes the queue at a time (the systemd timer and an interactive call must not submit twice): the second one just stops."""
    os.makedirs(os.path.dirname(QFILE), exist_ok=True)
    f = open(os.path.join(os.path.dirname(QFILE), '.lock'), 'w')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print('another queue.py is running, skipped')
        sys.exit(0)
    return f


def notify(body):
    env = dict(os.environ)
    env.setdefault('DBUS_SESSION_BUS_ADDRESS', 'unix:path=/run/user/%d/bus' % os.getuid())
    env.setdefault('DISPLAY', ':0')
    try:
        subprocess.run(['notify-send', '-a', 'CLFB', 'Jean Zay queue', body], env=env, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        pass


def load():
    if not os.path.exists(QFILE):
        return {'limit': LIMIT, 'entries': []}
    return json.load(open(QFILE))


def save(q):
    os.makedirs(os.path.dirname(QFILE), exist_ok=True)
    json.dump(q, open(QFILE, 'w'), indent=1)


def ssh(script, timeout=180):
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', HOST, 'bash -l -s'], input=script, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[-400:] or 'ssh failed')
    return r.stdout


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def cluster_jobs():
    """(number of my submitted jobs, {jobid: state}) with job arrays expanded: this is what the limit counts."""
    out = ssh('squeue -u "$USER" -h -r -o "%i|%T"').strip().splitlines()
    return len(out), dict(l.split('|') for l in out if '|' in l)


def cluster_counts():
    """Submitted jobs per dev partition ('v100' = qos_gpu-dev, 'a100' = the others): the 10-job limit is per QoS."""
    out = ssh('squeue -u "$USER" -h -r -o "%q"').strip().splitlines()
    c = {'a100': 0, 'v100': 0}
    for q in out:
        c['v100' if q.strip() == 'qos_gpu-dev' else 'a100'] += 1
    return c


def group_of(e):
    try:
        return 'v100' if 'vcj@v100' in open(os.path.join(ROOT, e['script'])).read() else 'a100'
    except OSError:
        return 'a100'


def add(a):
    q = load()
    if any(e['name'] == a.name for e in q['entries']):
        sys.exit('an entry named %s exists' % a.name)
    path = os.path.join(ROOT, 'scripts', 'queue', a.name + '.sh')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joined = '; '.join(a.cmd).replace('"', '\\"')
    with open(path, 'w') as f:
        gpu = a.gpu or ('v100' if all('eval_ckpt=' in c for c in a.cmd) else 'a100')
        f.write((HEADER_V100 if gpu == 'v100' else HEADER).format(time=a.time, joined=joined))
        for c in a.cmd:
            f.write(c + '\n')
    q['entries'].append({'name': a.name, 'title': a.title or a.name, 'script': 'scripts/queue/%s.sh' % a.name, 'slots': a.slots, 'state': 'queued', 'job_id': None, 'added': datetime.now().isoformat(timespec='seconds')})
    save(q)
    print('added %s (%d command(s), %s)' % (a.name, len(a.cmd), a.time))


def status(a=None):
    q = load()
    try:
        n, st = cluster_jobs()
        print('cluster: %d of %d job slots used (%s)' % (n, q.get('limit', LIMIT), ', '.join('%s %d' % (k, list(st.values()).count(k)) for k in sorted(set(st.values()))) or 'empty'))
    except Exception as e:
        print('cluster not reachable: %s' % e)
    for i, e in enumerate(q['entries'], 1):
        print('%2d. %-26s %-10s %s  %s' % (i, e['name'], e['state'], e['job_id'] or '-', e['title']))
    cnt = {}
    for e in q['entries']:
        cnt[e['state']] = cnt.get(e['state'], 0) + 1
    print(cnt)


def sync(a=None):
    q = load()
    ids = [e['job_id'] for e in q['entries'] if e['job_id'] and e['state'] in ('submitted', 'running')]
    if not ids:
        return
    out = ssh('sacct -j %s -X -n -P --format=JobID,State' % ','.join(ids))
    states = dict(l.split('|') for l in out.strip().splitlines() if '|' in l)
    for e in q['entries']:
        s = states.get(e['job_id'])
        if not s:
            continue
        s = s.split()[0].rstrip('+')
        e['state'] = {'COMPLETED': 'done', 'RUNNING': 'running', 'PENDING': 'submitted'}.get(s, 'failed' if s in ('FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'CANCELLED') else e['state'])
        if s.startswith('CANCELLED'):
            e['state'] = 'cancelled'
    save(q)


def submit(a):
    q = load()
    queued = [e for e in q['entries'] if e['state'] == 'queued']
    if not queued:
        print('nothing queued')
        return
    sync()
    q = load()
    queued = [e for e in q['entries'] if e['state'] == 'queued']
    n, _ = cluster_jobs()
    counts = cluster_counts()
    gfree = {g: q.get('limit', LIMIT) - counts[g] for g in counts}
    free = max(gfree.values())
    print('%d jobs (a100 %d, v100 %d, %d free each), %d queued' % (n, counts['a100'], counts['v100'], q.get('limit', LIMIT), len(queued)))
    if free <= 0 or not queued:
        return
    # the cluster needs the scripts: they must be committed and pushed
    dirty = git('status', '--porcelain', 'scripts/queue', 'tools', 'fb_continual_temp.py', 'offline_rl_approx.py')
    subprocess.run(['git', 'fetch', '-q'], cwd=ROOT)
    unpushed = git('rev-list', '--count', 'origin/main..HEAD')
    if dirty or unpushed not in ('0', ''):
        sys.exit('not submitted: uncommitted changes in scripts/tools/*.py or %s unpushed commit(s): commit and push first' % unpushed)
    todo = []
    for e in queued:
        g = group_of(e)
        if e['slots'] <= gfree[g]:
            todo.append(e)
            gfree[g] -= e['slots']
    if a.dry_run:
        print('would submit:', [e['name'] for e in todo])
        return
    if not todo:
        return
    script = 'cd "%s" || exit 1\ngit pull --ff-only -q 2>&1 | tail -2\nmkdir -p .slurm_logs\n' % REMOTE
    for e in todo:
        script += 'echo "@@%s $(sbatch %s)"\n' % (e['name'], e['script'])
    out = ssh(script)
    done, bad = [], []
    for line in out.splitlines():
        m = re.match(r'@@(\S+) Submitted batch job (\d+)', line)
        if m:
            for e in q['entries']:
                if e['name'] == m.group(1):
                    e.update(state='submitted', job_id=m.group(2), submitted=datetime.now().isoformat(timespec='seconds'))
            print('submitted %s -> job %s' % (m.group(1), m.group(2)))
            done.append('%s (%s)' % (m.group(1), m.group(2)))
        elif line.startswith('@@'):
            print('FAILED', line)
            bad.append(line)
    save(q)
    if a.notify and (done or bad):
        notify('submitted %d: %s%s' % (len(done), ', '.join(done), ('; %d failed' % len(bad)) if bad else ''))


def prio(a):
    """Move the named queued entries to the front of the queue, in the order given."""
    q = load()
    by = {e['name']: e for e in q['entries']}
    front = [by[n] for n in a.names if n in by]
    q['entries'] = front + [e for e in q['entries'] if e['name'] not in a.names]
    save(q)
    print('front of the queue:', [e['name'] for e in q['entries'] if e['state'] == 'queued'][:8])


def requeue(a):
    q = load()
    for e in q['entries']:
        if e['name'] == a.name:
            e.update(state='queued', job_id=None)
    save(q)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('add'); p.add_argument('name'); p.add_argument('--time', default='1:00:00'); p.add_argument('--cmd', action='append', required=True)
    p.add_argument('--title', default=''); p.add_argument('--slots', type=int, default=1); p.add_argument('--gpu', choices=['a100', 'v100'], default=''); p.set_defaults(f=add)
    sp.add_parser('status').set_defaults(f=status)
    sp.add_parser('sync').set_defaults(f=sync)
    p = sp.add_parser('submit'); p.add_argument('--dry-run', action='store_true'); p.add_argument('--notify', action='store_true', help='desktop notification when something is submitted'); p.set_defaults(f=submit)
    p = sp.add_parser('requeue'); p.add_argument('name'); p.set_defaults(f=requeue)
    p = sp.add_parser('prio'); p.add_argument('names', nargs='+'); p.set_defaults(f=prio)
    a = ap.parse_args()
    if a.cmd in ('add', 'submit', 'sync', 'requeue', 'prio'):
        _lock = lock()        # kept open until the process ends
    a.f(a)


if __name__ == '__main__':
    main()
