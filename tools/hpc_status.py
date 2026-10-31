#!/usr/bin/env python3
"""Collect the state of the Slurm jobs launched from scripts/ on the HPC (read-only).

One `ssh jz` call lists the jobs (sacct); a second one reads, for the jobs that are new or not final yet,
the first lines of their .slurm_logs/*.out (command, run folder) and the tail of the .err of failed jobs.

Finished jobs also get `results` (final / best return, see job_results.py) from the synced logs in hpc/logs.
Writes hpc/status/jobs.json (all known jobs, keyed by job id) and prints a short summary.
`--diff` also writes hpc/status/snapshot.json ({checkedAt, jobs}), the content of the tracker's document
snapshot/jobs, and hpc/status/curves.json ({curves: {jobId: learning curve}}), the content of the document
snapshot/curves; it says whether either changed since the last push (`--mark-pushed` records the push).
The tracker flags (tracked, title, description) live apart, in the collection `tracking`, so a refresh never touches them.

A desktop notification is sent when jobs start, finish, fail or are cancelled (one per job array and kind of change).

Usage:
    tools/hpc_status.py [--since 2026-10-30] [--diff] [--mark-pushed] [--no-notify] [--notify-dry]
Exits quietly (code 0) when the HPC cannot be reached, so it is safe to run from a timer.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import job_results

HOST = 'jz'
REMOTE_DIR = '$WORK/CLFB'
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(ROOT, 'hpc', 'status')
JOBS = os.path.join(OUT, 'jobs.json')
PUSHED = os.path.join(OUT, 'pushed.json')       # hash of the job fields last pushed to the tracker
SNAPSHOT = os.path.join(OUT, 'snapshot.json')
CURVES = os.path.join(OUT, 'curves.json')       # content of the tracker's document snapshot/curves

FINAL = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL', 'DEADLINE', 'BOOT_FAIL'}
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
SACCT_FIELDS = 'JobIDRaw,JobID,JobName,State,Elapsed,Submit,Start,End,ExitCode,Partition,NodeList'


def ssh(script, timeout=120):
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', HOST, 'bash -l -s'],
                       input=script, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or 'ssh failed (%d)' % r.returncode)
    return r.stdout


def status_of(state):
    s = state.split()[0].rstrip('+')
    if s in ('PENDING', 'REQUEUED', 'RESIZING'):
        return 'pending'
    if s in ('RUNNING', 'COMPLETING', 'SUSPENDED'):
        return 'running'
    if s == 'COMPLETED':
        return 'finished'
    if s.startswith('CANCELLED'):
        return 'cancelled'
    return 'failed'


def list_jobs(since):
    out = ssh('cd "%s" || exit 1\necho @@SCRIPTS; ls scripts\necho @@SACCT\n'
              'sacct -u "$USER" -S %s -X -n -P --format=%s\n'
              'echo @@SQUEUE; squeue -h -u "$USER" -t PENDING -o "%%i|%%R"\n' % (REMOTE_DIR, since, SACCT_FIELDS))
    scripts, jobs, mode, reasons = set(), {}, None, {}
    for line in out.splitlines():
        if line.startswith('@@'):
            mode = line[2:]
        elif mode == 'SCRIPTS' and line.strip():
            scripts.add(line.strip())
        elif mode == 'SQUEUE' and '|' in line:
            reasons[line.split('|')[0].strip()] = line.split('|', 1)[1].strip()
        elif mode == 'SACCT' and line.strip():
            f = line.split('|')
            if len(f) < len(SACCT_FIELDS.split(',')):
                continue
            raw, jid, name, state, elapsed, submit, start, end, exitc, part, node = f[:11]
            if name not in scripts:          # only jobs launched from scripts/
                continue
            array_id, _, idx = jid.partition('_')
            if submit < since:                          # older than the window
                continue
            if idx.startswith('['):                     # array range whose tasks have not started: one row, keyed by its sacct id
                if status_of(state) != 'pending':       # (a cancelled range never ran: not listed)
                    continue
                raw = jid
            jobs[raw] = {
                'jobId': raw, 'arrayJob': array_id if idx else None, 'arrayIndex': idx or None,
                'script': name, 'state': state.split()[0], 'status': status_of(state), 'elapsed': elapsed,
                'submit': submit, 'start': start, 'end': end, 'exit': exitc, 'partition': part, 'node': node,
            }
    for raw, j in jobs.items():                         # why a pending job waits (QOSGrpCpuLimit, Priority, Resources, ...)
        if j['status'] == 'pending':
            j['reason'] = reasons.get(raw) or reasons.get(j['arrayJob'] or '', '') or reasons.get('%s_%s' % (j['arrayJob'], j['arrayIndex']), '')
    return jobs


def read_details(jobs, ids):
    """command, run folders, last output line (and .err tail if failed) for the given job ids."""
    if not ids:
        return {}
    parts = ['cd "%s" || exit 1' % REMOTE_DIR]
    for raw in ids:
        j = jobs[raw]
        f = '.slurm_logs/%s_%s' % (j['script'], raw)
        parts.append(
            'echo "@@JOB %s"; if [ -f "%s.out" ]; then grep -a -m1 "^Running:" "%s.out"; '
            'grep -a "^LOGS_FOLDER" "%s.out" | sed "s/^/@@FOLDER /"; '
            'echo "@@LAST"; tail -c 3000 "%s.out" | tr "\\r" "\\n" | grep -a -v "^$" | tail -n 1 | cut -c1-240; fi; '
            'if [ -f "%s.err" ]; then echo "@@ERR"; tr "\\r" "\\n" < "%s.err" | grep -a -v -E "it/s|s/it|^[[:space:]]*$" | tail -n 8 | cut -c1-240; fi' % (raw, f, f, f, f, f, f))
    out = ssh('\n'.join(parts))
    res, cur, mode = {}, None, None
    for line in out.splitlines():
        if line.startswith('@@JOB '):
            cur = line.split()[1]
            res[cur] = {'command': None, 'runFolders': [], 'lastLine': '', 'err': ''}
            mode = None
        elif cur is None:
            continue
        elif mode and not line.startswith('@@'):      # inside @@LAST / @@ERR: never a command or a folder
            res[cur][mode] = (res[cur][mode] + '\n' + line).strip()
        elif line.startswith('Running:'):
            res[cur]['command'] = line[len('Running:'):].strip()
        elif line.startswith('@@FOLDER'):
            v = line.split(':', 1)[1].strip()
            res[cur]['runFolders'].append(v[2:].split('/', 1)[-1] if v.startswith('./') else v)
        elif line.startswith('@@LAST'):
            mode = 'lastLine'
        elif line.startswith('@@ERR'):
            mode = 'err'
        elif mode:
            res[cur][mode] = (res[cur][mode] + '\n' + line).strip()
    return res


def digest(job):
    keys = ('state', 'reason', 'elapsed', 'start', 'end', 'exit', 'node', 'command', 'runFolders', 'lastLine', 'err', 'results')
    return hashlib.md5(json.dumps([job.get(k) for k in keys], sort_keys=True).encode()).hexdigest()


def whole_digest(jobs):
    return hashlib.md5(''.join(sorted(r + digest(j) for r, j in jobs.items())).encode()).hexdigest()


def curves_digest(c):
    return hashlib.md5(json.dumps(c, sort_keys=True).encode()).hexdigest()


def load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def transitions(before, fresh):
    """{(array or job id, script, new status): [job ids]} for jobs whose status changed since the last check."""
    out = {}
    for raw, j in fresh.items():
        new = j['status']
        if before.get(raw) != new and new in ('running', 'finished', 'failed', 'cancelled'):
            out.setdefault((j['arrayJob'] or raw, j['script'], new), []).append(raw)
    return out


def notify(trans, dry=False):
    """One desktop notification per job array, listing how many jobs went to each status."""
    by_array = {}
    for (arr, script, status), ids in trans.items():
        by_array.setdefault((arr, script), {})[status] = len(ids)
    env = dict(os.environ)
    env.setdefault('DBUS_SESSION_BUS_ADDRESS', 'unix:path=/run/user/%d/bus' % os.getuid())
    env.setdefault('DISPLAY', ':0')
    for (arr, script), counts in by_array.items():
        title = 'Jean Zay: job %s' % arr
        body = '%s\n%s' % (script, ', '.join('%d %s' % (n, st) for st, n in sorted(counts.items())))
        urgency = 'critical' if 'failed' in counts else 'normal'
        if dry:
            print('notify [%s] %s | %s' % (urgency, title, body.replace('\n', ' | ')))
        else:
            try:
                subprocess.run(['notify-send', '-a', 'CLFB', '-u', urgency, title, body], env=env, timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', default='2026-10-30', help='ignore jobs submitted before this date')
    ap.add_argument('--diff', action='store_true', help='write hpc/status/to_push.json')
    ap.add_argument('--no-notify', action='store_true', help='do not send desktop notifications')
    ap.add_argument('--notify-dry', action='store_true', help='print the notifications instead of sending them')
    ap.add_argument('--mark-pushed', action='store_true', help='record to_push.json as pushed to the tracker')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if args.mark_pushed:
        with open(PUSHED, 'w') as f:
            json.dump({'hash': whole_digest(load(JOBS, {})), 'curves': curves_digest(load(CURVES, {}))}, f)
        print('marked as pushed')
        return

    try:
        fresh = list_jobs(args.since)
        known = {r: j for r, j in load(JOBS, {}).items() if (j.get('submit') or '') >= args.since and not str(j.get('arrayIndex') or '').startswith('[')}
        need = [r for r, j in fresh.items() if r not in known or known[r].get('state') not in FINAL
                or known[r].get('status') != j['status'] or not known[r].get('command')]
        details = read_details(fresh, need)
    except (RuntimeError, subprocess.TimeoutExpired, OSError) as e:
        print('hpc_status: HPC not reachable, skipped (%s)' % str(e).splitlines()[0][:120], file=sys.stderr)
        return

    for raw, j in fresh.items():
        d = details.get(raw) or {k: known.get(raw, {}).get(k) for k in ('command', 'runFolders', 'lastLine', 'err')}
        err = '\n'.join(l.rstrip() for l in ANSI.sub('', d.get('err') or '').splitlines() if l.strip()) if j['status'] == 'failed' else ''
        j.update({'command': d.get('command'), 'runFolders': d.get('runFolders') or [],
                  'lastLine': ANSI.sub('', d.get('lastLine') or '')[-200:], 'err': err[-900:]})
    first_run = not os.path.exists(JOBS)
    trans = transitions({r: j.get('status') for r, j in known.items()}, fresh)
    known.update(fresh)
    for j in known.values():          # headline numbers from the synced logs (run tools/sync_from_hpc.sh first)
        j['results'] = job_results.headline(j)
    with open(JOBS, 'w') as f:
        json.dump(known, f, indent=1, sort_keys=True)

    if trans and not first_run and not args.no_notify:
        notify(trans, dry=args.notify_dry)

    counts = {}
    for j in known.values():
        counts[j['status']] = counts.get(j['status'], 0) + 1
    print('%d jobs:' % len(known), ', '.join('%s %d' % kv for kv in sorted(counts.items())))

    if args.diff:
        curves = {r: c for r, c in ((r, job_results.curve(j)) for r, j in known.items()) if c}
        pushed = load(PUSHED, {})
        changed = pushed.get('hash') != whole_digest(known) or pushed.get('curves') != curves_digest({'curves': curves})
        now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        with open(SNAPSHOT, 'w') as f:
            json.dump({'checkedAt': now, 'jobs': known}, f)
        with open(CURVES, 'w') as f:
            json.dump({'curves': curves}, f)
        print('changed since last push' if changed else 'no change since last push')

if __name__ == '__main__':
    main()
