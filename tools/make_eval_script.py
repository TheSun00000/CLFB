#!/usr/bin/env python3
"""Write an sbatch script that evaluates, on ALL the benchmark tasks, the final checkpoints of finished runs (eval-only mode of fb_continual_temp.py).

Usage:  tools/make_eval_script.py <array job id> [<array job id> ...] [--per-job 8] [--out scripts/a100_dev_eval_all_tasks.sh]
Reads hpc/status/snapshot.json (run `tools/hpc_status.py --diff` first): for every finished child run (or finished job with one run folder) of the given
arrays it adds `python fb_continual_temp.py ... eval_ckpt=models/<run folder>/fb_stage_<k>_<task>.pickle`, plus the pretrained checkpoint as a baseline.
The models exist on the cluster only: check them there before submitting. Results: tools/eval_summary.py after tools/sync_from_hpc.sh.
"""
import argparse, json, os, re, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def flag(cmd, k, default=None):
    m = re.findall(r'(?:^|\s)%s=(\S+)' % re.escape(k), cmd)
    return m[-1] if m else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arrays', nargs='+')
    ap.add_argument('--per-job', type=int, default=8)
    ap.add_argument('--out', default='scripts/a100_dev_eval_all_tasks.sh')
    ap.add_argument('--time', default='0:45:00')
    ap.add_argument('--eval-labels', default=None, help='evaluate only these benchmark tasks (eval_labels=...)')
    a = ap.parse_args()
    jobs = json.load(open(os.path.join(ROOT, 'hpc', 'status', 'snapshot.json')))['jobs']
    runs, skipped = [], []
    for k, j in sorted(jobs.items()):
        if j.get('arrayJob') not in a.arrays or j.get('children'):
            continue
        cmd, folders = j.get('command') or '', j.get('runFolders') or []
        if j.get('status') != 'finished' or len(folders) != 1 or 'fb_continual' not in cmd or flag(cmd, 'finetune') not in (None, 'none'):
            skipped.append((k, j.get('status'))); continue
        tasks = [t for t in flag(cmd, 'tasks').strip('[]').split(',') if t]
        runs.append((k, cmd, folders[0], tasks))
    if not runs:
        sys.exit('no finished run found in arrays %s (skipped: %s)' % (a.arrays, skipped))
    ck = flag(runs[0][1], 'checkpoint')
    env = flag(runs[0][1], 'env_name')
    extra = (' eval_labels=%s' % a.eval_labels) if a.eval_labels else ''
    cmds = [('pretrained model (baseline)', 'python fb_continual_temp.py env_name=%s checkpoint=$CKPT tasks=%s eval_ckpt=$CKPT%s' % (env, ','.join(runs[0][3]), extra))]
    for k, cmd, folder, tasks in runs:
        stage = len(tasks)
        path = 'models/%s/fb_stage_%d_%s.pickle' % (folder, stage, tasks[-1])
        cmds.append((k, 'python fb_continual_temp.py env_name=%s checkpoint=$CKPT tasks=%s eval_ckpt=%s%s' % (flag(cmd, 'env_name'), ','.join(tasks), path, extra)))
    head = open(os.path.join(ROOT, 'scripts', 'a100_dev_return.sh')).read().split('\n# Round 4')[0]
    head = re.sub(r'--time=\S+', '--time=' + a.time, head)
    n_jobs = (len(cmds) + a.per_job - 1) // a.per_job
    head = re.sub(r'--array=\S+ # .*', '--array=0-%d # %d eval-only jobs' % (n_jobs - 1, n_jobs), head)
    body = '\n# Eval-only: every benchmark task, zero-shot (latent from fresh labels, 20 episodes), on the final checkpoint of finished runs of arrays %s.\n' % ' '.join(a.arrays)
    body += '# No training. The log folders are logs/<time>_FBeval-*; their info.txt records the source checkpoint. Summary: tools/eval_summary.py.\n'
    body += 'CKPT=%s\n\nCOMMANDS=(\n' % ck
    for i in range(n_jobs):
        chunk = cmds[i * a.per_job:(i + 1) * a.per_job]
        body += '    # job %d: %s\n' % (i, ', '.join(str(c[0]) for c in chunk))
        body += '    "%s"\n' % '; '.join(c[1] for c in chunk)
    body += ')\n\nCMD="${COMMANDS[$SLURM_ARRAY_TASK_ID]}"\necho "Running: $CMD"\n\neval "$CMD"\n'
    out = os.path.join(ROOT, a.out)
    open(out, 'w').write(head.rstrip('\n') + '\n' + body)
    print('%d checkpoints (+ baseline) in %d jobs -> %s%s' % (len(runs), n_jobs, a.out, ('; skipped %s' % skipped) if skipped else ''))


if __name__ == '__main__':
    main()
