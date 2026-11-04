#!/usr/bin/env python3
"""Launcher of the fast-adaptation experiments (stage 2 with B frozen, short schedules, dense evaluation).

    python tools/overnight.py submit NAME --qos a100|v100 --cmd "python $SCRIPT ..." [--cmd ...]    # one sbatch job running the commands one after the other
    python tools/overnight.py slots                                                                 # submitted jobs per QoS on the cluster

Scripts are written to scripts/overnight/NAME.sh (local), copied to the cluster with noclobber (an existing script is never overwritten) and submitted with
SCRIPT=fb_continual_temp_adapter.py. Limits: 10 submitted jobs per QoS (qos_gpu_a100-dev, qos_gpu-dev). `submit` refuses to go over LIMIT.
"""
import argparse, os, subprocess, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
REMOTE = '/lustre/fswork/projects/rech/vcj/uyo96iu/CLFB'
LIMIT = 10
HEAD = {
    'a100': """#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
""",
    'v100': """#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@v100
#SBATCH --qos=qos_gpu-dev
"""}
BODY = """
module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module unload cudnn
module load cudnn/9.8.0.87-cuda
conda activate jax_env

SCRIPT=${SCRIPT:-fb_continual_temp.py}
PRE=models/2026-10-29_14-44-51_FBpretrain-walker/fb_agent_latest.pickle
mkdir -p models logs
"""
QOS = {'a100': 'qos_gpu_a100-dev', 'v100': 'qos_gpu-dev'}


def ssh(cmd, stdin=None):
    return subprocess.run(['ssh', 'jz', "bash -l -c '%s'" % cmd], input=stdin, capture_output=True, text=True)


def slots():
    out = ssh("squeue -u $USER -h -o %q").stdout.split()
    return {k: sum(1 for q in out if q == v) for k, v in QOS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['submit', 'slots'])
    ap.add_argument('name', nargs='?')
    ap.add_argument('--qos', default='a100', choices=['a100', 'v100'])
    ap.add_argument('--cmd', action='append', default=[])
    ap.add_argument('--time', default=None)
    a = ap.parse_args()
    if a.action == 'slots':
        print(slots())
        return
    used = slots()[a.qos]
    if used >= LIMIT:
        sys.exit('no free slot on %s (%d submitted)' % (a.qos, used))
    head = HEAD[a.qos].replace('--time=2:00:00', '--time=' + a.time) if a.time else HEAD[a.qos]
    lines = []
    for c in a.cmd:
        lines.append('echo "Running: %s"' % c.replace('"', '\\"'))
        lines.append(c)
    text = head + BODY + '\n'.join(lines) + '\n'
    os.makedirs(os.path.join(ROOT, 'scripts', 'overnight'), exist_ok=True)
    path = os.path.join(ROOT, 'scripts', 'overnight', a.name + '.sh')
    open(path, 'w').write(text)
    r = ssh("cd %s && set -o noclobber && mkdir -p scripts/overnight && cat > scripts/overnight/%s.sh && sbatch --job-name=%s --export=ALL,SCRIPT=fb_continual_temp_adapter.py scripts/overnight/%s.sh"
            % (REMOTE, a.name, a.name, a.name), stdin=text)
    print(r.stdout.strip(), r.stderr.strip())


if __name__ == '__main__':
    main()
