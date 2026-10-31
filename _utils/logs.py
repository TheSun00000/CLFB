import os
import random
import shutil
import socket
import subprocess
import sys
import csv


def auto_cast(x):
    for cast in (int, float):
        try:
            return cast(x)
        except:
            pass
    return x

def parse_kwargs(argv):
    out = {}
    for item in argv:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        out[k] = auto_cast(v)
    return out


def resolve_seed(kwargs, key='seed'):
    """The seed given as key=, else a random one. Written back into kwargs so that info.txt records the seed actually used."""
    given = key in kwargs
    seed = int(kwargs[key]) if given else random.SystemRandom().randrange(10 ** 7)
    kwargs[key] = seed
    print(f'{key}: {seed}' + ('' if given else ' (random)'))
    return seed


def _git(*args):
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ''


def run_info():
    """Text for info.txt: the Slurm job, the git commit (flagged if tracked files were modified) and the full command."""
    env = os.environ.get
    fields = [('slurm_job_id', env('SLURM_JOB_ID')), ('array_job_id', env('SLURM_ARRAY_JOB_ID')),
              ('array_task_id', env('SLURM_ARRAY_TASK_ID')), ('job_name', env('SLURM_JOB_NAME')),
              ('node', env('SLURMD_NODENAME') or socket.gethostname())]
    job = ' '.join(f'{k}={v}' for k, v in fields if v)
    commit = _git('rev-parse', 'HEAD')
    dirty = _git('status', '--porcelain', '--untracked-files=no')
    git = (commit[:12] + (' (uncommitted changes)' if dirty else '')) if commit else 'unknown'
    return f"job: {job}\ngit: {git}\ncommand: {' '.join(['python'] + sys.argv)}\n"


def add_text_to_file(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
        
        
        
# CSV ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

def init_csv_logger(path, fieldnames):
    file_exists = os.path.isfile(path)

    if file_exists:
        return  # don't overwrite existing file

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def log_csv_row(path, row_dict):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        writer.writerow(row_dict)