#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=0:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@v100
#SBATCH --qos=qos_gpu-dev

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
module unload cudnn
module load cudnn/9.8.0.87-cuda
conda activate jax_env

# Runs after the last job of round 7 (sbatch --dependency=afterany:...): writes the whole report of rounds 6b and 7 into logs/fa_reports/round7_report.txt
mkdir -p logs/fa_reports
OUT=logs/fa_reports/round7_report.txt
{
echo "== generated $(date)"
echo "== round 7: stage-2 refinements (r7_*) and jump_any with a stronger stage-1 retention (r7_jump_any_mb*)"
python tools/fa_report.py r7
echo
echo "== round 6b (share <= 0.5)"
python tools/fa_report.py r6b
echo
echo "== stage-1 quality of the mu_b 1000 / 3000 B for jump_any (b_diag; eps = fit error, pre -> new)"
for f in .slurm_logs/fa_r7_mb*.out; do
    echo "-- $f"
    grep -E "B: R\^B|\] (jump_any|run|stand|flip|split|supine) +eps" "$f" | cut -c1-170
done
echo
echo "== per-task detail of the main settings"
python tools/fa_detail.py r7_jump_any_mb1000_lr0.0003_f0.5_ka15 r7_jump_any_mb3000_lr0.0003_f0.5_ka15 r7_jump_any_lr0.0003_f0.5_ka15 r7_run_lr0.0003_f0.5_ka15 r7_jump_any_lr0.0005_f0.5_ka15 r7_run_lr0.0005_f0.5_ka15 r7_jump_any_lr0.0003_f0.75_ka15 r7_run_lr0.0003_f0.75_ka15
} > "$OUT" 2>&1
echo "report written to $OUT"
