#!/usr/bin/env bash
set -euo pipefail

# FairMetrics paper batch runner.
#
# Executes the full experiment grid across both datasets using:
#   - multiclass mode: mul
#   - one-vs-rest binary modes: bin0, bin1, bin2
#
# Scheduled runs:
#   - Adult:    mul, bin0, bin1, bin2
#   - Students: mul, bin0, bin1, bin2
#
# Output:
#   Writes one JSON artifact per dataset/mode under ./results
#   (e.g., results/adult_mul.json, results/students_bin2.json).
#
# Typical background execution:
#   nohup bash scripts/multi_run.sh > jobs_log/submit.log 2>&1 &

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate fairmetrics 2>/dev/null || true

RUNPATH="${RUNPATH:-$PWD}"
cd "$RUNPATH" || {
  echo "could not cd into $RUNPATH" >&2
  exit 1
}

export PYTHONPATH="${PYTHONPATH:-}:$RUNPATH"
mkdir -p results

for mode in mul bin0 bin1 bin2; do
  echo "[RUN] adult $mode"
  python -m fairmetrics_experiments.run_hpo --dataset adult --mode "$mode" --output "results/adult_${mode}.json"
done

for mode in mul bin0 bin1 bin2; do
  echo "[RUN] students $mode"
  python -m fairmetrics_experiments.run_hpo --dataset students --mode "$mode" --output "results/students_${mode}.json"
done

echo "All runs completed. Results saved under ./results"