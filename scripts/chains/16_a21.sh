#!/usr/bin/env bash
# Amendment A21 (docs/preregistration.md; plan: docs/countering-limitations.md item 2): MEMIT at 17
# edit sites on the three B1 folds, each uncapped, norm-capped and as norm-matched noise; one site
# rebuilt for the reproducibility check; the audit of layers L1/L2/L3/L5 (hf side); the analysis.
# Writes runs/a21/ and runs/a21_repro/. Extra key=value arguments go to every model stage, e.g.
# `scripts/chains/16_a21.sh device=cuda audit.cells_per_process=0`.
# As run (CUDA, two audit shards, plus the Mac-vs-puget check): scripts/puget/run_a21.sh.
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=scripts/lib.sh
source scripts/lib.sh
export HF_HUB_OFFLINE=1

inputs() {
  mkdir -p runs/a21/interventions
  [[ -e runs/a21/interventions/b2_split.json ]] ||
    cp runs/interventions/b2_split.json runs/a21/interventions/
}

run inputs inputs
run edits uv run python interventions/run_a21.py out_dir=runs/a21 "$@"
run "repro memit" uv run python interventions/run_a21.py \
  "a21.sites=[10]" "a21.folds=[0]" out_dir=runs/a21_repro "$@"
run "check repro memit" uv run python checks/check_a20_a21_repro.py memit
loop "audit hf" uv run python audit/run_audit.py \
  grid.behaviour=b1 cells=edit side=hf out_dir=runs/a21 "$@"
run analyze uv run python audit/a21_analyze.py out_dir=runs/a21
echo "=== done $(date)"
