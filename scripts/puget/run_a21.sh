#!/bin/bash
# A21 on puget (docs/preregistration.md A21): MEMIT at every editable layer on the three B1 folds,
# capped and noise cells, one site rebuilt for the reproducibility check, layer 0 against the Mac's
# MEMIT circuit cells, then the hf_layers audit of every cell. Waits for A20 to finish.
# shellcheck source=scripts/puget/env.sh
# shellcheck disable=SC2086  # $A, $SMALL and $R are split into key=value words on purpose
source /dev/shm/cs/env.sh
cd /dev/shm/cs/circuit-steering || exit 1
set -o pipefail
run() {
  echo "=== $1 $(date)"
  shift
  "$@" 2>&1 | grep --line-buffered -v "Redirects\|Loading weights\|it/s\]\|Warning\|warn(" || {
    echo "=== FAILED"
    exit 1
  }
}
until grep -q "^=== done" logs_puget/a20.log; do
  grep -q "^=== FAILED" logs_puget/a20.log && exit 1
  sleep 60
done
sha256sum docs/preregistration.md
cp docs/preregistration.md logs_puget/preregistration_at_a21_start.md
# Two audit shards share the GPU at 15 GB each (30 GB together); each logs to its own file.
shards() {
  local label=$1
  shift
  echo "=== $label $(date)"
  local pids=()
  for i in 0 1; do
    CS_GPU_CAP_GB=15 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@" shard=$i/2 >"logs_puget/${label// /_}_shard$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || {
    echo "=== FAILED $label"
    exit 1
  }; done
  grep -h " hf [0-9]*s" logs_puget/${label// /_}_shard*.log | tail -3
}
A="device=cuda audit.cells_per_process=0 out_dir=runs_a21"
mkdir -p runs_a21/interventions && cp runs/interventions/b2_split.json runs_a21/interventions/
run "edits" .venv/bin/python interventions/run_a21.py $A
run "repro memit" .venv/bin/python interventions/run_a21.py "a21.sites=[10]" "a21.folds=[0]" device=cuda out_dir=runs_a21_repro
run "check repro memit" .venv/bin/python checks/check_a20_a21_repro.py memit
run "check vs mac" .venv/bin/python checks/check_a20_a21_repro.py mac
shards "a21 audit hf" .venv/bin/python audit/run_audit.py grid.behaviour=b1 cells=edit side=hf $A
echo "=== done $(date)"
