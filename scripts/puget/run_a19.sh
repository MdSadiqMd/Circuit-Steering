#!/bin/bash
# A19 on puget: run_a19.sh's stages without its A18 gate (A19 reads cell artifacts, not A18 outputs)
# and without the analysis (it reads A18's hf.json and runs on the Mac after the copy back). CUDA has
# no allocator growth, so cells run in one process (audit.cells_per_process=0).
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
A="device=cuda audit.cells_per_process=0"
run "a19 validate" .venv/bin/python audit/l4_a19.py stage=validate $A
run "a19 base" .venv/bin/python audit/l4_a19.py stage=base $A
until [ -f $CS/.runs_synced ]; do sleep 30; done
for b in b1 b2; do run "a19 cells $b" .venv/bin/python audit/l4_a19.py stage=cells grid.behaviour=$b $A; done
echo "=== done $(date)"
