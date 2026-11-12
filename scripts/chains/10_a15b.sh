#!/usr/bin/env bash
# After 09_a15.sh: CAD on the pillar-set mask for every cell from the kept edges
# (audit/l4_pillarmask.py), then the analysis for both behaviours.
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=scripts/lib.sh
source scripts/lib.sh
export HF_HUB_OFFLINE=1

for b in b1 b2; do
  run "l4 pillar mask $b" uv run python audit/l4_pillarmask.py grid.behaviour="$b"
done
run tables uv run python -m aya_circuits.table
analyze_both
echo "=== done $(date)"
