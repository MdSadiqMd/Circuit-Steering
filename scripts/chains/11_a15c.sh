#!/usr/bin/env bash
# A15 follow-up after 10_a15b.sh: the first run of 09_a15.sh stopped the steering grid after its
# first three cells (exit 3 was only re-invoked for LoRA), so the random_joint / inverse_joint
# steering cells were built here. Builds them, audits them (HF side, L4, per-type and pillar masks,
# ripple, arbitration), regenerates tables and analyses.
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=scripts/lib.sh
source scripts/lib.sh
export HF_HUB_OFFLINE=1

joint="interventions.conditions=[circuit_joint,random_joint,inverse_joint]"
loop "grid b1 steer joint" uv run python interventions/run_grid.py \
  grid.behaviour=b1 grid.intervention=steer "$joint"
loop "audit b1 steer" uv run python audit/run_audit.py grid.behaviour=b1 cells=steer
run "l4 cellset b1 steer" uv run python audit/l4_cellset.py grid.behaviour=b1 cells=steer
run ripple uv run python audit/ripple.py
run arbitration uv run python audit/arbitration.py grid.behaviour=b1
run "l4 pillar mask b1" uv run python audit/l4_pillarmask.py grid.behaviour=b1
run "l4 shared b1" uv run python audit/l4_shared.py grid.behaviour=b1
run tables uv run python -m aya_circuits.table
analyze_both
echo "=== done $(date)"
