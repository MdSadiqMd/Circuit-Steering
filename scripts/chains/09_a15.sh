#!/usr/bin/env bash
# Amendment A15 (2026-11-07): B1 LoRA and steering cells on the joint-ranked (faithful) circuit,
# with count-matched random and inverse sets; full audit on every layer, both L4 mask definitions,
# ripple, arbitration; then the L4 power check (audit/l4_power.py), tables and analysis. Runs after
# 08_gaps_final.sh.
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=scripts/lib.sh
source scripts/lib.sh
export HF_HUB_OFFLINE=1

joint="interventions.conditions=[circuit_joint,random_joint,inverse_joint]"
loop "grid b1 lora joint" uv run python interventions/run_grid.py \
  grid.behaviour=b1 grid.intervention=lora "$joint"
loop "grid b1 steer joint" uv run python interventions/run_grid.py \
  grid.behaviour=b1 grid.intervention=steer "$joint"
loop "audit b1 lora+steer" uv run python audit/run_audit.py grid.behaviour=b1 cells=lora,steer
run "l4 cellset b1" uv run python audit/l4_cellset.py grid.behaviour=b1 cells=lora,steer
run ripple uv run python audit/ripple.py
run arbitration uv run python audit/arbitration.py grid.behaviour=b1
run "l4 power" uv run python audit/l4_power.py
run tables uv run python -m aya_circuits.table
run "analyze b1" uv run python audit/analyze.py grid.behaviour=b1
echo "=== done $(date)"
