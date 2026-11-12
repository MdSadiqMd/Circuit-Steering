#!/usr/bin/env bash
# Amendment A6: rebuild the B1 steering cells with fold-disjoint source prompts, re-audit them on all
# layers (HF side, L4 on both circuit definitions) and regenerate the analyses. Runs after
# 04_cellset.sh.
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=scripts/lib.sh
source scripts/lib.sh
: "${HF_TOKEN:?set HF_TOKEN (Cohere licence accepted on the Hub)}"

loop "grid b1 steer" uv run python interventions/run_grid.py grid.behaviour=b1 grid.intervention=steer
loop "audit b1 steer" uv run python audit/run_audit.py grid.behaviour=b1 cells=steer
run "l4 cellset b1 steer" uv run python audit/l4_cellset.py grid.behaviour=b1 cells=steer
analyze_both
echo "=== done $(date)"
