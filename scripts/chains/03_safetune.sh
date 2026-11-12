#!/usr/bin/env bash
# Amendment A4: SafeTune refusal-direction baseline (B2 condition safetune_rd): build, audit (new
# steering cells only; cached cells are skipped), analysis for both behaviours (the
# judge-robustness section). Runs after 02_tail.sh.
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=scripts/lib.sh
source scripts/lib.sh
: "${HF_TOKEN:?set HF_TOKEN (Cohere licence accepted on the Hub)}"

run build uv run python interventions/refusal_direction.py
loop "audit b2" uv run python audit/run_audit.py grid.behaviour=b2 cells=steer
analyze_both
echo "=== done $(date)"
