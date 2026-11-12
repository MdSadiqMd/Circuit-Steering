#!/usr/bin/env bash
# Third-audit checks that need the GPU after 07_gaps_followup.sh: HF <-> TransformerLens
# equivalence incl. k/v (GQA) and gate/up LoRA deltas, then the final tables, analysis and the data
# manifest.
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=scripts/lib.sh
source scripts/lib.sh
export HF_HUB_OFFLINE=1

run "apply equivalence" uv run python checks/check_apply_equivalence.py
run tables uv run python -m aya_circuits.table
analyze_both
run "data manifest" uv run python data/manifest.py
echo "=== done $(date)"
