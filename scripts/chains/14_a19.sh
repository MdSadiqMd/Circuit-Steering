#!/usr/bin/env bash
# Amendment A19 (docs/preregistration.md; plan: docs/countering-limitations.md item 3). Stage 0
# measures every quadrature rule on the base model and seals the choice; stage 1 writes per-example
# base scores per seed; stage 2 attributes every main cell of the A19 conditions with the cell
# applied; then the A19 analysis for both behaviours. Runs after 13_a18.sh (the analysis reads
# A18's audits). As run (on a CUDA machine): scripts/puget/run_a19.sh.
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=scripts/lib.sh
source scripts/lib.sh
export HF_HUB_OFFLINE=1

run "a19 validate" uv run python audit/l4_a19.py stage=validate
run "a19 base" uv run python audit/l4_a19.py stage=base
for b in b1 b2; do
  loop "a19 cells $b" uv run python audit/l4_a19.py stage=cells grid.behaviour="$b"
done
for b in b1 b2; do
  run "a19 analyze $b" uv run python audit/a19_analyze.py grid.behaviour="$b"
done
echo "=== done $(date)"
