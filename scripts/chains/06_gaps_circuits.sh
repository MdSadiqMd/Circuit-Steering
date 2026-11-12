#!/usr/bin/env bash
# Brief-gap closure, circuits part (third audit, 2026-11-06): seeds 1 and 2 for the non-English fact
# circuits (brief §7.4 "3 seeds" for every circuit), and the stability pillar + the 1..30 % size
# curve for every circuit that lacked them (brief §12.1). faithfulness.py runs in fill mode on
# existing files: numbers already there are kept, only absent pieces are computed. The per-type
# cellset evaluation is not run for the new non-English seeds: those circuits are never intervened
# on, so it adds about 3 h for a table row nothing depends on. Runs after 05_a6.sh.
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=scripts/lib.sh
source scripts/lib.sh
export HF_HUB_OFFLINE=1

for t in b1_hi b1_ar b1_fr b1_sw; do
  for s in 1 2; do
    run "discover $t $s" uv run python -m aya_circuits.discover task="$t" seed="$s"
  done
done
for t in b1_en b2_en ioi langid_en b1_hi b1_ar b1_fr b1_sw; do
  for s in 0 1 2; do
    bs=4
    [[ $t == langid_en ]] && bs=2
    run "faithfulness $t $s" uv run python -m aya_circuits.faithfulness task="$t" seed="$s" \
      "faithfulness.curve_tasks=null" faithfulness.batch_size="$bs"
  done
done
for t in b1_hi b1_ar b1_fr b1_sw; do
  for s in 1 2; do
    run "controls $t $s" uv run python -m aya_circuits.random_baseline task="$t" seed="$s"
  done
done
run overlap uv run python -m aya_circuits.overlap
run tables uv run python -m aya_circuits.table
echo "=== done $(date)"
