#!/usr/bin/env bash
# Discover and evaluate every circuit (targets and bystanders) for every seed; a subset of
# 01_full.sh. TASKS and SEEDS narrow it; extra key=value arguments go to every stage, e.g.
#   TASKS="b1_en" SEEDS="0" scripts/chains/circuits.sh discovery.batch_size=2
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=scripts/lib.sh
source scripts/lib.sh
: "${HF_TOKEN:?set HF_TOKEN (Cohere licence accepted on the Hub)}"

read -r -a tasks <<<"${TASKS:-b1_en b2_en ioi langid_en b1_hi b1_ar b1_fr b1_sw}"
read -r -a seeds <<<"${SEEDS:-0 1 2}"
for t in "${tasks[@]}"; do
  for s in "${seeds[@]}"; do
    run "discover $t seed $s" uv run python -m aya_circuits.discover task="$t" seed="$s" "$@"
  done
done
for t in "${tasks[@]}"; do
  for s in "${seeds[@]}"; do
    run "faithfulness $t seed $s" \
      uv run python -m aya_circuits.faithfulness task="$t" seed="$s" "$@"
  done
done
echo "=== done $(date)"
