#!/usr/bin/env bash
# Amendment A5 (second audit, B1/B2): pillars, controls and necessity on the pre-registered
# (per-type) circuit for every discovered circuit, then the L4 side recomputed on that definition for
# every audited cell (edge scores kept), then the analyses. Runs after 03_safetune.sh.
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=scripts/lib.sh
source scripts/lib.sh
: "${HF_TOKEN:?set HF_TOKEN (Cohere licence accepted on the Hub)}"

for t in b1_en b2_en; do
  for s in 0 1 2; do
    run "cellset $t $s" uv run python -m aya_circuits.cellset task="$t" seed="$s"
  done
done
for s in 0 1 2; do
  run "cellset ioi $s" uv run python -m aya_circuits.cellset task=ioi seed="$s"
done
for s in 0 1 2; do # batch 4 ran out of memory on seed 1 (long prompts), as in 02_tail.sh
  run "cellset langid_en $s" \
    uv run python -m aya_circuits.cellset task=langid_en seed="$s" faithfulness.batch_size=2
done
for t in b1_hi b1_ar b1_fr b1_sw; do
  run "cellset $t 0" uv run python -m aya_circuits.cellset task="$t" seed=0
done
run tables uv run python -m aya_circuits.table
for b in b1 b2; do
  run "l4 cellset $b" uv run python audit/l4_cellset.py grid.behaviour="$b"
done
analyze_both
echo "=== done $(date)"
