#!/usr/bin/env bash
# Amendment A16 (2026-11-07): the built-in IOI bystander is mis-specified on this tokenizer (metric
# read at the answer token, double BOS, misaligned pairs; checks/check_ioi_position.py). Discover
# and evaluate `ioi_fixed` (data/ioi_fixed.py) and the length-matched refusal task `b2m_en`
# (data/b2_matched.py) on three seeds (pillars, size curve, stability, same-path controls, per-type
# cellset), then re-attribute `ioi_fixed` on every audited cell of both behaviours
# (audit/l4_task.py) and re-analyse. Runs after 11_a15c.sh.
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=scripts/lib.sh
source scripts/lib.sh
export HF_HUB_OFFLINE=1

for s in 0 1 2; do
  run "discover ioi_fixed $s" uv run python -m aya_circuits.discover task=ioi_fixed seed="$s"
  run "faithfulness ioi_fixed $s" uv run python -m aya_circuits.faithfulness \
    task=ioi_fixed seed="$s" "faithfulness.curve_tasks=null"
  run "controls ioi_fixed $s" uv run python -m aya_circuits.random_baseline task=ioi_fixed seed="$s"
  run "cellset ioi_fixed $s" uv run python -m aya_circuits.cellset task=ioi_fixed seed="$s"
done
for s in 0 1 2; do
  run "discover b2m_en $s" uv run python -m aya_circuits.discover \
    task=b2m_en seed="$s" discovery.batch_size=2
  run "faithfulness b2m_en $s" uv run python -m aya_circuits.faithfulness \
    task=b2m_en seed="$s" "faithfulness.curve_tasks=null"
  run "controls b2m_en $s" uv run python -m aya_circuits.random_baseline task=b2m_en seed="$s"
  run "cellset b2m_en $s" uv run python -m aya_circuits.cellset task=b2m_en seed="$s"
done
run overlap uv run python -m aya_circuits.overlap
run tables uv run python -m aya_circuits.table
for b in b1 b2; do
  run "l4 ioi_fixed $b" uv run python audit/l4_task.py grid.behaviour="$b" task=ioi_fixed
done
analyze_both
echo "=== done $(date)"
