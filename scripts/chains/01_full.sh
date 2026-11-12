#!/usr/bin/env bash
# Full pipeline: circuits -> faithfulness -> freeze -> grids -> perturbation controls -> circuit
# controls -> audits -> analysis. Every stage skips cached outputs; rerun after any interruption.
# Needs HF_TOKEN (the model is downloaded on first use).
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=scripts/lib.sh
source scripts/lib.sh
: "${HF_TOKEN:?set HF_TOKEN (Cohere licence accepted on the Hub)}"

for t in b1_en b2_en ioi langid_en; do
  for s in 0 1 2; do
    bs=4
    [[ $t == langid_en || $t == b2_en ]] && bs=2
    run "discover $t $s" \
      uv run python -m aya_circuits.discover task="$t" seed="$s" discovery.batch_size="$bs"
  done
done
for t in b1_hi b1_ar b1_fr b1_sw; do
  run "discover $t 0" uv run python -m aya_circuits.discover task="$t" seed=0
done
for t in b1_en b2_en ioi langid_en; do
  for s in 0 1 2; do
    run "faithfulness $t $s" uv run python -m aya_circuits.faithfulness task="$t" seed="$s"
  done
done
for t in b1_hi b1_ar b1_fr b1_sw; do
  run "faithfulness $t 0" uv run python -m aya_circuits.faithfulness task="$t" seed=0
done
run overlap uv run python -m aya_circuits.overlap
# refuses once docs/preregistration.md is frozen, which is the expected outcome of any rerun
try freeze uv run python checks/freeze_prereg.py
for b in b1 b2; do
  for i in steer lora edit; do
    loop "grid $b $i" \
      uv run python interventions/run_grid.py grid.behaviour="$b" grid.intervention="$i"
  done
done
run perturb uv run python -m interventions.perturb --all
for t in b1_en b2_en ioi langid_en; do
  for s in 0 1 2; do
    run "controls $t $s" uv run python -m aya_circuits.random_baseline task="$t" seed="$s"
  done
done
for t in b1_hi b1_ar b1_fr b1_sw; do
  run "controls $t 0" uv run python -m aya_circuits.random_baseline task="$t" seed=0
done
run tables uv run python -m aya_circuits.table
for b in b1 b2; do
  loop "audit $b" uv run python audit/run_audit.py grid.behaviour="$b"
done
analyze_both
echo "=== done $(date)"
