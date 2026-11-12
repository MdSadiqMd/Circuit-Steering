#!/usr/bin/env bash
# After 01_full.sh: the B1 LoRA cells that ran out of memory before the label-position loss
# (interventions/lora.py), their perturbation controls, the faithfulness run that ran out of memory
# at batch 4 (langid_en seed 1) and its control, then both audits again (new B1 LoRA cells on the
# HF side; the whole L4 side, which ran out of memory at batch 4), tables, analysis.
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=scripts/lib.sh
source scripts/lib.sh
: "${HF_TOKEN:?set HF_TOKEN (Cohere licence accepted on the Hub)}"

loop "grid b1 lora" uv run python interventions/run_grid.py grid.behaviour=b1 grid.intervention=lora
run perturb uv run python -m interventions.perturb --all
run "faithfulness langid_en 1 bs2" uv run python -m aya_circuits.faithfulness \
  task=langid_en seed=1 faithfulness.batch_size=2
run "controls langid_en 1 bs2" uv run python -m aya_circuits.random_baseline \
  task=langid_en seed=1 faithfulness.batch_size=2
run tables uv run python -m aya_circuits.table
for b in b1 b2; do # HF side cached; L4 side at batch 2
  loop "audit $b" uv run python audit/run_audit.py grid.behaviour="$b"
done
analyze_both
echo "=== done $(date)"
