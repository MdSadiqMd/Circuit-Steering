#!/usr/bin/env bash
# Brief-gap closure, everything after the circuits part (third audit, 2026-11-06): the L4 base-edge
# rescale fix (audit/l4_rescore.py); §7.4 ROME smoke, §11 profiler + determinism, §3.3 B1
# back-translation, §12.5 steering direction check; H5 causal tracing + MEMIT cells at the traced
# layer (A8) with perturbation controls and full audit; L1 one-hop ripple (B1), L5 relearning probe
# (B2), L5 evidence arbitration (both); tables and analysis. The four stand-alone checks log a
# failure and the chain moves on; every other stage stops it. Runs after 06_gaps_circuits.sh.
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=scripts/lib.sh
source scripts/lib.sh
export HF_HUB_OFFLINE=1

# third audit: B1 seed-0 base edges of b1_en/b2_en/ioi were cached at batch 4 (half scale)
run "l4 rescore b1 s0" uv run python audit/l4_rescore.py \
  grid.behaviour=b1 disc_seed=0 "tasks=[b1_en,b2_en,ioi]"
try "rome smoke" uv run python checks/check_rome_smoke.py device=cpu
try "profile lora" uv run python checks/profile_lora.py
try "b1 back-translation" uv run python checks/check_b1_backtranslation.py
try "steering direction" uv run python checks/check_steer_direction.py
run "causal trace" uv run python interventions/causal_trace.py
loop "grid b1 edit causal_trace" uv run python interventions/run_grid.py \
  grid.behaviour=b1 grid.intervention=edit \
  "interventions.conditions=[causal_trace]" "interventions.editing.extra_conditions=[]"
run perturb uv run python -m interventions.perturb --all
loop "audit b1 edit" uv run python audit/run_audit.py grid.behaviour=b1 cells=edit
run "l4 cellset b1 edit" uv run python audit/l4_cellset.py grid.behaviour=b1 cells=edit
run ripple uv run python audit/ripple.py
run relearn uv run python audit/relearn.py
run arbitration uv run python audit/arbitration.py
run tables uv run python -m aya_circuits.table
analyze_both
echo "=== done $(date)"
