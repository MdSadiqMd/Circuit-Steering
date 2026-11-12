#!/usr/bin/env bash
# Amendment A20 (docs/preregistration.md; plan: docs/countering-limitations.md item 1): a certified
# circuit for b1_en on each seed, the B1 steering and LoRA cells on it and on its controls, one
# LoRA cell rebuilt for the reproducibility check, the audit of layers L1/L2/L3/L5 (hf side) and L4
# under A19's rule, then the analysis. Writes runs/a20/ and runs/a20_repro/; needs the main study's
# circuits and A19's rule and base (scripts/chains/14_a19.sh). Extra key=value arguments go to every
# model stage, e.g. `scripts/chains/15_a20.sh device=cuda audit.cells_per_process=0`.
# As run (CUDA, two audit shards): scripts/puget/run_a20.sh.
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=scripts/lib.sh
source scripts/lib.sh
export HF_HUB_OFFLINE=1

# Inputs shared with the main study, copied once; anything already in runs/a20/ is kept.
inputs() {
  local t d
  for t in b1_en b2_en langid_en ioi_fixed; do
    mkdir -p "runs/a20/circuits/$t"
    for d in "runs/circuits/$t"/seed*; do
      [[ -e runs/a20/circuits/$t/${d##*/} ]] || cp -R "$d" "runs/a20/circuits/$t/"
    done
  done
  mkdir -p runs/a20/interventions runs/a20/audit/a19
  [[ -e runs/a20/interventions/b2_split.json ]] ||
    cp runs/interventions/b2_split.json runs/a20/interventions/
  for d in rule.json base; do
    [[ -e runs/a20/audit/a19/$d ]] || cp -R "runs/audit/a19/$d" runs/a20/audit/a19/
  done
}

repro_inputs() {
  mkdir -p runs/a20_repro/circuits
  [[ -e runs/a20_repro/circuits/b1_en ]] || cp -R runs/a20/circuits/b1_en runs/a20_repro/circuits/
}

run inputs inputs
for s in 0 1 2; do
  run "certify s$s" uv run python -m aya_circuits.certify task=b1_en seed="$s" out_dir=runs/a20 "$@"
done
run "cells steer" uv run python interventions/run_a20.py grid.intervention=steer out_dir=runs/a20 "$@"
run "cells lora" uv run python interventions/run_a20.py grid.intervention=lora out_dir=runs/a20 "$@"
run "repro inputs" repro_inputs
run "repro lora" uv run python interventions/run_a20.py \
  grid.intervention=lora "only=[cert_s0]" out_dir=runs/a20_repro "$@"
run "check repro lora" uv run python checks/check_a20_a21_repro.py lora
loop "audit hf" uv run python audit/run_audit.py \
  grid.behaviour=b1 cells=steer,lora side=hf out_dir=runs/a20 "$@"
loop "audit l4 (A19 rule)" uv run python audit/a20_l4.py \
  stage=cells grid.behaviour=b1 cells=steer,lora out_dir=runs/a20 "$@"
run analyze uv run python audit/a20_analyze.py out_dir=runs/a20
echo "=== done $(date)"
