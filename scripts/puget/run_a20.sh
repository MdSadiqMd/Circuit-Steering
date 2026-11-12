#!/bin/bash
# A20 on puget (docs/preregistration.md A20): certify, build B1 steering + LoRA cells, rebuild one
# LoRA cell for the reproducibility check, audit L1/L2/L3/L5 (hf_layers) and L4 under A19's rule.
# Waits for A19 to finish (the GPU runs one job at a time and A20's L4 reuses A19's rule and base).
# shellcheck source=scripts/puget/env.sh
# shellcheck disable=SC2086  # $A, $SMALL and $R are split into key=value words on purpose
source /dev/shm/cs/env.sh
cd /dev/shm/cs/circuit-steering || exit 1
set -o pipefail
run() {
  echo "=== $1 $(date)"
  shift
  "$@" 2>&1 | grep --line-buffered -v "Redirects\|Loading weights\|it/s\]\|Warning\|warn(" || {
    echo "=== FAILED"
    exit 1
  }
}
until grep -q "^=== done" logs_puget/a19.log; do
  grep -q "^=== FAILED" logs_puget/a19.log && exit 1
  sleep 60
done
sha256sum docs/preregistration.md
cp docs/preregistration.md logs_puget/preregistration_at_a20_start.md
# Two audit shards share the GPU at 15 GB each (30 GB together); each logs to its own file.
shards() {
  local label=$1
  shift
  echo "=== $label $(date)"
  local pids=()
  for i in 0 1; do
    CS_GPU_CAP_GB=15 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@" shard=$i/2 >"logs_puget/${label// /_}_shard$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || {
    echo "=== FAILED $label"
    exit 1
  }; done
  grep -h " hf [0-9]*s" logs_puget/${label// /_}_shard*.log | tail -3
}
A="device=cuda audit.cells_per_process=0 out_dir=runs_a20"
R=$CS/tmp/a19_repro
rm -rf $R
mkdir -p $R/interventions/b1/lora $R/audit
ln -s $PWD/runs/circuits $R/circuits
ln -s $PWD/runs/audit/a19 $R/audit/a19
ln -s $PWD/runs/interventions/b1/lora/circuit_s0 $R/interventions/b1/lora/circuit_s0
run "a19 repro cell" .venv/bin/python audit/l4_a19.py stage=cells grid.behaviour=b1 cells=lora device=cuda out_dir=$R
run "check a19 repro" .venv/bin/python checks/check_a19_repro.py $R
for t in b1_en b2_en langid_en ioi_fixed; do
  mkdir -p runs_a20/circuits/$t
  cp -rn runs/circuits/$t/seed* runs_a20/circuits/$t/
done
mkdir -p runs_a20/interventions runs_a20/audit/a19 && cp runs/interventions/b2_split.json runs_a20/interventions/
cp -rn runs/audit/a19/rule.json runs/audit/a19/base runs_a20/audit/a19/
for s in 0 1 2; do run "certify s$s" .venv/bin/python -m aya_circuits.certify task=b1_en seed=$s $A; done
run "cells steer" .venv/bin/python interventions/run_a20.py grid.intervention=steer $A
run "cells lora" .venv/bin/python interventions/run_a20.py grid.intervention=lora $A
mkdir -p runs_a20_repro/circuits && cp -r runs_a20/circuits/b1_en runs_a20_repro/circuits/
run "repro lora" .venv/bin/python interventions/run_a20.py grid.intervention=lora "only=[cert_s0]" device=cuda out_dir=runs_a20_repro
run "check repro lora" .venv/bin/python checks/check_a20_a21_repro.py lora
shards "a20 audit hf" .venv/bin/python audit/run_audit.py grid.behaviour=b1 cells=steer,lora side=hf $A
run "audit l4 (A19 rule)" .venv/bin/python audit/a20_l4.py stage=cells grid.behaviour=b1 cells=steer,lora $A
echo "=== done $(date)"
